from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import torch
from PIL import Image
from transformers import AutoProcessor

from vlm_robot_eval.core.action_parser import (
    build_action_fallback,
    parse_action_json,
    parse_objects_json,
)


QWEN_MODEL_PATH = "/mnt/data/modelscope_cache/hub/Qwen/Qwen2.5-VL-3B-Instruct"


PROMPT_TEMPLATE = """Return one JSON object only. No explanation, no markdown.

instruction: {instruction}
candidate_targets: {objects}

schema:
{{"action_sequence":[{{"action":"move_to","target":"cup_1"}},{{"action":"grasp","target":"cup_1"}}]}}

constraints:
- action in [move_to, grasp, release, push, rotate]
- target must come from candidate_targets
- if impossible, output {{"action_sequence":[]}}
"""


OBJECTS_PROMPT = """Carefully examine this image and identify the scene type and ONLY the top 3 most salient objects.

Output strict JSON with NO extra text:
{{
  "objects": ["object1", "object2", "object3"],
  "scene_type": "one of: dining_table, office, kitchen, indoor, unknown"
}}

Rules:
1. List at most 3 objects as lowercase English nouns; do NOT hallucinate
2. scene_type must be one of: dining_table, office, kitchen, indoor, unknown
3. If uncertain about an object, omit it
4. If nothing is identifiable, return: {{"objects": [], "scene_type": "unknown"}}
"""


def _resolve_max_memory() -> Dict[Any, str] | None:
    if not torch.cuda.is_available():
        return None
    try:
        total_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        ratio = float(os.getenv("VLM_GPU_MEMORY_RATIO", "0.95"))
        reserve_gib = float(os.getenv("VLM_GPU_RESERVE_GB", "0.25"))
        gpu_gib = max(1, int(total_gib * ratio - reserve_gib))
    except Exception:
        return None
    cpu_mem = os.getenv("VLM_CPU_MAX_MEMORY", "64GiB")
    return {0: f"{gpu_gib}GiB", "cpu": cpu_mem}


class QwenVLModel:
    name = "Qwen2.5-VL-3B-Instruct"

    def __init__(
        self,
        model_path: str = QWEN_MODEL_PATH,
        dtype: torch.dtype = torch.float16,
        device_map: str = "auto",
        max_new_tokens: int = 256,
        deterministic: bool = False,
    ):
        self.model_path = model_path
        self.dtype = dtype
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self.deterministic = bool(deterministic)

        offload_dir = os.getenv("VLM_OFFLOAD_DIR", "/tmp/vlm_offload")
        os.makedirs(offload_dir, exist_ok=True)

        dm_raw = str(device_map).strip().lower() if device_map is not None else ""
        resolved_device_map: Any = None if dm_raw in {"", "none", "null", "off"} else device_map

        load_kwargs: Dict[str, Any] = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        if resolved_device_map is not None:
            load_kwargs["device_map"] = resolved_device_map

        if resolved_device_map == "auto":
            mm = _resolve_max_memory()
            if mm:
                load_kwargs["max_memory"] = mm
            load_kwargs["offload_buffers"] = True
            load_kwargs["offload_folder"] = offload_dir

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                **load_kwargs,
            )
        except Exception:
            from transformers import AutoModelForVision2Seq

            self.model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                **load_kwargs,
            )

        self.processor = AutoProcessor.from_pretrained(model_path)

        # Avoid noisy warning when deterministic decoding is used but model config carries temperature.
        try:
            if getattr(self.model, "generation_config", None) is not None and self.deterministic:
                self.model.generation_config.temperature = None
                self.model.generation_config.top_p = None
                self.model.generation_config.top_k = None
        except Exception:
            pass

    def _generate(self, image: Image.Image, prompt: str) -> Dict[str, Any]:
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        start = time.perf_counter()

        text_out: Optional[str] = None
        with torch.inference_mode():
            if hasattr(self.processor, "apply_chat_template"):
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                chat_text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.processor(
                    text=chat_text,
                    images=image,
                    return_tensors="pt",
                )
            else:
                inputs = self.processor(text=prompt, images=image, return_tensors="pt")

            if hasattr(self.model, "device"):
                device = self.model.device
                inputs = {k: v.to(device) for k, v in inputs.items()}

            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens": self.max_new_tokens,
            }
            if self.deterministic:
                gen_kwargs["do_sample"] = False
            else:
                gen_kwargs.update({"do_sample": True, "temperature": 0.2, "top_p": 0.9})

            if self.processor.tokenizer.pad_token_id is not None:
                gen_kwargs["pad_token_id"] = self.processor.tokenizer.pad_token_id
            elif self.processor.tokenizer.eos_token_id is not None:
                gen_kwargs["pad_token_id"] = self.processor.tokenizer.eos_token_id
            if self.processor.tokenizer.eos_token_id is not None:
                gen_kwargs["eos_token_id"] = self.processor.tokenizer.eos_token_id

            generated_ids = self.model.generate(**inputs, **gen_kwargs)

            input_len = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else 0
            # decoder-only models return [prompt + generated], but encoder-decoder may return generated-only
            if input_len > 0 and generated_ids.shape[-1] > input_len:
                decode_ids = generated_ids[:, input_len:]
            else:
                decode_ids = generated_ids

            if getattr(decode_ids, "numel", lambda: 0)() == 0:
                text_out = ""
            elif hasattr(self.processor, "batch_decode"):
                text_out = self.processor.batch_decode(decode_ids, skip_special_tokens=True)[0]
            else:
                text_out = self.processor.tokenizer.batch_decode(
                    decode_ids, skip_special_tokens=True
                )[0]

        if use_cuda:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        end = time.perf_counter()

        mem = int(torch.cuda.max_memory_allocated()) if use_cuda else 0
        return {
            "raw_text": text_out or "",
            "inference_time": float(end - start),
            "gpu_memory": mem,
        }

    def _build_prompt(self, instruction: str, objects: list | None = None) -> str:
        obj_str = ", ".join(str(o) for o in objects) if objects else "unknown"
        return PROMPT_TEMPLATE.format(instruction=instruction, objects=obj_str)

    def infer(self, image: Image.Image, instruction: str, objects: list | None = None) -> Dict[str, Any]:
        obj_list = objects or []
        prompt = self._build_prompt(instruction, objects=obj_list)
        gen = self._generate(image=image, prompt=prompt)
        parsed = parse_action_json(gen.get("raw_text", ""))
        seq = parsed.get("action_sequence", [])
        used_fallback = False
        if not seq:
            used_fallback = True
            seq = build_action_fallback(instruction=instruction, objects=obj_list)
        return {
            "action_sequence": seq,
            "used_fallback": used_fallback,
            **gen,
        }

    def infer_objects(self, image: Image.Image) -> Dict[str, Any]:
        gen = self._generate(image=image, prompt=OBJECTS_PROMPT)
        parsed = parse_objects_json(gen.get("raw_text", ""))
        return {
            "objects": parsed.get("objects", []),
            "scene_type": parsed.get("scene_type", ""),
            **gen,
        }
