from __future__ import annotations

import os
import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from vlm_robot_eval.core.action_parser import (
    build_action_fallback,
    parse_action_json,
    parse_objects_json,
)


SMOL_MODEL_PATH = "/mnt/data/modelscope_cache/hub/HuggingFaceTB/SmolVLM-256M-Instruct"


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


PROMPT_TEMPLATE = """只返回一个 JSON 对象，不要任何解释/前后缀/Markdown。

instruction: {instruction}
candidate_targets: {objects}

schema:
{{"action_sequence":[{{"action":"move_to","target":"cup_1"}},{{"action":"grasp","target":"cup_1"}}]}}

constraints:
- action must be one of: move_to, grasp, release, push, rotate
- target must be chosen from candidate_targets
- if task cannot be executed, return: {{"action_sequence":[]}}
"""


OBJECTS_PROMPT = """Carefully examine this image and output strict JSON only. No explanation, no markdown.

Required format:
{"objects":["object1","object2","object3"],"scene_type":"LABEL"}

Rules:
1) objects: list the top-3 most salient physical objects visible (lowercase English nouns)
2) scene_type MUST be exactly one of: dining_table, office, kitchen, indoor, unknown
   - dining_table: table with food/dishes/meals
   - office: desk, computer, keyboard, monitor
   - kitchen: kitchen appliances, stove, sink, countertop
   - indoor: other indoor room (bedroom, living room, shelf, etc.)
   - unknown: cannot determine
3) Output ONLY the JSON object, nothing else.
4) If uncertain, use: {"objects":[],"scene_type":"unknown"}
"""

SCENE_ONLY_PROMPT = """Look at this image carefully.
Return ONLY one word from this list (no punctuation, no explanation):
dining_table
office
kitchen
indoor
unknown

Your answer:"""

_VALID_SCENES = {"dining_table", "office", "kitchen", "indoor", "unknown"}

# Extended alias map for scene normalization fallback
_SCENE_ALIAS_MAP: dict = {
    "dining": "dining_table",
    "diningtable": "dining_table",
    "dinner": "dining_table",
    "table": "dining_table",
    "restaurant": "dining_table",
    "cafeteria": "dining_table",
    "kitchen": "kitchen",
    "cook": "kitchen",
    "cooking": "kitchen",
    "counter": "kitchen",
    "stove": "kitchen",
    "sink": "kitchen",
    "office": "office",
    "desk": "office",
    "workspace": "office",
    "computer": "office",
    "monitor": "office",
    "keyboard": "office",
    "laptop": "office",
    "lab": "office",
    "indoor": "indoor",
    "room": "indoor",
    "bedroom": "indoor",
    "living": "indoor",
    "shelf": "indoor",
    "hallway": "indoor",
    "corridor": "indoor",
    "unknown": "unknown",
    "none": "unknown",
    "other": "unknown",
}

# Object-keyword → scene heuristic (expanded)
_OBJ_SCENE_MAP: dict = {
    "laptop": "office", "keyboard": "office", "mouse": "office", "monitor": "office",
    "computer": "office", "desk": "office", "pen": "office", "pencil": "office",
    "book": "office", "paper": "office", "notebook": "office",
    "bottle": "kitchen", "bowl": "kitchen", "plate": "kitchen", "fork": "kitchen",
    "spoon": "kitchen", "cup": "kitchen", "knife": "kitchen", "mug": "kitchen",
    "pan": "kitchen", "pot": "kitchen", "glass": "kitchen", "can": "kitchen",
    "apple": "kitchen", "orange": "kitchen", "banana": "kitchen", "food": "kitchen",
    "pizza": "dining_table", "burger": "dining_table", "sandwich": "dining_table",
    "table": "dining_table", "chair": "dining_table", "wine": "dining_table",
    "sofa": "indoor", "couch": "indoor", "bed": "indoor", "pillow": "indoor",
    "shelf": "indoor", "remote": "indoor", "tv": "indoor", "lamp": "indoor",
}


class SmolVLMModel:
    name = "SmolVLM-256M-Instruct"

    def __init__(
        self,
        model_path: str = SMOL_MODEL_PATH,
        dtype: torch.dtype = torch.float16,
        device_map: str = "auto",
        max_new_tokens: int = 128,
    ):
        self.model_path = model_path
        self.dtype = dtype
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens

        offload_dir = os.getenv("VLM_OFFLOAD_DIR", "/tmp/vlm_offload")
        os.makedirs(offload_dir, exist_ok=True)

        load_kwargs: Dict[str, Any] = {
            "torch_dtype": dtype,
            "device_map": device_map,
            "low_cpu_mem_usage": True,
        }
        if device_map == "auto":
            mm = _resolve_max_memory()
            if mm:
                load_kwargs["max_memory"] = mm
            load_kwargs["offload_buffers"] = True
            load_kwargs["offload_folder"] = offload_dir

        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            **load_kwargs,
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

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
                inputs = self.processor(text=chat_text, images=image, return_tensors="pt")
            else:
                inputs = self.processor(text=prompt, images=image, return_tensors="pt")

            if hasattr(self.model, "device"):
                device = self.model.device
                inputs = {k: v.to(device) for k, v in inputs.items()}

            # Use greedy decoding to avoid long-tail sampling hangs on small models.
            # Also set pad/eos tokens to prevent infinite generation.
            gen_kwargs: dict = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": False,
            }
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

    def _build_prompt(self, instruction: str, objects: list = None) -> str:
        obj_str = ", ".join(str(o) for o in objects) if objects else "（根据图像判断）"
        return PROMPT_TEMPLATE.format(instruction=instruction, objects=obj_str)

    def infer(self, image: Image.Image, instruction: str, objects: list = None) -> Dict[str, Any]:
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

    def _extract_scene_label(self, text: str) -> str:
        """Extract scene label from free-form text with alias mapping + regex fallback."""
        s = str(text or "").strip().lower().replace("-", "_")
        s = re.sub(r"[^a-z0-9_ ]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        if not s:
            return ""

        # 1) direct token match against valid scenes
        for tok in s.replace("_", " ").split():
            if tok in _VALID_SCENES:
                return tok

        # 2) alias map: check single words and bigrams
        words = s.split()
        for i in range(len(words)):
            w = words[i]
            label = _SCENE_ALIAS_MAP.get(w)
            if label:
                return label
            if i + 1 < len(words):
                bigram = words[i] + "_" + words[i + 1]
                label = _SCENE_ALIAS_MAP.get(bigram) or _SCENE_ALIAS_MAP.get(words[i] + " " + words[i + 1])
                if label:
                    return label

        # 3) keyword regex scan (imported from action_parser via _normalize_scene_type)
        from vlm_robot_eval.core.action_parser import _normalize_scene_type
        label = _normalize_scene_type(text)
        if label:
            return label

        return ""

    def _scene_from_objects(self, objs: List[str]) -> str:
        """Infer scene type from recognized objects using keyword heuristics."""
        counts: Counter[str] = Counter()
        for obj in objs:
            key = str(obj).strip().lower()
            label = _OBJ_SCENE_MAP.get(key)
            if label:
                counts[label] += 1
        if not counts:
            return ""
        # return the most common inferred scene
        return counts.most_common(1)[0][0]

    def infer_objects(self, image: Image.Image) -> Dict[str, Any]:
        # Pass 1: structured JSON prompt
        gen = self._generate(image=image, prompt=OBJECTS_PROMPT)
        raw_text = str(gen.get("raw_text", "") or "")
        parsed = parse_objects_json(raw_text)

        objects: List[str] = parsed.get("objects", [])
        if not isinstance(objects, list):
            objects = []

        scene_type = str(parsed.get("scene_type", "")).strip().lower()
        if scene_type not in _VALID_SCENES:
            scene_type = ""

        # 2) regex/alias extraction from pass-1 raw text
        if not scene_type:
            scene_type = self._extract_scene_label(raw_text)

        # 3) second-pass: forced single-label prompt
        scene_raw = ""
        if not scene_type:
            scene_gen = self._generate(image=image, prompt=SCENE_ONLY_PROMPT)
            scene_raw = str(scene_gen.get("raw_text", "") or "")
            scene_type = self._extract_scene_label(scene_raw)

        # 4) object-keyword heuristic
        if not scene_type:
            scene_type = self._scene_from_objects(objects)

        # 5) final safety net: never emit empty string
        if scene_type not in _VALID_SCENES:
            scene_type = "unknown"

        return {
            "objects": objects,
            "scene_type": scene_type,
            **gen,
            "scene_raw_text": scene_raw,
        }
