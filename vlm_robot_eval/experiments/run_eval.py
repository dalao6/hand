from __future__ import annotations

import argparse
import csv
import json
import math
import os
import inspect
import statistics
from collections import Counter
from typing import Any, Dict, List, Set, Tuple

from tqdm import tqdm

import torch
from PIL import Image

from vlm_robot_eval.core.metrics import actions_to_text, consistency_rate, pick_representative
from vlm_robot_eval.core.semantic_constraint import check_executable, _target_matches, LEGAL_ACTIONS
from vlm_robot_eval.models.qwen_vl import QwenVLModel
from vlm_robot_eval.models.smol_vlm import SmolVLMModel


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _load_image(img_path: str) -> Image.Image:
    if os.path.exists(img_path):
        return Image.open(img_path).convert("RGB")
    raise FileNotFoundError(f"Image not found: {img_path}")


def _seq_key(action_sequence: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for a in action_sequence:
        if not isinstance(a, dict):
            continue
        parts.append(f"{a.get('action','')}::{a.get('target','')}")
    return "|".join(parts)


def _pick_rep_raw(actions_runs: List[List[Dict[str, Any]]], raw_text_runs: List[str]) -> str:
    rep = pick_representative(actions_runs)
    rep_key = _seq_key(rep)
    for i, seq in enumerate(actions_runs):
        if _seq_key(seq) == rep_key:
            return raw_text_runs[i] if i < len(raw_text_runs) else ""
    return raw_text_runs[0] if raw_text_runs else ""


def _cosine(u: torch.Tensor, v: torch.Tensor) -> float:
    denom = (u.norm(p=2) * v.norm(p=2)).clamp_min(1e-12)
    return float((u @ v) / denom)


def _micro_precision_recall(pairs: List[Tuple[List[str], List[str]]]) -> Tuple[float, float]:
    tp = fp = fn = 0
    for pred, gt in pairs:
        ps = {str(x).strip().lower() for x in pred if str(x).strip()}
        gs = {str(x).strip().lower() for x in gt if str(x).strip()}
        tp += len(ps & gs)
        fp += len(ps - gs)
        fn += len(gs - ps)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float(precision), float(recall)


def _difficulty_bucket(instruction: str, gt_objects: List[str]) -> str:
    ins = instruction.lower().strip()
    n_obj = len([x for x in gt_objects if str(x).strip()])
    if " and " in ins or "then" in ins or "place" in ins:
        return "L4"
    if n_obj >= 5:
        return "L3"
    if any(k in ins for k in ["rotate", "push", "release"]):
        return "L2"
    return "L1"


def _model_infer(model: Any, image: Image.Image, instruction: str, objects: List[str]) -> Dict[str, Any]:
    sig = inspect.signature(model.infer)
    if "objects" in sig.parameters:
        return model.infer(image=image, instruction=instruction, objects=objects)
    return model.infer(image=image, instruction=instruction)


def _infer_with_retry(model: Any, image: Image.Image, instruction: str, objects: List[str], retries: int = 1) -> Tuple[Dict[str, Any], str]:
    last: Dict[str, Any] = {}
    for attempt in range(retries + 1):
        try:
            out = _model_infer(model=model, image=image, instruction=instruction, objects=objects)
            if isinstance(out, dict):
                seq = out.get("action_sequence", [])
                if isinstance(seq, list):
                    return out, "none"
            last = out if isinstance(out, dict) else {}
        except Exception as e:
            last = {"action_sequence": [], "raw_text": str(e), "inference_time": 0.0, "gpu_memory": 0}
    return last if isinstance(last, dict) else {"action_sequence": []}, "infer_failed"


def _infer_objects_with_retry(model: Any, image: Image.Image, retries: int = 1) -> Tuple[Dict[str, Any], str]:
    last: Dict[str, Any] = {}
    for _ in range(retries + 1):
        try:
            out = model.infer_objects(image=image)
            if isinstance(out, dict):
                objs = out.get("objects", [])
                scene = out.get("scene_type", "")
                if isinstance(objs, list) and isinstance(scene, str):
                    return out, "none"
            last = out if isinstance(out, dict) else {}
        except Exception as e:
            last = {"objects": [], "scene_type": "", "raw_text": str(e)}
    return last if isinstance(last, dict) else {"objects": [], "scene_type": ""}, "infer_objects_failed"


def _stress_gpu_memory(models: List[Any], dataset: List[Dict[str, Any]], exp_dir: str, out_dir: str, max_items: int = 24) -> Dict[str, float]:
    stress: Dict[str, float] = {m.name: 0.0 for m in models}
    if not torch.cuda.is_available():
        return stress

    usable = [x for x in dataset if isinstance(x, dict)]
    if max_items > 0:
        usable = usable[:max_items]
    if not usable:
        return stress

    scales = [224, 384, 512, 768]
    token_levels = [128, 256, 384]

    for model in models:
        peak = 0
        old_tokens = int(getattr(model, "max_new_tokens", 256))
        for i, item in enumerate(usable):
            image_rel = str(item.get("image", ""))
            instruction = str(item.get("instruction", ""))
            gt = item.get("ground_truth", {}) if isinstance(item.get("ground_truth", {}), dict) else {}
            gt_objects = gt.get("objects", []) if isinstance(gt.get("objects", []), list) else []

            img_path = image_rel if os.path.isabs(image_rel) else os.path.join(exp_dir, image_rel)
            if not os.path.exists(img_path):
                continue
            image = _load_image(img_path)
            side = scales[i % len(scales)]
            image = image.resize((side, side), Image.BILINEAR)

            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

            setattr(model, "max_new_tokens", token_levels[i % len(token_levels)])
            out, _ = _infer_with_retry(model=model, image=image, instruction=instruction, objects=gt_objects, retries=0)
            peak = max(peak, int(out.get("gpu_memory", 0)) if isinstance(out, dict) else 0)

        setattr(model, "max_new_tokens", old_tokens)
        stress[model.name] = float(peak)

    stress_path = os.path.join(out_dir, "gpu_stress_results.csv")
    with open(stress_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "stress_gpu_memory"])
        w.writeheader()
        for k, v in stress.items():
            w.writerow({"model": k, "stress_gpu_memory": v})

    return stress


def _write_coverage_report(records: List[Dict[str, Any]], models: List[Any], out_dir: str, n_items: int) -> None:
    metrics = [
        "avg_action_len",
        "consistency_rate",
        "empty_output_rate",
        "executable_rate",
        "avg_gpu_memory",
        "illegal_action_rate",
        "precision",
        "recall",
        "scene_accuracy",
        "semantic_similarity",
        "target_mismatch_rate",
    ]
    by_model: Dict[str, List[Dict[str, Any]]] = {m.name: [] for m in models}
    for r in records:
        by_model.setdefault(str(r.get("model", "")), []).append(r)

    out_rows: List[Dict[str, Any]] = []
    for m in models:
        rs = by_model.get(m.name, [])
        denom = max(n_items, 1)
        for metric in metrics:
            present = sum(1 for r in rs if r.get(metric) is not None)
            out_rows.append(
                {
                    "model": m.name,
                    "metric": metric,
                    "expected": denom,
                    "present": present,
                    "coverage": float(present / denom),
                }
            )

    p = os.path.join(out_dir, "coverage_report.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "metric", "expected", "present", "coverage"])
        w.writeheader()
        w.writerows(out_rows)


def _write_level_results(records: List[Dict[str, Any]], out_dir: str) -> None:
    if not records:
        return

    by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in records:
        k = (str(r.get("model", "")), str(r.get("difficulty_level", "L1")))
        by_key.setdefault(k, []).append(r)

    fields = [
        "avg_action_len",
        "consistency_rate",
        "empty_output_rate",
        "executable_rate",
        "avg_gpu_memory",
        "illegal_action_rate",
        "precision",
        "recall",
        "scene_accuracy",
        "semantic_similarity",
        "target_mismatch_rate",
    ]

    rows: List[Dict[str, Any]] = []
    for (model, level), arr in by_key.items():
        row: Dict[str, Any] = {"model": model, "difficulty_level": level, "count": len(arr)}
        for f in fields:
            vals = [float(x.get(f, 0.0)) for x in arr]
            row[f] = float(sum(vals) / len(vals)) if vals else 0.0
        rows.append(row)

    p = os.path.join(out_dir, "level_results.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "difficulty_level", "count", *fields])
        w.writeheader()
        w.writerows(rows)


def run(
    dataset_path: str,
    out_dir: str,
    repeat: int = 5,
    max_items: int = 0,
) -> str:
    _ensure_dir(out_dir)

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    if not isinstance(dataset, list):
        raise ValueError("ground_truth_dataset.json must be a list")

    exp_dir = os.path.dirname(dataset_path)

    # Load models (fp16, device_map=auto inside constructors)
    smol = SmolVLMModel()
    qwen = QwenVLModel()
    models = [smol, qwen]

    # Sentence-BERT for semantic similarity
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError("Missing dependency: sentence-transformers") from e

    sbert_device = "cuda" if torch.cuda.is_available() else "cpu"
    sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=sbert_device)

    # Per-model accumulators
    per_model: Dict[str, Dict[str, Any]] = {}
    for m in models:
        per_model[m.name] = {
            "consistency_rates": [],
            "semantic_sims": [],
            "executable_rates": [],
            "correction_costs": [],
            "latencies": [],
            "gpu_mems": [],
            "obj_pairs": [],  # (pred, gt)
            "scene_pairs": [],  # (pred_scene, gt_scene)
            "empty_output_rates": [],
            "illegal_action_rates": [],
            "target_mismatch_rates": [],
            "action_lengths": [],
            "non_empty_semantic_sims": [],
            "non_empty_executable_rates": [],
            "non_empty_action_lengths": [],
            "fail_reasons": Counter(),
            "levels": Counter(),
        }

    debug_path = os.path.join(out_dir, "debug.jsonl")
    dbg_f = open(debug_path, "w", encoding="utf-8")
    records: List[Dict[str, Any]] = []
    uniq_keys: Set[str] = set()

    run_cfg = {
        "dataset_path": dataset_path,
        "out_dir": out_dir,
        "repeat": int(repeat),
        "max_items": int(max_items),
        "models": [m.name for m in models],
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": str(torch.__version__),
    }
    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_cfg, f, ensure_ascii=False, indent=2)

    n_total = min(max_items, len(dataset)) if max_items else len(dataset)
    n_items = 0
    pbar = tqdm(total=n_total, desc="Evaluating samples", unit="sample", dynamic_ncols=True)
    for idx, item in enumerate(dataset):
        if max_items and idx >= max_items:
            break
        if not isinstance(item, dict):
            continue

        image_rel = str(item.get("image", ""))
        instruction = str(item.get("instruction", ""))
        gt = item.get("ground_truth", {})
        if not isinstance(gt, dict):
            gt = {}

        gt_objects = gt.get("objects", [])
        if not isinstance(gt_objects, list):
            gt_objects = []

        hw = gt.get("hardware_constraints", {})
        if not isinstance(hw, dict):
            hw = {}

        gt_scene = gt.get("scene_type", "")
        if not isinstance(gt_scene, str):
            gt_scene = ""
        gt_scene = gt_scene.strip()

        img_path = image_rel
        if not os.path.isabs(img_path):
            img_path = os.path.join(exp_dir, image_rel)

        image = _load_image(img_path)
        n_items += 1
        pbar.set_postfix({"img": os.path.basename(img_path), "n": n_items})

        # Experiment 1/3: repeated action inference
        for model in models:
            actions_runs: List[List[Dict[str, Any]]] = []
            raw_text_runs: List[str] = []
            times: List[float] = []
            mems: List[int] = []

            fail_reason = "none"
            repeat_bar = tqdm(
                range(max(int(repeat), 1)),
                desc=f"  {model.name} repeats",
                unit="run",
                leave=False,
                dynamic_ncols=True,
            )
            for _ in repeat_bar:
                out, fr = _infer_with_retry(
                    model=model,
                    image=image,
                    instruction=instruction,
                    objects=gt_objects,
                    retries=0,
                )
                if fr != "none":
                    fail_reason = fr
                seq = out.get("action_sequence", []) if isinstance(out, dict) else []
                if not isinstance(seq, list):
                    seq = []
                actions_runs.append(seq)

                raw_text = out.get("raw_text", "") if isinstance(out, dict) else ""
                raw_text_runs.append(str(raw_text) if raw_text is not None else "")

                t = out.get("inference_time", 0.0)
                m = out.get("gpu_memory", 0)
                times.append(float(t) if t is not None else 0.0)
                mems.append(int(m) if m is not None else 0)

            rep_actions = pick_representative(actions_runs)
            rep_raw_text = _pick_rep_raw(actions_runs, raw_text_runs)

            cr = consistency_rate(actions_runs)
            action_text = actions_to_text(rep_actions)

            # semantic similarity (instruction vs action text)
            if action_text.strip():
                emb = sbert.encode([instruction, action_text], convert_to_tensor=True, normalize_embeddings=False)
                sim = _cosine(emb[0], emb[1])
            else:
                sim = 0.0

            # Experiment 2: executability + correction cost
            corrected_actions, executable_rate, correction_cost = check_executable(
                action_sequence=rep_actions,
                ground_truth_objects=gt_objects,
                hardware_constraints=hw,
            )

            # Extra per-item metrics
            empty_out = 1.0 if len(rep_actions) == 0 else 0.0
            avg_action_len = float(len(rep_actions))
            if rep_actions:
                illegal = sum(1 for a in rep_actions if a.get("action", "") not in LEGAL_ACTIONS)
                illegal_rate = illegal / len(rep_actions)
            else:
                illegal_rate = 0.0

            obj_set_lower = {str(o).strip().lower() for o in gt_objects if str(o).strip()}
            if rep_actions:
                mismatched = sum(
                    1 for a in rep_actions
                    if not _target_matches(str(a.get("target", "")).strip().lower(), obj_set_lower)
                )
                target_mismatch = mismatched / len(rep_actions)
            else:
                target_mismatch = 1.0

            # Experiment 4: object understanding (single pass, no retry to avoid OOM/hang)
            obj_out, fr_obj = _infer_objects_with_retry(model=model, image=image, retries=0)
            if fr_obj != "none" and fail_reason == "none":
                fail_reason = fr_obj
            pred_objects = obj_out.get("objects", []) if isinstance(obj_out, dict) else []
            pred_scene = obj_out.get("scene_type", "") if isinstance(obj_out, dict) else ""
            if not isinstance(pred_objects, list):
                pred_objects = []
            if not isinstance(pred_scene, str):
                pred_scene = ""
            pred_scene = pred_scene.strip()

            level = _difficulty_bucket(instruction=instruction, gt_objects=gt_objects)

            pm = per_model[model.name]
            pm["consistency_rates"].append(float(cr))
            pm["semantic_sims"].append(float(sim))
            pm["executable_rates"].append(float(executable_rate))
            pm["correction_costs"].append(float(correction_cost))
            pm["latencies"].extend(times)
            pm["gpu_mems"].extend(mems)
            pm["obj_pairs"].append((pred_objects, gt_objects))
            pm["scene_pairs"].append((pred_scene, gt_scene))
            pm["empty_output_rates"].append(empty_out)
            pm["illegal_action_rates"].append(illegal_rate)
            pm["target_mismatch_rates"].append(target_mismatch)
            pm["action_lengths"].append(avg_action_len)
            pm["fail_reasons"][fail_reason] += 1
            pm["levels"][level] += 1
            if not empty_out:
                pm["non_empty_semantic_sims"].append(float(sim))
                pm["non_empty_executable_rates"].append(float(executable_rate))
                pm["non_empty_action_lengths"].append(float(avg_action_len))

            dbg_rec = {
                "model": model.name,
                "id": item.get("id", idx),
                "image": img_path,
                "instruction": instruction,
                "difficulty_level": level,
                "ground_truth": gt,
                "repeat": int(repeat),
                "consistency_rate": float(cr),
                "semantic_similarity": float(sim),
                "executable_rate": float(executable_rate),
                "correction_cost": float(correction_cost),
                "empty_output": float(empty_out),
                "avg_action_len": float(avg_action_len),
                "illegal_action_rate": float(illegal_rate),
                "target_mismatch_rate": float(target_mismatch),
                "avg_inference_time": float(sum(times) / len(times) if times else 0.0),
                "latency_std": float(statistics.pstdev(times) if len(times) > 1 else 0.0),
                "avg_gpu_memory": int(sum(mems) / len(mems) if mems else 0),
                "fail_reason": fail_reason,
                "rep_actions": rep_actions,
                "corrected_actions": corrected_actions,
                "rep_raw_text": rep_raw_text,
                "raw_text_runs": raw_text_runs,
                "predicted_objects": pred_objects,
                "predicted_scene_type": pred_scene,
            }

            rec_key = f"{model.name}::{item.get('id', idx)}::{level}"
            if rec_key not in uniq_keys:
                uniq_keys.add(rec_key)
                records.append(
                    {
                        "model": model.name,
                        "sample_id": item.get("id", idx),
                        "difficulty_level": level,
                        "avg_action_len": float(avg_action_len),
                        "consistency_rate": float(cr),
                        "empty_output_rate": float(empty_out),
                        "executable_rate": float(executable_rate),
                        "avg_gpu_memory": float(sum(mems) / len(mems) if mems else 0.0),
                        "illegal_action_rate": float(illegal_rate),
                        "precision": float(
                            (len({str(x).strip().lower() for x in pred_objects if str(x).strip()} & obj_set_lower))
                            /
                            max(len({str(x).strip().lower() for x in pred_objects if str(x).strip()}), 1)
                        ),
                        "recall": float(
                            (len({str(x).strip().lower() for x in pred_objects if str(x).strip()} & obj_set_lower))
                            /
                            max(len(obj_set_lower), 1)
                        ),
                        "scene_accuracy": float(
                            1.0 if gt_scene and str(pred_scene).strip() == str(gt_scene).strip() else 0.0
                        ),
                        "semantic_similarity": float(sim),
                        "target_mismatch_rate": float(target_mismatch),
                        "fail_reason": fail_reason,
                    }
                )

            dbg_f.write(json.dumps(dbg_rec, ensure_ascii=False) + "\n")

        pbar.update(1)

    pbar.close()
    dbg_f.close()

    stress_mem_map = _stress_gpu_memory(models=models, dataset=dataset[:n_total], exp_dir=exp_dir, out_dir=out_dir, max_items=6)

    # Aggregate per model
    rows: List[Dict[str, Any]] = []
    for model in models:
        name = model.name
        pm = per_model[name]

        def _avg(lst: list) -> float:
            return float(sum(lst) / len(lst)) if lst else 0.0

        cons = _avg(pm["consistency_rates"])
        sem = _avg(pm["semantic_sims"])
        exe = _avg(pm["executable_rates"])
        cost = _avg(pm["correction_costs"])
        empty_rate = _avg(pm["empty_output_rates"])
        illegal_rate = _avg(pm["illegal_action_rates"])
        tgt_mismatch = _avg(pm["target_mismatch_rates"])
        avg_action_len = _avg(pm["action_lengths"])

        latencies = [float(x) for x in pm["latencies"]]
        avg_t = float(sum(latencies) / len(latencies) if latencies else 0.0)
        lat_std = float(statistics.pstdev(latencies) if len(latencies) > 1 else 0.0)

        mems = [int(x) for x in pm["gpu_mems"]]
        avg_mem = float(sum(mems) / len(mems) if mems else 0.0)

        precision, recall = _micro_precision_recall(pm["obj_pairs"])  # micro-average

        spairs = pm["scene_pairs"]
        sc_total = sum(1 for _, gt_scene in spairs if str(gt_scene).strip())
        sc_correct = sum(
            1 for pred_scene, gt_scene in spairs if str(gt_scene).strip() and str(pred_scene).strip() == str(gt_scene).strip()
        )
        scene_acc = float(sc_correct / sc_total if sc_total > 0 else 0.0)

        rows.append(
            {
                "model": name,
                "consistency_rate": cons,
                "semantic_similarity": sem,
                "semantic_similarity_non_empty": _avg(pm["non_empty_semantic_sims"]),
                "executable_rate": exe,
                "executable_rate_non_empty": _avg(pm["non_empty_executable_rates"]),
                "correction_cost": cost,
                "empty_output_rate": empty_rate,
                "illegal_action_rate": illegal_rate,
                "target_mismatch_rate": tgt_mismatch,
                "avg_action_len": avg_action_len,
                "avg_action_len_non_empty": _avg(pm["non_empty_action_lengths"]),
                "avg_inference_time": avg_t,
                "latency_std": lat_std,
                "avg_gpu_memory": avg_mem,
                "stress_gpu_memory": float(stress_mem_map.get(name, 0.0)),
                "precision": precision,
                "recall": recall,
                "scene_accuracy": scene_acc,
                "fail_reason_summary": json.dumps(dict(pm["fail_reasons"]), ensure_ascii=False),
                "level_distribution": json.dumps(dict(pm["levels"]), ensure_ascii=False),
            }
        )

    csv_path = os.path.join(out_dir, "all_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "consistency_rate",
                "semantic_similarity",
                "semantic_similarity_non_empty",
                "executable_rate",
                "executable_rate_non_empty",
                "correction_cost",
                "empty_output_rate",
                "illegal_action_rate",
                "target_mismatch_rate",
                "avg_action_len",
                "avg_action_len_non_empty",
                "avg_inference_time",
                "latency_std",
                "avg_gpu_memory",
                "stress_gpu_memory",
                "precision",
                "recall",
                "scene_accuracy",
                "fail_reason_summary",
                "level_distribution",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    sample_csv = os.path.join(out_dir, "sample_metrics.csv")
    with open(sample_csv, "w", newline="", encoding="utf-8") as f:
        if records:
            w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            w.writeheader()
            w.writerows(records)

    _write_coverage_report(records=records, models=models, out_dir=out_dir, n_items=n_items)
    _write_level_results(records=records, out_dir=out_dir)

    _plot_summary(rows, out_dir)
    _print_summary(rows, n_items=n_items, repeat=repeat, out_dir=out_dir)
    return csv_path


def _plot_summary(rows: List[Dict[str, Any]], out_dir: str) -> None:
    import matplotlib
    import matplotlib.pyplot as plt

    # Resolve a CJK-capable font so Chinese titles render correctly
    _cjk_candidates = [
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "Noto Sans CJK SC",
        "Noto Sans SC",
        "SimHei",
        "Microsoft YaHei",
        "PingFang SC",
        "AR PL UMing CN",
        "DejaVu Sans",  # fallback (no CJK, but won't crash)
    ]
    _available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    _chosen = next((f for f in _cjk_candidates if f in _available), None)
    if _chosen:
        plt.rcParams["font.family"] = _chosen
    plt.rcParams["axes.unicode_minus"] = False  # fix minus sign rendering

    models = [str(r.get("model", "")) for r in rows]

    def get(metric: str) -> List[float]:
        out: List[float] = []
        for r in rows:
            v = r.get(metric, 0.0)
            out.append(float(v) if v is not None else 0.0)
        return out

    # (metric_key, title, y_label, fname, value_fmt)
    # value_fmt: None -> auto (4f for rate/score, special for memory/time)
    plots = [
        ("consistency_rate",    "动作一致性对比",       "一致性（0~1）",         "consistency.png",         "{:.4f}"),
        ("semantic_similarity", "语义匹配度对比",       "语义相似度（0~1）",     "semantic_similarity.png", "{:.4f}"),
        ("executable_rate",     "可执行率对比",         "可执行率（0~1）",       "executable_rate.png",     "{:.4f}"),
        ("empty_output_rate",   "空输出率对比",         "空输出率（0~1）",       "empty_output_rate.png",   "{:.4f}"),
        ("illegal_action_rate", "非法动作率对比",       "非法动作率（0~1）",     "illegal_action_rate.png", "{:.4f}"),
        ("target_mismatch_rate","目标不匹配率对比",     "目标不匹配率（0~1）",   "target_mismatch_rate.png","{:.4f}"),
        ("avg_action_len",      "平均动作序列长度对比", "平均步数",              "avg_action_len.png",      "{:.2f}"),
        ("scene_accuracy",      "场景分类准确率对比",   "场景分类准确率（0~1）", "scene_accuracy.png",      "{:.4f}"),
    ]

    for metric, title, ylabel, fname, vfmt in plots:
        values = get(metric)
        plt.figure(figsize=(7, 5))
        bars = plt.bar(models, values, edgecolor="black", linewidth=0.8)
        plt.title(title, fontsize=13)
        plt.xlabel("模型", fontsize=11)
        plt.ylabel(ylabel, fontsize=11)
        ymax = max(values) if values else 0.0
        if ymax <= 0:
            plt.ylim(0.0, 1.0)
        else:
            plt.ylim(0.0, ymax * 1.20)
        for b, v in zip(bars, values):
            label_y = (v if v > 0 else 0.0) + (0.02 * (ymax if ymax > 0 else 1.0))
            plt.text(
                b.get_x() + b.get_width() / 2,
                label_y,
                vfmt.format(v),
                ha="center",
                va="bottom",
                fontsize=9,
            )
        plt.xticks(rotation=15, ha="right", fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=200)
        plt.close()

    # ── Inference time（单位：秒）──
    time_vals = get("avg_inference_time")
    plt.figure(figsize=(7, 5))
    bars = plt.bar(models, time_vals, edgecolor="black", linewidth=0.8)
    plt.title("推理时间对比", fontsize=13)
    plt.xlabel("模型", fontsize=11)
    plt.ylabel("平均推理时间（秒）", fontsize=11)
    t_ymax = max(time_vals) if time_vals else 0.0
    plt.ylim(0.0, t_ymax * 1.20 if t_ymax > 0 else 1.0)
    for b, v in zip(bars, time_vals):
        label_y = (v if v > 0 else 0.0) + (0.02 * (t_ymax if t_ymax > 0 else 1.0))
        plt.text(b.get_x() + b.get_width() / 2, label_y, f"{v:.2f} s",
                 ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=15, ha="right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "inference_time.png"), dpi=200)
    plt.close()

    # ── GPU Memory（字节 → MB）──
    mem_bytes = get("stress_gpu_memory")
    mem_mb = [v / (1024 ** 2) for v in mem_bytes]
    plt.figure(figsize=(7, 5))
    bars = plt.bar(models, mem_mb, edgecolor="black", linewidth=0.8)
    plt.title("显存占用对比（压力测试）", fontsize=13)
    plt.xlabel("模型", fontsize=11)
    plt.ylabel("峰值显存（MB）", fontsize=11)
    m_ymax = max(mem_mb) if mem_mb else 0.0
    plt.ylim(0.0, m_ymax * 1.20 if m_ymax > 0 else 1024.0)
    for b, v in zip(bars, mem_mb):
        label_y = (v if v > 0 else 0.0) + (0.02 * (m_ymax if m_ymax > 0 else 1024.0))
        plt.text(b.get_x() + b.get_width() / 2, label_y, f"{v:.0f} MB",
                 ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=15, ha="right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gpu_memory.png"), dpi=200)
    plt.close()

    # ── Precision / Recall 联合对比 ──
    precisions = get("precision")
    recalls = get("recall")

    xs = list(range(len(models)))
    w = 0.35
    plt.figure(figsize=(7, 5))
    b1 = plt.bar([x - w / 2 for x in xs], precisions, width=w, label="精确率", edgecolor="black", linewidth=0.8)
    b2 = plt.bar([x + w / 2 for x in xs], recalls, width=w, label="召回率", edgecolor="black", linewidth=0.8)
    pr_ymax = max((precisions + recalls), default=0.0)
    if pr_ymax <= 0:
        plt.ylim(0.0, 1.0)
    else:
        plt.ylim(0.0, pr_ymax * 1.20)
    for b, v in list(zip(b1, precisions)) + list(zip(b2, recalls)):
        label_y = (v if v > 0 else 0.0) + (0.02 * (pr_ymax if pr_ymax > 0 else 1.0))
        plt.text(b.get_x() + b.get_width() / 2, label_y, f"{v:.4f}",
                 ha="center", va="bottom", fontsize=9)
    plt.title("目标识别 Precision / Recall 对比", fontsize=13)
    plt.xlabel("模型", fontsize=11)
    plt.ylabel("分数（0~1）", fontsize=11)
    plt.xticks(xs, models, rotation=15, ha="right", fontsize=9)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "precision_recall.png"), dpi=200)
    plt.close()


def _print_summary(rows: List[Dict[str, Any]], n_items: int, repeat: int, out_dir: str) -> None:
    print("=============================")
    print("双 VLM 机器人控制评估结果汇总")
    print("=============================")
    print(f"samples={n_items}, repeat={repeat}")
    print(f"results_dir={out_dir}")
    for r in rows:
        print("---")
        print(f"model: {r.get('model')}")
        print(f"consistency_rate: {r.get('consistency_rate'):.4f}")
        print(f"semantic_similarity: {r.get('semantic_similarity'):.4f}")
        print(f"executable_rate: {r.get('executable_rate'):.4f}")
        print(f"correction_cost: {r.get('correction_cost'):.4f}")
        print(f"empty_output_rate: {r.get('empty_output_rate'):.4f}")
        print(f"illegal_action_rate: {r.get('illegal_action_rate'):.4f}")
        print(f"target_mismatch_rate: {r.get('target_mismatch_rate'):.4f}")
        print(f"avg_action_len: {r.get('avg_action_len'):.4f}")
        print(f"avg_inference_time: {r.get('avg_inference_time'):.4f}")
        print(f"latency_std: {r.get('latency_std'):.4f}")
        print(f"avg_gpu_memory: {r.get('avg_gpu_memory'):.0f}")
        print(f"stress_gpu_memory: {r.get('stress_gpu_memory'):.0f}")
        print(f"precision: {r.get('precision'):.4f}")
        print(f"recall: {r.get('recall'):.4f}")
        print(f"scene_accuracy: {r.get('scene_accuracy'):.4f}")


def main() -> None:
    # Unified entrypoint: delegate to core.evaluator.run_evaluation_v3
    from vlm_robot_eval.core.evaluator import run_evaluation_v3

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=os.path.join(os.path.dirname(__file__), "ground_truth_dataset_v3.json"),
        help="Path to v3 dataset json",
    )
    parser.add_argument(
        "--out_dir",
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "results"),
        help="Output directory",
    )
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--max_items", type=int, default=0, help="0 means all")
    parser.add_argument("--enable_stress_test", action="store_true", help="Enable GPU stress memory test")
    parser.add_argument("--disable_qwen_deterministic", action="store_true", help="Use sampling mode for Qwen generation")
    args = parser.parse_args()

    csv_path = run_evaluation_v3(
        dataset_path=args.dataset,
        out_dir=args.out_dir,
        repeat=args.repeat,
        max_items=args.max_items,
        enable_stress_test=bool(args.enable_stress_test),
        qwen_deterministic=not bool(args.disable_qwen_deterministic),
    )
    print(csv_path)


if __name__ == "__main__":
    main()
