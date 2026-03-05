from __future__ import annotations

import csv
import inspect
import json
import os
import statistics
import time
import traceback
from typing import Any, Dict, List, Set, Tuple

import matplotlib
import matplotlib.pyplot as plt
import torch
from PIL import Image
from tqdm import tqdm

from vlm_robot_eval.core.metrics import actions_to_text, consistency_rate, pick_representative
from vlm_robot_eval.core.semantic_constraint import LEGAL_ACTIONS, _target_matches, check_executable
from vlm_robot_eval.models.qwen_vl import QwenVLModel
from vlm_robot_eval.models.smol_vlm import SmolVLMModel


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _load_image(img_path: str) -> Image.Image:
    return Image.open(img_path).convert("RGB")


def _model_infer(model: Any, image: Image.Image, instruction: str, objects: List[str]) -> Dict[str, Any]:
    sig = inspect.signature(model.infer)
    if "objects" in sig.parameters:
        return model.infer(image=image, instruction=instruction, objects=objects)
    return model.infer(image=image, instruction=instruction)


def _seq_key(action_sequence: List[Dict[str, Any]]) -> str:
    return "|".join(f"{a.get('action','')}::{a.get('target','')}" for a in action_sequence if isinstance(a, dict))


def _cosine(u: torch.Tensor, v: torch.Tensor) -> float:
    denom = (u.norm(p=2) * v.norm(p=2)).clamp_min(1e-12)
    return float((u @ v) / denom)


def _normalize_target(s: str) -> str:
    return str(s).strip().lower()


_SCENE_LABELS = {"dining_table", "office", "kitchen", "indoor", "unknown"}


def _set_cjk_font() -> None:
    # Try to choose a CJK-capable font to avoid tofu when using Chinese titles
    candidates = [
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "Noto Sans CJK SC",
        "Noto Sans SC",
        "SimHei",
        "Microsoft YaHei",
        "PingFang SC",
        "AR PL UMing CN",
        "DejaVu Sans",  # fallback
    ]
    available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    chosen = next((f for f in candidates if f in available), None)
    if chosen:
        plt.rcParams["font.family"] = chosen
    plt.rcParams["axes.unicode_minus"] = False


def _normalize_scene_label(s: Any) -> str:
    x = str(s).strip().lower().replace("-", " ")
    x = " ".join(x.split())
    if not x:
        return ""
    if x in {"dining table", "table"}:
        return "dining_table"
    x2 = x.replace(" ", "_")
    if x2 in _SCENE_LABELS:
        return x2
    return ""


def _load_sbert() -> Any:
    model_id = os.getenv("SBERT_MODEL_PATH", "sentence-transformers/all-MiniLM-L6-v2")
    try:
        from sentence_transformers import SentenceTransformer

        device = os.getenv("SBERT_DEVICE", "cpu").strip().lower()
        if device not in {"cpu", "cuda", "auto"}:
            device = "cpu"
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        return SentenceTransformer(model_id, device=device)
    except Exception as e:
        print(f"[warn] sentence-transformers load failed for {model_id}; semantic similarity disabled. error: {e}")
        return None


def _is_seq_match(pred: List[Dict[str, Any]], gt: List[Dict[str, Any]], obj_set: Set[str]) -> bool:
    if len(pred) != len(gt):
        return False
    for a, b in zip(pred, gt):
        pa = str(a.get("action", "")).strip().lower()
        pb = str(b.get("action", "")).strip().lower()
        if pa != pb:
            return False
        ta = _normalize_target(a.get("target", ""))
        tb = _normalize_target(b.get("target", ""))
        if ta == tb:
            continue
        if not _target_matches(ta, obj_set):
            return False
        if not _target_matches(tb, obj_set):
            return False
    return True


def _safe_mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def _classify_error(exc: Exception) -> str:
    msg = str(exc).lower()
    if isinstance(exc, FileNotFoundError):
        return "file_not_found"
    if isinstance(exc, torch.cuda.OutOfMemoryError) or "cuda out of memory" in msg:
        return "cuda_oom"
    if "timeout" in msg:
        return "timeout"
    if isinstance(exc, ValueError):
        return "value_error"
    if isinstance(exc, RuntimeError):
        return "runtime_error"
    return "unknown_error"


def _traceback_text() -> str:
    return traceback.format_exc(limit=8)


def _micro_pr(pairs: List[Tuple[List[str], List[str]]]) -> Tuple[float, float]:
    tp = fp = fn = 0
    for pred, gt in pairs:
        ps = {str(x).strip().lower() for x in pred if str(x).strip()}
        gs = {str(x).strip().lower() for x in gt if str(x).strip()}
        tp += len(ps & gs)
        fp += len(ps - gs)
        fn += len(gs - ps)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float(p), float(r)


def _plot_bar(models: List[str], values: List[float], title: str, ylabel: str, out_path: str) -> None:
    plt.figure(figsize=(7, 5))
    bars = plt.bar(models, values, edgecolor="black", linewidth=0.8)
    ymax = max(values) if values else 0.0
    plt.ylim(0.0, ymax * 1.2 if ymax > 0 else 1.0)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Model")
    for b, v in zip(bars, values):
        dy = 0.02 * (ymax if ymax > 0 else 1.0)
        plt.text(b.get_x() + b.get_width() / 2, v + dy, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_precision_recall(models: List[str], ps: List[float], rs: List[float], out_path: str) -> None:
    xs = list(range(len(models)))
    w = 0.35
    plt.figure(figsize=(7, 5))
    b1 = plt.bar([x - w / 2 for x in xs], ps, width=w, label="精确率", edgecolor="black", linewidth=0.8)
    b2 = plt.bar([x + w / 2 for x in xs], rs, width=w, label="召回率", edgecolor="black", linewidth=0.8)
    ymax = max(ps + rs) if (ps or rs) else 0.0
    plt.ylim(0.0, ymax * 1.2 if ymax > 0 else 1.0)
    for b, v in list(zip(b1, ps)) + list(zip(b2, rs)):
        dy = 0.02 * (ymax if ymax > 0 else 1.0)
        plt.text(b.get_x() + b.get_width() / 2, v + dy, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(xs, models, rotation=15, ha="right")
    plt.ylabel("score")
    plt.title("精确率 / 召回率")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_difficulty_curve(records: List[Dict[str, Any]], models: List[str], out_path: str) -> None:
    levels = [1, 2, 3, 4, 5]
    plt.figure(figsize=(7, 5))
    for m in models:
        ys = []
        for lv in levels:
            arr = [r for r in records if r.get("model") == m and int(r.get("difficulty_level", 0)) == lv]
            ys.append(_safe_mean([float(r.get("task_success", 0.0)) for r in arr]))
        plt.plot(levels, ys, marker="o", label=m)
    plt.xticks(levels)
    plt.ylim(0.0, 1.0)
    plt.xlabel("difficulty_level")
    plt.ylabel("success_rate")
    plt.title("成功率 vs 难度")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_scene_diagnostics(models: List[str], acc: List[float], valid_rate: List[float], valid_acc: List[float], out_path: str) -> None:
    xs = list(range(len(models)))
    w = 0.25
    plt.figure(figsize=(8, 5))
    b1 = plt.bar([x - w for x in xs], acc, width=w, label="scene_acc", edgecolor="black", linewidth=0.8)
    b2 = plt.bar(xs, valid_rate, width=w, label="valid_scene_rate", edgecolor="black", linewidth=0.8)
    b3 = plt.bar([x + w for x in xs], valid_acc, width=w, label="valid_scene_acc", edgecolor="black", linewidth=0.8)
    ymax = max(acc + valid_rate + valid_acc) if (acc or valid_rate or valid_acc) else 0.0
    plt.ylim(0.0, ymax * 1.2 if ymax > 0 else 1.0)
    for b, v in list(zip(b1, acc)) + list(zip(b2, valid_rate)) + list(zip(b3, valid_acc)):
        dy = 0.02 * (ymax if ymax > 0 else 1.0)
        plt.text(b.get_x() + b.get_width() / 2, v + dy, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    plt.xticks(xs, models, rotation=15, ha="right")
    plt.ylabel("score")
    plt.title("场景指标")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def run_evaluation_v3(
    dataset_path: str,
    out_dir: str,
    repeat: int = 5,
    max_items: int = 0,
    enable_stress_test: bool = False,
    qwen_deterministic: bool = True,
) -> str:
    _ensure_dir(out_dir)
    plot_dir = os.path.join(out_dir, "plots")
    _ensure_dir(plot_dir)
    _set_cjk_font()

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    if not isinstance(dataset, list):
        raise ValueError("dataset must be a list")

    if max_items > 0:
        dataset = dataset[:max_items]

    # Enforce v3 scene labels to avoid evaluating task labels as scene labels.
    for i, item in enumerate(dataset):
        if not isinstance(item, dict):
            continue
        gt = item.get("ground_truth", {}) if isinstance(item.get("ground_truth"), dict) else {}
        raw_scene = gt.get("scene_type", "")
        if not isinstance(raw_scene, str) or not raw_scene.strip():
            continue
        if _normalize_scene_label(raw_scene) == "":
            raise ValueError(
                f"Invalid scene_type '{raw_scene}' at index={i}; please use v3 dataset with canonical scene labels"
            )

    models = [SmolVLMModel(), QwenVLModel(deterministic=qwen_deterministic)]

    sbert = _load_sbert()

    per_model: Dict[str, Dict[str, Any]] = {
        m.name: {
            "consistency": [],
            "semantic": [],
            "action_len": [],
            "executable": [],
            "correction": [],
            "illegal": [],
            "mismatch": [],
            "latency": [],
            "memory": [],
            "obj_pairs": [],
            "scene_pairs": [],
            "scene_valid": [],
            "scene_valid_correct": [],
            "complex_success": [],
            "times_for_std": [],
            "empty": [],
        }
        for m in models
    }

    debug_path = os.path.join(out_dir, "debug.jsonl")
    sample_rows: List[Dict[str, Any]] = []

    with open(debug_path, "w", encoding="utf-8") as dbg:
        pbar = tqdm(dataset, desc="Evaluating v3 samples", unit="sample", dynamic_ncols=True)
        for idx, item in enumerate(pbar, start=1):
            image_rel = str(item.get("image", ""))
            if not os.path.isabs(image_rel):
                image_path = os.path.join(os.path.dirname(dataset_path), image_rel)
            else:
                image_path = image_rel
            image = _load_image(image_path)
            pbar.set_postfix({"sample": item.get("id", idx)})
        

            instruction = str(item.get("instruction", ""))
            reasoning_type = str(item.get("reasoning_type", "simple"))
            difficulty = int(item.get("difficulty_level", 1))
            gt = item.get("ground_truth", {}) if isinstance(item.get("ground_truth"), dict) else {}

            gt_objs = gt.get("objects", []) if isinstance(gt.get("objects"), list) else []
            hw = gt.get("hardware_constraints", {}) if isinstance(gt.get("hardware_constraints"), dict) else {}
            gt_scene = _normalize_scene_label(gt.get("scene_type", ""))
            gt_expected = gt.get("expected_action_sequence", []) if isinstance(gt.get("expected_action_sequence"), list) else []

            id_to_cat = {str(o.get("id", "")).strip().lower(): str(o.get("category", "")).strip().lower() for o in gt_objs if isinstance(o, dict)}
            gt_id_set = {k for k in id_to_cat.keys() if k}
            gt_cat_set = {v for v in id_to_cat.values() if v}
            gt_target_space = sorted(gt_id_set | gt_cat_set)

            rel_gt_objs = []
            target_id = str(gt.get("target_object_id", "")).strip().lower()
            rel_id = str(gt.get("relation_object_id", "")).strip().lower()
            if target_id and target_id in id_to_cat:
                rel_gt_objs.append(id_to_cat[target_id])
            if rel_id and rel_id in id_to_cat:
                rel_gt_objs.append(id_to_cat[rel_id])
            if not rel_gt_objs:
                rel_gt_objs = sorted(gt_cat_set)

            for model in models:
                runs: List[List[Dict[str, Any]]] = []
                raws: List[str] = []
                ts: List[float] = []
                ms: List[int] = []

                infer_error_type = "none"
                infer_traceback = ""
                for _ in range(max(1, int(repeat))):
                    t0 = time.time()
                    try:
                        out = _model_infer(model=model, image=image, instruction=instruction, objects=gt_target_space)
                    except Exception as e:
                        out = {
                            "action_sequence": [],
                            "raw_text": str(e),
                            "inference_time": 0.0,
                            "gpu_memory": 0,
                        }
                        infer_error_type = _classify_error(e)
                        infer_traceback = _traceback_text()
                    t1 = time.time()
                    seq = out.get("action_sequence", []) if isinstance(out, dict) else []
                    runs.append(seq if isinstance(seq, list) else [])
                    raws.append(str(out.get("raw_text", "")) if isinstance(out, dict) else "")
                    ts.append(float(out.get("inference_time", max(t1 - t0, 0.0))) if isinstance(out, dict) else max(t1 - t0, 0.0))
                    ms.append(int(out.get("gpu_memory", 0)) if isinstance(out, dict) else 0)

                rep = pick_representative(runs)
                cons = consistency_rate(runs)
                txt = actions_to_text(rep)
                sim = 0.0
                if sbert is not None and txt.strip():
                    try:
                        emb = sbert.encode([instruction, txt], convert_to_tensor=True, normalize_embeddings=True)
                        sim = max(0.0, float(emb[0] @ emb[1]))
                    except Exception as e:
                        sim = 0.0
                        print(f"[warn] sbert encoding failed; semantic similarity set to 0. error: {e}")

                corrected, exe, cost = check_executable(rep, gt_target_space, hw)
                empty = 1.0 if len(rep) == 0 else 0.0
                alen = float(len(rep))

                if rep:
                    illegal = sum(1 for a in rep if str(a.get("action", "")) not in LEGAL_ACTIONS) / len(rep)
                    mismatch = sum(1 for a in rep if not _target_matches(str(a.get("target", "")).strip().lower(), set(gt_target_space))) / len(rep)
                else:
                    illegal = 0.0
                    mismatch = 1.0

                if reasoning_type == "negative":
                    success = 1.0 if len(rep) == 0 else 0.0
                else:
                    success = 1.0 if _is_seq_match(rep, gt_expected, set(gt_target_space)) else 0.0

                obj_error_type = "none"
                obj_traceback = ""
                try:
                    obj_out = model.infer_objects(image=image)
                except Exception as e:
                    obj_out = {"objects": [], "scene_type": "", "raw_text": str(e)}
                    obj_error_type = _classify_error(e)
                    obj_traceback = _traceback_text()
                pred_objs = obj_out.get("objects", []) if isinstance(obj_out, dict) and isinstance(obj_out.get("objects"), list) else []
                pred_objs = pred_objs[:5]
                pred_scene = _normalize_scene_label(obj_out.get("scene_type", "") if isinstance(obj_out, dict) else "")

                p, r = _micro_pr([(pred_objs, rel_gt_objs)])
                scene_valid = 1.0 if pred_scene in _SCENE_LABELS else 0.0
                scene_acc = 1.0 if gt_scene and pred_scene == gt_scene else 0.0
                scene_valid_correct = 1.0 if (scene_valid > 0.0 and gt_scene and pred_scene == gt_scene) else 0.0

                pm = per_model[model.name]
                pm["consistency"].append(float(cons))
                pm["semantic"].append(float(sim))
                pm["action_len"].append(float(alen))
                pm["executable"].append(float(exe))
                pm["correction"].append(float(cost))
                pm["illegal"].append(float(illegal))
                pm["mismatch"].append(float(mismatch))
                pm["latency"].append(float(sum(ts) / len(ts) if ts else 0.0))
                pm["times_for_std"].extend(ts)
                pm["memory"].append(float(sum(ms) / len(ms) if ms else 0.0))
                pm["obj_pairs"].append((pred_objs, rel_gt_objs))
                pm["scene_pairs"].append((pred_scene, gt_scene))
                pm["scene_valid"].append(scene_valid)
                pm["scene_valid_correct"].append(scene_valid_correct)
                pm["empty"].append(empty)
                if reasoning_type in {"spatial_relation", "multi_step"}:
                    pm["complex_success"].append(success)

                rec = {
                    "model": model.name,
                    "sample_id": item.get("id", ""),
                    "reasoning_type": reasoning_type,
                    "difficulty_level": difficulty,
                    "consistency_rate": float(cons),
                    "semantic_similarity": float(sim),
                    "avg_action_len": float(alen),
                    "executable_rate": float(exe),
                    "correction_cost": float(cost),
                    "illegal_action_rate": float(illegal),
                    "target_mismatch_rate": float(mismatch),
                    "avg_inference_time": float(sum(ts) / len(ts) if ts else 0.0),
                    "latency_std": float(statistics.pstdev(ts) if len(ts) > 1 else 0.0),
                    "avg_gpu_memory": float(sum(ms) / len(ms) if ms else 0.0),
                    "precision": float(p),
                    "recall": float(r),
                    "scene_accuracy": float(scene_acc),
                    "valid_scene": float(scene_valid),
                    "valid_scene_accuracy": float(scene_valid_correct),
                    "empty_output_rate": float(empty),
                    "task_success": float(success),
                    "infer_error_type": infer_error_type,
                    "obj_error_type": obj_error_type,
                }
                sample_rows.append(rec)
                dbg.write(
                    json.dumps(
                        {
                            **rec,
                            "rep_actions": rep,
                            "corrected_actions": corrected,
                            "raw_text_runs": raws,
                            "infer_traceback": infer_traceback,
                            "obj_traceback": obj_traceback,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        pbar.close()

    # stress memory (disabled by default to avoid host freeze)
    stress_map: Dict[str, float] = {m.name: 0.0 for m in models}
    if enable_stress_test and torch.cuda.is_available() and dataset:
        stress_pool = dataset[: min(24, len(dataset))]
        scales = [224, 384, 512, 768]
        for m in models:
            peak = 0
            old_tokens = int(getattr(m, "max_new_tokens", 256))
            for i, item in enumerate(stress_pool):
                image_rel = str(item.get("image", ""))
                image_path = image_rel if os.path.isabs(image_rel) else os.path.join(os.path.dirname(dataset_path), image_rel)
                image = _load_image(image_path).resize((scales[i % len(scales)], scales[i % len(scales)]), Image.BILINEAR)
                out = _model_infer(model=m, image=image, instruction=str(item.get("instruction", "")), objects=[])
                peak = max(peak, int(out.get("gpu_memory", 0)) if isinstance(out, dict) else 0)
            setattr(m, "max_new_tokens", old_tokens)
            stress_map[m.name] = float(peak)

    rows: List[Dict[str, Any]] = []
    for m in models:
        pm = per_model[m.name]
        p, r = _micro_pr(pm["obj_pairs"])
        sp = pm["scene_pairs"]
        scene_total = sum(1 for _, gt in sp if str(gt).strip())
        scene_ok = sum(1 for pred, gt in sp if str(gt).strip() and str(pred).strip() == str(gt).strip())
        scene_valid_cnt = sum(1 for pred, gt in sp if str(gt).strip() and str(pred).strip() in _SCENE_LABELS)
        scene_valid_ok = sum(1 for pred, gt in sp if str(gt).strip() and str(pred).strip() in _SCENE_LABELS and str(pred).strip() == str(gt).strip())
        scene_acc = scene_ok / scene_total if scene_total > 0 else 0.0
        valid_scene_rate = scene_valid_cnt / scene_total if scene_total > 0 else 0.0
        valid_scene_acc = scene_valid_ok / scene_valid_cnt if scene_valid_cnt > 0 else 0.0

        rows.append(
            {
                "model": m.name,
                "consistency_rate": _safe_mean(pm["consistency"]),
                "semantic_similarity": _safe_mean(pm["semantic"]),
                "avg_action_len": _safe_mean(pm["action_len"]),
                "executable_rate": _safe_mean(pm["executable"]),
                "correction_cost": _safe_mean(pm["correction"]),
                "illegal_action_rate": _safe_mean(pm["illegal"]),
                "target_mismatch_rate": _safe_mean(pm["mismatch"]),
                "avg_inference_time": _safe_mean(pm["latency"]),
                "latency_std": float(statistics.pstdev(pm["times_for_std"]) if len(pm["times_for_std"]) > 1 else 0.0),
                "avg_gpu_memory": _safe_mean(pm["memory"]),
                "stress_gpu_memory": float(stress_map.get(m.name, 0.0)),
                "precision": float(p),
                "recall": float(r),
                "scene_accuracy": float(scene_acc),
                "valid_scene_rate": float(valid_scene_rate),
                "valid_scene_accuracy": float(valid_scene_acc),
                "empty_output_rate": _safe_mean(pm["empty"]),
                "complex_reasoning_success_rate": _safe_mean(pm["complex_success"]),
            }
        )

    all_csv = os.path.join(out_dir, "all_results.csv")
    with open(all_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "consistency_rate",
                "semantic_similarity",
                "avg_action_len",
                "executable_rate",
                "correction_cost",
                "illegal_action_rate",
                "target_mismatch_rate",
                "avg_inference_time",
                "latency_std",
                "avg_gpu_memory",
                "stress_gpu_memory",
                "precision",
                "recall",
                "scene_accuracy",
                "valid_scene_rate",
                "valid_scene_accuracy",
                "empty_output_rate",
                "complex_reasoning_success_rate",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    sample_csv = os.path.join(out_dir, "sample_metrics.csv")
    with open(sample_csv, "w", newline="", encoding="utf-8") as f:
        if sample_rows:
            w = csv.DictWriter(f, fieldnames=list(sample_rows[0].keys()))
            w.writeheader()
            w.writerows(sample_rows)

    names = [r["model"] for r in rows]
    _plot_bar(names, [float(r["consistency_rate"]) for r in rows], "一致性", "consistency", os.path.join(plot_dir, "consistency.png"))
    _plot_bar(names, [float(r["semantic_similarity"]) for r in rows], "语义相似度", "semantic", os.path.join(plot_dir, "semantic.png"))
    _plot_bar(names, [float(r["executable_rate"]) for r in rows], "可执行率", "rate", os.path.join(plot_dir, "executable.png"))

    # inference time
    plt.figure(figsize=(7, 5))
    vals = [float(r["avg_inference_time"]) for r in rows]
    bars = plt.bar(names, vals, edgecolor="black", linewidth=0.8)
    ymax = max(vals) if vals else 0.0
    plt.ylim(0.0, ymax * 1.2 if ymax > 0 else 1.0)
    plt.title("推理时间")
    plt.ylabel("秒")
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.02 * (ymax if ymax > 0 else 1.0), f"{v:.2f}s", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "inference_time.png"), dpi=200)
    plt.close()

    # memory
    plt.figure(figsize=(7, 5))
    mem = [float(r["stress_gpu_memory"]) / (1024 ** 2) for r in rows]
    bars = plt.bar(names, mem, edgecolor="black", linewidth=0.8)
    ymax = max(mem) if mem else 0.0
    plt.ylim(0.0, ymax * 1.2 if ymax > 0 else 1024.0)
    plt.title("显存占用（压力测试）")
    plt.ylabel("MB")
    for b, v in zip(bars, mem):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.02 * (ymax if ymax > 0 else 1024.0), f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "memory.png"), dpi=200)
    plt.close()

    _plot_precision_recall(names, [float(r["precision"]) for r in rows], [float(r["recall"]) for r in rows], os.path.join(plot_dir, "precision_recall.png"))
    _plot_scene_diagnostics(
        names,
        [float(r["scene_accuracy"]) for r in rows],
        [float(r["valid_scene_rate"]) for r in rows],
        [float(r["valid_scene_accuracy"]) for r in rows],
        os.path.join(plot_dir, "scene_diagnostics.png"),
    )
    _plot_difficulty_curve(sample_rows, names, os.path.join(plot_dir, "difficulty_curve.png"))

    print("=============================")
    print("双 VLM 机器人控制评估系统（升级版 v3）")
    print("=============================")
    print(f"samples={len(dataset)}, repeat={repeat}")
    print(f"results_dir={out_dir}")
    for r in rows:
        print("---")
        for k, v in r.items():
            if k == "model":
                print(f"model: {v}")
            else:
                print(f"{k}: {float(v):.4f}" if isinstance(v, (float, int)) else f"{k}: {v}")

    return all_csv
