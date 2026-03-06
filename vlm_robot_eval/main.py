from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Tuple


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _validate_v3(results_dir: str, expected_repeat: int, expected_samples: int, focus_metrics_only: bool = False) -> Tuple[bool, List[str]]:
    issues: List[str] = []
    all_csv = os.path.join(results_dir, "all_results.csv")
    sample_csv = os.path.join(results_dir, "sample_metrics.csv")
    debug_jsonl = os.path.join(results_dir, "debug.jsonl")
    plot_dir = os.path.join(results_dir, "plots")

    for p in [all_csv, sample_csv, debug_jsonl, plot_dir]:
        if not os.path.exists(p):
            issues.append(f"missing: {p}")

    expected_plots = [
        "empty_output_rate.png",
        "illegal_action_rate.png",
        "precision_recall.png",
        "scene_accuracy.png",
        "scene_diagnostics.png",
    ] if focus_metrics_only else [
        "consistency.png",
        "semantic.png",
        "executable.png",
        "inference_time.png",
        "memory.png",
        "precision_recall.png",
        "difficulty_curve.png",
    ]
    for f in expected_plots:
        p = os.path.join(plot_dir, f)
        if not os.path.exists(p):
            issues.append(f"missing plot: {p}")

    rows: List[Dict[str, Any]] = []
    if os.path.exists(all_csv):
        with open(all_csv, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    for r in rows:
        model = str(r.get("model", ""))
        empty_rate = _safe_float(r.get("empty_output_rate", 0.0))
        avg_len = _safe_float(r.get("avg_action_len", 0.0))
        cons = _safe_float(r.get("consistency_rate", 0.0))
        if empty_rate >= 0.999 and avg_len <= 1e-12 and cons > 1e-12:
            issues.append(f"{model}: invalid aggregate (empty=1, action_len=0, consistency>0)")

    if os.path.exists(sample_csv):
        with open(sample_csv, "r", encoding="utf-8") as f:
            sample_rows = list(csv.DictReader(f))
        model_count = len({str(x.get("model", "")) for x in sample_rows})
        if expected_samples > 0 and model_count > 0:
            exp = expected_samples * model_count
            if len(sample_rows) != exp:
                issues.append(f"sample_metrics coverage mismatch: expected={exp}, got={len(sample_rows)}")

        # repeat sanity via debug line count is hard; ensure exists and non-empty
        if expected_repeat > 0 and os.path.exists(debug_jsonl) and os.path.getsize(debug_jsonl) == 0:
            issues.append("debug.jsonl is empty")

    return len(issues) == 0, issues


def _default_coco_root() -> str:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    for p in [
        os.getenv("COCO_ROOT"),
        os.getenv("COCO2017_ROOT"),
        os.path.join(repo_root, "coco2017"),
        os.path.join(repo_root, "vlm_robot_eval", "coco2017"),
        os.path.expanduser("~/coco2017"),
    ]:
        if p and os.path.isdir(p):
            return p
    return os.path.join(repo_root, "coco2017")


def main() -> None:
    here = os.path.dirname(__file__)
    dataset_v3 = os.path.join(here, "experiments", "ground_truth_dataset_v3.json")
    results_root = os.path.join(here, "results")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coco_root",
        default=_default_coco_root(),
        help="COCO 2017 root dir containing val2017/ and annotations/",
    )
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=-1, help="-1 means random seed")
    parser.add_argument("--enable_stress_test", action="store_true", help="Enable GPU stress memory test")
    parser.add_argument("--disable_qwen_deterministic", action="store_true", help="Use sampling mode for Qwen generation")
    parser.add_argument("--results_root", default=results_root)
    parser.add_argument("--run_name", default="", help="results subdir name")
    parser.add_argument("--dataset", default=dataset_v3, help="Path to v3 dataset json")
    parser.add_argument("--focus_metrics_only", action="store_true", help="Only output focused metrics and plots")
    args = parser.parse_args()

    run_name = args.run_name.strip() if isinstance(args.run_name, str) else ""
    if not run_name:
        run_name = datetime.now().strftime("v3_eval_%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.results_root, run_name)
    os.makedirs(out_dir, exist_ok=True)

    dataset_path = args.dataset
    need_build = True
    if os.path.exists(dataset_path):
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            need_build = not (isinstance(data, list) and len(data) >= args.samples)
        except Exception:
            need_build = True

    if need_build:
        from vlm_robot_eval.core.dataset_builder_v3 import build_ground_truth_dataset_v3

        print("[main] building v3 dataset...")
        build_ground_truth_dataset_v3(
            coco_root=args.coco_root,
            out_json_path=dataset_path,
            num_samples=args.samples,
            seed=None if args.seed < 0 else args.seed,
            copy_images=True,
        )

    from vlm_robot_eval.core.evaluator import run_evaluation_v3

    csv_path = run_evaluation_v3(
        dataset_path=dataset_path,
        out_dir=out_dir,
        repeat=args.repeat,
        max_items=args.samples,
        enable_stress_test=bool(args.enable_stress_test),
        qwen_deterministic=not bool(args.disable_qwen_deterministic),
        focus_metrics_only=bool(args.focus_metrics_only),
    )

    ok, issues = _validate_v3(
        results_dir=out_dir,
        expected_repeat=args.repeat,
        expected_samples=args.samples,
        focus_metrics_only=bool(args.focus_metrics_only),
    )

    print("\n[validate] result_dir:", out_dir)
    if ok:
        print("[validate] PASS")
    else:
        print("[validate] FAIL")
        for i, msg in enumerate(issues, start=1):
            print(f"  {i}. {msg}")
        sys.exit(2)

    print(csv_path)


if __name__ == "__main__":
    main()
