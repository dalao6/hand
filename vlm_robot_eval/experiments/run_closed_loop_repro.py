from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from vlm_robot_eval.core.evaluator import run_evaluation_v3
from vlm_robot_eval.experiments.pybullet_sim_minimal import run_minimal_sim


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _default_dataset() -> str:
    here = os.path.dirname(__file__)
    return os.path.join(here, "ground_truth_dataset_v3.json")


def _default_results_root() -> str:
    here = os.path.dirname(__file__)
    return os.path.join(os.path.dirname(here), "results")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducible closed-loop eval: language planning -> PyBullet execution")
    parser.add_argument("--dataset", default=_default_dataset())
    parser.add_argument("--results_root", default=_default_results_root())
    parser.add_argument("--run_name", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--sim_max_samples", type=int, default=12)
    parser.add_argument("--sim_gui", action="store_true")
    parser.add_argument("--sim_enable_domain_randomization", action="store_true")
    parser.add_argument("--sim_domain_randomization_strength", type=float, default=0.2)
    parser.add_argument("--sim_camera_pose_jitter", type=float, default=0.0)
    parser.add_argument("--sim_camera_width", type=int, default=640)
    parser.add_argument("--sim_camera_height", type=int, default=480)
    parser.add_argument("--sim_camera_calibration_out", default="")
    parser.add_argument("--enable_stress_test", action="store_true")
    parser.add_argument("--disable_qwen_deterministic", action="store_true")
    parser.add_argument("--focus_metrics_only", action="store_true")
    args = parser.parse_args()

    _set_seed(args.seed)

    run_name = str(args.run_name).strip()
    if not run_name:
        run_name = f"closed_loop_seed{int(args.seed)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = os.path.join(args.results_root, run_name)
    os.makedirs(out_dir, exist_ok=True)

    all_csv = run_evaluation_v3(
        dataset_path=args.dataset,
        out_dir=out_dir,
        repeat=int(args.repeat),
        max_items=int(args.samples),
        enable_stress_test=bool(args.enable_stress_test),
        qwen_deterministic=not bool(args.disable_qwen_deterministic),
        focus_metrics_only=bool(args.focus_metrics_only),
    )

    sim_out_dir = os.path.join(out_dir, "pybullet_closed_loop")
    grid_path = run_minimal_sim(
        dataset_path=args.dataset,
        out_dir=sim_out_dir,
        debug_jsonl=os.path.join(out_dir, "debug.jsonl"),
        max_samples=max(1, int(args.sim_max_samples)),
        gui=bool(args.sim_gui),
        seed=int(args.seed),
        enable_domain_randomization=bool(args.sim_enable_domain_randomization),
        domain_randomization_strength=float(args.sim_domain_randomization_strength),
        camera_pose_jitter=float(args.sim_camera_pose_jitter),
        camera_width=max(16, int(args.sim_camera_width)),
        camera_height=max(16, int(args.sim_camera_height)),
        camera_calibration_out=str(args.sim_camera_calibration_out),
    )

    manifest: Dict[str, Any] = {
        "seed": int(args.seed),
        "dataset": os.path.abspath(args.dataset),
        "repeat": int(args.repeat),
        "samples": int(args.samples),
        "sim_max_samples": int(args.sim_max_samples),
        "sim_enable_domain_randomization": bool(args.sim_enable_domain_randomization),
        "sim_domain_randomization_strength": float(args.sim_domain_randomization_strength),
        "sim_camera_pose_jitter": float(args.sim_camera_pose_jitter),
        "sim_camera_width": int(args.sim_camera_width),
        "sim_camera_height": int(args.sim_camera_height),
        "out_dir": os.path.abspath(out_dir),
        "all_results_csv": os.path.abspath(all_csv),
        "debug_jsonl": os.path.abspath(os.path.join(out_dir, "debug.jsonl")),
        "sim_summary_csv": os.path.abspath(os.path.join(sim_out_dir, "summary_metrics_sim.csv")),
        "sim_by_model_csv": os.path.abspath(os.path.join(sim_out_dir, "summary_metrics_sim_by_model.csv")),
        "sim_sample_csv": os.path.abspath(os.path.join(sim_out_dir, "sample_metrics_sim.csv")),
        "sim_camera_calibration_json": os.path.abspath(
            str(args.sim_camera_calibration_out).strip() if str(args.sim_camera_calibration_out).strip() else os.path.join(sim_out_dir, "camera_calibration.json")
        ),
        "grid_path": os.path.abspath(grid_path),
    }
    with open(os.path.join(out_dir, "closed_loop_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(all_csv)
    print(grid_path)
    print(os.path.join(out_dir, "closed_loop_manifest.json"))


if __name__ == "__main__":
    main()
