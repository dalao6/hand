from __future__ import annotations

import json
import os
import random
import shutil
import urllib.request
import zipfile
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from pycocotools.coco import COCO

from vlm_robot_eval.core.relation_engine import (
    above,
    bbox_area,
    closest_to,
    dedup_by_iou,
    largest,
    left_of,
    leftmost,
    right_of,
    rightmost,
    smallest,
)


TARGET_CATEGORIES = [
    "cup",
    "bottle",
    "book",
    "chair",
    "backpack",
    "laptop",
    "mouse",
    "remote",
    "bowl",
    "dining table",
]

TASK_PLAN = {
    "simple": 12,
    "multi_instance": 18,
    "spatial_relation": 15,
    "multi_step": 9,
    "negative": 6,
}

COCO_VAL2017_ZIP = "https://images.cocodataset.org/zips/val2017.zip"
COCO_ANN_ZIP = "https://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _download(url: str, dst: str) -> None:
    _ensure_dir(os.path.dirname(dst))
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return
    tmp = dst + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dst)


def _extract_zip(zip_path: str, dst_dir: str, must_have: Optional[str] = None) -> None:
    if must_have and os.path.exists(os.path.join(dst_dir, must_have)):
        return
    _ensure_dir(dst_dir)
    dst_abs = os.path.abspath(dst_dir)
    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.namelist():
            target = os.path.abspath(os.path.join(dst_dir, member))
            if not target.startswith(dst_abs + os.sep) and target != dst_abs:
                raise RuntimeError(f"Unsafe zip member path detected: {member}")
        z.extractall(dst_dir)


def _prepare_coco(coco_root: str) -> Tuple[str, str]:
    ann_dir = os.path.join(coco_root, "annotations")
    img_dir = os.path.join(coco_root, "val2017")
    inst_json = os.path.join(ann_dir, "instances_val2017.json")

    if os.path.exists(img_dir) and os.path.exists(inst_json):
        return img_dir, inst_json

    _ensure_dir(coco_root)
    ann_zip = os.path.join(coco_root, "_downloads", "annotations_trainval2017.zip")
    val_zip = os.path.join(coco_root, "_downloads", "val2017.zip")

    if not os.path.exists(inst_json):
        _download(COCO_ANN_ZIP, ann_zip)
        _extract_zip(ann_zip, coco_root, must_have=os.path.join("annotations", "instances_val2017.json"))

    if not os.path.exists(img_dir):
        _download(COCO_VAL2017_ZIP, val_zip)
        _extract_zip(val_zip, coco_root, must_have=os.path.join("val2017", "000000000139.jpg"))

    return img_dir, inst_json


def _scene_type_from_categories(cats: List[str]) -> str:
    s = set(cats)
    if "dining table" in s:
        return "dining_table"
    if "laptop" in s or "mouse" in s or "book" in s:
        return "office"
    if "cup" in s or "bottle" in s or "bowl" in s:
        return "kitchen"
    if "chair" in s:
        return "indoor"
    return "unknown"


def _mk_action(action: str, target: str) -> Dict[str, str]:
    return {"action": action, "target": target}


def _to_record_obj(ann: Dict[str, Any], cat_name: str, k: int) -> Dict[str, Any]:
    return {
        "id": f"{cat_name.replace(' ', '_')}_{k}",
        "category": cat_name,
        "bbox": [float(x) for x in ann.get("bbox", [0, 0, 0, 0])],
    }


def _build_simple(objs: List[Dict[str, Any]], rng: random.Random) -> Optional[Dict[str, Any]]:
    t = rng.choice(objs)
    ins = f"Pick up the {t['category']}"
    return {
        "instruction": ins,
        "reasoning_type": "simple",
        "difficulty_level": 1,
        "target_object_id": t["id"],
        "relation_object_id": "",
        "expected_action_sequence": [_mk_action("move_to", t["id"]), _mk_action("grasp", t["id"])],
    }


def _build_multi_instance(objs: List[Dict[str, Any]], rng: random.Random) -> Optional[Dict[str, Any]]:
    by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for o in objs:
        by_cat[str(o["category"])].append(o)
    cands = [v for v in by_cat.values() if len(v) >= 2]
    if not cands:
        return None
    group = rng.choice(cands)
    mode = rng.choice(["left", "right", "largest", "smallest"])

    if mode == "left":
        t = leftmost(group)
        ins = f"Pick up the left {t['category']}"
    elif mode == "right":
        t = rightmost(group)
        ins = f"Pick up the right {t['category']}"
    elif mode == "largest":
        t = largest(group)
        ins = f"Move the largest {t['category']}"
    else:
        t = smallest(group)
        ins = f"Move the smallest {t['category']}"

    act2 = "grasp" if "Pick up" in ins else "push"
    return {
        "instruction": ins,
        "reasoning_type": "multi_instance",
        "difficulty_level": 2,
        "target_object_id": t["id"],
        "relation_object_id": "",
        "expected_action_sequence": [_mk_action("move_to", t["id"]), _mk_action(act2, t["id"])],
    }


def _build_spatial(objs: List[Dict[str, Any]], rng: random.Random) -> Optional[Dict[str, Any]]:
    if len(objs) < 2:
        return None
    a = rng.choice(objs)
    others = [o for o in objs if o["id"] != a["id"]]
    if not others:
        return None
    b = rng.choice(others)

    rel = rng.choice(["left_of", "right_of", "above", "next_to"])
    if rel == "left_of":
        if not left_of(a, b):
            a, b = b, a
        ins = f"Move the {a['category']} left of the {b['category']}"
    elif rel == "right_of":
        if not right_of(a, b):
            a, b = b, a
        ins = f"Move the {a['category']} right of the {b['category']}"
    elif rel == "above":
        if not above(a, b):
            a, b = b, a
        ins = f"Pick up the {a['category']} above the {b['category']}"
    else:
        near = closest_to(a, others)
        if near is None:
            return None
        b = near
        ins = f"Pick up the {a['category']} next to the {b['category']}"

    act2 = "grasp" if "Pick up" in ins else "push"
    return {
        "instruction": ins,
        "reasoning_type": "spatial_relation",
        "difficulty_level": 3,
        "target_object_id": a["id"],
        "relation_object_id": b["id"],
        "expected_action_sequence": [_mk_action("move_to", a["id"]), _mk_action(act2, a["id"])],
    }


def _build_multi_step(objs: List[Dict[str, Any]], rng: random.Random) -> Optional[Dict[str, Any]]:
    if len(objs) < 2:
        return None
    a = rng.choice(objs)
    others = [o for o in objs if o["id"] != a["id"]]
    b = rng.choice(others)
    ins = f"Pick up the {a['category']} and place it next to the {b['category']}"
    return {
        "instruction": ins,
        "reasoning_type": "multi_step",
        "difficulty_level": 4,
        "target_object_id": a["id"],
        "relation_object_id": b["id"],
        "expected_action_sequence": [
            _mk_action("move_to", a["id"]),
            _mk_action("grasp", a["id"]),
            _mk_action("move_to", b["id"]),
            _mk_action("release", a["id"]),
        ],
    }


def _build_negative(objs: List[Dict[str, Any]], rng: random.Random) -> Optional[Dict[str, Any]]:
    by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for o in objs:
        by_cat[str(o["category"])].append(o)

    one_inst = [cat for cat, arr in by_cat.items() if len(arr) == 1]
    if one_inst and rng.random() < 0.6:
        cat = rng.choice(one_inst)
        ins = f"Pick up the right {cat}"
    else:
        cat = rng.choice(list(by_cat.keys()))
        ins = f"Pick up the red {cat}"

    return {
        "instruction": ins,
        "reasoning_type": "negative",
        "difficulty_level": 5,
        "target_object_id": "",
        "relation_object_id": "",
        "expected_action_sequence": [],
    }


def _build_task(task_type: str, objs: List[Dict[str, Any]], rng: random.Random) -> Optional[Dict[str, Any]]:
    if task_type == "simple":
        return _build_simple(objs, rng)
    if task_type == "multi_instance":
        return _build_multi_instance(objs, rng)
    if task_type == "spatial_relation":
        return _build_spatial(objs, rng)
    if task_type == "multi_step":
        return _build_multi_step(objs, rng)
    if task_type == "negative":
        return _build_negative(objs, rng)
    return None


def build_ground_truth_dataset_v3(
    coco_root: str,
    out_json_path: str,
    num_samples: int = 60,
    seed: Optional[int] = None,
    copy_images: bool = True,
) -> str:
    img_dir, inst_json = _prepare_coco(coco_root)
    coco = COCO(inst_json)

    cat_ids = coco.getCatIds(catNms=TARGET_CATEGORIES)
    id_to_name = {c["id"]: c["name"] for c in coco.loadCats(cat_ids)}

    img_ids = set()
    for cid in cat_ids:
        img_ids.update(coco.getImgIds(catIds=[cid]))
    img_ids = list(img_ids)

    rng = random.Random(seed) if seed is not None else random.Random()
    rng.shuffle(img_ids)

    out_dir = os.path.dirname(out_json_path)
    _ensure_dir(out_dir)
    subset_dir = os.path.join(out_dir, "coco_val2017_subset_v3")
    if copy_images:
        _ensure_dir(subset_dir)

    candidates: List[Dict[str, Any]] = []
    for img_id in img_ids:
        img_info = coco.loadImgs([img_id])[0]
        W = float(img_info.get("width", 0))
        H = float(img_info.get("height", 0))
        if W <= 0 or H <= 0:
            continue
        image_area = W * H

        ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=cat_ids, iscrowd=False)
        anns = coco.loadAnns(ann_ids)

        by_cat_count: Dict[str, int] = defaultdict(int)
        objs_raw: List[Dict[str, Any]] = []
        for a in anns:
            cid = int(a.get("category_id", -1))
            cat = id_to_name.get(cid, "")
            if cat not in TARGET_CATEGORIES:
                continue
            b = a.get("bbox", [0, 0, 0, 0])
            if not isinstance(b, list) or len(b) != 4:
                continue
            area = bbox_area([float(x) for x in b])
            if area <= 0.01 * image_area:
                continue
            by_cat_count[cat] += 1
            objs_raw.append(_to_record_obj(a, cat, by_cat_count[cat]))

        if len(objs_raw) < 2:
            continue

        objs = dedup_by_iou(objs_raw, iou_threshold=0.8)
        if len(objs) < 2:
            continue

        cat_set = {str(o["category"]) for o in objs}
        if len(cat_set) < 2:
            continue

        image_file = str(img_info.get("file_name", ""))
        if not image_file:
            continue

        src_img = os.path.join(img_dir, image_file)
        rel_img = image_file
        if copy_images:
            dst_img = os.path.join(subset_dir, image_file)
            if not os.path.exists(dst_img):
                shutil.copy2(src_img, dst_img)
            rel_img = os.path.join("coco_val2017_subset_v3", image_file)
        else:
            rel_img = src_img

        candidates.append(
            {
                "image_id": int(img_id),
                "image": rel_img,
                "objects": objs,
                "scene_type": _scene_type_from_categories(list(cat_set)),
            }
        )

    if not candidates:
        raise RuntimeError("No eligible images for v3 dataset after filtering.")

    target_counts = dict(TASK_PLAN)
    if num_samples != 60:
        total = float(sum(TASK_PLAN.values()))
        target_counts = {k: int(round(num_samples * v / total)) for k, v in TASK_PLAN.items()}
        diff = num_samples - sum(target_counts.values())
        keys = list(TASK_PLAN.keys())
        i = 0
        while diff != 0:
            k = keys[i % len(keys)]
            target_counts[k] += 1 if diff > 0 else -1
            diff += -1 if diff > 0 else 1
            i += 1

    dataset: List[Dict[str, Any]] = []
    running_id = 1

    for task_type, need in target_counts.items():
        made = 0
        pool = candidates[:]
        rng.shuffle(pool)
        attempts = 0
        max_attempts = max(need * 30, 200)

        while made < need and attempts < max_attempts:
            attempts += 1
            item = rng.choice(pool)
            spec = _build_task(task_type, item["objects"], rng)
            if spec is None:
                continue

            rec = {
                "id": running_id,
                "image": item["image"],
                "instruction": spec["instruction"],
                "reasoning_type": spec["reasoning_type"],
                "difficulty_level": int(spec["difficulty_level"]),
                "ground_truth": {
                    "objects": item["objects"],
                    "target_object_id": spec["target_object_id"],
                    "relation_object_id": spec["relation_object_id"],
                    "scene_type": item["scene_type"],
                    "expected_action_sequence": spec["expected_action_sequence"],
                    "hardware_constraints": {
                        "joint_angle_range": [-90, 90],
                        "grip_force_range": [0, 30],
                        "move_speed_range": [0, 1.0],
                    },
                },
            }
            dataset.append(rec)
            running_id += 1
            made += 1

        if made < need:
            raise RuntimeError(f"Unable to build enough samples for task={task_type}, need={need}, got={made}")

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    return out_json_path
