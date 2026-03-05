from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import urllib.request
import zipfile
from typing import Any, Dict, List, Tuple


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


def _extract_zip(zip_path: str, dst_dir: str, must_have: str | None = None) -> None:
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
    """Ensure COCO val2017 images and instances_val2017.json exist.

    Returns:
        (val_images_dir, instances_json_path)
    """

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


def _choose_instruction(objects: List[str]) -> Tuple[str, str, List[str], List[Dict[str, Any]]]:
    obj_set = {o for o in objects}

    if "cup" in obj_set and "dining table" in obj_set:
        instruction = "Pick up the cup and place it on the dining table"
        scene_type = "pick_and_place"
        targets = ["cup", "dining table"]
        expected = [
            {"action": "move_to", "target": "cup"},
            {"action": "grasp", "target": "cup"},
            {"action": "move_to", "target": "dining table"},
            {"action": "release", "target": "cup"},
        ]
        return instruction, scene_type, targets, expected

    if "bottle" in obj_set:
        instruction = "Pick up the bottle"
        scene_type = "pick"
        targets = ["bottle"]
        expected = [
            {"action": "move_to", "target": "bottle"},
            {"action": "grasp", "target": "bottle"},
        ]
        return instruction, scene_type, targets, expected

    if "chair" in obj_set:
        instruction = "Move the chair"
        scene_type = "move"
        targets = ["chair"]
        expected = [
            {"action": "move_to", "target": "chair"},
            {"action": "push", "target": "chair"},
        ]
        return instruction, scene_type, targets, expected

    if "book" in obj_set:
        instruction = "Pick up the book"
        scene_type = "pick"
        targets = ["book"]
        expected = [
            {"action": "move_to", "target": "book"},
            {"action": "grasp", "target": "book"},
        ]
        return instruction, scene_type, targets, expected

    # Fallback for remaining target categories
    for x in ["backpack", "laptop", "mouse", "remote", "bowl", "cup", "dining table"]:
        if x in obj_set:
            if x == "dining table":
                instruction = "Move to the dining table"
                scene_type = "navigate"
                targets = ["dining table"]
                expected = [{"action": "move_to", "target": "dining table"}]
                return instruction, scene_type, targets, expected
            instruction = f"Pick up the {x}"
            scene_type = "pick"
            targets = [x]
            expected = [
                {"action": "move_to", "target": x},
                {"action": "grasp", "target": x},
            ]
            return instruction, scene_type, targets, expected

    instruction = "Move to the object"
    scene_type = "unknown"
    targets = []
    expected = []
    return instruction, scene_type, targets, expected


def build_ground_truth_dataset(
    coco_root: str,
    out_json_path: str,
    num_images: int = 30,
    seed: int = 42,
    copy_images: bool = True,
) -> str:
    from pycocotools.coco import COCO

    img_dir, inst_json = _prepare_coco(coco_root)

    coco = COCO(inst_json)
    cat_ids = coco.getCatIds(catNms=TARGET_CATEGORIES)
    # getImgIds with multiple catIds uses AND logic; use OR by querying each category separately
    img_id_set: set = set()
    for cid in cat_ids:
        img_id_set.update(coco.getImgIds(catIds=[cid]))
    img_ids = list(img_id_set)

    rnd = random.Random(seed)
    img_ids = list(img_ids)
    if len(img_ids) == 0:
        raise RuntimeError("No COCO images found for target categories.")

    if num_images > len(img_ids):
        num_images = len(img_ids)
    chosen = rnd.sample(img_ids, num_images)

    out_dir = os.path.dirname(out_json_path)
    _ensure_dir(out_dir)

    subset_dir = os.path.join(out_dir, "coco_val2017_subset")
    if copy_images:
        _ensure_dir(subset_dir)

    dataset: List[Dict[str, Any]] = []

    for idx, img_id in enumerate(chosen):
        img_info = coco.loadImgs([img_id])[0]
        file_name = img_info.get("file_name", "")
        if not isinstance(file_name, str) or not file_name:
            continue

        ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        present_cat_ids = {int(a.get("category_id")) for a in anns if isinstance(a, dict) and "category_id" in a}
        present_names = []
        for cid in present_cat_ids:
            c = coco.loadCats([cid])[0]
            nm = c.get("name", "")
            if isinstance(nm, str) and nm in TARGET_CATEGORIES:
                present_names.append(nm)
        present_names = sorted(set(present_names))

        instruction, scene_type, target_objects, expected_seq = _choose_instruction(present_names)

        src_img = os.path.join(img_dir, file_name)
        rel_img = file_name
        if copy_images:
            dst_img = os.path.join(subset_dir, file_name)
            if not os.path.exists(dst_img):
                shutil.copy2(src_img, dst_img)
            rel_img = os.path.join("coco_val2017_subset", file_name)
        else:
            rel_img = src_img

        item = {
            "id": int(img_id),
            "image": rel_img,
            "instruction": instruction,
            "ground_truth": {
                "scene_type": scene_type,
                "objects": present_names,
                "target_objects": target_objects,
                "expected_action_sequence": expected_seq,
                "hardware_constraints": {
                    "joint_angle_range": [-90, 90],
                    "grip_force_range": [0, 30],
                    "move_speed_range": [0, 1.0],
                },
            },
        }
        dataset.append(item)

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    return out_json_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coco_root",
        default=os.getenv("COCO2017_ROOT", os.path.expanduser("~/coco2017")),
        help="COCO 2017 root dir containing val2017/ and annotations/",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "ground_truth_dataset.json"),
        help="Output dataset json path",
    )
    parser.add_argument("--num_images", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_copy_images", action="store_true")
    args = parser.parse_args()

    out = build_ground_truth_dataset(
        coco_root=args.coco_root,
        out_json_path=args.out,
        num_images=args.num_images,
        seed=args.seed,
        copy_images=not args.no_copy_images,
    )
    print(out)


if __name__ == "__main__":
    main()
