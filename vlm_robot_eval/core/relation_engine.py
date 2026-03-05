from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple


BBox = List[float]
Obj = Dict[str, object]


def bbox_center(bbox: BBox) -> Tuple[float, float]:
    x, y, w, h = [float(v) for v in bbox]
    return x + w / 2.0, y + h / 2.0


def bbox_area(bbox: BBox) -> float:
    _, _, w, h = [float(v) for v in bbox]
    return max(w, 0.0) * max(h, 0.0)


def iou(b1: BBox, b2: BBox) -> float:
    x1, y1, w1, h1 = [float(v) for v in b1]
    x2, y2, w2, h2 = [float(v) for v in b2]
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)
    inter_w = max(0.0, xb - xa)
    inter_h = max(0.0, yb - ya)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    union = bbox_area(b1) + bbox_area(b2) - inter
    return inter / union if union > 0 else 0.0


def leftmost(objs: List[Obj]) -> Optional[Obj]:
    return min(objs, key=lambda o: bbox_center(o["bbox"])[0]) if objs else None


def rightmost(objs: List[Obj]) -> Optional[Obj]:
    return max(objs, key=lambda o: bbox_center(o["bbox"])[0]) if objs else None


def largest(objs: List[Obj]) -> Optional[Obj]:
    return max(objs, key=lambda o: bbox_area(o["bbox"])) if objs else None


def smallest(objs: List[Obj]) -> Optional[Obj]:
    return min(objs, key=lambda o: bbox_area(o["bbox"])) if objs else None


def left_of(a: Obj, b: Obj) -> bool:
    return bbox_center(a["bbox"])[0] < bbox_center(b["bbox"])[0]


def right_of(a: Obj, b: Obj) -> bool:
    return bbox_center(a["bbox"])[0] > bbox_center(b["bbox"])[0]


def above(a: Obj, b: Obj) -> bool:
    return bbox_center(a["bbox"])[1] < bbox_center(b["bbox"])[1]


def closest_to(anchor: Obj, candidates: List[Obj]) -> Optional[Obj]:
    if not candidates:
        return None
    ax, ay = bbox_center(anchor["bbox"])

    def _dist(o: Obj) -> float:
        bx, by = bbox_center(o["bbox"])
        return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

    return min(candidates, key=_dist)


def dedup_by_iou(objs: List[Obj], iou_threshold: float = 0.8) -> List[Obj]:
    kept: List[Obj] = []
    for o in sorted(objs, key=lambda x: bbox_area(x["bbox"]), reverse=True):
        if any(iou(o["bbox"], k["bbox"]) >= iou_threshold for k in kept):
            continue
        kept.append(o)
    return kept
