from __future__ import annotations

from typing import Any, Dict, List, Tuple


LEGAL_ACTIONS = ["move_to", "grasp", "release", "push", "rotate"]

# Color adjectives stripped when doing fuzzy target matching
_COLOR_WORDS = {
    "red", "blue", "green", "yellow", "white", "black", "orange", "purple",
    "pink", "brown", "grey", "gray", "light", "dark",
}

# Articles stripped when doing fuzzy target matching
_ARTICLES = {"the", "a", "an"}


def _normalize(s: str) -> str:
    """Lowercase, strip leading/trailing whitespace."""
    return s.strip().lower()


def _strip_articles(s: str) -> str:
    """Remove leading articles (the/a/an) from a string."""
    tokens = s.split()
    while tokens and tokens[0] in _ARTICLES:
        tokens = tokens[1:]
    return " ".join(tokens)


def _strip_colors(s: str) -> str:
    """Remove leading color adjectives from a target string."""
    tokens = s.split()
    while tokens and tokens[0] in _COLOR_WORDS:
        tokens = tokens[1:]
    return " ".join(tokens)


def _target_matches(pred: str, gt_set: set) -> bool:
    """Fuzzy target matching:
    1. Exact match in gt_set
    2. Article-stripped pred matches gt_set entry
    3. Color-stripped pred matches any gt_set entry (color-stripped)
    4. pred is a substring of any gt_set entry, or vice-versa
    """
    pred_norm = _normalize(pred)
    if pred_norm in gt_set:
        return True

    pred_no_art = _strip_articles(pred_norm)
    if pred_no_art in gt_set:
        return True

    pred_stripped = _strip_colors(pred_no_art)
    for gt in gt_set:
        gt_norm = _normalize(gt)
        gt_no_art = _strip_articles(gt_norm)
        gt_stripped = _strip_colors(gt_no_art)
        if pred_stripped and pred_stripped == gt_stripped:
            return True
        if pred_norm and (pred_norm in gt_norm or gt_norm in pred_norm):
            return True
        if pred_no_art and (pred_no_art in gt_norm or gt_norm in pred_no_art):
            return True
    return False


def _clamp(v: float, lo: float, hi: float) -> Tuple[float, bool]:
    if v < lo:
        return lo, True
    if v > hi:
        return hi, True
    return v, False


def check_executable(
    action_sequence: List[Dict[str, Any]],
    ground_truth_objects: List[str],
    hardware_constraints: Dict[str, Any] | None = None,
) -> Tuple[List[Dict[str, Any]], float, float]:
    """Check legality + target existence + optional hardware constraints.

    Returns:
        corrected_actions, executable_rate, correction_cost

    correction_cost:
        - removing an illegal action or invalid target counts as 1
        - clamping any numeric parameter (angle/force/speed) counts as 1
        - final cost is averaged by max(len(original_actions), 1)

    Notes:
        Current unified action schema requires at least {action,target}.
        Optional numeric params supported: angle (rotate), force (grasp), speed (move_to/push).
    """

    if not isinstance(action_sequence, list):
        action_sequence = []
    if not isinstance(ground_truth_objects, list):
        ground_truth_objects = []

    obj_set = {str(o).strip().lower() for o in ground_truth_objects if str(o).strip()}

    constraints = hardware_constraints or {}
    joint_lo, joint_hi = (-90.0, 90.0)
    grip_lo, grip_hi = (0.0, 30.0)
    speed_lo, speed_hi = (0.0, 1.0)

    jr = constraints.get("joint_angle_range")
    if isinstance(jr, list) and len(jr) == 2 and all(isinstance(x, (int, float)) for x in jr):
        joint_lo, joint_hi = float(jr[0]), float(jr[1])

    gr = constraints.get("grip_force_range")
    if isinstance(gr, list) and len(gr) == 2 and all(isinstance(x, (int, float)) for x in gr):
        grip_lo, grip_hi = float(gr[0]), float(gr[1])

    sr = constraints.get("move_speed_range")
    if isinstance(sr, list) and len(sr) == 2 and all(isinstance(x, (int, float)) for x in sr):
        speed_lo, speed_hi = float(sr[0]), float(sr[1])

    corrected: List[Dict[str, Any]] = []
    cost = 0.0

    for a in action_sequence:
        if not isinstance(a, dict):
            cost += 1.0
            continue

        action = a.get("action")
        target = a.get("target")
        if not isinstance(action, str) or not isinstance(target, str):
            cost += 1.0
            continue

        action = action.strip()
        target = target.strip()
        if action not in LEGAL_ACTIONS:
            cost += 1.0
            continue
        if not _target_matches(target, obj_set):
            cost += 1.0
            continue

        out: Dict[str, Any] = {"action": action, "target": target}

        # Optional parameter checks
        if action == "rotate" and isinstance(a.get("angle"), (int, float)):
            angle = float(a["angle"])
            angle2, changed = _clamp(angle, joint_lo, joint_hi)
            out["angle"] = angle2
            if changed:
                cost += 1.0

        if action == "grasp" and isinstance(a.get("force"), (int, float)):
            force = float(a["force"])
            force2, changed = _clamp(force, grip_lo, grip_hi)
            out["force"] = force2
            if changed:
                cost += 1.0

        if action in ("move_to", "push") and isinstance(a.get("speed"), (int, float)):
            speed = float(a["speed"])
            speed2, changed = _clamp(speed, speed_lo, speed_hi)
            out["speed"] = speed2
            if changed:
                cost += 1.0

        corrected.append(out)

    denom = len(action_sequence) if len(action_sequence) > 0 else 1
    executable_rate = len(corrected) / denom
    correction_cost = cost / denom
    return corrected, float(executable_rate), float(correction_cost)


def apply_constraints(action_sequence: List[Dict[str, Any]], detected_objects: List[str]) -> Tuple[List[Dict[str, Any]], float]:
    filtered, rate, _ = check_executable(action_sequence, detected_objects, hardware_constraints=None)
    return filtered, rate
