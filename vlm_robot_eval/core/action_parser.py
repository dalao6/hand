import json
import re
from typing import Any, Dict, List


_VALID_SCENE_LABELS = {"dining_table", "kitchen", "office", "indoor", "unknown"}

# Alias map: normalized-lowercase-key -> canonical label
_SCENE_ALIASES: Dict[str, str] = {
    # dining_table variants
    "dining table": "dining_table",
    "dining-table": "dining_table",
    "dining room": "dining_table",
    "dining_room": "dining_table",
    "table": "dining_table",
    "dinner table": "dining_table",
    "dinner_table": "dining_table",
    "restaurant": "dining_table",
    "cafeteria": "dining_table",
    # kitchen variants
    "kitchen": "kitchen",
    "kitchen counter": "kitchen",
    "kitchen_counter": "kitchen",
    "kitchenette": "kitchen",
    "food": "kitchen",
    # office variants
    "office": "office",
    "workspace": "office",
    "work space": "office",
    "desk": "office",
    "study": "office",
    "lab": "office",
    "laboratory": "office",
    "computer": "office",
    # indoor variants
    "indoor": "indoor",
    "indoors": "indoor",
    "room": "indoor",
    "living room": "indoor",
    "living_room": "indoor",
    "bedroom": "indoor",
    "hallway": "indoor",
    "corridor": "indoor",
    "shelf": "indoor",
    "shelves": "indoor",
    # unknown
    "unknown": "unknown",
    "n/a": "unknown",
    "none": "unknown",
    "other": "unknown",
    "unrecognized": "unknown",
}

# Regex patterns for keyword-based fallback (checked in priority order)
_SCENE_KEYWORD_PATTERNS: List[tuple] = [
    (re.compile(r"\bdining[\s_-]*table\b|\bdining[\s_-]*room\b|\bdinner[\s_-]*table\b", re.I), "dining_table"),
    (re.compile(r"\bkitchen\b|\bcooking\b|\bcupboard\b|\bcounter\b|\bcountertop\b", re.I), "kitchen"),
    (re.compile(r"\boffice\b|\bworkspace\b|\bdesk\b|\blaboratory\b|\blab\b|\bcomputer\b|\bmonitor\b", re.I), "office"),
    (re.compile(r"\bindoor\b|\bliving[\s_-]*room\b|\bbedroom\b|\bhallway\b|\bshelf\b|\bshelves\b", re.I), "indoor"),
    (re.compile(r"\bunknown\b", re.I), "unknown"),
]


def _normalize_scene_type(raw: Any) -> str:
    """Normalize a raw scene-type string to a canonical label.

    Priority:
    1) Exact canonical match (with underscore/hyphen normalization)
    2) Alias dictionary lookup
    3) Keyword regex scan
    """
    if not isinstance(raw, str):
        return ""
    s = raw.strip().lower().replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    if not s:
        return ""
    s2 = s.replace(" ", "_")
    # 1) direct canonical
    if s2 in _VALID_SCENE_LABELS:
        return s2
    # 2) alias dict
    alias = _SCENE_ALIASES.get(s) or _SCENE_ALIASES.get(s2)
    if alias:
        return alias
    # 3) keyword regex scan over the raw string
    for pat, label in _SCENE_KEYWORD_PATTERNS:
        if pat.search(s):
            return label
    return ""


def _regex_extract_scene(text: str) -> str:
    """Fallback: extract scene label from free-form text via alias/keyword matching."""
    if not isinstance(text, str) or not text.strip():
        return ""
    # Try to find an explicit scene_type key in the text
    m = re.search(
        r'"?scene_type"?\s*[=:]\s*"?([A-Za-z0-9_\- ]{2,40})"?',
        text,
        re.IGNORECASE,
    )
    if m:
        candidate = _normalize_scene_type(m.group(1))
        if candidate:
            return candidate
    # Keyword scan over full text
    s = text.strip().lower()
    for pat, label in _SCENE_KEYWORD_PATTERNS:
        if pat.search(s):
            return label
    return ""


def _regex_extract_objects(text: str) -> List[str]:
    """Fallback: pull quoted words from an 'objects' array in free-form text."""
    if not isinstance(text, str) or not text.strip():
        return []
    m = re.search(r'"?objects"?\s*[=:]\s*\[([^\]]*)\]', text, re.IGNORECASE)
    if not m:
        return []
    raw_items = m.group(1)
    items = re.findall(r'"([^"]{1,40})"', raw_items)
    seen: set = set()
    out: List[str] = []
    for it in items:
        s = it.strip().lower()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out[:5]


def _first_json_value(text: str) -> Any:
    if not isinstance(text, str) or not text.strip():
        return None

    s = text.strip()
    if "```" in s:
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)

    try:
        return json.loads(s)
    except Exception:
        pass

    decoder = json.JSONDecoder()
    last: Any = None
    for i, ch in enumerate(s):
        if ch not in "[{":
            continue
        try:
            obj, _ = decoder.raw_decode(s[i:])
            last = obj
        except Exception:
            continue
    return last


def _clean_target(raw: str) -> str:
    t = re.sub(r"[^a-zA-Z0-9_\- ]+", " ", raw).strip().lower()
    t = re.sub(r"\s+", " ", t)
    stop = {"it", "them", "object", "objects", "one", "there", "here", "and", "then"}
    if t in stop:
        return ""
    return t


def _keyword_to_action(clause: str) -> str:
    c = clause.lower()
    if re.search(r"\b(pick|pickup|pick up|grab|grasp|take)\b", c):
        return "grasp"
    if re.search(r"\b(place|put|drop|release)\b", c):
        return "release"
    if re.search(r"\b(move|push|go to|approach)\b", c):
        return "push"
    if re.search(r"\b(rotate|turn)\b", c):
        return "rotate"
    return ""


def _extract_target_from_clause(clause: str) -> str:
    c = clause.lower()
    patterns = [
        r"(?:the|to|toward|towards|of|near|next to)\s+([a-zA-Z0-9_\- ]+)",
        r"\b([a-zA-Z]+_[0-9]+)\b",
    ]
    for p in patterns:
        m = re.search(p, c)
        if not m:
            continue
        tgt = _clean_target(m.group(1))
        if tgt:
            return tgt

    toks = [x for x in re.findall(r"[a-zA-Z0-9_\-]+", c) if x not in {"pick", "up", "move", "place", "put", "and", "then"}]
    return _clean_target(toks[-1]) if toks else ""


def _keyword_fallback(text: str) -> List[Dict[str, Any]]:
    if not isinstance(text, str) or not text.strip():
        return []

    clauses = re.split(r"[\n\.;]|\band\b|\bthen\b", text, flags=re.IGNORECASE)
    out: List[Dict[str, Any]] = []
    for cl in clauses:
        c = cl.strip()
        if not c:
            continue
        action = _keyword_to_action(c)
        if not action:
            continue
        target = _extract_target_from_clause(c)
        if not target:
            continue
        out.append({"action": action, "target": target})
    return out


def _pick_target_from_instruction(instruction: str, objects: List[str]) -> str:
    inst = _clean_target(instruction)
    if not objects:
        return ""

    norm_objs = [str(o).strip().lower() for o in objects if str(o).strip()]
    if not norm_objs:
        return ""

    # Prefer id-like targets (e.g., cup_1) that are explicitly mentioned
    for o in norm_objs:
        if o in inst:
            return o

    # Then match by category tokens against id-like object names
    words = set(re.findall(r"[a-zA-Z0-9_\-]+", inst))
    for o in norm_objs:
        if "_" in o:
            cat = o.split("_")[0]
            if cat in words:
                return o

    return norm_objs[0]


def build_action_fallback(instruction: str, objects: List[str]) -> List[Dict[str, Any]]:
    inst = str(instruction or "").lower()
    target = _pick_target_from_instruction(inst, objects)
    if not target:
        return []

    if re.search(r"\b(pick|pickup|pick up|grab|grasp|take)\b", inst):
        return [{"action": "move_to", "target": target}, {"action": "grasp", "target": target}]
    if re.search(r"\b(place|put|drop|release)\b", inst):
        return [{"action": "move_to", "target": target}, {"action": "release", "target": target}]
    if re.search(r"\b(push)\b", inst):
        return [{"action": "move_to", "target": target}, {"action": "push", "target": target}]
    if re.search(r"\b(rotate|turn)\b", inst):
        return [{"action": "rotate", "target": target}]
    if re.search(r"\b(move|go to|approach|navigate)\b", inst):
        return [{"action": "move_to", "target": target}]
    return [{"action": "move_to", "target": target}]


def parse_action_json(text: str) -> Dict[str, Any]:
    """Parse model output to unified {"action_sequence": [...]}.

    Robustness policy:
    1) Try strict/partial JSON extraction first
    2) If failed, try regex pseudo-JSON extraction
    3) If still failed, apply keyword mapping fallback (pick/place/move/...)
    4) Only then return empty output
    """

    obj = _first_json_value(text)

    if isinstance(obj, dict) and isinstance(obj.get("action_sequence"), list):
        seq = obj.get("action_sequence", [])
    elif isinstance(obj, list):
        # 兼容部分 VLM 输出: [{"action_sequence": [...]}]
        if len(obj) == 1 and isinstance(obj[0], dict) and isinstance(obj[0].get("action_sequence"), list):
            seq = obj[0].get("action_sequence", [])
        else:
            seq = obj
    elif isinstance(obj, dict) and isinstance(obj.get("action"), str) and isinstance(obj.get("target"), str):
        seq = [obj]
    else:
        seq = []
        for m in re.finditer(
            r"action\s*[\":=]+\s*\"?([a-zA-Z_]+)\"?.{0,120}?target\s*[\":=]+\s*\"?([a-zA-Z0-9_\- ]+)\"?",
            text or "",
            flags=re.IGNORECASE | re.DOTALL,
        ):
            seq.append({"action": m.group(1), "target": m.group(2)})

    if not seq:
        seq = _keyword_fallback(text)

    if not isinstance(seq, list):
        return {"action_sequence": []}

    cleaned: List[dict] = []
    for item in seq:
        if not isinstance(item, dict):
            continue

        action = item.get("action")
        target = item.get("target")
        if not isinstance(action, str) or not isinstance(target, str):
            continue

        action = action.strip().lower()
        target = _clean_target(target)
        if not action or not target:
            continue

        out: Dict[str, Any] = {"action": action, "target": target}
        for k in ("angle", "force", "speed"):
            v = item.get(k)
            if isinstance(v, (int, float)):
                out[k] = float(v)

        cleaned.append(out)

    return {"action_sequence": cleaned}


def parse_objects_json(text: str) -> Dict[str, Any]:
    """Parse model output for objects + scene_type.

    Robustness policy:
    1) Strict/partial JSON extraction
    2) If JSON missing / scene empty: regex key-value extraction
    3) Keyword regex scan over full text
    4) Always return a valid dict (never None)
    """
    obj = _first_json_value(text)
    raw_objs: Any = []
    scene_type = ""

    if isinstance(obj, dict):
        raw_objs = obj.get("objects", [])
        scene_type = _normalize_scene_type(obj.get("scene_type", ""))

    # -- objects --
    out_objs: List[str] = []
    seen: set = set()
    if isinstance(raw_objs, list):
        for x in raw_objs:
            if not isinstance(x, str):
                continue
            s = x.strip().lower()
            if not s or s in seen:
                continue
            seen.add(s)
            out_objs.append(s)
    out_objs = out_objs[:5]

    # Regex fallback for objects if JSON parsing yielded nothing
    if not out_objs:
        out_objs = _regex_extract_objects(text)

    # -- scene_type --
    # If JSON gave nothing, try regex extraction from full text
    if not scene_type:
        scene_type = _regex_extract_scene(text)

    return {"objects": out_objs, "scene_type": scene_type}
