from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List


def _seq_key(action_sequence: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for a in action_sequence:
        if not isinstance(a, dict):
            continue
        parts.append(f"{a.get('action','')}::{a.get('target','')}")
    return "|".join(parts)


def pick_representative(actions_runs: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Pick the most frequent action_sequence among repeated runs."""
    if not actions_runs:
        return []
    keys = [_seq_key(seq) for seq in actions_runs]
    most_common_key, _ = Counter(keys).most_common(1)[0]
    for seq in actions_runs:
        if _seq_key(seq) == most_common_key:
            return seq
    return actions_runs[0]


def consistency_rate(actions_runs: List[List[Dict[str, Any]]]) -> float:
    """Consistency = (max identical action_sequence count) / repeat.

    Special case: if ALL runs return empty sequences, consistency is 0.0
    (an empty output is not a valid consistent action plan).
    """
    if not actions_runs:
        return 0.0
    # If every run produced an empty sequence, treat as inconsistent/failed
    if all(len(seq) == 0 for seq in actions_runs):
        return 0.0
    keys = [_seq_key(seq) for seq in actions_runs]
    most = Counter(keys).most_common(1)[0][1] if keys else 0
    return most / len(actions_runs)


def actions_to_text(action_sequence: List[Dict[str, Any]]) -> str:
    """Convert action sequence into a plain text string for semantic similarity."""
    parts: List[str] = []
    for a in action_sequence:
        if not isinstance(a, dict):
            continue
        act = a.get("action", "")
        tgt = a.get("target", "")
        if not isinstance(act, str) or not isinstance(tgt, str):
            continue
        act = act.strip()
        tgt = tgt.strip()
        if not act or not tgt:
            continue
        parts.append(f"{act} {tgt}")
    return " ".join(parts)
