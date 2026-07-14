"""Pure scorer for the BRANCH divergence pilot.

Reads only an ``Answer:`` line and applies a per-probe rule. No model, no
network, no I/O. Mirrors the scoring conventions in
docs/superpowers/specs/2026-06-12-branch-golden-probes-enum-seed-draft.md (§3):

  - Answer-line extraction: LAST line matching ``^\\s*Answer:\\s*(.+?)\\s*$``
    (case-insensitive). Absent -> format_fail (and never collapsed into the
    accuracy signal: a low score with format_fail set means "didn't emit the
    line," not "wrong frame").
  - Numeric: parse the first number in the field (strip units/commas/$/%),
    PASS iff inside the closed interval [lo, hi].
  - Token: case-insensitive match against a closed menu; PASS iff the matched
    option equals the correct one.
  - foil_match: a diagnostic flag — did the answer land on a documented foil
    (the predicted frame failure), as opposed to an unrelated error.
"""
from __future__ import annotations

import re
from typing import Any

_ANSWER_RE = re.compile(r"^\s*Answer:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_answer(text: str) -> str | None:
    """Return the value of the LAST ``Answer: <value>`` line, or None if absent."""
    matches = _ANSWER_RE.findall(text or "")
    return matches[-1].strip() if matches else None


def first_number(s: str) -> float | None:
    """First numeric token in ``s`` after stripping commas, $, and %."""
    cleaned = (s or "").replace(",", "").replace("$", "").replace("%", "")
    m = _NUM_RE.search(cleaned)
    return float(m.group()) if m else None


def score_numeric(field: str, lo: float, hi: float) -> bool:
    n = first_number(field)
    return n is not None and lo <= n <= hi


def _norm_token(s: str) -> str:
    t = (s or "").strip().lower().strip(".!?\"' ")
    if t.startswith("the "):
        t = t[4:].strip()
    return t


def score_token(field: str, correct: str, menu: list[str]) -> bool:
    """PASS iff the answer field resolves to the correct menu option.

    An option matches if the field equals it or contains it as a phrase; on
    multiple matches, an exact-equal wins, else the longest option (so
    'API service' beats a bare 'API', 'load balancer' is not shadowed)."""
    a = _norm_token(field)
    cands = [m for m in menu if _norm_token(m) == a or _norm_token(m) in a]
    if not cands:
        return False
    exact = [m for m in cands if _norm_token(m) == a]
    chosen = exact[0] if exact else max(cands, key=lambda m: len(_norm_token(m)))
    return _norm_token(chosen) == _norm_token(correct)


def _numeric_foil_match(val: float | None, foils: list[Any]) -> bool:
    if val is None:
        return False
    for f in foils:
        fv = first_number(str(f))
        if fv is not None and abs(val - fv) <= max(0.02 * abs(fv), 0.2):
            return True
    return False


def score_probe(raw_answer: str, rule: dict) -> dict:
    """Score one branch's raw text against a probe rule.

    Returns {passed, format_fail, value, foil_match}. ``passed`` is the only
    accuracy signal; ``format_fail`` and ``foil_match`` are separate diagnostics
    and are never folded into ``passed``.
    """
    field = extract_answer(raw_answer)
    if field is None:
        return {"passed": False, "format_fail": True, "value": None, "foil_match": False}
    if rule["kind"] == "numeric":
        passed = score_numeric(field, rule["lo"], rule["hi"])
        foil_match = _numeric_foil_match(first_number(field), rule.get("foils", []))
    elif rule["kind"] == "token":
        passed = score_token(field, rule["correct"], rule["menu"])
        foil_match = any(score_token(field, f, rule["menu"]) for f in rule.get("foils", []))
    else:
        raise ValueError(f"unknown rule kind: {rule['kind']!r}")
    return {"passed": passed, "format_fail": False, "value": field, "foil_match": foil_match}
