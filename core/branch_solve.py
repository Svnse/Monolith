"""BRANCH solve pass — place the classified frame at the top, then solve.

The end-to-end BRANCH turn, V1 (advisory commit, single frame):

    task --classify--> problem-type --frame_instruction--> a top-placed steer
         --solve--> answer

The frame steer is built from the type's own enum gloss (core/problem_types.py),
so the system is general: it works for any task the classifier can place, not
just the probe set. The frame goes at the TOP of the solve prompt — the only
place that conditions the trajectory on a no-KV-cache API (the order primitive).

An ``other:<free>`` or unrecognized type yields NO frame (None) — the solve runs
unframed (the baseline). This is deliberate: the enum is closed, and a task the
classifier cannot place onto a real cell must not be steered by a fabricated
frame. Both calls are injectable so the orchestration is unit-tested offline.
"""
from __future__ import annotations

from typing import Callable

from core import branch_classify as bc
from core import problem_types as pt

_FRAME_TEMPLATE = (
    "REASONING FRAME (apply strictly): treat this as a {type_id} problem — "
    "{gloss}{approach} Work the problem only under this frame, then end with a "
    "single line `Answer: <value>` and nothing after it."
)


def frame_instruction(type_id: str | None) -> str | None:
    """Top-placed steer for a classified type, or None for an unplaceable type
    (``other:<free>`` / None) — which falls through to an unframed solve.

    Carries both the descriptive gloss (the coordinate) and the operational
    approach (the steer). The e2e value test showed the gloss alone is too weak
    to override a salient in-task wrong-attractor; the approach is the strength."""
    if not type_id or pt.is_other(type_id):
        return None
    gloss = pt.PROBLEM_TYPES.get(type_id)
    if not gloss:
        return None
    approach = pt.get_approach(type_id)
    approach_clause = f" Approach: {approach}" if approach else ""
    return _FRAME_TEMPLATE.format(type_id=type_id, gloss=gloss, approach=approach_clause)


def branch_turn(
    task: str,
    *,
    classify_call: Callable[[str], str | None],
    solve_call: Callable[[str, str | None], str | None],
) -> dict:
    """Run one staked BRANCH turn.

    classify_call : prompt -> raw classifier text (HTTP or in-app)
    solve_call    : (task, frame_or_None) -> raw solver text

    Returns {type, frame, answer}. Commit is advisory at V1: the single framed
    answer IS the turn output; the frozen `type`/`frame` are the audit record.
    """
    type_id = bc.classify(task, call=classify_call)
    frame = frame_instruction(type_id)
    answer = solve_call(task, frame)
    return {"type": type_id, "frame": frame, "answer": answer}
