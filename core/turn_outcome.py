"""TurnOutcome — composite success-signal computation per task type.

Deterministic ledger view of "did this turn succeed?" Components are
recorded individually; the composite is a per-task-type AND of those
components. Null components are *not failing* — they are excluded from
the AND unless the recipe marks that component as ``null_fails=True``.

Components (all optional booleans):
  no_parse_errors                — verifier callsite has zero parse retries
  no_consecutive_tool_failures   — verifier callsite has no streak of failures
  no_unresolved_assumptions      — len(assumptions) == 0
  last_tool_ok                   — last tool outcome was success
  terminal_step_reached          — path parser hit a terminal step

In Monolith v1, several callsite signals are not yet tracked
(no per-turn tool outcomes, no path parser). Callers pass None for
unavailable components — recipes were designed to tolerate that for
chat/analysis/retrieval/creative without false-firing. The code/debug
recipes mark ``last_tool_ok`` as ``null_fails=True`` so a tool-less
debug turn fails composite — that is intentional, but the v1 verifier
hardcodes ``task_type="chat"`` until per-turn tool tracking lands.
"""
from __future__ import annotations

from typing import Any


# ── per-task composite recipe ──────────────────────────────────────────
#
# Recipe table:  task_type → ((component_name, null_fails), ...).
# Components not listed are recorded but excluded from the AND.

_COMPOSITE_RECIPE: dict[str, tuple[tuple[str, bool], ...]] = {
    "debug": (
        ("no_parse_errors", False),
        ("no_consecutive_tool_failures", False),
        ("last_tool_ok", True),
        ("terminal_step_reached", False),
    ),
    "code": (
        ("no_parse_errors", False),
        ("no_consecutive_tool_failures", False),
        ("last_tool_ok", True),
        ("terminal_step_reached", False),
    ),
    "analysis": (
        ("no_parse_errors", False),
        ("no_unresolved_assumptions", False),
        ("terminal_step_reached", False),
    ),
    "retrieval": (
        ("no_parse_errors", False),
        ("no_unresolved_assumptions", False),
        ("terminal_step_reached", False),
    ),
    "creative": (
        ("no_parse_errors", False),
        ("no_unresolved_assumptions", False),
        ("terminal_step_reached", False),
    ),
    "chat": (
        ("no_parse_errors", False),
        ("no_unresolved_assumptions", False),
    ),
}


def compute_success_signal(
    task_type: str,
    *,
    no_parse_errors: bool | None,
    no_consecutive_tool_failures: bool | None,
    no_unresolved_assumptions: bool | None,
    last_tool_ok: bool | None,
    terminal_step_reached: bool | None,
) -> dict[str, Any]:
    """Build the ``success_signal`` dict for a turn.

    Returns
    -------
    dict
        ``{"composite": bool, "components": { ... }}``. Components
        are recorded verbatim (None preserved) so calibration analysis
        can re-derive composites without replaying turns.

    The composite is False iff at least one *applicable* component is
    explicitly False — or is None when the recipe marks that component
    as ``null_fails=True``. Otherwise null components are ignored.
    An unrecognized task_type degenerates to the chat recipe.
    """
    components: dict[str, bool | None] = {
        "no_parse_errors": no_parse_errors,
        "no_consecutive_tool_failures": no_consecutive_tool_failures,
        "no_unresolved_assumptions": no_unresolved_assumptions,
        "last_tool_ok": last_tool_ok,
        "terminal_step_reached": terminal_step_reached,
    }
    recipe = _COMPOSITE_RECIPE.get(task_type, _COMPOSITE_RECIPE["chat"])
    composite = True
    for name, null_fails in recipe:
        v = components.get(name)
        if v is None:
            if null_fails:
                composite = False
                break
            continue
        if not v:
            composite = False
            break
    return {"composite": composite, "components": components}
