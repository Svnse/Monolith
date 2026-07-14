"""Ephemeral coalescer - single-insert section aggregator.

Aggregates ephemeral contributors (runtime_state, review_loop, observer,
last_turn, confidence_trajectory, rating_telemetry, context_refresh) into ONE
inserted block before the latest non-ephemeral
user message. Two benefits:

  1. KV-cache prefix stability: one moving insertion point instead of many,
     so anything downstream of the ephemeral block stays in cache turn-to-turn.
  2. Total-char budget (default 4000, tunable via MONOLITH_EPHEMERAL_BUDGET_CHARS).
     Lowest-priority sections drop on overflow instead of every contributor
     independently deciding to fire without regard for total size.

Drop order under budget pressure (first listed = kept longest):
  runtime_state -> review_loop -> observer -> last_turn -> confidence_trajectory
  -> rating_telemetry -> context_refresh

Direct-inject scaffolds and repair/state blocks (prompt, monothink, tool,
bearing, command_feedback) are NOT contributors here. They register before
this coalescer in bootstrap.py so load-bearing scaffolds and typed repair
instructions are never silently dropped by this prose-budget gate.

Audit defects #6 (KV-cache prefix invalidation from 5 moving inserts) and
#8 (no coalescing or budget layer for ephemerals).

Each contributor module exports:

    def contribute_section(messages: list[dict], config: dict) -> SectionResult | None

Returning ``None`` means the contributor declined this turn (flag off, gate
not satisfied, store empty, etc.). The contributor's pre-existing
``X_interceptor`` function continues to work standalone — bootstrap registers
only the coalescer, but per-module tests against ``X_interceptor`` are not
broken by this module.

Independence: this module imports only from other ``core/`` interceptor
modules plus system addons. No engine or monokernel dependencies; ACU access
is owned by the runtime_state contributor.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class SectionResult:
    """A contributor's proposed ephemeral block.

    ``on_commit`` runs only if this section survives the budget cut AND the
    final block lands in the message list. Use it for state mutations that
    should reflect actual injection (e.g. context_refresh's last-refresh
    marker) so a dropped section doesn't update gates as if it had fired.
    """
    name: str
    text: str
    on_commit: Callable[[], None] | None = None


# Lower index = HIGHER priority (kept under pressure). Names at the END of
# this tuple drop first when the total-char budget is exceeded.
_DROP_ORDER: tuple[str, ...] = (
    "runtime_state",           # 0 - coalesced identity/continuity/recall/runtime/time
    "review_loop",             # 1 — unresolved high-severity substrate routing
    "self_check",              # 1 — last-turn verifier verdict (Self-Check Loop)
    "observer",                # 1 — advisory turn-boundary substrate read
    "last_turn",               # 1 — prior-turn action history (useful, not load-bearing)
    "confidence_trajectory",   # 2 — calibration history (useful, not load-bearing)
    "rating_telemetry",        # 4
    "context_refresh",         # 5
    # Direct-inject blocks stay outside this tuple. See bootstrap.py.
)

_BUDGET_ENV = "MONOLITH_EPHEMERAL_BUDGET_CHARS"
_DEFAULT_BUDGET = 4000
_COALESCER_SOURCE = "ephemeral_coalescer"


def _budget_chars() -> int:
    raw = str(os.environ.get(_BUDGET_ENV, _DEFAULT_BUDGET)).strip()
    try:
        return max(0, int(raw))
    except (ValueError, TypeError):
        return _DEFAULT_BUDGET


def _contributors() -> list[tuple[str, Callable[[list[dict], dict], "SectionResult | None"]]]:
    """Lazy-import contributor modules to avoid bootstrap-time circulars.

    Direct-inject blocks are intentionally excluded; this registry is only for
    prose sections that can share the coalescer's budget and insertion point.
    """
    from core import runtime_state_projection, review_loop, last_turn, confidence_trajectory, rating_telemetry, context_refresh, fault_telemetry, active_agents
    from addons.system import observer

    return [
        ("runtime_state", runtime_state_projection.contribute_section),
        ("active_agents", active_agents.contribute_section),
        ("agent_recap", active_agents.contribute_recap_section),
        ("review_loop", review_loop.contribute_section),
        ("self_check", fault_telemetry.contribute_section),
        ("observer", observer.contribute_section),
        ("last_turn", last_turn.contribute_section),
        ("confidence_trajectory", confidence_trajectory.contribute_section),
        ("rating_telemetry", rating_telemetry.contribute_section),
        ("context_refresh", context_refresh.contribute_section),
    ]


def gather_sections(messages: list[dict], config: dict) -> list[SectionResult]:
    """Call each contributor, drop None / invalid / empty results.

    One contributor raising must not break the others — same isolation
    semantics as ``apply_interceptors``.
    """
    out: list[SectionResult] = []
    for _name, fn in _contributors():
        try:
            sec = fn(messages, config)
        except Exception:
            continue
        if sec is None or not isinstance(sec, SectionResult):
            continue
        if not sec.text or not sec.text.strip():
            continue
        out.append(sec)
    return out


def apply_budget(sections: list[SectionResult], budget: int) -> list[SectionResult]:
    """Drop lowest-priority sections until total text chars <= budget.

    Returns sections sorted high-to-low priority. Budget <= 0 drops all.
    Unknown section names sort to the bottom (drop first).
    """
    if not sections or budget <= 0:
        return []
    rank = {name: i for i, name in enumerate(_DROP_ORDER)}
    ordered = sorted(sections, key=lambda s: rank.get(s.name, 9999))
    kept = list(ordered)
    while kept and sum(len(s.text) for s in kept) > budget:
        kept.pop()
    return kept


def _find_last_non_ephemeral_user_idx(messages: list[dict]) -> int:
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") == "user" and not msg.get("ephemeral"):
            return i
    return -1


def _already_injected(messages: list[dict]) -> bool:
    for msg in messages:
        if msg.get("source") == _COALESCER_SOURCE:
            return True
    return False


def ephemeral_coalescer_interceptor(
    messages: list[dict], config: dict
) -> list[dict] | None:
    """Aggregate all contributor sections into ONE inserted block.

    Returns ``None`` when:
      - the coalesced block is already injected in this turn (double-fire defense)
      - no contributor returned a section
      - the budget drops everything (e.g. budget=0)
      - no non-ephemeral user message exists to insert before
    """
    if _already_injected(messages):
        return None

    sections = gather_sections(messages, config)
    if not sections:
        return None

    budgeted = apply_budget(sections, _budget_chars())
    if not budgeted:
        return None

    last_user_idx = _find_last_non_ephemeral_user_idx(messages)
    if last_user_idx < 0:
        return None

    # Fire on_commit only AFTER confirming the block will land — honors the
    # SectionResult contract ("runs only if ... the final block lands in the
    # message list") so a section's gate (e.g. context_refresh's high-water
    # mark) never advances when nothing is injected (when-plane fix #7).
    for sec in budgeted:
        if sec.on_commit is not None:
            try:
                sec.on_commit()
            except Exception:
                pass

    body = "\n\n".join(sec.text for sec in budgeted)
    result = list(messages)
    result.insert(
        last_user_idx,
        {
            "role": "user",
            "content": body,
            "ephemeral": True,
            "source": _COALESCER_SOURCE,
            "sections": tuple(sec.name for sec in budgeted),
        },
    )
    return result
