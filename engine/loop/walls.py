"""
Walls — hard runtime-enforced boundaries.

The agent cannot bypass these.  The runtime checks walls *before*
executing any actions in a cycle.  If a wall is hit the run terminates
immediately with ``RunResult.wall_hit`` set.

Wall taxonomy:
  budget      — max cycles, tool calls, or elapsed time exceeded
  repetition  — same intent repeated N times in a window
  stall       — progress has not increased for N consecutive cycles
  empty_action_stall — consecutive empty-action non-complete steps
  malformed   — LLM output could not be parsed after retries (per-cycle)
"""

from __future__ import annotations

import time
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.loop.contracts import RunContext, Step


class WallChecker:
    """Stateless wall evaluator — all state lives in RunContext."""

    __slots__ = ()

    def check(self, ctx: RunContext, step: Step) -> str | None:
        """Return the wall name if any boundary is violated, else ``None``."""
        return (
            self._budget(ctx, step)
            or self._repetition(ctx, step)
            or self._stall(ctx)
        )

    # ---- individual walls ----

    @staticmethod
    def _budget(ctx: RunContext, step: Step) -> str | None:
        if ctx.cycle > ctx.policy.max_cycles:
            return "max_cycles"
        pending = len(step.actions)
        if ctx.total_tool_calls + pending > ctx.policy.max_tool_calls:
            return "max_tool_calls"
        elapsed = max(0.0, (time.time() - ctx.start_time) - float(getattr(ctx, "paused_sec", 0.0) or 0.0))
        if elapsed > ctx.policy.max_elapsed_sec:
            return "max_elapsed"
        return None

    @staticmethod
    def _repetition(ctx: RunContext, step: Step) -> str | None:
        window = ctx.policy.repetition_window
        if window < 2:
            return None
        recent = ctx.recent_intents[-(window - 1):]
        if len(recent) < window - 1:
            return None
        current = _canonical_intent(step.intent)
        if not current:
            return None
        normalized_recent = [_canonical_intent(i) for i in recent]
        if any(not x for x in normalized_recent):
            return None
        # All recent intents canonically match the current one
        if all(i == current for i in normalized_recent):
            return "repetition"
        return None

    @staticmethod
    def _stall(ctx: RunContext) -> str | None:
        hist = ctx.pad._progress_history
        window = ctx.policy.stall_window
        if len(hist) < window:
            return None
        # Allow early exploration cycles to remain at 0% without tripping stall.
        if max(hist) <= 0.0:
            return None
        tail = hist[-window:]
        # Progress hasn't changed across the whole window
        if len(set(tail)) == 1:
            return "stall"
        return None


_INTENT_TOKEN_RE = re.compile(r"[a-zA-Z]+")
_INTENT_STOPWORDS = {"i", "will", "the", "a", "an", "to", "now", "next", "then", "and"}


def _canonical_intent(intent: str) -> str:
    if not isinstance(intent, str):
        return ""
    tokens = []
    for raw in _INTENT_TOKEN_RE.findall(intent.lower()):
        if raw in _INTENT_STOPWORDS:
            continue
        tok = raw
        if len(tok) > 5 and tok.endswith("ing"):
            tok = tok[:-3]
        elif len(tok) > 4 and tok.endswith("ed"):
            tok = tok[:-2]
        elif len(tok) > 4 and tok.endswith("es"):
            tok = tok[:-2]
        elif len(tok) > 3 and tok.endswith("s"):
            tok = tok[:-1]
        tokens.append(tok)
    if not tokens:
        return ""
    return " ".join(tokens[:4])
