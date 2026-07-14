"""Plan-scoped grounding resolver for the done-gate.

Builds the `Resolver` (Callable[[str], int|None]) that `plans.attest_criterion`
injects — the same injection pattern `grounded_verdict` uses, so `plans.py` stays
pure. A cited handle resolves to a constant authority IFF it names a real ground
THIS plan accumulated; unknown / cross-plan / malformed → None (fabricated). This
is the resolution-time laundering guard, mirroring `core.recall_handles.resolve`.

Handle form: `obs:<id>` → a `plan_observations` row belonging to `plan_uid`.
The gate is binary (resolves vs fabricated); `ground_kind` carries the real
classification, so the authority value is a constant.
"""
from __future__ import annotations

from collections.abc import Callable

from core import plans

_GROUND_AUTHORITY = 1


def make_plan_resolver(plan_uid: str) -> Callable[[str], "int | None"]:
    """Return a resolver bound to one plan's accumulated grounds."""

    def _resolve(handle: str) -> "int | None":
        h = str(handle or "").strip()
        if not h.lower().startswith("obs:"):
            return None
        raw = h.split(":", 1)[1].strip()
        if not raw.isdigit():
            return None
        row = plans.get_observation(plan_uid, int(raw))
        return _GROUND_AUTHORITY if row is not None else None

    return _resolve
