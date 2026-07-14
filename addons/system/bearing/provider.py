"""BearingProvider — read-only surface for kernel-adjacent consumers.

The provider is the ONLY public read path for kernel code; classifier
(and any other kernel consumer) MUST NOT import store.py directly.

Pattern parallels how `effort.set_world_state` exposes the effort plane
to the rest of the kernel — via a DI handle, not direct import. See
`bootstrap.py:14-33` for the convention.

Consumers:
  - turn_classifier: reads active_goal for carry-forward heuristic,
    reads stakes for complexity scoring
  - bearing compiler: reads current_frame + channel context for
    staleness detection
"""
from __future__ import annotations

from typing import Any

from . import store
from .schema import Bearing


class BearingProvider:
    """Read-only view onto the Bearing store.

    Constructed at addon-boot time; passed to kernel consumers via setter
    (e.g. `core.turn_classifier.set_bearing_provider`). The kernel never
    holds a reference to the store module; it only sees this provider.
    """

    def get_active_goal(self) -> str:
        """Return the current active_goal, or empty string if none."""
        try:
            return store.get_bearing().active_goal
        except Exception:
            return ""

    def get_current_frame(self) -> str:
        """Return the current_frame, or empty string if none."""
        try:
            return store.get_bearing().current_frame
        except Exception:
            return ""

    def get_stakes(self) -> dict[str, Any] | None:
        """Return stakes as a dict, or None if unset."""
        try:
            b = store.get_bearing()
            if b.stakes is None:
                return None
            return {
                "reversibility": b.stakes.reversibility,
                "urgency": b.stakes.urgency,
                "cost_if_wrong": b.stakes.cost_if_wrong,
            }
        except Exception:
            return None

    def get_open_tension_count(self) -> int:
        """Return the count of open tensions."""
        try:
            return len(store.get_bearing().open_tensions)
        except Exception:
            return 0

    def get_last_modified_turn(self) -> str:
        """Return the turn ID when bearing was last modified, or empty string."""
        try:
            return store.get_bearing().updated_at_turn
        except Exception:
            return ""
