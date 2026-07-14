"""Shared availability marker for tests that exercise external Monoline code."""
from __future__ import annotations

import pytest

from engine import monoline_bridge


def _monoline_available() -> bool:
    try:
        monoline_bridge._resolve_monoline_root()
    except RuntimeError:
        return False
    return True


requires_monoline = pytest.mark.skipif(
    not _monoline_available(),
    reason=(
        "external Monoline checkout is not configured; set "
        "MONOLITH_MONOLINE_ROOT to run integration tests"
    ),
)
