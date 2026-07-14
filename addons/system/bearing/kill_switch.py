"""Bearing kill switch — env-var gating for MONOLITH_BEARING_V1.

Single source of truth for the on/off flag. Other Bearing files import
is_enabled() instead of re-reading os.environ — keeps semantics
consistent and the truthy-set centralized.

Mirrors the pattern in core.continuity._flag_enabled (continuity.py:68-70)
and core.plane_loader.PlaneLoader._flag_enabled (plane_loader.py:101-103).

Flag values resolving to enabled: "1", "true", "yes", "on" (any case).
Anything else (including unset → defaults to "1") resolves to disabled
only when explicitly set to a falsy value.
"""
from __future__ import annotations

import os

FLAG_ENV = "MONOLITH_BEARING_V1"
STALENESS_FLAG_ENV = "MONOLITH_BEARING_STALENESS_V2"
FRAME_NUDGE_FLAG_ENV = "MONOLITH_FRAME_NUDGE_V1"
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def is_enabled() -> bool:
    """Return True when MONOLITH_BEARING_V1 is unset or truthy."""
    raw = str(os.environ.get(FLAG_ENV, "1")).strip().lower()
    return raw in _TRUTHY


def staleness_is_enabled() -> bool:
    """Return True only when MONOLITH_BEARING_STALENESS_V2 is explicitly truthy.

    Defaults OFF (unset → disabled), unlike is_enabled(): the V1 staleness
    closure loop ships dark. Flag-off keeps the legacy single-nudge path, so
    activating is one line in monolith.bat and reverting is deleting it.
    """
    raw = str(os.environ.get(STALENESS_FLAG_ENV, "0")).strip().lower()
    return raw in _TRUTHY


def frame_nudge_is_enabled() -> bool:
    """Return True only when MONOLITH_FRAME_NUDGE_V1 is explicitly truthy.

    Defaults OFF (unset → disabled). Ships dark — activating is one line in
    monolith.bat and reverting is deleting it. Mirrors staleness_is_enabled()
    exactly (same truthy parsing, same default-off semantics).
    """
    raw = str(os.environ.get(FRAME_NUDGE_FLAG_ENV, "0")).strip().lower()
    return raw in _TRUTHY


FRAME_COMMIT_FLAG_ENV = "MONOLITH_FRAME_COMMIT_V1"


def frame_commit_is_enabled() -> bool:
    """Return True only when MONOLITH_FRAME_COMMIT_V1 is explicitly truthy.

    Defaults OFF (unset → disabled). Ships dark — activating is one line in
    monolith.bat and reverting is deleting it. Mirrors frame_nudge_is_enabled()
    exactly (same truthy parsing, same default-off semantics).
    """
    raw = str(os.environ.get(FRAME_COMMIT_FLAG_ENV, "0")).strip().lower()
    return raw in _TRUTHY
