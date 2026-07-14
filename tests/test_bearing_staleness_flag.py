"""MONOLITH_BEARING_STALENESS_V2 gates the V1 closure loop and defaults OFF
(unlike MONOLITH_BEARING_V1, which defaults ON). Flag-off → byte-identical to
the legacy single-nudge path."""
from __future__ import annotations

import pytest

from addons.system.bearing import kill_switch


def test_staleness_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("MONOLITH_BEARING_STALENESS_V2", raising=False)
    assert kill_switch.staleness_is_enabled() is False


def test_staleness_enabled_when_truthy(monkeypatch) -> None:
    for v in ("1", "true", "YES", "on"):
        monkeypatch.setenv("MONOLITH_BEARING_STALENESS_V2", v)
        assert kill_switch.staleness_is_enabled() is True


def test_staleness_disabled_when_falsy(monkeypatch) -> None:
    for v in ("0", "off", "false", "no", ""):
        monkeypatch.setenv("MONOLITH_BEARING_STALENESS_V2", v)
        assert kill_switch.staleness_is_enabled() is False


def test_v1_flag_unaffected_still_defaults_on(monkeypatch) -> None:
    # guard: adding the staleness flag must not change MONOLITH_BEARING_V1 semantics
    monkeypatch.delenv("MONOLITH_BEARING_V1", raising=False)
    assert kill_switch.is_enabled() is True
