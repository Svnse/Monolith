from __future__ import annotations

from core.acatalepsy import canonical_log_kinds as k


def test_m2_identity_kinds_registered() -> None:
    for kind in (
        "identity_emergence_detected",
        "identity_amendment_proposed",
        "identity_amendment_applied",
        "identity_milestone_snapshot",
    ):
        assert k.is_valid_kind(kind), f"{kind} must be a known canonical_log kind"


def test_kind_version_bumped_for_m2() -> None:
    assert k.KIND_VERSION >= 6


def test_expedition_tick_kind_registered() -> None:
    # MonoExplore's _log_tick writes this every tick; unregistered, it raised
    # UnknownKind and was swallowed (silent observability failure).
    assert k.is_valid_kind("expedition_tick")
    k.assert_valid_kind("expedition_tick")  # must not raise
