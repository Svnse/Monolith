"""Tests for compiler._apply_frame_selection_contract — the standing-recorder
prompt contract injection.

When MONOLITH_MONOFRAME_V1 is on, the model is instructed to emit a
<frame_selection> block (its committed frame choice) before its answer — so the
recorder fires EVERY turn, automatically, not only when asked. Flag-gated:
byte-identical when off.
"""
from __future__ import annotations

import pytest


def test_appends_contract_when_enabled(monkeypatch):
    from addons.system.bearing import compiler
    monkeypatch.setenv("MONOLITH_MONOFRAME_V1", "1")
    out = compiler._apply_frame_selection_contract("[BEARING] x [/BEARING]")
    assert "[BEARING] x [/BEARING]" in out
    assert "<frame_selection>" in out
    assert "CANDIDATES" in out and "SELECTED" in out and "REJECTED" in out and "REASON" in out


def test_noop_when_disabled(monkeypatch):
    from addons.system.bearing import compiler
    monkeypatch.delenv("MONOLITH_MONOFRAME_V1", raising=False)
    block = "[BEARING] x [/BEARING]"
    assert compiler._apply_frame_selection_contract(block) == block
