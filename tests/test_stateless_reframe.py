"""Tests for addons.system.bearing.stateless_reframe — the Stage 1 producer.

A tap on the synchronous generate seam (the off-chat-lock path monoline/planner/
expedition/resample use), NOT a chat-loop rewire. Produces clean (bearing OFF),
control (bearing ON, same call-shape), and clean' frames, then assess+record.
Model call is dependency-injected so tests never hit a live model.
"""
from __future__ import annotations

import threading

import pytest


# ---------------------------------------------------------------------------
# build_reframe_messages — pure; call-shape fixed, bearing toggled
# ---------------------------------------------------------------------------

class TestBuildReframeMessages:
    def test_clean_has_no_bearing(self):
        from addons.system.bearing import stateless_reframe as sr
        msgs = sr.build_reframe_messages(["what is a mutex?"], bearing_block=None)
        joined = " ".join(m["content"] for m in msgs)
        assert "[BEARING" not in joined
        assert "mutex" in joined

    def test_control_includes_bearing(self):
        from addons.system.bearing import stateless_reframe as sr
        msgs = sr.build_reframe_messages(
            ["what is a mutex?"], bearing_block="[BEARING] current_frame: old topic [/BEARING]"
        )
        joined = " ".join(m["content"] for m in msgs)
        assert "current_frame: old topic" in joined

    def test_call_shape_fixed_same_instruction(self):
        from addons.system.bearing import stateless_reframe as sr
        clean = sr.build_reframe_messages(["x"], bearing_block=None)
        control = sr.build_reframe_messages(["x"], bearing_block="[BEARING] y [/BEARING]")
        # The instruction (system message) must be identical — only bearing presence differs.
        assert clean[0] == control[0]
