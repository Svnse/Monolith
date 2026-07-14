"""Tests for compiler._apply_correction_example — the live read-side injection.

At prompt-assembly, the nearest TRAINABLE human card is appended to the [BEARING]
block as ONE worked example. Flag-gated (MONOLITH_MONOFRAME_V1); no-op until a
human card exists, so the chat path is byte-identical by default.
"""
from __future__ import annotations

import pytest


def _store_human_card(st, tmp_path, better):
    from addons.system.bearing import correction_card as cc
    card = cc.CorrectionCard(
        bad_frame="reasoning about vehicle ethics", better_frame=better,
        process_move="drop the carried topic; anchor the live ask",
        anchor_type=cc.AnchorType.IMPLIED_TASK, anchor_error="mirrored_loud_noun",
        aperture=cc.Aperture.COLLAPSE, stateless_control="ctrl",
        source=cc.Source.HUMAN, promoted=True,
    )
    st.store_card(card)


def test_appends_nearest_card_when_enabled(monkeypatch, tmp_path):
    from addons.system.bearing import compiler, correction_store as st
    monkeypatch.setenv("MONOLITH_MONOFRAME_V1", "1")
    monkeypatch.setattr(st, "_STORE", tmp_path / "cards.jsonl")
    _store_human_card(st, tmp_path, "answering the mutex question")
    messages = [{"role": "user", "content": "how does a mutex lock work", "source": "user"}]
    out = compiler._apply_correction_example("[BEARING] x [/BEARING]", messages)
    assert "answering the mutex question" in out
    assert "[BEARING] x [/BEARING]" in out  # original block preserved


def test_noop_when_flag_off(monkeypatch, tmp_path):
    from addons.system.bearing import compiler, correction_store as st
    monkeypatch.delenv("MONOLITH_MONOFRAME_V1", raising=False)
    monkeypatch.setattr(st, "_STORE", tmp_path / "cards.jsonl")
    block = "[BEARING] x [/BEARING]"
    messages = [{"role": "user", "content": "mutex", "source": "user"}]
    assert compiler._apply_correction_example(block, messages) == block


def test_noop_when_no_trainable_cards(monkeypatch, tmp_path):
    from addons.system.bearing import compiler, correction_store as st
    monkeypatch.setenv("MONOLITH_MONOFRAME_V1", "1")
    monkeypatch.setattr(st, "_STORE", tmp_path / "cards.jsonl")
    block = "[BEARING] x [/BEARING]"
    messages = [{"role": "user", "content": "mutex", "source": "user"}]
    assert compiler._apply_correction_example(block, messages) == block
