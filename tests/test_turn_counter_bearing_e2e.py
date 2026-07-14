"""End-to-end: turn-counter → bearing stamp → readable age render.

Covers the full production chain MINUS the Qt-bound glue in engine/llm.py that
sets config['_turn_n'] (6 guarded lines over the primitives exercised here):

    turn_counter.next_turn()  →  config['_turn_n']
    chat_finalize._process_bearing_envelope  →  updater.process_turn_output(turn_n=)
    store  →  Bearing.updated_at_turn_n
    compiler.bearing_interceptor  →  "N turns ago"
"""
from __future__ import annotations

import pytest

from addons.system.bearing import audit, compiler
from addons.system.bearing import store
from core import chat_finalize
import core.llm_config as llm_config
from core import turn_counter as tc


@pytest.fixture
def wired(monkeypatch, tmp_path):
    monkeypatch.setenv("MONOLITH_TURN_COUNTER_V1", "1")
    monkeypatch.setenv("MONOLITH_BEARING_V1", "1")
    monkeypatch.setattr(store, "_STORE_PATH", tmp_path / "bearing.json")
    monkeypatch.setattr(audit, "_AUDIT_PATH", tmp_path / "bearing.audit.jsonl")
    monkeypatch.setattr(tc, "_counter_path", lambda: tmp_path / "turn_counter.json")
    # chat_finalize does `from core.llm_config import get_current_model_id` at
    # call time, so patching the source module attribute takes effect.
    monkeypatch.setattr(llm_config, "get_current_model_id", lambda: "m-test")
    yield tmp_path


def test_full_chain_stamps_and_renders_age(wired) -> None:
    # Turn 7: the engine would do config["_turn_n"] = next_turn().
    for _ in range(7):
        tc.next_turn()
    config = {"_turn_id": "uuid-7", "_turn_n": tc.current_turn()}
    assert config["_turn_n"] == 7

    # Model commits a bearing this turn → finalize stamps the readable count.
    raw = '<bearing_update>{"active_goal": {"new": "ship it", "reason": "user asked"}}</bearing_update>'
    chat_finalize._process_bearing_envelope(raw, config)
    assert store.get_bearing().updated_at_turn_n == 7

    # 20 turns later, the bearing block renders a readable age.
    later = {"_turn_id": "uuid-27", "_turn_n": 27}
    result = compiler.bearing_interceptor([{"role": "user", "content": "later"}], later)
    assert result is not None
    assert "updated_at_turn: 7 (20 turns ago)" in result[0]["content"]


def test_full_chain_flag_off_is_uuid(monkeypatch, tmp_path) -> None:
    """With the flag off, the engine never sets _turn_n; the stamp stays 0 and
    the render is the UUID — byte-identical to pre-feature."""
    monkeypatch.delenv("MONOLITH_TURN_COUNTER_V1", raising=False)
    monkeypatch.setenv("MONOLITH_BEARING_V1", "1")
    monkeypatch.setattr(store, "_STORE_PATH", tmp_path / "bearing.json")
    monkeypatch.setattr(audit, "_AUDIT_PATH", tmp_path / "bearing.audit.jsonl")
    monkeypatch.setattr(llm_config, "get_current_model_id", lambda: "m-test")

    # Engine with flag off → no _turn_n in config.
    config = {"_turn_id": "uuid-x"}
    raw = '<bearing_update>{"active_goal": {"new": "g", "reason": "r"}}</bearing_update>'
    chat_finalize._process_bearing_envelope(raw, config)
    assert store.get_bearing().updated_at_turn_n == 0

    result = compiler.bearing_interceptor([{"role": "user", "content": "u"}], config)
    assert result is not None
    assert "updated_at_turn: uuid-x" in result[0]["content"]
    assert "turns ago" not in result[0]["content"]


def test_mid_session_toggle_off_stops_stamping_and_renders_uuid(monkeypatch, tmp_path) -> None:
    """ON->OFF mid-session: after the toggle the engine (via resolve_turn_n)
    yields 0, so config['_turn_n'] is not written, the stamp stays 0, and the
    render is UUID — flag-off byte-identical even after the feature was on."""
    monkeypatch.setattr(store, "_STORE_PATH", tmp_path / "bearing.json")
    monkeypatch.setattr(audit, "_AUDIT_PATH", tmp_path / "bearing.audit.jsonl")
    monkeypatch.setattr(tc, "_counter_path", lambda: tmp_path / "turn_counter.json")
    monkeypatch.setattr(llm_config, "get_current_model_id", lambda: "m-test")

    def engine_glue(prev_n: int, is_outer: bool, turn_id: str) -> dict:
        """Faithful mirror of the engine/llm.py glue lines."""
        n = tc.resolve_turn_n(prev_n, is_outer)
        cfg = {"_turn_id": turn_id}
        if n:
            cfg["_turn_n"] = n
        return n, cfg

    # --- flag ON: an outer turn stamps the count ---
    monkeypatch.setenv("MONOLITH_TURN_COUNTER_V1", "1")
    last_n, cfg_on = engine_glue(0, True, "u1")
    assert cfg_on.get("_turn_n") == 1

    # --- flag toggled OFF mid-session: next outer turn forgets it ---
    monkeypatch.delenv("MONOLITH_TURN_COUNTER_V1", raising=False)
    last_n, cfg_off = engine_glue(last_n, True, "u2")
    assert "_turn_n" not in cfg_off  # not written → byte-identical

    # a bearing update on the OFF turn stamps 0; the block renders UUID
    raw = '<bearing_update>{"active_goal": {"new": "g", "reason": "r"}}</bearing_update>'
    chat_finalize._process_bearing_envelope(raw, cfg_off)
    assert store.get_bearing().updated_at_turn_n == 0
    result = compiler.bearing_interceptor([{"role": "user", "content": "x"}], cfg_off)
    assert "turns ago" not in result[0]["content"]
