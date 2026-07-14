"""Tests for Observer V0 — advisory turn-boundary substrate reader.

Covers: runtime snapshot building, IRP labeling of observer output, store
persistence, compiler section emission, and the chat_finalize wiring that
fires the turn boundary.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from addons.system.observer import runtime, store, compiler
from core import irp
from core.acu_store import ACUStore


# ── fixtures ────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolate_observer_store(tmp_path: Path, monkeypatch):
    """Redirect observer.json to tmp_path so tests don't touch real config."""
    monkeypatch.setattr(store, "_STORE_PATH", tmp_path / "observer.json")


@pytest.fixture
def _isolate_acu_store(tmp_path: Path, monkeypatch):
    from core import db_connect as _dbc
    monkeypatch.setattr(_dbc, "DB_PATH", tmp_path / "acatalepsy.sqlite3", raising=True)
    monkeypatch.setenv("MONOLITH_DB_AUTHORIZER_STRICT", "1")
    from core.acatalepsy import schema, intake, canonical_log
    schema.migrate()
    for mod in (intake, canonical_log):
        tl = getattr(mod, "_tl", None)
        if tl is not None:
            for a in ("writer_conn", "reader_conn", "writer", "reader"):
                if hasattr(tl, a):
                    delattr(tl, a)


# ── runtime: is_enabled ─────────────────────────────────────────────


def test_observer_enabled_by_default(monkeypatch):
    monkeypatch.delenv("MONOLITH_OBSERVER_V0", raising=False)
    assert runtime.is_enabled() is True


def test_observer_disabled_when_env_zero(monkeypatch):
    monkeypatch.setenv("MONOLITH_OBSERVER_V0", "0")
    assert runtime.is_enabled() is False


def test_observer_enabled_when_env_true(monkeypatch):
    monkeypatch.setenv("MONOLITH_OBSERVER_V0", "true")
    assert runtime.is_enabled() is True


# ── runtime: build_observer_snapshot (no substrate) ─────────────────


def test_snapshot_empty_when_no_substrate(monkeypatch):
    monkeypatch.setattr(runtime, "_locked_identity_claims", lambda limit=3: [])
    monkeypatch.setattr(runtime, "_bearing_input", lambda: {})
    monkeypatch.setattr(runtime, "_recent_log_events", lambda limit=8: [])
    # No-substrate must control EVERY reader the observer consults (M2/M3/M1).
    monkeypatch.setattr(runtime, "_latest_emergence_signal", lambda: None)
    monkeypatch.setattr(runtime, "_latest_curiosity_signal", lambda: None)
    monkeypatch.setattr(runtime, "_active_plan_line", lambda: "")

    snap = runtime.build_observer_snapshot(turn_id="t1")
    assert snap["schema_version"] == 1
    assert snap["authority"] == "advisory"
    assert snap["mutation_power"] == "none"
    assert snap["block"] == ""
    assert snap["lines"] == []


def test_snapshot_includes_locked_identity_claims(monkeypatch):
    fake_row = {"canonical": "Origin 0 / What I value: Precision", "locked": 1}
    monkeypatch.setattr(runtime, "_locked_identity_claims", lambda limit=3: [fake_row])
    monkeypatch.setattr(runtime, "_bearing_input", lambda: {})
    monkeypatch.setattr(runtime, "_recent_log_events", lambda limit=8: [])
    monkeypatch.setattr(runtime, "_latest_emergence_signal", lambda: None)
    monkeypatch.setattr(runtime, "_latest_curiosity_signal", lambda: None)
    monkeypatch.setattr(runtime, "_active_plan_line", lambda: "")

    snap = runtime.build_observer_snapshot(turn_id="t2")
    assert len(snap["lines"]) == 1
    assert "[LOCKED]" in snap["lines"][0]
    assert "Precision" in snap["lines"][0]
    assert "[OBSERVER]" in snap["block"]
    assert "[/OBSERVER]" in snap["block"]


def test_snapshot_includes_bearing_goal_and_next_move(monkeypatch):
    monkeypatch.setattr(runtime, "_locked_identity_claims", lambda limit=3: [])
    monkeypatch.setattr(runtime, "_bearing_input", lambda: {
        "active_goal": "build IRP labeler",
        "next_move": "write tests",
        "trajectory": "V0 scope",
    })
    monkeypatch.setattr(runtime, "_recent_log_events", lambda limit=8: [])

    snap = runtime.build_observer_snapshot(turn_id="t3")
    assert any("Bearing active goal:" in ln for ln in snap["lines"])
    assert any("Bearing next move:" in ln for ln in snap["lines"])
    assert any("Bearing trajectory:" in ln for ln in snap["lines"])
    for ln in snap["lines"]:
        assert "[PROVISIONAL]" in ln


def test_snapshot_caps_at_8_lines(monkeypatch):
    many_claims = [
        {"canonical": f"claim {i}", "locked": 1}
        for i in range(20)
    ]
    monkeypatch.setattr(runtime, "_locked_identity_claims", lambda limit=3: many_claims[:3])
    monkeypatch.setattr(runtime, "_bearing_input", lambda: {
        "active_goal": "a", "next_move": "b", "trajectory": "c",
    })

    class FakeEvent:
        def __init__(self, kind, payload):
            self.kind = kind
            self.payload = payload

    monkeypatch.setattr(runtime, "_recent_log_events", lambda limit=8: [
        FakeEvent(f"kind_{i}", {"text": f"msg {i}"}) for i in range(8)
    ])

    snap = runtime.build_observer_snapshot(turn_id="t4")
    assert len(snap["lines"]) == 8


# ── runtime: _short truncation ──────────────────────────────────────


def test_short_truncates_long_text():
    long = "x" * 300
    result = runtime._short(long, limit=50)
    assert len(result) <= 52
    assert result.endswith("...")


def test_short_preserves_short_text():
    assert runtime._short("hello", limit=50) == "hello"


# ── store: read/write/clear ─────────────────────────────────────────


def test_store_roundtrip():
    snap = {"schema_version": 1, "block": "[OBSERVER] test [/OBSERVER]", "lines": ["test"]}
    store.write_latest(snap)
    got = store.read_latest()
    assert got is not None
    assert got["block"] == snap["block"]


def test_store_empty_when_no_file():
    assert store.read_latest() is None


def test_store_clear():
    store.write_latest({"block": "x"})
    store.clear()
    assert store.read_latest() is None


# ── compiler: contribute_section ─────────────────────────────────────


def test_compiler_returns_none_when_disabled(monkeypatch):
    monkeypatch.setenv("MONOLITH_OBSERVER_V0", "0")
    assert compiler.contribute_section([], {}) is None


def test_compiler_returns_none_when_no_snapshot(monkeypatch):
    monkeypatch.setenv("MONOLITH_OBSERVER_V0", "1")
    assert compiler.contribute_section([], {}) is None


def test_compiler_returns_section_from_stored_snapshot(monkeypatch):
    monkeypatch.setenv("MONOLITH_OBSERVER_V0", "1")
    store.write_latest({
        "block": "[OBSERVER] test block [/OBSERVER]",
        "lines": ["test"],
    })

    result = compiler.contribute_section([], {})
    assert result is not None
    assert result.name == "observer"
    assert "[OBSERVER]" in result.text


# ── fire_turn_boundary integration ──────────────────────────────────


def test_fire_turn_boundary_persists_and_compiles(monkeypatch):
    monkeypatch.setenv("MONOLITH_OBSERVER_V0", "1")
    monkeypatch.setattr(runtime, "_locked_identity_claims", lambda limit=3: [
        {"canonical": "Origin 0 / Identity: test claim", "locked": 1}
    ])
    monkeypatch.setattr(runtime, "_bearing_input", lambda: {"active_goal": "ship V0"})
    monkeypatch.setattr(runtime, "_recent_log_events", lambda limit=8: [])

    snap = runtime.fire_turn_boundary(turn_id="t5")
    assert snap is not None
    assert "[OBSERVER]" in snap["block"]

    section = compiler.contribute_section([], {})
    assert section is not None
    assert "[OBSERVER]" in section.text
    assert "test claim" in section.text


def test_fire_turn_boundary_returns_none_when_disabled(monkeypatch):
    monkeypatch.setenv("MONOLITH_OBSERVER_V0", "0")
    assert runtime.fire_turn_boundary(turn_id="t6") is None


# ── IRP labeling on observer output ─────────────────────────────────


def test_observer_lines_are_irp_labeled():
    labeled = irp.label_text("advisory note", scope="observer", label="PROVISIONAL")
    assert labeled == "[PROVISIONAL] advisory note"


def test_observer_scope_passes_through_irp():
    labeled = irp.label_text("identity", scope="observer", label="LOCKED")
    assert labeled == "[LOCKED] identity"


# ── chat_finalize wiring ────────────────────────────────────────────


def test_chat_finalize_fires_observer(monkeypatch):
    fired = []

    import addons.system.observer as obs_pkg
    monkeypatch.setattr(obs_pkg, "fire_turn_boundary", lambda turn_id: fired.append(turn_id))

    from core.chat_finalize import _process_observer_boundary
    _process_observer_boundary({"_turn_id": "t7"})
    assert fired == ["t7"]


def test_chat_finalize_skips_observer_when_no_turn_id(monkeypatch):
    fired = []
    monkeypatch.setattr(runtime, "fire_turn_boundary", lambda turn_id: fired.append(turn_id))

    from core.chat_finalize import _process_observer_boundary
    _process_observer_boundary({})
    assert fired == []


def test_chat_finalize_observer_never_raises(monkeypatch):
    """Observer boundary errors must be swallowed — never break chat path."""
    def boom(turn_id):
        raise RuntimeError("observer crash")

    monkeypatch.setattr(runtime, "fire_turn_boundary", boom)

    from core.chat_finalize import _process_observer_boundary
    _process_observer_boundary({"_turn_id": "t8"})  # must not raise


# ── locked ACU integration with Observer ────────────────────────────


def test_locked_acus_appear_in_observer_snapshot(
    _isolate_acu_store, monkeypatch,
):
    """Origin 0 ACUs loaded via identity_acus show up in the Observer block."""
    s = ACUStore()
    try:
        s.ingest_locked("Origin 0 / Identity: I do not invent confidence")
    finally:
        s.close()

    monkeypatch.setattr(runtime, "_bearing_input", lambda: {})
    monkeypatch.setattr(runtime, "_recent_log_events", lambda limit=8: [])

    snap = runtime.build_observer_snapshot(turn_id="t9")
    assert any("I do not invent confidence" in ln for ln in snap["lines"])
    assert any("[LOCKED]" in ln for ln in snap["lines"])


# ── canonical_log event emission (Auditor pairing) ──────────────────


def test_fire_emits_observer_fired_to_canonical_log(monkeypatch, tmp_path):
    """Observer writes an observer_fired event so the Auditor can see it."""
    monkeypatch.setenv("MONOLITH_OBSERVER_V0", "1")
    monkeypatch.setattr(runtime, "_locked_identity_claims", lambda limit=3: [
        {"canonical": "test claim", "locked": 1}
    ])
    monkeypatch.setattr(runtime, "_bearing_input", lambda: {"active_goal": "test"})
    monkeypatch.setattr(runtime, "_recent_log_events", lambda limit=8: [])

    emitted = []

    def fake_append(kind, payload=None, **kw):
        emitted.append({"kind": kind, "payload": payload})
        return 1

    import core.acatalepsy.canonical_log as cl
    monkeypatch.setattr(cl, "append", fake_append)

    runtime.fire_turn_boundary(turn_id="t10")

    assert len(emitted) == 1
    assert emitted[0]["kind"] == "observer_fired"
    p = emitted[0]["payload"]
    assert p["turn_id"] == "t10"
    assert p["lines_count"] >= 1
    assert p["block_chars"] > 0
    assert "LOCKED" in p["labels_used"]
