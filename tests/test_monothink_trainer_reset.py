"""Client smoke test — monothink_trainer.py --reset.

Verifies the trainer's --reset verb (a) POSTs /reset and (b) unlinks the local
.last_turn.json (stale-turn-id hygiene), without needing a running Monolith.
We load the skill module by path and monkeypatch its _req so the test is
hermetic. If the skill file isn't present (running the repo without the user's
~/.claude skills), the test skips rather than fails.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_TRAINER = Path.home() / ".claude" / "skills" / "monothink-trainer" / "monothink_trainer.py"


def _load_trainer():
    if not _TRAINER.exists():
        pytest.skip(f"trainer skill not present at {_TRAINER}")
    spec = importlib.util.spec_from_file_location("monothink_trainer", _TRAINER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_reset_posts_route_and_unlinks_state(tmp_path, monkeypatch):
    mod = _load_trainer()

    sent = {}

    def fake_req(method, path, body=None, timeout=600):
        sent["method"] = method
        sent["path"] = path
        sent["body"] = body
        return {"ok": True, "dispatched": True}

    monkeypatch.setattr(mod, "_req", fake_req)

    # Point the state file at a temp dir and seed a stale turn.
    state = tmp_path / ".last_turn.json"
    state.write_text(json.dumps({"turn_id": "STALE-1"}), encoding="utf-8")
    monkeypatch.setattr(mod, "_STATE", state)

    rc = mod._cmd_reset()

    assert rc == 0
    assert sent["method"] == "POST"
    assert sent["path"] == "/reset"
    assert sent["body"] == {}           # empty body for header consistency
    assert not state.exists()           # stale turn id wiped


def test_reset_returns_nonzero_when_server_not_ok(tmp_path, monkeypatch):
    mod = _load_trainer()
    monkeypatch.setattr(mod, "_req", lambda *a, **k: {"ok": False, "error": "down"})
    state = tmp_path / ".last_turn.json"
    monkeypatch.setattr(mod, "_STATE", state)  # missing file — unlink must not raise
    assert mod._cmd_reset() == 1
