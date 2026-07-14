from __future__ import annotations

import json

import pytest

import engine.external_agents as ext


def test_add_peer_rejects_url_with_credentials(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(ext, "_PEERS_PATH", tmp_path / "peers.json")
    ext.load_peers()

    result = ext.add_peer("alpha", "Alpha", "http://user:pass@example.com")

    assert result["ok"] is False
    assert "credentials" in result["error"]


def test_add_peer_enforces_maximum(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(ext, "_PEERS_PATH", tmp_path / "peers.json")
    monkeypatch.setattr(ext, "_MAX_PEERS", 1)
    ext.load_peers()
    assert ext.add_peer("one", "One", "http://example.com")["ok"] is True

    blocked = ext.add_peer("two", "Two", "http://example.org")
    assert blocked["ok"] is False
    assert "too many peers" in blocked["error"]


def test_save_peers_is_atomic(monkeypatch, tmp_path) -> None:
    path = tmp_path / "peers.json"
    path.write_text(json.dumps([{"name": "old", "label": "Old", "url": "http://old", "enabled": True}]), encoding="utf-8")
    monkeypatch.setattr(ext, "_PEERS_PATH", path)

    def _fail_replace(_src, _dst):
        raise OSError("replace failed")

    monkeypatch.setattr("engine.external_agents.os.replace", _fail_replace)

    with pytest.raises(OSError):
        ext.save_peers([{"name": "new", "label": "New", "url": "http://new", "enabled": True}])

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded[0]["name"] == "old"
    assert list(tmp_path.glob("peers.json.*.tmp")) == []


def test_dispatch_rejects_oversized_message(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(ext, "_PEERS_PATH", tmp_path / "peers.json")
    ext.save_peers([{"name": "peer", "label": "Peer", "url": "http://example.com", "enabled": True}])
    errors: list[str] = []
    replies: list[str] = []

    ext.dispatch(
        "peer",
        "x" * (ext._MAX_MESSAGE_CHARS + 1),
        on_reply=lambda _label, text: replies.append(text),
        on_error=lambda _label, err: errors.append(err),
    )

    assert replies == []
    assert errors
    assert "message too large" in errors[0]
