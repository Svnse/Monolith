"""
engine/external_agents.py — Peer agent registry and @mention dispatch.

External agents (Gemini CLI, Codex, Kimi, other Monolith instances, etc.)
connect as peers by registering a name + HTTP endpoint URL. No API keys.

Each peer runs a compatible server exposing:
    POST /chat   body: {"message": "...", "agent": "monolith"}
                 returns: {"ok": true, "response": "..."}

The Monolith agent_server at 7821 is itself a compatible peer endpoint,
so any two Monolith instances can talk to each other out of the box.
agentchattr's MCP bridge can also be wrapped with a thin /chat shim.

Peers are stored in ~/.monolith/peers.json and managed via the
Connections page UI.
"""

from __future__ import annotations

import json
import os
import re
import threading
import urllib.parse
import urllib.error
import urllib.request
from pathlib import Path
import tempfile
from typing import Callable


_PEERS_PATH = Path.home() / ".monolith" / "peers.json"
_MAX_PEERS = 128
_MAX_MESSAGE_CHARS = 20_000

# name → {name, label, url, enabled}
_PEERS: dict[str, dict] = {}
_PEERS_LOCK = threading.Lock()


# ── persistence ───────────────────────────────────────────────────────────────

def load_peers() -> dict[str, dict]:
    """Load peer config from disk. Returns the registry dict."""
    _PEERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _PEERS_PATH.exists():
        _PEERS_PATH.write_text("[]", encoding="utf-8")
    try:
        entries = json.loads(_PEERS_PATH.read_text(encoding="utf-8"))
    except Exception:
        entries = []

    peers: dict[str, dict] = {}
    for entry in entries if isinstance(entries, list) else []:
        name = str(entry.get("name", "")).strip().lower()
        if not name:
            continue
        if not entry.get("enabled", True):
            continue
        peers[name] = {
            "name": name,
            "label": str(entry.get("label", name)).strip() or name,
            "url": str(entry.get("url", "")).strip().rstrip("/"),
            "enabled": True,
        }

    with _PEERS_LOCK:
        _PEERS.clear()
        _PEERS.update(peers)

    return peers


def save_peers(peers: list[dict]) -> None:
    """Persist peer list to disk."""
    _PEERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(peers, indent=2, ensure_ascii=False)
    tmp_fd = -1
    tmp_path = ""
    try:
        tmp_fd, tmp_path = tempfile.mkstemp(prefix=f"{_PEERS_PATH.name}.", suffix=".tmp", dir=_PEERS_PATH.parent)
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        tmp_fd = -1
        os.replace(tmp_path, _PEERS_PATH)
    finally:
        if tmp_fd != -1:
            try:
                os.close(tmp_fd)
            except OSError:
                pass
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    load_peers()


def add_peer(name: str, label: str, url: str) -> dict:
    """Add or update a peer. Returns the peer dict."""
    name = name.strip().lower()[:64]
    label = (label.strip() or name)[:128]
    url = url.strip().rstrip("/")[:2048]
    if not name:
        return {"ok": False, "error": "name is required"}
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return {"ok": False, "error": "url must start with http:// or https://"}
    if parsed.username or parsed.password:
        return {"ok": False, "error": "url must not contain user credentials"}

    with _PEERS_LOCK:
        current = list(_PEERS.values())

    # Update if exists, append if new
    updated = False
    for peer in current:
        if peer["name"] == name:
            peer["label"] = label
            peer["url"] = url
            peer["enabled"] = True
            updated = True
            break
    if not updated:
        if len(current) >= _MAX_PEERS:
            return {"ok": False, "error": f"too many peers (max {_MAX_PEERS})"}
        current.append({"name": name, "label": label, "url": url, "enabled": True})

    save_peers(current)
    return {"ok": True, "name": name, "label": label, "url": url}


def remove_peer(name: str) -> bool:
    name = name.strip().lower()
    with _PEERS_LOCK:
        current = [p for p in _PEERS.values() if p["name"] != name]
    save_peers(current)
    return True


def get_peers() -> dict[str, dict]:
    with _PEERS_LOCK:
        return dict(_PEERS)


def peer_names() -> list[str]:
    with _PEERS_LOCK:
        return list(_PEERS.keys())


# ── @mention parsing ──────────────────────────────────────────────────────────

def parse_mentions(text: str) -> list[str]:
    """Return list of @mentioned peer names (lowercase) found in text."""
    with _PEERS_LOCK:
        names = list(_PEERS.keys())
    if not names:
        return []
    pattern = re.compile(
        r"@(" + "|".join(re.escape(n) for n in names) + r")\b",
        re.IGNORECASE,
    )
    return [m.group(1).lower() for m in pattern.finditer(text)]


def strip_mention(text: str, name: str) -> str:
    """Remove @name from text."""
    return re.sub(r"@" + re.escape(name) + r"\b", "", text, flags=re.IGNORECASE).strip()


# ── dispatch ──────────────────────────────────────────────────────────────────

def dispatch(
    peer_name: str,
    message: str,
    on_reply: Callable[[str, str], None],
    on_error: Callable[[str, str], None] | None = None,
    history: list[dict] | None = None,
    url: str | None = None,
) -> None:
    """
    Fire-and-forget: POST message to peer's /chat endpoint in a background thread.

    The peer server must implement:
        POST /chat   {"message": "...", "agent": "monolith", "history": [...]}
        → {"ok": true, "response": "..."}

    history is a list of {"role": "user"|"assistant", "text": "..."} dicts.
    url overrides the registered peer URL (for session-joined participants).
    on_reply(label, text) is called when the peer responds.
    on_error(label, error_text) is called on failure.
    """
    with _PEERS_LOCK:
        peer = dict(_PEERS.get(peer_name) or {})

    # Allow dispatch to participants that joined with a URL but aren't in peers registry
    if not peer and url:
        peer = {"name": peer_name, "label": peer_name, "url": url.rstrip("/")}

    if not peer:
        if on_error:
            on_error(peer_name, f"[{peer_name}: not connected]")
        return
    if url:
        peer = dict(peer)
        peer["url"] = url.rstrip("/")

    if len(str(message or "")) > _MAX_MESSAGE_CHARS:
        err = f"[{peer_name}: message too large (max {_MAX_MESSAGE_CHARS} chars)]"
        if on_error:
            on_error(peer_name, err)
        else:
            on_reply(peer_name, err)
        return

    thread = threading.Thread(
        target=_dispatch_worker,
        args=(peer, message, on_reply, on_error, history or []),
        daemon=True,
        name=f"peer-dispatch-{peer_name}",
    )
    thread.start()


def _dispatch_worker(
    peer: dict,
    message: str,
    on_reply: Callable,
    on_error: Callable | None,
    history: list[dict] | None = None,
) -> None:
    label = peer["label"]
    url = peer["url"].rstrip("/") + "/chat"
    try:
        auth_token = str(os.getenv("MONOLITH_PEER_TOKEN", "")).strip()
        headers = {"Content-Type": "application/json"}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        body: dict = {"message": message, "agent": "monolith"}
        if history:
            body["history"] = history
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            method="POST",
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        if not data.get("ok"):
            raise ValueError(data.get("error") or "peer returned ok=false")

        reply = str(data.get("response", "")).strip()
        on_reply(label, reply)

    except urllib.error.URLError as exc:
        err = f"[{label}: could not reach {url} — {exc.reason}]"
        if on_error:
            on_error(label, err)
        else:
            on_reply(label, err)
    except Exception as exc:
        err = f"[{label}: {exc}]"
        if on_error:
            on_error(label, err)
        else:
            on_reply(label, err)


# ── peer health check ─────────────────────────────────────────────────────────

def ping_peer(name: str) -> bool:
    """Synchronous health check. Returns True if peer responds to GET /health."""
    with _PEERS_LOCK:
        peer = _PEERS.get(name)
    if not peer:
        return False
    url = peer["url"].rstrip("/") + "/health"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return bool(data.get("ok"))
    except Exception:
        return False
