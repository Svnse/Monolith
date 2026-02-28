"""MCP server for Monolith Relay agent tools.

Two transports:
  streamable-http  :8200  — Claude Code, Codex, Kimi
  SSE              :8201  — Gemini
"""

import json
import threading
import time
import logging

log = logging.getLogger(__name__)

# Injected by process worker before servers start
store = None
router = None

_presence: dict[str, float] = {}
_presence_lock = threading.Lock()
_cursors: dict[str, int] = {}
_cursors_lock = threading.Lock()

PRESENCE_TIMEOUT = 300   # seconds

_INSTRUCTIONS = (
    "Monolith Relay — shared coordination channel between AI agents and humans. "
    "Use chat_send to post messages. Use chat_read to check for new messages. "
    "Use chat_join to announce your presence when you connect. "
    "Use your own agent name as sender — never impersonate others."
)


# ─────────────────────────────────────────────────────────────────────────────
# Tool implementations
# ─────────────────────────────────────────────────────────────────────────────

def chat_send(sender: str, message: str, image_path: str = "", reply_to: int = -1) -> str:
    """Send a message to Monolith Relay. Use your agent name as sender.

    Optionally attach a local image via image_path (absolute path).
    Optionally reply to a message by providing its ID in reply_to."""
    if not message.strip() and not image_path:
        return "Empty message — not sent."

    attachments = []
    if image_path:
        import shutil
        import uuid
        from pathlib import Path
        src = Path(image_path)
        if not src.exists():
            return f"Image not found: {image_path}"
        if src.suffix.lower() not in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"):
            return f"Unsupported image type: {src.suffix}"
        upload_dir = Path("./data/relay_uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{uuid.uuid4().hex[:8]}{src.suffix}"
        shutil.copy2(str(src), str(upload_dir / fname))
        attachments.append({"name": src.name, "url": f"/relay/uploads/{fname}"})

    reply_id = reply_to if reply_to >= 0 else None
    if reply_id is not None and store.get_by_id(reply_id) is None:
        return f"Message #{reply_to} not found."

    msg = store.add(sender, message.strip(), attachments=attachments, reply_to=reply_id)
    with _presence_lock:
        _presence[sender] = time.time()
    return f"Sent (id={msg['id']})"


def _serialize(msgs: list[dict]) -> str:
    out = []
    for m in msgs:
        entry = {
            "id":     m["id"],
            "sender": m["sender"],
            "text":   m["text"],
            "type":   m["type"],
            "time":   m["time"],
        }
        if m.get("attachments"):
            entry["attachments"] = m["attachments"]
        if m.get("reply_to") is not None:
            entry["reply_to"] = m["reply_to"]
        out.append(entry)
    return json.dumps(out, indent=2, ensure_ascii=False) if out else "No new messages."


def _update_cursor(sender: str, msgs: list[dict]) -> None:
    if sender and msgs:
        with _cursors_lock:
            _cursors[sender] = msgs[-1]["id"]


def chat_read(sender: str = "", since_id: int = 0, limit: int = 20) -> str:
    """Read messages from Monolith Relay. Returns JSON array.

    - First call with sender returns the last `limit` messages (full context).
    - Subsequent calls return only new messages since the last read.
    - Pass since_id to read from a specific point.
    - Omit sender to always get the last `limit` messages."""
    if since_id:
        msgs = store.get_since(since_id)
    elif sender:
        with _cursors_lock:
            cursor = _cursors.get(sender, 0)
        msgs = store.get_since(cursor) if cursor else store.get_recent(limit)
    else:
        msgs = store.get_recent(limit)

    msgs = msgs[-limit:]
    _update_cursor(sender, msgs)
    return _serialize(msgs)


def chat_resync(sender: str, limit: int = 50) -> str:
    """Full context fetch — returns latest `limit` messages and resets your read cursor."""
    if not sender.strip():
        return "Error: sender required."
    msgs = store.get_recent(limit)
    _update_cursor(sender, msgs)
    return _serialize(msgs)


def chat_join(name: str, color: str = "", label: str = "") -> str:
    """Announce connection to Monolith Relay. Call this when your session starts."""
    with _presence_lock:
        _presence[name] = time.time()
    entry = router.registry.join(name, color=color, label=label, kind="external")
    store.add(name, f"{name} connected", msg_type="join")
    online = _get_online()
    return f"Joined as '{entry['label']}'. Online: {', '.join(online)}"


def chat_who() -> str:
    """List who is currently online in Monolith Relay."""
    online = _get_online()
    return f"Online: {', '.join(online)}" if online else "Nobody else online."


def chat_todo(
    sender: str,
    action: str = "list",
    message: str = "",
    msg_id: int = 0,
    status: str = "",
) -> str:
    """Manage todos. Actions: create, done, reopen, remove, list."""
    action = action.lower().strip()

    if action == "create":
        if msg_id:
            store.add_todo(msg_id)
            return f"Added todo from message #{msg_id}"
        if message.strip():
            msg = store.add(sender, message.strip())
            store.add_todo(msg["id"])
            return f"Todo created (id={msg['id']})"
        return "Provide msg_id or message text."

    if action == "done":
        return f"Marked #{msg_id} done" if store.complete_todo(msg_id) else f"#{msg_id} not a todo"

    if action == "reopen":
        return f"Reopened #{msg_id}" if store.reopen_todo(msg_id) else f"#{msg_id} not a todo"

    if action == "remove":
        return f"Removed #{msg_id}" if store.remove_todo(msg_id) else f"#{msg_id} not a todo"

    # list
    filt = status if status in ("todo", "done") else None
    items = store.get_todo_messages(filt)
    if not items:
        return "No todos."
    todos = store.get_todos()
    lines = []
    for m in items:
        s = todos.get(m["id"], "?")
        mark = "[ ]" if s == "todo" else "[x]"
        lines.append(f"{mark} #{m['id']} [{m['time']}] {m['sender']}: {m['text']}")
    return "\n".join(lines)


def _get_online() -> list[str]:
    now = time.time()
    with _presence_lock:
        return [n for n, ts in _presence.items() if now - ts < PRESENCE_TIMEOUT]


def is_online(name: str) -> bool:
    now = time.time()
    with _presence_lock:
        return name in _presence and now - _presence[name] < PRESENCE_TIMEOUT


# ─────────────────────────────────────────────────────────────────────────────
# Server creation (called inside worker subprocess)
# ─────────────────────────────────────────────────────────────────────────────

_ALL_TOOLS = [chat_send, chat_read, chat_resync, chat_join, chat_who, chat_todo]


def create_servers(http_port: int = 8200, sse_port: int = 8201):
    """Create and return (mcp_http, mcp_sse). Call after injecting store/router."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        return None, None

    def _make(port: int):
        srv = FastMCP(
            "monolith-relay",
            host="127.0.0.1",
            port=port,
            log_level="ERROR",
            instructions=_INSTRUCTIONS,
        )
        for fn in _ALL_TOOLS:
            srv.tool()(fn)
        return srv

    return _make(http_port), _make(sse_port)
