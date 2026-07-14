"""
engine/agent_server.py — Monolith Agent Server

Two transports, one send_message tool:

  MCP stdio  (Claude Code, Cursor, etc.)
    Add to .mcp.json:
      {
        "monolith": {
          "type": "stdio",
          "command": "python",
          "args": ["C:/path/to/Monolith/engine/agent_server.py", "--stdio"],
          "cwd": "C:/path/to/Monolith"
        }
      }

  HTTP REST  (Kimi, Gemini, any agent without MCP)
    Start:  python engine/agent_server.py --http [--port 7821]
    Use:    POST http://localhost:7821/chat
            body: {"message": "hello", "agent": "Kimi"}
            returns: {"ok": true, "response": "..."}

    Also:   GET  /health        ->  {"ok": true}
            GET  /state         ->  current model status, recent messages, queue depth
            POST /chat/stream   ->  SSE stream of typed events (token, thinking, tool_call, done)
            POST /hooks         ->  register webhook callback URLs for event push
            DELETE /hooks       ->  unregister a webhook
            POST /nudge         ->  inject ephemeral system context into next N generations

In embedded mode (ConnectionsPage owns the server), the server is
wired to the active PageChat session via callbacks set by the page.
In standalone mode (--http / --stdio without Monolith running),
/chat returns an error until Monolith connects.
"""

from __future__ import annotations

import hmac
import ipaddress
import json
import os
import queue
import re
import sys
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer

from core.curiosity_capture import strip_curiosity_blocks
from core.internal_tags import EXTERNAL_STRIP_TAGS, strip_tag_blocks
from core.version import APP_VERSION


def _public_error(label: str, exc: BaseException) -> str:
    """Return a useful client error without leaking local paths or secrets."""
    message = str(exc).strip()
    has_path = bool(
        re.search(r"(?i)\b[A-Z]:[\\/]", message)
        or re.search(
            r"(?<![:\w])/(?:Users|home|var|tmp|private|opt|srv|mnt|media|run)/",
            message,
        )
        or re.search(r"\\\\[^\\\s]+\\", message)
    )
    has_secret = bool(
        re.search(r"(?i)(?:api[_ -]?key|token|secret)\s*[:=]\s*\S+", message)
    )
    if not message or has_path or has_secret:
        return f"{label} ({type(exc).__name__})"
    return f"{label}: {message}"


def _is_loopback_host(host: str) -> bool:
    """True only for explicit loopback bind targets."""
    value = str(host or "").strip().lower().strip("[]")
    if value == "localhost":
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


# Thinking-stream lane parser still needs its own pattern because it must
# capture the tag *name* per match (to disambiguate think/analysis/reasoning
# in the lane router). The tag NAMES come from EXTERNAL_STRIP_TAGS so the
# stream parser and the external strip stay aligned, but the SHAPE
# (named-group capture vs. simple removal) is purpose-specific.
_THINK_BLOCK_RE = re.compile(
    r"<(?P<tag>think|analysis|reasoning)>(?P<content>.*?)</(?P=tag)>",
    re.DOTALL | re.IGNORECASE,
)


def _clean_agent_response(text: str) -> str:
    """Strip internal runtime markers from text before sending to external agents.

    Removes every tag in core/internal_tags.EXTERNAL_STRIP_TAGS plus the
    `[TOOL_LOOP_DONE]` sentinel — anything Restore's runtime emits that
    external agents should not see by default. Tag set is centralized so
    adding a new internal tag in one place propagates here automatically.
    """
    text = strip_curiosity_blocks(text)
    text = strip_tag_blocks(text, EXTERNAL_STRIP_TAGS)
    # Strip embedded transport/role bracket tags anywhere in the body — not just
    # a leading one. The model sometimes echoes [CHANNEL: connect/...] or
    # [AGENT:Name] mid-response; those are runtime markup a peer must not see.
    text = re.sub(r"\[(?:CHANNEL|AGENT)\b[^\]]*\]", "", text, flags=re.IGNORECASE)
    text = text.replace("[TOOL_LOOP_DONE]", "")
    return text.strip()


def _extract_thinking_text(text: str) -> str:
    """Return concatenated internal thinking blocks from a raw model response."""
    chunks: list[str] = []
    for match in _THINK_BLOCK_RE.finditer(str(text or "")):
        body = str(match.group("content") or "").strip()
        if body:
            chunks.append(body)
    return "\n\n".join(chunks)


class _ThinkingStreamParser:
    """Incrementally split raw token chunks into answer vs thinking lanes."""

    _OPEN_RE = re.compile(r"<(think|analysis|reasoning)>", re.IGNORECASE)
    _CLOSE_RE = re.compile(r"</(think|analysis|reasoning)>", re.IGNORECASE)
    _OPEN_TAGS = ("<think>", "<analysis>", "<reasoning>")
    _CLOSE_TAGS = ("</think>", "</analysis>", "</reasoning>")

    def __init__(self) -> None:
        self._buf = ""
        self._in_thinking = False

    def feed(self, chunk: str) -> list[tuple[str, str]]:
        self._buf += str(chunk or "")
        return self._drain(final=False)

    def flush(self) -> list[tuple[str, str]]:
        return self._drain(final=True)

    def _drain(self, *, final: bool) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        while self._buf:
            regex = self._CLOSE_RE if self._in_thinking else self._OPEN_RE
            match = regex.search(self._buf)
            if match is None:
                if not final:
                    tags = self._CLOSE_TAGS if self._in_thinking else self._OPEN_TAGS
                    keep = self._partial_tag_suffix_len(self._buf, tags)
                else:
                    keep = 0
                if not final and len(self._buf) <= keep:
                    break
                emit = self._buf if keep == 0 else self._buf[:-keep]
                self._buf = "" if keep == 0 else self._buf[-keep:]
                if emit:
                    lane = EVENT_THINKING if self._in_thinking else EVENT_TOKEN
                    out.append((lane, emit))
                break

            prefix = self._buf[:match.start()]
            if prefix:
                lane = EVENT_THINKING if self._in_thinking else EVENT_TOKEN
                out.append((lane, prefix))
            self._buf = self._buf[match.end():]
            self._in_thinking = not self._in_thinking
        return out

    @staticmethod
    def _partial_tag_suffix_len(text: str, tags: tuple[str, ...]) -> int:
        lower = text.lower()
        best = 0
        for tag in tags:
            max_probe = min(len(lower), len(tag) - 1)
            for size in range(1, max_probe + 1):
                if lower.endswith(tag[:size]):
                    best = max(best, size)
        return best


# ── Debug-read helpers (used by /trace, /session/messages, /db/query) ─────


_DB_QUERY_STORES: dict[str, frozenset[str]] = {
    "turn_trace": frozenset({"stage_traces", "frame_traces", "outcome_traces", "schema_version"}),
    "acu": frozenset({"acus"}),
}


_FORBIDDEN_SQL_RE = re.compile(
    r"(?:^|[\s;(])(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|ATTACH|REPLACE|TRUNCATE|PRAGMA|VACUUM)\b",
    re.IGNORECASE,
)


def _db_path_for_store(store: str):
    """Resolve a DB store name to a path under LOG_DIR."""
    try:
        from core.paths import LOG_DIR
    except Exception:
        return None
    if store == "turn_trace":
        return LOG_DIR / "turn_trace.sqlite3"
    if store == "acu":
        return LOG_DIR / "acatalepsy.sqlite3"
    return None


def _run_db_query(db_path, allowed_tables: frozenset, *, table: str = "", sql: str = "", limit=20) -> dict:
    """Bounded read against an allowlisted store. SELECT-only; stacked
    statements rejected; mutation keywords rejected; default LIMIT applied.
    """
    import sqlite3
    from pathlib import Path

    clean_table = str(table or "").strip()
    clean_sql = str(sql or "").strip()
    try:
        safe_limit = max(1, min(int(limit), 100))
    except (TypeError, ValueError):
        safe_limit = 20

    if not clean_table and not clean_sql:
        return {"ok": False, "error": "provide either 'table' or 'sql'"}
    if not Path(db_path).exists():
        return {"ok": False, "error": "database not found"}

    if clean_sql:
        normalized = clean_sql.lstrip().upper()
        if not normalized.startswith("SELECT") and not normalized.startswith("WITH"):
            return {"ok": False, "error": "only SELECT (or WITH ... SELECT) queries are allowed"}
        forbidden_match = _FORBIDDEN_SQL_RE.search(clean_sql)
        if forbidden_match:
            kw = forbidden_match.group(1).upper()
            return {"ok": False, "error": f"'{kw}' not allowed in queries"}
        no_trailing = clean_sql.rstrip().rstrip(";").rstrip()
        if ";" in no_trailing:
            return {"ok": False, "error": "stacked statements not allowed"}
        query = clean_sql if "LIMIT" in normalized else f"{clean_sql} LIMIT {safe_limit}"
    elif clean_table in allowed_tables:
        order_by = " ORDER BY id DESC" if clean_table != "schema_version" else ""
        query = f"SELECT * FROM {clean_table}{order_by} LIMIT {safe_limit}"
    else:
        return {
            "ok": False,
            "error": f"unknown table '{clean_table}'",
            "allowed": sorted(allowed_tables),
        }

    try:
        conn = sqlite3.connect(str(db_path), timeout=3)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(query).fetchall()
        finally:
            conn.close()
    except sqlite3.OperationalError as exc:
        return {"ok": False, "error": _public_error("database query failed", exc)}
    except Exception as exc:
        return {"ok": False, "error": _public_error("database query failed", exc)}

    serialized = [dict(r) for r in rows]
    return {"ok": True, "query": query, "count": len(serialized), "rows": serialized}


def _summary_to_dict(r) -> dict:
    """Serialize a turn_trace.TurnTraceSummary for JSON."""
    return {
        "turn_id": r.turn_id,
        "parent_turn_id": r.parent_turn_id,
        "captured_at": r.captured_at,
        "backend": r.backend,
        "stage_count": r.stage_count,
        "errored_stage_count": r.errored_stage_count,
        "total_chars": r.total_chars,
        "frame_present": r.frame_present,
    }


def _joined_to_dict(joined) -> dict:
    """Serialize a turn_trace.TurnTraceJoined for JSON."""
    return {
        "turn_id": joined.turn_id,
        "parent_turn_id": joined.parent_turn_id,
        "summary": dict(joined.summary or {}),
        "stages": [s.to_payload() for s in joined.stages],
        "frame": joined.frame.to_payload() if joined.frame is not None else None,
        "outcomes": [o.to_dict() for o in joined.outcomes],
    }


def _strip_message(msg: dict) -> dict:
    """Apply `_clean_agent_response` to the `text` field of a session message
    dict (preserves role/time/i/other fields). Used by /session/messages?raw=false.
    """
    if not isinstance(msg, dict):
        return msg
    out = dict(msg)
    if "text" in out and isinstance(out["text"], str):
        out["text"] = _clean_agent_response(out["text"])
    return out


class _QuietHTTPServer(HTTPServer):
    """Suppress noisy ConnectionResetError/BrokenPipeError tracebacks."""
    def handle_error(self, request, client_address):
        import sys
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
            return  # client dropped — not an error worth logging
        super().handle_error(request, client_address)
from typing import Callable


# -- Event types for SSE + webhooks --------------------------------------------

EVENT_TOKEN = "token"
EVENT_THINKING = "thinking"
EVENT_TOOL_CALL = "tool_call"
EVENT_TOOL_RESULT = "tool_result"
EVENT_ERROR = "error"
EVENT_DONE = "done"
EVENT_STATUS = "status"          # model status change (idle/generating/tool-loop)
EVENT_GENERATION_START = "generation_start"

ALL_EVENTS = frozenset({
    EVENT_TOKEN, EVENT_THINKING, EVENT_TOOL_CALL, EVENT_TOOL_RESULT,
    EVENT_ERROR, EVENT_DONE, EVENT_STATUS, EVENT_GENERATION_START,
})


# ── Plane pending-state registry (compressed step 2.6) ──────────────────────
#
# The 4 parallel plane pending-state clusters (effort/conversation/reasoning/
# linguency) shipped in step 2.5 are now compressed into a registry pattern:
# one _PendingPlane instance per plane, held in AgentServer._pending_planes.
# Generic setter/consumer methods take a plane name and dispatch through the
# registry. Per-plane public methods (set_pending_effort, etc.) survive as
# one-line wrappers so call sites stay stable and self-documenting.
#
# Adding a new plane is now one line in _PLANE_SPECS + one public-wrapper
# method (optional — callers can also use _set_pending(plane, value, once)
# directly).
#
# key_suffix: "tier" for effort (depth tier), "mode" for everything else
#   (categorical mode). Drives the world_state state keys:
#     baseline:  f"{name}_{key_suffix}"        — e.g. effort_tier, conversation_mode
#     once:      f"{name}_once_{key_suffix}"   — e.g. effort_once_tier
#
# module_import_path: lazy import target so module load order stays flexible
#   (matches the existing core/* import pattern elsewhere in this file).


@dataclass
class _PendingPlane:
    """Per-plane pending-state slot. One instance per plane on AgentServer."""
    name: str
    key_suffix: str
    module_import_path: str
    value: str | None = None
    once: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def baseline_state_key(self) -> str:
        return f"{self.name}_{self.key_suffix}"

    @property
    def once_state_key(self) -> str:
        return f"{self.name}_once_{self.key_suffix}"

    def get_module(self):
        """Lazy-import the plane's loader module (cached by Python's import system)."""
        return __import__(self.module_import_path, fromlist=["valid_modes"])


# (name, key_suffix, module_import_path) — order is insertion order for the
# _pending_planes dict, which also fixes the dispatch order in _apply_pending_all
# (effort first, linguency last). Don't reorder casually — the dispatch ordering
# parallels the bootstrap.py interceptor registration order (see proximity
# comment in bootstrap.py around the register_interceptor block).
_PLANE_SPECS: tuple[tuple[str, str, str], ...] = ()  # Legacy planes removed; /prompt system replaces them


# ── per-call request body parsing helpers (3 endpoints share these) ─────────
#
# /chat, /chat/stream, and MCP send_message all accept the same per-call body
# fields (effort, conversation, reasoning, linguency, plus their _once
# counterparts). The 3 endpoints previously had near-identical 30-line parsing
# blocks; these helpers compress that into one parse call + one tag-build call.


def _parse_per_call_modes(server: "AgentServer", data: dict, default_once: bool = True) -> dict[str, str]:
    """Parse plane-mode body fields from a request. Returns dict of {plane: applied_value}.

    For each plane: looks up data[plane] (e.g. data["effort"]) and an optional
    data[f"{plane}_once"] override. Applies via the generic _set_pending; only
    successfully-validated values appear in the returned dict.

    `default_once`: True for HTTP /chat and /chat/stream (which honor the body
    field), False for paths where the once-semantics are hardcoded. MCP
    send_message hardcodes once=True so it passes default_once=True.
    """
    applied: dict[str, str] = {}
    if not isinstance(data, dict):
        return applied
    for plane in server._pending_planes:
        raw = str(data.get(plane) or "").strip().lower()
        if not raw:
            continue
        once_key = f"{plane}_once"
        once = bool(data.get(once_key, default_once))
        if server._set_pending(plane, raw, once=once):
            applied[plane] = raw
    return applied


def _process_rating(data: dict) -> tuple[dict, int]:
    """Core of POST /rating — separated from the HTTP handler so it is unit-testable
    without a running server (this is the non-Qt path that exercises the SP1
    record_outcome metadata->hook seam).

    Validates the body, records a rating ``OutcomeTraceRecord`` on the SP1 contract
    (``failure_tags`` + ``surface_note`` in metadata, ``reason`` = auto-composed human
    echo), and returns ``(payload, http_status)``. ``record_outcome`` fires the SP1
    evolution hook, which itself gates on non-empty tags + a monothink-active turn.
    ``surface_note`` rides metadata and IS surfaced to the evolution decider as
    non-canonical context (the ``RATER_NOTE_NONCANONICAL`` block, threaded
    turn_trace -> monothink._compose_prompt); it can locate evidence or support a
    reserve/reject but cannot mint a ``failure_signature`` or authorize an apply (the
    promotion gate never reads it).
    """
    if not isinstance(data, dict):
        return {"ok": False, "error": "invalid body"}, 400
    turn_id = str(data.get("turn_id") or "").strip()
    if not turn_id:
        return {"ok": False, "error": "'turn_id' is required"}, 400
    rv = data.get("rating")
    if not isinstance(rv, int) or isinstance(rv, bool) or not (0 <= rv <= 100):
        return {"ok": False, "error": "'rating' must be an integer 0-100"}, 400
    raw_tags = data.get("failure_tags") or []
    if not isinstance(raw_tags, list):
        return {"ok": False, "error": "'failure_tags' must be a list of tag ids"}, 400

    from core.failure_tags import normalize_tags, is_valid_tag, compose_reasoning_why
    str_tags = [str(t) for t in raw_tags]
    tags = normalize_tags(str_tags)
    dropped = [t for t in str_tags if not is_valid_tag(t)]
    surface_note = str(data.get("surface_note") or "").strip() or None
    think_block = str(data.get("think_block") or "").strip() or None
    replay_input = str(data.get("replay_input") or "").strip() or None

    metadata: dict = {}
    if tags:
        metadata["failure_tags"] = tags
    if surface_note:
        metadata["surface_note"] = surface_note
    if think_block:
        metadata["think_block"] = think_block
    if replay_input:
        metadata["replay_input"] = replay_input

    from datetime import datetime, timezone
    from core import turn_trace as _tt
    record = _tt.OutcomeTraceRecord(
        turn_id=turn_id,
        recorded_at=datetime.now(timezone.utc).isoformat(),
        kind="rating",
        rating_value=rv,
        reason=(compose_reasoning_why(tags) if tags else surface_note),
        metadata=(metadata or None),
    )
    try:
        _tt.record_outcome(record)
    except Exception as exc:
        return {"ok": False, "error": _public_error("record_outcome failed", exc)}, 500

    will_evolve = bool(tags) and _tt.get_turn_monothink_active(turn_id)
    return {
        "ok": True,
        "turn_id": turn_id,
        "rating": rv,
        "failure_tags": tags,
        "dropped_unknown_tags": dropped,
        "will_evolve": will_evolve,
        "has_replay_input": bool(replay_input),
    }, 200


def _process_frame_correction(data: dict, *, runner=None) -> tuple[dict, int]:
    """Core of POST /frame — MonoFrame v2 CorrectionCard. Unit-testable (runner
    injected), no live model. Validates the body, maps source (human TRAINS /
    candidate is logged only), and dispatches process_correction_async off-thread.

    The HTTP handler fills bad_frame / recent_asks / base_config from live state
    (current bearing frame, session messages, engine config) before calling, so
    the trainer only has to send ``better_frame`` + ``source``. Source defaults to
    candidate so a malformed call never accidentally trains.
    """
    if not isinstance(data, dict):
        return {"ok": False, "error": "invalid body"}, 400
    better_frame = str(data.get("better_frame") or "").strip()
    if not better_frame:
        return {"ok": False, "error": "'better_frame' is required"}, 400

    from addons.system.bearing import correction_card as _cc
    src_raw = str(data.get("source") or "").strip().lower()
    source = _cc.Source.HUMAN if src_raw == "human" else _cc.Source.CLAUDE_CANDIDATE

    if runner is None:
        from addons.system.bearing.correction_runner import process_correction_async as runner

    recent_asks = data.get("recent_asks") or []
    if not isinstance(recent_asks, list):
        recent_asks = [str(recent_asks)]
    base_config = data.get("base_config")
    try:
        runner(
            str(data.get("turn_id") or ""),
            bad_frame=str(data.get("bad_frame") or ""),
            better_frame=better_frame,
            recent_asks=[str(a) for a in recent_asks],
            base_config=base_config if isinstance(base_config, dict) else {},
            source=source,
        )
    except Exception as exc:
        return {"ok": False, "error": _public_error("frame correction failed", exc)}, 500

    return {
        "ok": True,
        "source": source.value,
        "better_frame": better_frame,
        "trains": source is _cc.Source.HUMAN,
    }, 200


def _process_thinkpad(data: dict, *, run_live=None) -> tuple[dict, int]:
    """Core of POST /thinkpad — separated from the HTTP handler so it is
    unit-testable without a running server (the _process_rating pattern).

    Fans out N fenced reasoning branches over the live engine via
    ``core.thinkpad.run_thinkpad_live`` (each branch carries the monothink
    scaffold + the shared recall lane, no tools, single inference), then stamps
    every branch's MonoTrace ``monothink_active`` (see
    ``turn_trace.set_turn_monothink_active``) so a /rating on a branch
    trace_id passes the ``will_evolve`` gate. Returns ALL branches plus the
    ADVISORY grounded-verdict ranking — surfacing, not committing: nothing
    here writes to the conversation.
    """
    if not isinstance(data, dict):
        return {"ok": False, "error": "invalid body"}, 400
    message = str(data.get("message") or "").strip()
    if not message:
        return {"ok": False, "error": "'message' is required"}, 400
    try:
        n = int(data.get("n", 2))
    except (TypeError, ValueError):
        n = 2
    n = max(1, min(n, 4))  # hard cap: each branch is a real inference

    from core import turn_trace as _tt

    base_config: dict = {}
    if run_live is None:
        from core.llm_config import load_config
        from core.thinkpad import run_thinkpad_live as run_live  # noqa: F811
        base_config = load_config()

    try:
        latest = _tt.latest_outer_turn()
        parent_turn_id = latest.get("turn_id") if latest else None
    except Exception:
        parent_turn_id = None

    messages = [{"role": "user", "content": message}]
    # Phase 4 (shadow ablation): an explicit scaffold variant (lesson
    # present/absent) may ride the body. Branches run on it; the live
    # scaffold file is never touched. Absent → run_thinkpad_live loads
    # the live scaffold itself.
    scaffold = data.get("scaffold")
    scaffold = str(scaffold) if isinstance(scaffold, str) and scaffold.strip() else None
    try:
        if scaffold is not None:
            res, trace_ids = run_live(messages, base_config, n=n,
                                      parent_turn_id=parent_turn_id, scaffold=scaffold)
        else:
            res, trace_ids = run_live(messages, base_config, n=n,
                                      parent_turn_id=parent_turn_id)
    except Exception as exc:
        return {"ok": False, "error": _public_error("thinkpad failed", exc)}, 500

    stamped = [tid for tid in trace_ids if tid and _tt.set_turn_monothink_active(tid)]
    branches = [
        {
            "id": b.id,
            "answer": b.answer,
            "think": b.think,
            "cites": list(b.cites),
            "trace_id": b.trace_id,
        }
        for b in res.branches
    ]
    advisory = [
        {
            "id": r.candidate.id,
            "authority": r.authority,
            "grounded": r.grounded,
            "winning_cite": r.winning_cite,
        }
        for r in res.advisory
    ]
    return {
        "ok": True,
        "n": len(branches),
        "branches": branches,
        "advisory": advisory,
        "parent_turn_id": parent_turn_id,
        "stamped_trainable": stamped,
    }, 200


# Tag construction lives in core.channel_tag now — single source of truth
# shared with the UI's local-input path and history-replay tagging. Authored
# from world_state via PlaneLoader.peek_mode, so the tag reflects applied
# state (not validator-pass receipts that diverged when _world_state wasn't
# wired). See _handle_chat_result for the build site (post-_apply_pending_all).


class AgentServer:
    """
    Thread-safe HTTP + MCP stdio server with no Qt dependencies.
    ConnectionsPage sets the callbacks and drives token delivery.
    """

    RESPONSE_TIMEOUT = 120.0  # seconds before /chat gives up waiting
    BUSY_WAIT_TIMEOUT = 5.0
    MAX_BODY_BYTES = 1_000_000
    MAX_MESSAGE_CHARS = 20_000
    MAX_SSE_SUBSCRIBERS = 64
    MAX_WEBHOOKS = 128
    MAX_WEBHOOK_INFLIGHT = 32

    def __init__(self) -> None:
        self._host = "127.0.0.1"
        self._port = 7821
        self._httpd: HTTPServer | None = None
        self._http_thread: threading.Thread | None = None
        self._running = False
        self._auth_token = str(os.getenv("MONOLITH_AGENT_TOKEN", "")).strip()

        # One pending request at a time
        self._lock = threading.Lock()
        self._gate = threading.Condition(self._lock)
        self._busy = False
        self._pending_event: threading.Event | None = None
        self._pending_tokens: list[str] = []

        # SSE subscribers: list of queue.Queue that receive typed event dicts
        self._sse_subscribers: list[queue.Queue] = []
        self._sse_lock = threading.Lock()

        # Webhook registrations: [{url, events, id}, ...]
        self._webhooks: list[dict] = []
        self._webhook_lock = threading.Lock()
        self._webhook_counter = 0
        self._webhook_inflight = threading.BoundedSemaphore(self.MAX_WEBHOOK_INFLIGHT)

        # Nudges: [{text, ttl, remaining}, ...]
        self._nudges: list[dict] = []
        self._nudge_lock = threading.Lock()

        # MCP SSE sessions: session_id → Queue (for /mcp/sse transport)
        self._mcp_sse_sessions: dict[str, queue.Queue] = {}
        self._mcp_sse_lock = threading.Lock()

        # Connected participants: name → {name, joined_at}
        self._participants: dict[str, dict] = {}
        self._participants_lock = threading.Lock()

        # Pending plane modes — set by /chat body, /chat/stream body, or MCP
        # send_message body, applied to world_state immediately before the
        # next on_message dispatch (see _handle_chat_result around line 704).
        # Step-2.6 compression: one _PendingPlane instance per plane, held
        # here in a dict keyed by plane name. Per-plane public setters/getters
        # are thin wrappers around _set_pending / _apply_pending; the registry
        # pattern means adding a plane is one entry in _PLANE_SPECS, not a
        # new 3-line slot cluster here.
        self._pending_planes: dict[str, _PendingPlane] = {
            name: _PendingPlane(name=name, key_suffix=suffix, module_import_path=path)
            for (name, suffix, path) in _PLANE_SPECS
        }

        # Set by ConnectionsPage (called from server thread -> safe to emit Qt signals)
        self.on_message: Callable[[str, str], None] | None = None  # (agent_name, text)
        # Set by ConnectionsPage. () -> trigger a load of the model currently
        # selected/configured in Monolith, bringing the engine online so /chat
        # works after an autonomous (headless-ish) restart. Called from the
        # server thread → the callback hops to the Qt thread before touching
        # engine state (same pattern as on_message / on_reset).
        self.on_load_model: Callable[[], None] | None = None
        self.on_log: Callable[[str], None] | None = None
        # Set by ConnectionsPage — returns dict with current state snapshot
        self.on_state_request: Callable[[], dict] | None = None
        # Set by ConnectionsPage — called when participants join or leave
        self.on_participant_change: Callable[[], None] | None = None
        # Debug-read callbacks (set by ConnectionsPage when a PageChat exists).
        # Worker-thread reads of Qt-side state — implementations must return
        # a snapshot (no live references) so callers can serialize safely.
        self.on_session_messages: Callable[[int], list[dict]] | None = None
        self.on_interceptor_state: Callable[[], dict] | None = None
        # Set by ConnectionsPage — triggers a fresh/cold chat surface the same
        # way the UI "New Chat" does (PageChat._start_new_session, marshalled to
        # the Qt thread). Called from the server thread; the wired implementation
        # emits a Qt signal and returns a *dispatch* receipt (the actual reset
        # runs asynchronously on the Qt event loop — see _handle_reset).
        self.on_reset: Callable[[], dict] | None = None

    def _is_authorized(self, headers) -> bool:
        if not self._auth_token:
            return True
        auth_header = str(getattr(headers, "get", lambda *_: "")("Authorization", "") or "").strip()
        if auth_header.lower().startswith("bearer "):
            token = auth_header[7:].strip()
            if hmac.compare_digest(token, self._auth_token):
                return True
        api_key = str(getattr(headers, "get", lambda *_: "")("X-API-Key", "") or "").strip()
        return hmac.compare_digest(api_key, self._auth_token)

    # -- lifecycle ----------------------------------------------------------

    def start(self, port: int = 7821, host: str = "127.0.0.1") -> None:
        if self._running:
            return
        bind_host = str(host or "127.0.0.1").strip()
        if not _is_loopback_host(bind_host) and not self._auth_token:
            raise ValueError(
                "Refusing a non-loopback AgentServer bind without "
                "MONOLITH_AGENT_TOKEN"
            )
        self._host = bind_host
        self._port = port
        self._httpd = _QuietHTTPServer((self._host, port), self._make_handler())
        self._http_thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="monolith-agent-server",
            daemon=True,
        )
        self._running = True
        self._http_thread.start()
        self._log(f"server started on {self._host}:{port}")

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        httpd = self._httpd
        http_thread = self._http_thread
        self._httpd = None
        self._http_thread = None

        if httpd:
            httpd.shutdown()
            httpd.server_close()

        if (
            http_thread
            and http_thread.is_alive()
            and threading.current_thread() is not http_thread
        ):
            http_thread.join(timeout=2.0)

        # Unblock any waiting request
        with self._lock:
            if self._pending_event:
                self._pending_event.set()
        self._log("server stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    # -- Qt -> Server  (called from Qt main thread) -------------------------

    def push_token(self, token: str) -> None:
        """Feed an LLM token into the pending response buffer."""
        with self._lock:
            if self._pending_event is not None:
                self._pending_tokens.append(token)

    def push_done(self) -> None:
        """Signal that the LLM has finished. Unblocks the waiting HTTP handler."""
        with self._lock:
            if self._pending_event is not None:
                self._pending_event.set()

    # -- SSE broadcasting --------------------------------------------------

    def broadcast_event(self, event_type: str, data: dict | None = None) -> None:
        """
        Push a typed event to all SSE subscribers and registered webhooks.
        Called from Qt main thread (via ConnectionsPage signal handlers).
        """
        evt = {"event": event_type, "data": data or {}, "ts": time.time()}

        # SSE subscribers
        with self._sse_lock:
            dead = []
            for i, q in enumerate(self._sse_subscribers):
                try:
                    q.put_nowait(evt)
                except queue.Full:
                    dead.append(i)
            for i in reversed(dead):
                self._sse_subscribers.pop(i)

        # Webhooks (fire-and-forget in background thread)
        with self._webhook_lock:
            hooks = [h for h in self._webhooks if event_type in h["events"]]
        for hook in hooks:
            if not self._webhook_inflight.acquire(blocking=False):
                self._log("webhook dispatch skipped: in-flight limit reached")
                continue
            threading.Thread(
                target=self._fire_webhook_guarded,
                args=(hook["url"], evt),
                daemon=True,
            ).start()

    def _subscribe_sse(self) -> queue.Queue | None:
        """Create a new SSE subscription queue."""
        q: queue.Queue = queue.Queue(maxsize=512)
        with self._sse_lock:
            if len(self._sse_subscribers) >= self.MAX_SSE_SUBSCRIBERS:
                return None
            self._sse_subscribers.append(q)
        return q

    def _unsubscribe_sse(self, q: queue.Queue) -> None:
        with self._sse_lock:
            try:
                self._sse_subscribers.remove(q)
            except ValueError:
                pass

    def _fire_webhook(self, url: str, evt: dict) -> None:
        """POST event JSON to a webhook URL. Best-effort, no retries."""
        try:
            body = json.dumps(evt, ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(
                url, data=body, method="POST",
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as exc:
            self._log(f"webhook delivery failed for {url}: {exc!r}")

    def _fire_webhook_guarded(self, url: str, evt: dict) -> None:
        try:
            self._fire_webhook(url, evt)
        finally:
            try:
                self._webhook_inflight.release()
            except ValueError:
                pass

    # -- Nudge management --------------------------------------------------

    def add_nudge(self, text: str, ttl: int = 1) -> dict:
        """Register an ephemeral system hint for the next N generations."""
        with self._nudge_lock:
            nudge = {"text": text, "ttl": ttl, "remaining": ttl}
            self._nudges.append(nudge)
            return {"ok": True, "nudge": text, "ttl": ttl}

    def consume_nudges(self) -> list[str]:
        """
        Called by ConnectionsPage before each generation.
        Returns active nudge texts and decrements their remaining count.
        """
        with self._nudge_lock:
            active = [n["text"] for n in self._nudges if n["remaining"] > 0]
            for n in self._nudges:
                n["remaining"] -= 1
            self._nudges = [n for n in self._nudges if n["remaining"] > 0]
            return active

    def get_active_nudges(self) -> list[dict]:
        """Return current nudges (for /state)."""
        with self._nudge_lock:
            return [{"text": n["text"], "remaining": n["remaining"], "ttl": n["ttl"]}
                    for n in self._nudges if n["remaining"] > 0]

    # -- Webhook management ------------------------------------------------

    def register_webhook(self, url: str, events: list[str] | None = None) -> dict:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return {"ok": False, "error": "invalid webhook url (must be http/https)"}
        valid_events = set(events or ALL_EVENTS) & ALL_EVENTS
        if not valid_events:
            return {"ok": False, "error": f"no valid events - choose from: {sorted(ALL_EVENTS)}"}
        with self._webhook_lock:
            if len(self._webhooks) >= self.MAX_WEBHOOKS:
                return {"ok": False, "error": f"too many webhooks (max {self.MAX_WEBHOOKS})"}
            self._webhook_counter += 1
            hook = {"id": self._webhook_counter, "url": url, "events": valid_events}
            self._webhooks.append(hook)
            return {"ok": True, "id": hook["id"], "url": url, "events": sorted(valid_events)}

    def unregister_webhook(self, hook_id: int) -> dict:
        with self._webhook_lock:
            before = len(self._webhooks)
            self._webhooks = [h for h in self._webhooks if h["id"] != hook_id]
            removed = len(self._webhooks) < before
        return {"ok": removed, "removed": removed}

    # -- internal -----------------------------------------------------------

    def _log(self, text: str) -> None:
        if self.on_log:
            self.on_log(text)

    # Per-plane validation reads from the loader at call time — single source
    # of truth (S11 plane separation, drift-free). Earlier revisions held local
    # frozenset copies that drifted (experimental/monolith stayed in this
    # class's _VALID_EFFORT_TIERS for a session after the loader had removed
    # them); reading from the loader's accessor function makes drift
    # structurally impossible — when the loader's set changes, this class
    # sees the change on the next setter call without any code change here.
    # Lazy import inside each setter matches the existing core/* import
    # pattern in this file (no module-level core/* imports).

    def _log_canonical(
        self,
        kind: str,
        agent_name: str,
        text: str,
        *,
        extra_payload: dict | None = None,
    ) -> None:
        """Best-effort canonical_log write for chat dispatch events.

        Acatalepsy v1 producer loop: every user_message + assistant_message
        + session_open + session_close lands here. Failures are swallowed
        (logged via self._log) so substrate issues never break chat.

        v1.2: optional ``extra_payload`` merges into the canonical_log payload
        — used by the timeout path to mark a partial assistant capture with
        ``{"timeout": True, "partial": True}`` so the auditor can filter or
        weigh those entries separately from clean completions.

        See docs/specs/acatalepsy_v1_spec.md §6 + core/acatalepsy/canonical_log.py.
        """
        try:
            from core.acatalepsy import canonical_log as _cl
            session_id = f"connect:{agent_name}" if agent_name else "connect:anon"
            payload: dict = {"agent": agent_name}
            if kind in ("user_message", "assistant_message"):
                payload["text"] = text or ""
            elif text:
                payload["note"] = text
            if extra_payload:
                payload.update(extra_payload)
            _cl.append(kind, payload=payload, session_id=session_id)
        except Exception as exc:
            try:
                self._log(f"[canonical_log:{kind}] write failed: {exc!r}")
            except Exception:
                pass

    # ── plane pending-state: generic mechanism + per-plane wrappers ───
    #
    # Compressed step 2.6: one _set_pending / _apply_pending pair replaces
    # what was 4 near-identical setter/consumer clusters. Validation reads
    # from each plane's loader at call time (single source of truth, drift-
    # free per step 1). Per-plane public methods below are thin wrappers
    # preserving the established API surface.

    def _set_pending(self, plane: str, value: str | None, once: bool = True) -> bool:
        """Generic setter — dispatches to the plane's _PendingPlane slot.

        Returns True on accept (including clear-via-None). Returns False
        only when the value is non-empty but fails the plane's validator.
        """
        pp = self._pending_planes.get(plane)
        if pp is None:
            return False
        with pp.lock:
            if value is None or not str(value).strip():
                pp.value = None
                pp.once = False
                return True
            v = str(value).strip().lower()
            if v not in pp.get_module().valid_modes():
                return False
            pp.value = v
            pp.once = bool(once)
            return True

    def _apply_pending(self, plane: str) -> None:
        """Generic consumer — pipes pending value into the plane's world_state slot.

        Best-effort: if the plane's loader module isn't wired (testing /
        standalone), the value is silently dropped. Cleared after one dispatch
        regardless. Matches the per-plane behavior the wrappers had pre-
        compression.
        """
        pp = self._pending_planes.get(plane)
        if pp is None:
            return
        with pp.lock:
            value = pp.value
            once = pp.once
            pp.value = None
            pp.once = False
        if not value:
            return
        try:
            ws = getattr(pp.get_module(), "_world_state", None)
            if ws is None:
                return
            key = pp.once_state_key if once else pp.baseline_state_key
            ws.state[key] = value
            try:
                ws.mark_dirty()
            except Exception:
                pass
        except Exception as exc:
            self._log(f"_apply_pending({plane}) failed: {exc!r}")

    def _apply_pending_all(self) -> None:
        """Apply every plane's pending state in registry order.

        Dispatch order mirrors _PLANE_SPECS / interceptor registration order
        in bootstrap.py: effort, conversation, reasoning, linguency. Don't
        reorder casually — see proximity-ordering comment in bootstrap.py.
        """
        for plane in self._pending_planes:
            self._apply_pending(plane)

    # Per-plane public wrappers (preserve established API surface):

    def set_pending_effort(self, tier: str | None, once: bool = True) -> bool:
        """Set the effort tier for the next chat dispatch. Returns True on accept.

        Applied to world_state immediately before on_message fires — Layer 2/3
        in core/effort.py:_resolve_mode. `once=True` consumes after one turn;
        `once=False` sets the persistent baseline.
        """
        return self._set_pending("effort", tier, once=once)

    def _apply_pending_effort(self) -> None:
        self._apply_pending("effort")

    def set_pending_conversation_mode(self, mode: str | None, once: bool = True) -> bool:
        """Set the conversation mode for the next chat dispatch. Returns True on accept."""
        return self._set_pending("conversation", mode, once=once)

    def _apply_pending_conversation(self) -> None:
        self._apply_pending("conversation")

    def set_pending_reasoning_mode(self, mode: str | None, once: bool = True) -> bool:
        """Set the reasoning mode for the next chat dispatch. Returns True on accept."""
        return self._set_pending("reasoning", mode, once=once)

    def _apply_pending_reasoning(self) -> None:
        self._apply_pending("reasoning")

    def set_pending_linguency_mode(self, mode: str | None, once: bool = True) -> bool:
        """Set the linguency mode for the next chat dispatch. Returns True on accept."""
        return self._set_pending("linguency", mode, once=once)

    def _apply_pending_linguency(self) -> None:
        self._apply_pending("linguency")

    def _handle_chat(
        self,
        agent_name: str,
        message: str,
        *,
        transport: str | None = None,
    ) -> tuple[bool, str]:
        result = self._handle_chat_result(agent_name, message, transport=transport)
        return bool(result["ok"]), str(result["response"])

    def _handle_chat_result(
        self,
        agent_name: str,
        message: str,
        *,
        transport: str | None = None,
    ) -> dict:
        """
        Called from an HTTP handler thread.
        Notifies Qt (via on_message), then blocks waiting for the LLM response.
        Returns a dict with `ok`, cleaned `response`, and raw response text.

        `transport` names the inbound transport ("/chat blocking",
        "/chat/stream", "mcp send_message"). When set, this method strips any
        caller-supplied [CHANNEL: ...] prefix, applies pending plane modes,
        then injects a fresh tag via core.channel_tag.build_channel_tag — so
        the model sees a server-authored tag whose values match what the
        interceptors will actually apply this turn.
        """
        from core.channel_tag import build_channel_tag, strip_leading_channel_tag

        message = str(message or "")
        # Strip any caller-supplied [CHANNEL: ...] prefix — callers may lazy-
        # prepend their own; we re-author from world_state truth below.
        if transport is not None:
            message, _stripped = strip_leading_channel_tag(message)
        if len(message) > self.MAX_MESSAGE_CHARS:
            msg = f"message too large (max {self.MAX_MESSAGE_CHARS} chars)"
            return {"ok": False, "response": msg, "raw_response": ""}

        with self._gate:
            deadline = time.monotonic() + self.BUSY_WAIT_TIMEOUT
            while self._busy:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return {
                        "ok": False,
                        "response": "busy — timed out waiting for the previous request",
                        "raw_response": "",
                    }
                self._gate.wait(timeout=remaining)
            self._busy = True
            self._pending_event = threading.Event()
            self._pending_tokens = []

        self._log(f"[{agent_name}] {message[:80]}")

        if not self.on_message:
            # Standalone mode — Monolith not connected
            with self._gate:
                self._busy = False
                self._pending_event = None
                self._gate.notify()
            msg = "Monolith not connected (start server from within Monolith)"
            return {"ok": False, "response": msg, "raw_response": ""}

        event = self._pending_event
        # Pipe any pending effort tier into world_state before dispatch — must
        # happen here, after _busy is set but before on_message fires, so the
        # interceptor layer sees it on this turn's _resolve_tier call.
        self._apply_pending_all()
        # Acatalepsy v1: log the user message to canonical_log BEFORE
        # dispatch so it has a stable event_id the assistant turn can
        # reference. Best-effort — substrate failures must not break
        # the chat path. Logged tag-free; tag is a render-time addition
        # for the model, not a property of the message itself.
        self._log_canonical("user_message", agent_name, message)
        # Build the [CHANNEL: ...] tag from world_state truth (post-apply,
        # via peek so the interceptor still consumes the once key). The tag
        # is what the model sees prepended to its turn input.
        if transport is not None:
            channel_tag = build_channel_tag(
                f"connect/{agent_name}", transport=transport, include_modes=True
            )
            tagged_message = f"{channel_tag}\n\n{message}"
        else:
            tagged_message = message
        # SP2: snapshot the latest outer turn BEFORE dispatch so we can return
        # THIS turn's id (the new one that appears after) rather than a stale
        # prior turn. If no new outer turn appears (e.g. an ephemeral/tool-only
        # generation, or a competing writer), we omit turn_id instead of
        # returning the wrong one — turning silent mis-attribution into a
        # detectable absence. Precondition for clean attribution: no competing
        # outer-turn write during this blocking call (holds for the controlled
        # serial training loop; the live UI's emergence/curiosity heartbeats are
        # deterministic and do not produce outer turns).
        try:
            from core import turn_trace as _tt_pre
            _pre = _tt_pre.latest_outer_turn()
            _pre_turn_id = _pre["turn_id"] if _pre else None
        except Exception:
            _pre_turn_id = None
        try:
            self.on_message(agent_name, tagged_message)
        except Exception as exc:
            # If callback dispatch fails we must release the gate or future chats stall.
            with self._gate:
                self._pending_tokens = []
                self._pending_event = None
                self._busy = False
                self._gate.notify()
            self._log(f"callback dispatch failed: {exc!r}")
            msg = _public_error("failed to dispatch to Monolith", exc)
            return {"ok": False, "response": msg, "raw_response": ""}

        finished = event.wait(timeout=self.RESPONSE_TIMEOUT)

        with self._gate:
            response = "".join(self._pending_tokens)
            self._pending_tokens = []
            self._pending_event = None
            self._busy = False
            self._gate.notify()

        if not finished:
            # Acatalepsy v1.2: even on timeout, log whatever tokens finalized
            # before the cutoff so the response doesn't fall in the substrate
            # gap. Marks the entry with timeout=True + partial=True so the
            # auditor can filter / weigh these differently from clean
            # completions. Late-arriving tokens after _pending_tokens
            # cleanup are still lost — that's a v1.3 follow-up.
            if response and response.strip():
                clean_partial = _clean_agent_response(response)
                self._log_canonical(
                    "assistant_message",
                    agent_name,
                    clean_partial,
                    extra_payload={"timeout": True, "partial": True},
                )
            return {
                "ok": False,
                "response": "timeout waiting for LLM response",
                "raw_response": response,
            }
        clean = _clean_agent_response(response)
        # Acatalepsy v1: log the assistant turn after streaming completes.
        # Best-effort — substrate failures must not break the chat path.
        self._log_canonical("assistant_message", agent_name, clean)
        result = {"ok": True, "response": clean, "raw_response": response}
        # SP2: attach the turn_id of the turn this call produced — the NEW outer
        # turn that appeared since the pre-dispatch snapshot — plus whether it ran
        # monothink, so the caller can rate that exact turn via /rating.
        try:
            from core import turn_trace as _tt_post
            _post = _tt_post.latest_outer_turn()
            if _post and _post["turn_id"] != _pre_turn_id:
                result["turn_id"] = _post["turn_id"]
                result["monothink_active"] = _post["monothink_active"]
        except Exception:
            pass
        return result

    def _make_handler(self):
        server = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args) -> None:
                pass  # suppress CLF output

            def _json(self, data: dict, status: int = 200) -> None:
                body = json.dumps(data, ensure_ascii=False).encode("utf-8")
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(body)
                except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                    pass  # client disconnected before we could respond

            def do_OPTIONS(self) -> None:
                self.send_response(200)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods",
                                 "GET, POST, DELETE, OPTIONS")
                self.send_header(
                    "Access-Control-Allow-Headers",
                    "Content-Type, Authorization, X-API-Key",
                )
                self.end_headers()

            def _read_body(self) -> bytes:
                length = int(self.headers.get("Content-Length", 0))
                if length > server.MAX_BODY_BYTES:
                    self._json(
                        {
                            "ok": False,
                            "error": f"request body too large (max {server.MAX_BODY_BYTES} bytes)",
                        },
                        413,
                    )
                    return b"__body_too_large__"
                return self.rfile.read(length) if length else b""

            def _parse_json_body(self) -> dict | None:
                raw = self._read_body()
                if raw == b"__body_too_large__":
                    return None
                if not raw:
                    return {}
                try:
                    text = raw.decode("utf-8", errors="replace")
                    return json.loads(text)
                except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
                    self._json({"ok": False, "error": "invalid JSON body"}, 400)
                    return None

            # -- GET routes ------------------------------------------------

            def do_GET(self) -> None:
                path = self.path.split("?")[0]
                if path == "/health":
                    self._json({"ok": True, "server": "monolith", "port": server._port})
                elif path == "/state":
                    if not server._is_authorized(self.headers):
                        self._json({"ok": False, "error": "unauthorized"}, 401)
                        return
                    self._handle_state()
                elif path == "/hooks":
                    if not server._is_authorized(self.headers):
                        self._json({"ok": False, "error": "unauthorized"}, 401)
                        return
                    self._handle_list_hooks()
                elif path == "/mcp/sse":
                    if not server._is_authorized(self.headers):
                        self._json({"ok": False, "error": "unauthorized"}, 401)
                        return
                    self._handle_mcp_sse()
                elif path == "/who":
                    if not server._is_authorized(self.headers):
                        self._json({"ok": False, "error": "unauthorized"}, 401)
                        return
                    with server._participants_lock:
                        names = [p["name"] for p in server._participants.values()]
                    self._json({"ok": True, "participants": names})
                elif path == "/events":
                    if not server._is_authorized(self.headers):
                        self._json({"ok": False, "error": "unauthorized"}, 401)
                        return
                    # Persistent SSE stream — subscribes to all broadcast events.
                    # Agents open GET /events to receive real-time notifications
                    # without sending a chat message.
                    with server._sse_lock:
                        if len(server._sse_subscribers) >= server.MAX_SSE_SUBSCRIBERS:
                            self._json(
                                {"ok": False, "error": "too many active subscribers"},
                                503,
                            )
                            return
                        sub: queue.Queue = queue.Queue(maxsize=256)
                        server._sse_subscribers.append(sub)
                    try:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Connection", "keep-alive")
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.end_headers()
                        # Send a connected handshake event
                        handshake = json.dumps({"server": "monolith", "port": server._port})
                        self.wfile.write(f"event: connected\ndata: {handshake}\n\n".encode("utf-8"))
                        self.wfile.flush()
                    except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                        server._unsubscribe_sse(sub)
                        return
                    try:
                        while server._running:
                            try:
                                evt = sub.get(timeout=15.0)
                            except queue.Empty:
                                # keepalive ping
                                self.wfile.write(b": ping\n\n")
                                self.wfile.flush()
                                continue
                            event_type = evt.get("event", "unknown")
                            event_data = json.dumps(evt.get("data", {}), ensure_ascii=False)
                            self.wfile.write(
                                f"event: {event_type}\ndata: {event_data}\n\n".encode("utf-8")
                            )
                            self.wfile.flush()
                    except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
                        pass
                    finally:
                        server._unsubscribe_sse(sub)
                # ── Debug-read surface (read-only diagnostic endpoints) ─────
                elif path == "/interceptors/state":
                    if not server._is_authorized(self.headers):
                        self._json({"ok": False, "error": "unauthorized"}, 401)
                        return
                    self._handle_interceptors_state()
                elif path == "/memory/recall":
                    if not server._is_authorized(self.headers):
                        self._json({"ok": False, "error": "unauthorized"}, 401)
                        return
                    self._handle_memory_recall()
                elif path == "/memory/continuity":
                    if not server._is_authorized(self.headers):
                        self._json({"ok": False, "error": "unauthorized"}, 401)
                        return
                    self._handle_memory_continuity()
                elif path == "/trace/recent":
                    if not server._is_authorized(self.headers):
                        self._json({"ok": False, "error": "unauthorized"}, 401)
                        return
                    self._handle_trace_recent()
                elif path == "/trace/errors":
                    if not server._is_authorized(self.headers):
                        self._json({"ok": False, "error": "unauthorized"}, 401)
                        return
                    self._handle_trace_errors()
                elif path == "/session/messages":
                    if not server._is_authorized(self.headers):
                        self._json({"ok": False, "error": "unauthorized"}, 401)
                        return
                    self._handle_session_messages()
                elif path == "/log/tail":
                    if not server._is_authorized(self.headers):
                        self._json({"ok": False, "error": "unauthorized"}, 401)
                        return
                    self._handle_log_tail()
                elif path == "/log/since":
                    if not server._is_authorized(self.headers):
                        self._json({"ok": False, "error": "unauthorized"}, 401)
                        return
                    self._handle_log_since()
                elif path == "/db/query":
                    if not server._is_authorized(self.headers):
                        self._json({"ok": False, "error": "unauthorized"}, 401)
                        return
                    self._handle_db_query_get()
                elif (_m := re.fullmatch(r"/trace/([A-Za-z0-9._-]+)/prompt", path)):
                    if not server._is_authorized(self.headers):
                        self._json({"ok": False, "error": "unauthorized"}, 401)
                        return
                    self._handle_trace_prompt(_m.group(1))
                elif (_m := re.fullmatch(r"/trace/([A-Za-z0-9._-]+)", path)):
                    if not server._is_authorized(self.headers):
                        self._json({"ok": False, "error": "unauthorized"}, 401)
                        return
                    self._handle_trace_one(_m.group(1))
                elif (_m := re.fullmatch(r"/session/messages/(\d+)", path)):
                    if not server._is_authorized(self.headers):
                        self._json({"ok": False, "error": "unauthorized"}, 401)
                        return
                    self._handle_session_message_at(int(_m.group(1)))
                else:
                    self._json({"error": "not found"}, 404)

            # ── POST routes ───────────────────────────────────────────────────

            def do_POST(self) -> None:
                path = self.path.split("?")[0]
                if not server._is_authorized(self.headers):
                    self._json({"ok": False, "error": "unauthorized"}, 401)
                    return
                if path == "/chat":
                    self._handle_post_chat()
                elif path == "/chat/stream":
                    self._handle_stream_chat()
                elif path == "/hooks":
                    self._handle_register_hook()
                elif path == "/nudge":
                    self._handle_nudge()
                elif path == "/monothink":
                    self._handle_monothink()
                elif path == "/thinkpad":
                    self._handle_thinkpad()
                elif path == "/rating":
                    self._handle_rating()
                elif path == "/frame":
                    self._handle_frame()
                elif path == "/reset":
                    self._handle_reset()
                elif path == "/load_model":
                    self._handle_load_model()
                elif path == "/mcp":
                    self._handle_mcp_http()
                elif path == "/mcp/message":
                    self._handle_mcp_sse_message()
                elif path == "/join":
                    self._handle_join()
                elif path == "/leave":
                    self._handle_leave()
                elif path == "/db/query":
                    self._handle_db_query_post()
                else:
                    self._json({"error": "not found"}, 404)

            # -- DELETE routes ---------------------------------------------

            def do_DELETE(self) -> None:
                path = self.path.split("?")[0]
                if not server._is_authorized(self.headers):
                    self._json({"ok": False, "error": "unauthorized"}, 401)
                    return
                if path == "/hooks":
                    self._handle_delete_hook()
                else:
                    self._json({"error": "not found"}, 404)

            # -- /chat (blocking) ------------------------------------------

            def _handle_load_model(self) -> None:
                """POST /load_model — bring Monolith's engine online by loading the
                currently selected/configured model. Returns a dispatch receipt
                (202); the actual load runs async on the Qt/engine thread. Poll
                GET /health or /state to see READY."""
                cb = server.on_load_model
                if cb is None:
                    self._json({"ok": False, "error": "load_model unavailable (no UI bound)"}, 503)
                    return
                try:
                    cb()
                except Exception as exc:
                    server._log(f"load dispatch failed: {exc!r}")
                    self._json(
                        {"ok": False, "error": _public_error("load dispatch failed", exc)},
                        500,
                    )
                    return
                self._json({"ok": True, "status": "load triggered — loading the selected model"}, 202)

            def _handle_post_chat(self) -> None:
                data = self._parse_json_body()
                if data is None:
                    return
                debug_thinking = bool(data.get("debug_thinking", False))
                message = str(data.get("message") or "").strip()
                agent_name = str(
                    data.get("agent") or data.get("agent_name") or "Agent"
                ).strip()
                if not message:
                    self._json({"ok": False, "error": "'message' is required"}, 400)
                    return
                # Per-call plane modes (Layer 1 in each plane's _resolve): peers
                # can dial each plane's scaffold per turn via the request body.
                # `applied` is returned to the caller as a receipt; tag injection
                # happens server-side inside _handle_chat_result post-apply.
                applied = _parse_per_call_modes(server, data)
                result = server._handle_chat_result(
                    agent_name, message, transport="/chat blocking"
                )
                if result["ok"]:
                    payload = {
                        "ok": True,
                        "response": result["response"],
                        "agent": agent_name,
                    }
                    payload.update(applied)  # surface applied plane modes back to caller
                    # SP2: surface the turn_id this call produced (captured in
                    # _handle_chat_result via the pre/post snapshot) so the caller
                    # can rate that exact turn via /rating.
                    if result.get("turn_id"):
                        payload["turn_id"] = result["turn_id"]
                        payload["monothink_active"] = result.get("monothink_active", False)
                    if debug_thinking:
                        payload["debug"] = {
                            "thinking": _extract_thinking_text(result["raw_response"]),
                            "raw_response": result["raw_response"],
                        }
                    self._json(payload)
                else:
                    payload = {"ok": False, "error": result["response"]}
                    if debug_thinking and result["raw_response"]:
                        payload["debug"] = {
                            "thinking": _extract_thinking_text(result["raw_response"]),
                            "raw_response": result["raw_response"],
                        }
                    self._json(payload, 503)

            # -- /chat/stream (SSE) ----------------------------------------

            def _handle_stream_chat(self) -> None:
                data = self._parse_json_body()
                if data is None:
                    return
                debug_thinking = bool(data.get("debug_thinking", False))
                message = str(data.get("message") or "").strip()
                agent_name = str(
                    data.get("agent") or data.get("agent_name") or "Agent"
                ).strip()
                if not message:
                    self._json({"ok": False, "error": "'message' is required"}, 400)
                    return
                # Mirror /chat: per-call plane modes; tag injection happens
                # server-side inside _handle_chat_result post-apply. Transport
                # name distinguishes stream from blocking for the model.
                applied = _parse_per_call_modes(server, data)

                # Subscribe to SSE before sending message so we don't miss events
                sub = server._subscribe_sse()
                if sub is None:
                    self._json(
                        {"ok": False, "error": "too many active stream subscribers"},
                        503,
                    )
                    return

                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "keep-alive")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                    server._unsubscribe_sse(sub)
                    return

                # Dispatch the message (non-blocking — we'll stream events)
                # We need to handle_chat in a separate thread so we can stream
                chat_thread = threading.Thread(
                    target=server._handle_chat,
                    args=(agent_name, message),
                    kwargs={"transport": "/chat/stream"},
                    daemon=True,
                )
                chat_thread.start()

                # Stream events until we get a 'done' event or timeout
                deadline = time.monotonic() + server.RESPONSE_TIMEOUT + 5
                parser = _ThinkingStreamParser() if debug_thinking else None
                try:
                    while time.monotonic() < deadline:
                        try:
                            evt = sub.get(timeout=1.0)
                        except queue.Empty:
                            # Send keepalive comment
                            self.wfile.write(b": keepalive\n\n")
                            self.wfile.flush()
                            continue

                        event_type = evt.get("event", "unknown")
                        event_data = evt.get("data", {})

                        def _send_event(name: str, payload: dict) -> None:
                            body = json.dumps(payload, ensure_ascii=False)
                            sse_line = f"event: {name}\ndata: {body}\n\n"
                            self.wfile.write(sse_line.encode("utf-8"))
                            self.wfile.flush()

                        if parser is not None and event_type == EVENT_TOKEN:
                            raw_text = str(event_data.get("text") or "")
                            engine_key = str(event_data.get("engine") or "")
                            for lane, text_chunk in parser.feed(raw_text):
                                payload = {"text": text_chunk}
                                if engine_key:
                                    payload["engine"] = engine_key
                                _send_event(lane, payload)
                        else:
                            _send_event(event_type, event_data)

                        if event_type == EVENT_DONE:
                            if parser is not None:
                                engine_key = str(event_data.get("engine") or "")
                                for lane, text_chunk in parser.flush():
                                    payload = {"text": text_chunk}
                                    if engine_key:
                                        payload["engine"] = engine_key
                                    _send_event(lane, payload)
                            break
                except (ConnectionResetError, ConnectionAbortedError,
                        BrokenPipeError, OSError):
                    pass  # client disconnected
                finally:
                    server._unsubscribe_sse(sub)

            # -- /state (observer) -----------------------------------------

            def _handle_state(self) -> None:
                state = {"ok": True, "server": "monolith", "port": server._port}

                # Add busy/idle status
                with server._lock:
                    state["busy"] = server._busy

                # Active nudges
                state["nudges"] = server.get_active_nudges()

                # Webhook count
                with server._webhook_lock:
                    state["webhooks"] = len(server._webhooks)

                # SSE subscriber count
                with server._sse_lock:
                    state["sse_subscribers"] = len(server._sse_subscribers)

                # Ask ConnectionsPage for chat state if wired
                if server.on_state_request:
                    try:
                        chat_state = server.on_state_request()
                        if isinstance(chat_state, dict):
                            state.update(chat_state)
                    except Exception as exc:
                        state["chat"] = {"error": "state callback failed"}
                        server._log(f"state callback failed: {exc!r}")

                try:
                    from core import review_loop as _review_loop
                    state["review_loop"] = _review_loop.review_summary(limit=3)
                except Exception as exc:
                    server._log(f"review summary failed: {exc!r}")
                    state["review_loop"] = {
                        "error": _public_error("review summary failed", exc)
                    }

                self._json(state)

            # -- /hooks (webhook management) -------------------------------

            def _handle_register_hook(self) -> None:
                data = self._parse_json_body()
                if data is None:
                    return
                url = str(data.get("url") or "").strip()
                if not url:
                    self._json({"ok": False, "error": "'url' is required"}, 400)
                    return
                events = data.get("events")
                if events is not None and not isinstance(events, list):
                    self._json({"ok": False,
                                "error": "'events' must be a list"}, 400)
                    return
                result = server.register_webhook(url, events)
                self._json(result, 200 if result.get("ok") else 400)

            def _handle_delete_hook(self) -> None:
                data = self._parse_json_body()
                if data is None:
                    return
                hook_id = data.get("id")
                if not isinstance(hook_id, int):
                    self._json({"ok": False,
                                "error": "'id' (int) is required"}, 400)
                    return
                result = server.unregister_webhook(hook_id)
                self._json(result)

            def _handle_list_hooks(self) -> None:
                with server._webhook_lock:
                    hooks = [
                        {"id": h["id"], "url": h["url"],
                         "events": sorted(h["events"])}
                        for h in server._webhooks
                    ]
                self._json({"ok": True, "hooks": hooks})

            # ── /mcp  (HTTP transport — Codex, Claude Code HTTP mode) ────────
            # Streamable HTTP: single POST, JSON-RPC in, JSON-RPC out.

            def _handle_mcp_http(self) -> None:
                data = self._parse_json_body()
                if data is None:
                    return
                responses: list[dict] = []

                def _send(msg: dict) -> None:
                    responses.append(msg)

                if isinstance(data, list):
                    for item in data:
                        server._dispatch_mcp(item, _send)
                else:
                    server._dispatch_mcp(data, _send)

                body_obj = responses[0] if len(responses) == 1 else responses
                body = json.dumps(body_obj, ensure_ascii=False).encode("utf-8")
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(body)
                except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                    pass

            # ── /mcp/sse  (SSE transport — Gemini CLI) ────────────────────────
            # MCP 2024-11-05 SSE transport:
            #   GET /mcp/sse  → SSE stream; first event is 'endpoint' with POST URL
            #   POST /mcp/message?sessionId=X → JSON-RPC; responses arrive on SSE stream

            def _handle_mcp_sse(self) -> None:
                import uuid as _uuid
                session_id = _uuid.uuid4().hex
                q: queue.Queue = queue.Queue(maxsize=256)
                with server._mcp_sse_lock:
                    server._mcp_sse_sessions[session_id] = q
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "keep-alive")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    endpoint_url = f"/mcp/message?sessionId={session_id}"
                    self.wfile.write(
                        f"event: endpoint\ndata: {endpoint_url}\n\n".encode("utf-8")
                    )
                    self.wfile.flush()
                    deadline = time.monotonic() + 3600.0
                    while time.monotonic() < deadline:
                        try:
                            msg = q.get(timeout=15.0)
                        except queue.Empty:
                            self.wfile.write(b": ping\n\n")
                            self.wfile.flush()
                            continue
                        if msg is None:
                            break
                        payload = json.dumps(msg, ensure_ascii=False)
                        self.wfile.write(
                            f"event: message\ndata: {payload}\n\n".encode("utf-8")
                        )
                        self.wfile.flush()
                except (ConnectionResetError, ConnectionAbortedError,
                        BrokenPipeError, OSError):
                    pass
                finally:
                    with server._mcp_sse_lock:
                        server._mcp_sse_sessions.pop(session_id, None)

            def _handle_mcp_sse_message(self) -> None:
                full_path = self.path
                session_id = ""
                if "?" in full_path:
                    for part in full_path.split("?", 1)[1].split("&"):
                        if part.startswith("sessionId="):
                            session_id = part[len("sessionId="):]
                            break
                with server._mcp_sse_lock:
                    q = server._mcp_sse_sessions.get(session_id)
                if q is None:
                    self._json({"error": "unknown sessionId"}, 404)
                    return
                data = self._parse_json_body()
                if data is None:
                    return

                def _send(msg: dict) -> None:
                    try:
                        q.put_nowait(msg)
                    except queue.Full:
                        pass

                if isinstance(data, list):
                    for item in data:
                        server._dispatch_mcp(item, _send)
                else:
                    server._dispatch_mcp(data, _send)
                try:
                    self.send_response(202)
                    self.send_header("Content-Length", "0")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                    pass

            # ── /monothink (toggle the self-evolving reasoning scaffold) ──────

            def _handle_monothink(self) -> None:
                data = self._parse_json_body()
                if data is None:
                    return
                mode = str(data.get("mode") or "status").strip().lower()
                from core.monothink import set_monothink_toggle
                result = set_monothink_toggle(mode)
                server._log(f"[monothink] mode={mode} -> {result}")
                self._json(result, 200 if result.get("ok") else 400)

            # ── /thinkpad (reasoning-branch fan-out for the training loop) ────

            def _handle_thinkpad(self) -> None:
                """POST /thinkpad — fan out N fenced reasoning branches and
                return them all (+ advisory ranking). Blocking; takes the same
                busy gate as /chat so a fan-out can't collide with a live turn.
                """
                data = self._parse_json_body()
                if data is None:
                    return
                with server._gate:
                    if server._busy:
                        self._json({"ok": False, "error": "busy with another turn"}, 503)
                        return
                    server._busy = True
                try:
                    payload, status = _process_thinkpad(data)
                finally:
                    with server._gate:
                        server._busy = False
                        server._gate.notify()
                server._log(
                    f"[thinkpad] ok={payload.get('ok')} n={payload.get('n')} "
                    f"stamped={len(payload.get('stamped_trainable') or [])}"
                )
                self._json(payload, status)

            # ── /rating (rate a turn on the failure_tags contract) ────────────

            def _handle_rating(self) -> None:
                data = self._parse_json_body()
                if data is None:
                    return
                payload, status = _process_rating(data)
                if payload.get("ok"):
                    server._log(
                        f"[rating] turn={str(data.get('turn_id'))[:8]} "
                        f"score={data.get('rating')} tags={payload.get('failure_tags')} "
                        f"will_evolve={payload.get('will_evolve')}"
                    )
                self._json(payload, status)

            # ── /frame (MonoFrame v2 CorrectionCard; trainer = candidate) ─────

            def _handle_frame(self) -> None:
                data = self._parse_json_body()
                if data is None:
                    return
                # Fill live context the caller can't know: the current bearing
                # frame (bad_frame), the recent asks, and the engine config.
                try:
                    from core.llm_config import load_config
                    if not data.get("base_config"):
                        data["base_config"] = load_config()
                except Exception:
                    pass
                try:
                    from addons.system.bearing import store as _bstore
                    if not data.get("bad_frame"):
                        data["bad_frame"] = _bstore.get_bearing().current_frame or ""
                except Exception:
                    pass
                if not data.get("recent_asks"):
                    try:
                        from addons.system.bearing.stateless_reframe import session_asks as _sa
                        getter = getattr(server, "on_session_messages", None)
                        msgs = getter(12) if callable(getter) else []
                        data["recent_asks"] = _sa(msgs or [])
                    except Exception:
                        data["recent_asks"] = []
                payload, status = _process_frame_correction(data)
                if payload.get("ok"):
                    server._log(
                        f"[frame] source={payload.get('source')} "
                        f"trains={payload.get('trains')}"
                    )
                self._json(payload, status)

            # ── /reset (fresh/cold chat surface, headless "New Chat") ─────────

            def _handle_reset(self) -> None:
                """Start a fresh conversation the way the app's New Chat does.

                No request body is required. When ConnectionsPage has wired
                `on_reset`, it marshals to the Qt thread and invokes the live
                PageChat._start_new_session() (the faithful path). Because that
                hop is fire-and-forget (the reset runs later on the Qt event
                loop), the wired callback returns a *dispatch* receipt, not a
                completion claim.

                When `on_reset` is None (no UI wired — e.g. a headless test or
                a server with no PageChat), fall back to a best-effort headless
                clear: the working-memory slot is module-level and always
                clearable. The engine's conversation_history is owned by the
                UI-side LLMEngine and is NOT reachable from the server thread,
                so a headless reset cannot truncate it — we report that honestly
                rather than claim history_cleared.

                Never 500: all failure paths return {"ok": false, "error": ...}.
                """
                try:
                    # Cold-surface hygiene: also clear the cross-session BEARING.
                    # Unlike the UI's New Chat (which keeps bearing by design),
                    # /reset is the trainer's COLD reset — and a stale [BEARING]
                    # (frozen on a prior session, re-injected every turn by the
                    # bearing_interceptor) is exactly what keeps Monolith warm
                    # ACROSS a conversation reset. clear_bearing() is atomic and
                    # module-level (file-backed, no Qt), safe from the server
                    # thread. Best-effort — never let it block the reset.
                    bearing_cleared = False
                    try:
                        from addons.system.bearing import store as _bearing_store
                        _bearing_store.clear_bearing()
                        bearing_cleared = True
                    except Exception as _bexc:
                        server._log(f"[reset] bearing clear skipped: {_bexc}")

                    cb = server.on_reset
                    if cb is not None:
                        receipt = cb() or {}
                        payload = {"ok": True, "dispatched": True,
                                   "bearing_cleared": bearing_cleared}
                        if isinstance(receipt, dict):
                            payload.update(receipt)
                        server._log("[reset] new chat surface dispatched (Qt thread); "
                                    f"bearing_cleared={bearing_cleared}")
                        self._json(payload)
                        return
                    # Headless fallback — no UI wired.
                    from core.continuity import clear_working_memory
                    clear_working_memory()
                    server._log("[reset] headless working_memory+bearing clear "
                                "(no UI wired; conversation_history not reachable)")
                    self._json({
                        "ok": True,
                        "headless": True,
                        "working_memory_cleared": True,
                        "bearing_cleared": bearing_cleared,
                        "conversation_history_cleared": False,
                        "note": ("no PageChat wired to on_reset; cleared working "
                                 "memory + bearing only. Restart Monolith with the "
                                 "/reset wiring for a full New-Chat reset."),
                    })
                except Exception as exc:
                    server._log(f"[reset] error: {exc}")
                    self._json({"ok": False, "error": _public_error("reset failed", exc)})

            # ── /nudge (ephemeral context injection) ──────────────────────────

            def _handle_nudge(self) -> None:
                data = self._parse_json_body()
                if data is None:
                    return
                text = str(data.get("text") or "").strip()
                if not text:
                    self._json({"ok": False,
                                "error": "'text' is required"}, 400)
                    return
                ttl = data.get("ttl", 1)
                if not isinstance(ttl, int) or ttl < 1:
                    ttl = 1
                result = server.add_nudge(text, ttl)
                server._log(f"[nudge] ttl={ttl}: {text[:60]}")
                self._json(result)

            # ── /join + /leave (participant registry) ─────────────────────────

            def _handle_join(self) -> None:
                data = self._parse_json_body()
                if data is None:
                    return
                name = str(data.get("name") or "").strip()
                if not name:
                    self._json({"ok": False, "error": "'name' is required"}, 400)
                    return
                if len(name) > 32:
                    self._json({"ok": False, "error": "name too long (max 32)"}, 400)
                    return
                url = str(data.get("url") or "").strip().rstrip("/")
                entry = {"name": name, "joined_at": time.time(), "url": url}
                with server._participants_lock:
                    server._participants[name.lower()] = entry
                # Auto-register as a peer so @name routing works immediately
                if url:
                    from engine.external_agents import add_peer as _add_peer
                    _add_peer(name, name, url)
                server._log(f"[join] {name}" + (f" url={url}" if url else ""))
                server._log_canonical("session_open", name, f"url={url}" if url else "")
                server.broadcast_event("participant_join", {"name": name, "url": url})
                if server.on_participant_change:
                    server.on_participant_change()
                self._json({
                    "ok": True,
                    "name": name,
                    "commands": [t["name"] for t in _MCP_TOOLS],
                    "hint": (
                        f"You are now connected as '{name}'. "
                        f"Use send_message to talk to Monolith, "
                        f"read_history to catch up, who to see participants."
                        + (f" Your /chat endpoint is registered — others can @{name.lower()} you." if url else "")
                    ),
                })

            def _handle_leave(self) -> None:
                data = self._parse_json_body()
                if data is None:
                    return
                name = str(data.get("name") or "").strip()
                with server._participants_lock:
                    entry = server._participants.pop(name.lower(), None)
                # Remove auto-registered peer if it was session-only (joined with url)
                if entry and entry.get("url"):
                    from engine.external_agents import remove_peer as _remove_peer
                    _remove_peer(name)
                server._log(f"[leave] {name}")
                server._log_canonical("session_close", name, "")
                server.broadcast_event("participant_leave", {"name": name})
                if server.on_participant_change:
                    server.on_participant_change()
                self._json({"ok": True, "name": name})

            # ── Debug-read handlers (added 2026-05-11) ───────────────────

            def _query_params(self) -> dict:
                try:
                    parsed = urllib.parse.urlparse(self.path)
                    return {k: v[-1] for k, v in urllib.parse.parse_qs(parsed.query).items()}
                except Exception:
                    return {}

            def _handle_interceptors_state(self) -> None:
                cb = server.on_interceptor_state
                if cb is None:
                    self._json({"ok": False, "error": "no interceptor_state callback wired"}, 503)
                    return
                try:
                    payload = cb() or {}
                except Exception as exc:
                    server._log(f"interceptor_state failed: {exc!r}")
                    self._json(
                        {"ok": False, "error": _public_error("interceptor_state failed", exc)},
                        500,
                    )
                    return
                self._json({"ok": True, **payload})

            def _handle_memory_recall(self) -> None:
                params = self._query_params()
                query = str(params.get("q") or params.get("query") or "").strip()
                if not query:
                    self._json({"ok": False, "error": "missing 'q'"}, 400)
                    return
                try:
                    limit = max(1, min(int(params.get("limit", 10) or 10), 50))
                except (TypeError, ValueError):
                    limit = 10
                try:
                    from core.acu_store import ACUStore
                    store = ACUStore()
                    results = store.search(query, max_results=limit)
                except Exception as exc:
                    server._log(f"recall failed: {exc!r}")
                    self._json(
                        {"ok": False, "error": _public_error("recall failed", exc)}, 500
                    )
                    return
                self._json({"ok": True, "query": query, "count": len(results), "results": results})

            def _handle_memory_continuity(self) -> None:
                params = self._query_params()
                include_retired = str(params.get("include_retired", "")).lower() in {"1", "true", "yes"}
                try:
                    from core.continuity import read as continuity_read
                    snap = continuity_read(include_retired=include_retired, retired_limit=5)
                except Exception as exc:
                    server._log(f"continuity read failed: {exc!r}")
                    self._json(
                        {"ok": False, "error": _public_error("continuity read failed", exc)},
                        500,
                    )
                    return
                self._json({"ok": True, **snap})

            def _handle_trace_recent(self) -> None:
                params = self._query_params()
                try:
                    limit = max(1, min(int(params.get("limit", 20) or 20), 200))
                except (TypeError, ValueError):
                    limit = 20
                try:
                    from core import turn_trace as _tt
                    rows = _tt.list_recent_turns(limit=limit)
                except Exception as exc:
                    server._log(f"trace list failed: {exc!r}")
                    self._json(
                        {"ok": False, "error": _public_error("trace list failed", exc)}, 500
                    )
                    return
                payload = [_summary_to_dict(r) for r in rows]
                self._json({"ok": True, "count": len(payload), "rows": payload})

            def _handle_trace_errors(self) -> None:
                params = self._query_params()
                try:
                    limit = max(1, min(int(params.get("limit", 20) or 20), 200))
                except (TypeError, ValueError):
                    limit = 20
                try:
                    from core import turn_trace as _tt
                    rows = _tt.search_turns(has_errored_stage=True, limit=limit)
                except Exception as exc:
                    server._log(f"trace search failed: {exc!r}")
                    self._json(
                        {"ok": False, "error": _public_error("trace search failed", exc)},
                        500,
                    )
                    return
                payload = [_summary_to_dict(r) for r in rows]
                self._json({"ok": True, "count": len(payload), "rows": payload})

            def _handle_trace_one(self, turn_id: str) -> None:
                try:
                    from core import turn_trace as _tt
                    joined = _tt.get_turn_trace(turn_id)
                except Exception as exc:
                    server._log(f"trace fetch failed: {exc!r}")
                    self._json(
                        {"ok": False, "error": _public_error("trace fetch failed", exc)}, 500
                    )
                    return
                if joined is None:
                    self._json({"ok": False, "error": f"turn_id '{turn_id}' not found"}, 404)
                    return
                self._json({"ok": True, **_joined_to_dict(joined)})

            def _handle_trace_prompt(self, turn_id: str) -> None:
                """Return final_messages — the actual prompt the LLM saw."""
                try:
                    from core import turn_trace as _tt
                    joined = _tt.get_turn_trace(turn_id)
                except Exception as exc:
                    server._log(f"trace fetch failed: {exc!r}")
                    self._json(
                        {"ok": False, "error": _public_error("trace fetch failed", exc)}, 500
                    )
                    return
                if joined is None or joined.frame is None:
                    self._json({"ok": False, "error": f"frame for '{turn_id}' not found"}, 404)
                    return
                msgs = [
                    {
                        "role": m.role,
                        "ephemeral": m.ephemeral,
                        "source": m.source,
                        "chars": m.content_chars,
                        "preview": m.content_preview,
                        "content_hash": m.content_hash,
                    }
                    for m in joined.frame.final_messages
                ]
                self._json({
                    "ok": True,
                    "turn_id": turn_id,
                    "backend": joined.frame.backend,
                    "system_prompt_chars": joined.frame.system_prompt_chars,
                    "user_prompt_chars": joined.frame.user_prompt_chars,
                    "total_chars": joined.frame.total_chars,
                    "effort_tier": joined.frame.effort_tier,
                    "messages": msgs,
                })

            def _handle_session_messages(self) -> None:
                params = self._query_params()
                try:
                    n = max(1, min(int(params.get("n", 20) or 20), 200))
                except (TypeError, ValueError):
                    n = 20
                raw = str(params.get("raw", "false")).lower() in {"1", "true", "yes"}
                cb = server.on_session_messages
                if cb is None:
                    self._json({"ok": False, "error": "no session_messages callback wired"}, 503)
                    return
                try:
                    messages = list(cb(n) or [])
                except Exception as exc:
                    server._log(f"session_messages failed: {exc!r}")
                    self._json(
                        {"ok": False, "error": _public_error("session_messages failed", exc)},
                        500,
                    )
                    return
                if not raw:
                    messages = [_strip_message(m) for m in messages]
                self._json({"ok": True, "count": len(messages), "raw": raw, "messages": messages})

            def _handle_session_message_at(self, index: int) -> None:
                params = self._query_params()
                raw = str(params.get("raw", "false")).lower() in {"1", "true", "yes"}
                cb = server.on_session_messages
                if cb is None:
                    self._json({"ok": False, "error": "no session_messages callback wired"}, 503)
                    return
                try:
                    messages = list(cb(200) or [])
                except Exception as exc:
                    server._log(f"session_messages failed: {exc!r}")
                    self._json(
                        {"ok": False, "error": _public_error("session_messages failed", exc)},
                        500,
                    )
                    return
                if not (0 <= index < len(messages)):
                    self._json(
                        {"ok": False, "error": f"index {index} out of range (0..{max(0, len(messages) - 1)})"},
                        404,
                    )
                    return
                msg = messages[index]
                if not raw:
                    msg = _strip_message(msg)
                self._json({"ok": True, "index": index, "raw": raw, "message": msg})

            def _handle_log_tail(self) -> None:
                params = self._query_params()
                try:
                    n = max(1, min(int(params.get("n", 200) or 200), 1000))
                except (TypeError, ValueError):
                    n = 200
                from core.log_mirror import get_ring
                ring = get_ring()
                lines = ring.tail(n)
                self._json({
                    "ok": True,
                    "count": len(lines),
                    "head_seq": ring.head_seq(),
                    "latest_seq": ring.latest_seq(),
                    "capacity": ring.capacity,
                    "lines": lines,
                })

            def _handle_log_since(self) -> None:
                params = self._query_params()
                try:
                    seq = int(params.get("seq", 0) or 0)
                except (TypeError, ValueError):
                    seq = 0
                from core.log_mirror import get_ring
                ring = get_ring()
                lines = ring.since(seq)
                head = ring.head_seq()
                gap = bool(lines) and lines[0]["seq"] > max(seq + 1, head)
                self._json({
                    "ok": True,
                    "count": len(lines),
                    "head_seq": head,
                    "latest_seq": ring.latest_seq(),
                    "since_seq": seq,
                    "gap": gap,
                    "lines": lines,
                })

            def _handle_db_query_get(self) -> None:
                params = self._query_params()
                self._dispatch_db_query(
                    store=params.get("store") or "turn_trace",
                    table=params.get("table") or "",
                    sql="",
                    limit=params.get("limit") or 20,
                )

            def _handle_db_query_post(self) -> None:
                data = self._parse_json_body()
                if data is None:
                    return
                self._dispatch_db_query(
                    store=str(data.get("store") or "turn_trace"),
                    table=str(data.get("table") or ""),
                    sql=str(data.get("sql") or ""),
                    limit=data.get("limit", 20),
                )

            def _dispatch_db_query(self, *, store: str, table: str, sql: str, limit) -> None:
                store_lc = str(store or "").strip().lower()
                allowed = _DB_QUERY_STORES.get(store_lc)
                if allowed is None:
                    self._json(
                        {"ok": False, "error": f"unknown store '{store}'",
                         "allowed": sorted(_DB_QUERY_STORES)},
                        400,
                    )
                    return
                db_path = _db_path_for_store(store_lc)
                if db_path is None:
                    self._json({"ok": False, "error": f"store '{store}' has no path"}, 500)
                    return
                result = _run_db_query(db_path, allowed, table=table, sql=sql, limit=limit)
                status = 200 if result.get("ok") else 400
                self._json(result, status)

        return _Handler

    # ── MCP stdio ----------------------------------------------------------

    def run_stdio(self) -> None:
        """Run as an MCP stdio server (blocking). Spawn in a thread or separate process."""

        def send(msg: dict) -> None:
            line = json.dumps(msg, ensure_ascii=False) + "\n"
            sys.stdout.buffer.write(line.encode("utf-8"))
            sys.stdout.buffer.flush()

        buf = b""
        stdin = sys.stdin.buffer
        while True:
            chunk = stdin.read(1)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._dispatch_mcp(msg, send)

    def _dispatch_mcp(self, msg: dict, send: Callable) -> None:
        method = msg.get("method", "")
        msg_id = msg.get("id")
        params = msg.get("params") or {}

        if msg_id is None:
            return  # notification — no response

        if method == "initialize":
            send({
                "jsonrpc": "2.0", "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "monolith", "version": APP_VERSION},
                },
            })

        elif method == "tools/list":
            send({
                "jsonrpc": "2.0", "id": msg_id,
                "result": {"tools": _MCP_TOOLS},
            })

        elif method == "tools/call":
            name = params.get("name", "")
            args = params.get("arguments") or {}

            if name == "send_message":
                message = str(args.get("message") or "").strip()
                agent_name = str(args.get("agent") or "Agent").strip()
                if not message:
                    send({"jsonrpc": "2.0", "id": msg_id, "result": {
                        "content": [{"type": "text", "text": "Error: 'message' is required"}],
                        "isError": True,
                    }})
                    return
                # Per-call plane modes: MCP send_message uses args (not data)
                # and the MCP schema doesn't expose per-plane _once fields, so
                # default_once=True is hardcoded — every applied plane is a
                # one-turn override. _parse_per_call_modes works with args
                # passed as `data` since both are plain dicts.
                applied = _parse_per_call_modes(self, args, default_once=True)
                ok, response = self._handle_chat(
                    agent_name, message, transport="mcp send_message"
                )
                send({"jsonrpc": "2.0", "id": msg_id, "result": {
                    "content": [{"type": "text", "text": response if ok else f"Error: {response}"}],
                    "isError": not ok,
                }})

            elif name == "join":
                agent_name = str(args.get("name") or "Agent").strip()
                url = str(args.get("url") or "").strip().rstrip("/")
                entry = {"name": agent_name, "joined_at": time.time(), "url": url}
                with self._participants_lock:
                    self._participants[agent_name.lower()] = entry
                if url:
                    from engine.external_agents import add_peer as _add_peer
                    _add_peer(agent_name, agent_name, url)
                self._log(f"[join via mcp] {agent_name}" + (f" url={url}" if url else ""))
                self._log_canonical("session_open", agent_name, f"url={url}" if url else "")
                self.broadcast_event("participant_join", {"name": agent_name, "url": url})
                if self.on_participant_change:
                    self.on_participant_change()
                send({"jsonrpc": "2.0", "id": msg_id, "result": {
                    "content": [{"type": "text", "text": (
                        f"Joined as '{agent_name}'. "
                        "Commands: send_message, read_history, who, leave."
                        + (f" Registered as peer @{agent_name.lower()} at {url}." if url else "")
                    )}],
                }})

            elif name == "read_history":
                limit = min(int(args.get("limit") or 20), 50)
                snap = self.on_state_request() if self.on_state_request else {}
                recent = snap.get("recent_messages", [])[-limit:]
                lines = [f"[{m.get('role','?')}] {m.get('text','')[:200]}" for m in recent]
                send({"jsonrpc": "2.0", "id": msg_id, "result": {
                    "content": [{"type": "text", "text": "\n".join(lines) or "(no history)"}],
                }})

            elif name == "who":
                with self._participants_lock:
                    names = [p["name"] for p in self._participants.values()]
                text = "Connected: " + ", ".join(names) if names else "No agents connected."
                send({"jsonrpc": "2.0", "id": msg_id, "result": {
                    "content": [{"type": "text", "text": text}],
                }})

            elif name == "leave":
                agent_name = str(args.get("name") or "").strip()
                with self._participants_lock:
                    entry = self._participants.pop(agent_name.lower(), None)
                if entry and entry.get("url"):
                    from engine.external_agents import remove_peer as _remove_peer
                    _remove_peer(agent_name)
                self._log_canonical("session_close", agent_name, "")
                self.broadcast_event("participant_leave", {"name": agent_name})
                if self.on_participant_change:
                    self.on_participant_change()
                send({"jsonrpc": "2.0", "id": msg_id, "result": {
                    "content": [{"type": "text", "text": f"'{agent_name}' left."}],
                }})

            elif name == "set_prompts":
                prompts_raw = args.get("prompts") or []
                once = bool(args.get("once", True))
                if not prompts_raw or prompts_raw == []:
                    ws = getattr(self, "_world_state", None)
                    if ws:
                        ws.clear_prompts()
                    text = "[set_prompts: cleared]"
                else:
                    from core.prompt_library import valid_prompts
                    valid = valid_prompts()
                    names = [str(p).strip().lower() for p in prompts_raw if isinstance(p, str)]
                    bad = [n for n in names if n not in valid]
                    if bad:
                        text = f"Error: unknown prompts: {', '.join(bad)}. Available: {', '.join(sorted(valid))}"
                        send({"jsonrpc": "2.0", "id": msg_id, "result": {
                            "content": [{"type": "text", "text": text}],
                            "isError": True,
                        }})
                        return
                    ws = getattr(self, "_world_state", None)
                    if ws:
                        if once:
                            ws.set_prompts_once(names)
                        else:
                            ws.set_active_prompts(names)
                    scope = "once" if once else "baseline"
                    text = f"[set_prompts: {'+'.join(names)} scope={scope}]"

            elif name == "set_effort":
                _EFFORT_MAP = {"low": "direct", "med": "grounded", "high": "orient", "xhigh": "falsify", "ultimate": "orchestrate"}
                tier_raw = str(args.get("tier") or "").strip().lower()
                once = bool(args.get("once", True))
                mapped = _EFFORT_MAP.get(tier_raw)
                if not tier_raw or tier_raw == "off":
                    ws = getattr(self, "_world_state", None)
                    if ws:
                        ws.clear_prompts()
                    text = "[set_effort: cleared (deprecated — use set_prompts)]"
                elif mapped:
                    ws = getattr(self, "_world_state", None)
                    if ws:
                        if once:
                            ws.set_prompts_once([mapped])
                        else:
                            ws.set_active_prompts([mapped])
                    text = f"[set_effort: {tier_raw}→{mapped} (deprecated — use set_prompts)]"
                else:
                    text = f"Error: tier '{tier_raw}' not valid. Deprecated — use set_prompts with: {', '.join(sorted(_EFFORT_MAP.values()))}"
                    send({"jsonrpc": "2.0", "id": msg_id, "result": {
                        "content": [{"type": "text", "text": text}],
                        "isError": True,
                    }})
                    return
                send({"jsonrpc": "2.0", "id": msg_id, "result": {
                    "content": [{"type": "text", "text": text}],
                }})

            elif name == "inspect_trace":
                verb = str(args.get("verb") or "").strip().lower()
                if not verb:
                    send({"jsonrpc": "2.0", "id": msg_id, "result": {
                        "content": [{"type": "text", "text": "Error: 'verb' is required (recent, errors, or one)"}],
                        "isError": True,
                    }})
                    return
                try:
                    raw_limit = args.get("limit")
                    try:
                        limit_int = int(raw_limit) if raw_limit is not None else 5
                    except (TypeError, ValueError):
                        limit_int = 5
                    limit_int = max(1, min(limit_int, 50))
                    from core import turn_trace as _tt
                    if verb == "recent":
                        rows = _tt.list_recent_turns(limit=limit_int)
                        payload = [
                            {
                                "turn_id": r.turn_id,
                                "parent_turn_id": r.parent_turn_id,
                                "captured_at": r.captured_at,
                                "backend": r.backend,
                                "stage_count": r.stage_count,
                                "errored_stage_count": r.errored_stage_count,
                                "total_chars": r.total_chars,
                                "frame_present": r.frame_present,
                            }
                            for r in rows
                        ]
                        text = f"[inspect_trace:recent count={len(payload)}]\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
                    elif verb == "errors":
                        rows = _tt.search_turns(has_errored_stage=True, limit=limit_int)
                        payload = [
                            {
                                "turn_id": r.turn_id,
                                "parent_turn_id": r.parent_turn_id,
                                "captured_at": r.captured_at,
                                "errored_stage_count": r.errored_stage_count,
                                "stage_count": r.stage_count,
                                "backend": r.backend,
                                "frame_present": r.frame_present,
                            }
                            for r in rows
                        ]
                        text = f"[inspect_trace:errors count={len(payload)}]\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
                    elif verb == "one":
                        turn_id = str(args.get("turn_id") or "").strip()
                        if not turn_id:
                            text = "Error: verb=one requires 'turn_id'"
                            send({"jsonrpc": "2.0", "id": msg_id, "result": {
                                "content": [{"type": "text", "text": text}],
                                "isError": True,
                            }})
                            return
                        joined = _tt.get_turn_trace(turn_id)
                        if joined is None:
                            text = f"[inspect_trace:one] no turn with id={turn_id}"
                        else:
                            stages = [
                                {
                                    "seq": s.seq,
                                    "stage_name": s.stage_name,
                                    "outcome": s.outcome,
                                    "outcome_reason": s.outcome_reason,
                                    "items_added": [it.to_dict() for it in s.items_added],
                                }
                                for s in joined.stages
                            ]
                            frame_payload: dict | None = None
                            if joined.frame is not None:
                                frame_payload = {
                                    "captured_at": joined.frame.captured_at,
                                    "backend": joined.frame.backend,
                                    "effort_tier": joined.frame.effort_tier,
                                    "system_prompt_chars": joined.frame.system_prompt_chars,
                                    "user_prompt_chars": joined.frame.user_prompt_chars,
                                    "total_chars": joined.frame.total_chars,
                                }
                            payload = {
                                "turn_id": joined.turn_id,
                                "parent_turn_id": joined.parent_turn_id,
                                "stages": stages,
                                "frame": frame_payload,
                                "summary": joined.summary,
                            }
                            text = f"[inspect_trace:one turn_id={turn_id}]\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
                    else:
                        text = f"Error: unknown verb '{verb}'. Use one of: recent, errors, one"
                        send({"jsonrpc": "2.0", "id": msg_id, "result": {
                            "content": [{"type": "text", "text": text}],
                            "isError": True,
                        }})
                        return
                except Exception as exc:
                    self._log(f"inspect_trace failed: {exc!r}")
                    text = "Error: " + _public_error("inspect_trace failed", exc)
                    send({"jsonrpc": "2.0", "id": msg_id, "result": {
                        "content": [{"type": "text", "text": text}],
                        "isError": True,
                    }})
                    return
                send({"jsonrpc": "2.0", "id": msg_id, "result": {
                    "content": [{"type": "text", "text": text}],
                }})

            else:
                send({"jsonrpc": "2.0", "id": msg_id,
                      "error": {"code": -32601, "message": f"Unknown tool: {name}"}})

        else:
            send({
                "jsonrpc": "2.0", "id": msg_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            })


# MCP tools exposed to all connected agents
# ── module-level server reference (set by ConnectionsPage) ────────────────────

_active_server: "AgentServer | None" = None


def get_server() -> "AgentServer | None":
    """Return the running AgentServer instance, or None if CONNECT is not active."""
    return _active_server


def _set_active_server(server: "AgentServer | None") -> None:
    """Called by ConnectionsPage when the server starts/stops."""
    global _active_server
    _active_server = server


# ── MCP tool definitions ───────────────────────────────────────────────────────

_MCP_TOOLS = [
    {
        "name": "join",
        "description": "Join the Monolith session with a display name. Call this first. Optionally provide your /chat URL to be @mention-able by others.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Your display name (e.g. 'Gemini', 'Codex')"},
                "url":  {"type": "string", "description": "Your /chat endpoint URL (e.g. 'http://localhost:8300'). If provided, others can @mention you."},
            },
            "required": ["name"],
        },
    },
    {
        "name": "send_message",
        "description": "Send a message to the active Monolith chat. The local LLM responds. Use 'prompts' to set composable scaffolds (e.g. [\"falsify\", \"descent\"]). Legacy per-plane fields (effort/conversation/reasoning/linguency) still work as deprecated aliases.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "message":      {"type": "string", "description": "The message to send"},
                "agent":        {"type": "string", "description": "Your display name"},
                "prompts":      {"type": "array", "items": {"type": "string"}, "description": "Prompt scaffolds for this turn (composable, ordered). e.g. [\"falsify\", \"descent\"]. Available: direct, grounded, orient, falsify, orchestrate, explore, descent, scorecard, think_test"},
                "monothink":    {"type": "boolean", "description": "Enable monothink (self-evolving reasoning scaffold) for this turn."},
                "effort":       {"type": "string", "description": "[Deprecated → prompts] Effort tier — off, low, med, high, xhigh, ultimate."},
                "conversation": {"type": "string", "description": "[Deprecated → prompts] Conversation mode — off, default, experimental."},
                "reasoning":    {"type": "string", "description": "[Deprecated → /monothink] Reasoning mode — off, monothink."},
                "linguency":    {"type": "string", "description": "[Deprecated → prompts] Linguency mode — off, monolith, audit-scorecard."},
            },
            "required": ["message"],
        },
    },
    {
        "name": "read_history",
        "description": "Read recent messages from the active chat session.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Number of messages to return (max 50, default 20)"},
            },
        },
    },
    {
        "name": "who",
        "description": "List all agents currently connected to this Monolith session.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "leave",
        "description": "Leave the Monolith session.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Your display name"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "set_prompts",
        "description": "Set prompt scaffolds for the NEXT send_message call. Composable: multiple prompts fire together, first-listed = highest priority.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompts": {"type": "array", "items": {"type": "string"}, "description": "Prompt names. Available: direct, grounded, orient, falsify, orchestrate, explore, descent, scorecard, think_test. Pass empty array to clear."},
                "once": {"type": "boolean", "description": "If true (default), applies for one turn only. If false, sets the persistent baseline."},
            },
            "required": ["prompts"],
        },
    },
    {
        "name": "set_effort",
        "description": "[Deprecated → set_prompts] Set the depth-tier scaffold. Maps: low→direct, med→grounded, high→orient, xhigh→falsify, ultimate→orchestrate.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tier": {"type": "string", "description": "Effort tier — off, low, med, high, xhigh, or ultimate."},
                "once": {"type": "boolean", "description": "If true (default), applies for one turn only."},
            },
            "required": ["tier"],
        },
    },
    {
        "name": "inspect_trace",
        "description": "Read the turn-trace store — what each interceptor ran on a turn, and (when Layer B is writing) the final assembled prompt the model saw. Verbs: `recent` returns the last N turns; `errors` returns recent turns with at least one errored stage; `one` returns the full joined record for a specific turn_id.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "verb":    {"type": "string", "description": "One of: recent, errors, one"},
                "limit":   {"type": "integer", "description": "Number of turns to return (max 50, default 5). Ignored for verb=one."},
                "turn_id": {"type": "string", "description": "Required for verb=one. The turn UUID to fetch."},
            },
            "required": ["verb"],
        },
    },
]

# Keep for backward compat
_SEND_MESSAGE_TOOL = _MCP_TOOLS[1]


# -- standalone entry point -------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Monolith Agent Server — standalone mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Note: standalone mode can start the server but chat requests require
Monolith to be running with the Connections tab open and wired.

examples:
  Claude Code (auto-spawned via .mcp.json):
    { "command": "python", "args": ["engine/agent_server.py", "--stdio"] }

  Kimi / HTTP agents:
    python engine/agent_server.py --http --port 7821
    POST http://localhost:7821/chat  {"message": "hello", "agent": "Kimi"}
""",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--stdio", action="store_true",
                       help="MCP stdio transport (Claude Code, Cursor)")
    group.add_argument("--http", action="store_true",
                       help="HTTP REST transport (Kimi, custom agents)")
    parser.add_argument("--port", type=int, default=7821,
                        help="HTTP listen port (default: 7821)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="HTTP listen host (default: 127.0.0.1)")
    args = parser.parse_args()

    server = AgentServer()
    server.on_log = lambda msg: print(f"[monolith] {msg}", flush=True)

    if args.stdio:
        server.run_stdio()
    else:
        server.start(args.port, host=args.host)
        base = f"http://{args.host}:{args.port}"
        print(f"[monolith] Endpoints:", flush=True)
        print(f"[monolith]   POST {base}/chat          — blocking chat", flush=True)
        print(f"[monolith]   POST {base}/chat/stream   — SSE streaming chat", flush=True)
        print(f"[monolith]   GET  {base}/state         — observe model state", flush=True)
        print(f"[monolith]   POST {base}/hooks         — register webhook", flush=True)
        print(f"[monolith]   POST {base}/nudge         — inject ephemeral context", flush=True)
        print(f"[monolith]   GET  {base}/health        — health check", flush=True)
        try:
            input("[monolith] Press Enter to stop...\n")
        except KeyboardInterrupt:
            pass
        server.stop()


if __name__ == "__main__":
    main()
