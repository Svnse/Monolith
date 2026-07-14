"""Session-querying executor — reads canonical_log for current-session events.

Backed by core/acatalepsy/canonical_log.py — the audit floor that captures
every user_message and assistant_message dispatch keyed by session_id.
"""
import json
from datetime import datetime, timezone


def _coerce_int(value, default, lo, hi):
    try:
        v = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


def _connect_reader():
    from core.db_connect import connect_acatalepsy
    return connect_acatalepsy(role="reader")


def _current_session_id():
    # Most-recent event with non-null session_id is the active session.
    # session_open isn't reliably emitted across all chat surfaces (CONNECT
    # peers write user_message / assistant_message directly without one).
    conn = _connect_reader()
    try:
        cur = conn.execute(
            "SELECT session_id FROM canonical_log "
            "WHERE session_id IS NOT NULL "
            "ORDER BY event_id DESC LIMIT 1"
        )
        row = cur.fetchone()
        return str(row[0]) if row and row[0] else None
    finally:
        conn.close()


def _format_event(event_id, ts, kind, payload_json) -> str:
    when = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M:%S")
    body = ""
    if payload_json:
        try:
            payload = json.loads(payload_json)
            if isinstance(payload, dict):
                text = payload.get("text") or payload.get("content") or payload.get("message")
                if text:
                    body = str(text)[:200].replace("\n", " ")
                else:
                    body = json.dumps(payload, ensure_ascii=False)[:200]
        except Exception:
            body = str(payload_json)[:200]
    return f"#{event_id} [{when}] {kind}: {body}".rstrip()


def run(cmd: dict, _ctx) -> str:
    verb = str(cmd.get("verb", "")).strip().lower()
    if not verb:
        return "[session: no verb provided - expected state/recent/search]"

    try:
        session_id = _current_session_id()
    except Exception as exc:
        return f"[session: cannot read canonical_log - {exc}]"

    if not session_id:
        return "[session: no active session in canonical_log]"

    if verb == "state":
        conn = _connect_reader()
        try:
            cur = conn.execute(
                "SELECT MIN(ts), COUNT(*) FROM canonical_log WHERE session_id = ?",
                (session_id,),
            )
            row = cur.fetchone()
            started_at = row[0] if row else None
            count = int(row[1]) if row else 0
            cur = conn.execute(
                "SELECT kind, COUNT(*) FROM canonical_log WHERE session_id = ? GROUP BY kind ORDER BY 2 DESC",
                (session_id,),
            )
            kind_counts = [(r[0], int(r[1])) for r in cur.fetchall()]
        finally:
            conn.close()

        when = (
            datetime.fromtimestamp(started_at, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            if started_at else "unknown"
        )
        kind_summary = ", ".join(f"{k}={v}" for k, v in kind_counts) or "none"
        return (
            f"[session state]\n"
            f"session_id: {session_id}\n"
            f"started_at: {when}\n"
            f"events: {count}\n"
            f"kinds: {kind_summary}"
        )

    elif verb == "recent":
        limit = _coerce_int(cmd.get("limit", 10), 10, 1, 100)
        conn = _connect_reader()
        try:
            cur = conn.execute(
                "SELECT event_id, ts, kind, payload FROM canonical_log "
                "WHERE session_id = ? ORDER BY event_id DESC LIMIT ?",
                (session_id, limit),
            )
            rows = cur.fetchall()
        finally:
            conn.close()
        if not rows:
            return f"[session recent: no events for session {session_id}]"
        rows = list(reversed(rows))
        lines = [f"[session recent: {len(rows)} event(s), session {session_id}]"]
        for row in rows:
            lines.append(_format_event(row[0], row[1], row[2], row[3]))
        return "\n".join(lines)

    elif verb == "search":
        pattern = str(cmd.get("pattern", "")).strip()
        if not pattern:
            return "[session search: no pattern provided]"
        limit = _coerce_int(cmd.get("limit", 20), 20, 1, 100)
        conn = _connect_reader()
        try:
            cur = conn.execute(
                "SELECT event_id, ts, kind, payload FROM canonical_log "
                "WHERE session_id = ? AND payload LIKE ? "
                "ORDER BY event_id ASC LIMIT ?",
                (session_id, f"%{pattern}%", limit),
            )
            rows = cur.fetchall()
        finally:
            conn.close()
        if not rows:
            return f"[session search: no matches for '{pattern}' in session {session_id}]"
        lines = [f"[session search: {len(rows)} match(es) for '{pattern}']"]
        for row in rows:
            lines.append(_format_event(row[0], row[1], row[2], row[3]))
        return "\n".join(lines)

    else:
        return f"[session: unknown verb '{verb}' - expected state/recent/search]"
