"""Lag-bug verification: does axis_interceptor fire gates that don't match the current user turn?

Reads $APPDATA/Monolith/logs/turn_trace.sqlite3. For each turn, prints the
latest non-ephemeral user message preview and the [GATE] names that fired
in that turn's axis_interceptor stage. Then flags suspected misfires.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from pathlib import Path

DB_PATH = Path(os.environ["APPDATA"]) / "Monolith" / "logs" / "turn_trace.sqlite3"


def latest_user(final_messages_json: str) -> str:
    msgs = json.loads(final_messages_json or "[]")
    users = [m for m in msgs if m.get("role") == "user" and not m.get("ephemeral")]
    if not users:
        return ""
    return (users[-1].get("content_preview") or "").replace("\n", " ")


def gates_fired(conn: sqlite3.Connection, turn_id: str) -> list[str]:
    rows = list(conn.execute(
        """
        SELECT items_added_json FROM stage_traces
        WHERE turn_id = ? AND stage_name = 'axis_interceptor' AND outcome = 'ran'
        """,
        (turn_id,),
    ))
    if not rows:
        return []
    items = json.loads(rows[0]["items_added_json"] or "[]")
    found: list[str] = []
    for it in items:
        preview = it.get("content_preview") or ""
        for m in re.finditer(r"\[([A-Z][A-Z\s\-]+[A-Z])\]", preview):
            name = m.group(1).strip()
            if name.startswith("AXIS"):
                continue
            found.append(name)
    return found


def main() -> int:
    if not DB_PATH.exists():
        print(f"NO DB at {DB_PATH}")
        return 1
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    frames = list(conn.execute(
        "SELECT turn_id, captured_at, final_messages_json FROM frame_traces ORDER BY captured_at"
    ))
    print(f"Total frames: {len(frames)}\n")
    print(f"{'#':>4} | {'CAPTURED':<28} | {'USER':<60} | GATES")
    print("-" * 145)

    misfires: list[tuple[int, str, list[str], str]] = []
    for i, f in enumerate(frames):
        user = latest_user(f["final_messages_json"])
        gates = gates_fired(conn, f["turn_id"])
        u_short = user[:58]
        gates_str = ", ".join(gates) if gates else "-"
        print(f"{i:>4} | {f['captured_at']:<28} | {u_short:<60} | {gates_str}")

        u_lower = user.lower().strip()
        is_greeting = len(user) < 30 and any(
            u_lower.startswith(g) for g in ("hey", "hi", "hello", "yo ")
        )
        if is_greeting and gates:
            misfires.append((i, u_short, gates, "trivial greeting with gates firing"))
        if "plan" not in u_lower and "PLANNING POSTURE" in gates:
            misfires.append((i, u_short, gates, "PLANNING POSTURE without 'plan' in user msg"))
        if "SELF-CRITICISM PASS" in gates and is_greeting:
            misfires.append((i, u_short, gates, "SELF-CRITICISM PASS on a greeting"))

    print(f"\nSuspected lag/misfire events: {len(misfires)}")
    for idx, msg, gates, why in misfires:
        print(f"  turn #{idx}: {why}")
        print(f"    user : {msg}")
        print(f"    gates: {gates}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
