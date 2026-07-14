"""Frame-drift extractor (read-only falsifier, step 1 of the world-model build).

Pulls, per turn, the injected bearing `current_frame` and that turn's actual
user ask out of frame_traces, so we can judge whether the frame the model was
handed still matched reality. This is the cross-turn frame *maintenance* axis
BRANCH never measured (BRANCH measured per-task frame *selection*).

Read-only (mode=ro). Never writes. Dumps tuples to drift_pairs.json + prints
feasibility stats (does current_frame survive the 500-char preview?).
"""
from __future__ import annotations

import json
import os
import re
import sqlite3

_DB = os.path.join(os.environ["APPDATA"], "Monolith", "logs", "turn_trace.sqlite3")
_OUT = os.path.join(os.path.dirname(__file__), "drift_pairs.json")

_FRAME_RE = re.compile(r"current_frame:\s*(.+)")


def _preview(m: dict) -> str:
    # FrameMessage.to_dict — tolerate key drift across versions.
    for k in ("content_preview", "content", "preview", "text"):
        v = m.get(k)
        if isinstance(v, str) and v:
            return v
    return ""


def _extract_current_frame(msgs: list) -> str | None:
    for m in msgs:
        prev = _preview(m)
        if "current_frame:" in prev:
            mt = _FRAME_RE.search(prev)
            if mt:
                # take just the frame line (stop at next block line)
                return mt.group(1).split("\n")[0].strip()
    return None


_NON_ASK_PREFIXES = (
    "Tool results", "[TOOL", "[SUBAGENT_RESULT", "[BEARING", "[RUNTIME STATE",
    "[OBSERVER", "[SELF-CHECK", "[LAST TURN", "[REVIEW QUEUE",
)


def _is_real_user_ask(m: dict) -> bool:
    if m.get("role") != "user":
        return False
    if m.get("ephemeral"):
        return False
    src = str(m.get("source") or "")
    if src and src not in ("user", "chat", ""):
        return False
    prev = _preview(m).strip()
    if not prev:
        return False
    head = prev.lstrip("[ ")[:30]
    if any(prev.lstrip().startswith(p) for p in _NON_ASK_PREFIXES):
        return False
    return True


def _last_user_ask(msgs: list) -> str | None:
    # last *real* user message this turn (skip ephemeral blocks + tool results).
    for m in reversed(msgs):
        if _is_real_user_ask(m):
            prev = _preview(m).strip()
            # strip a leading [CHANNEL: ...] tag so the ask reads cleanly
            prev = re.sub(r"^\[CHANNEL:[^\]]*\]\s*", "", prev)
            return prev
    return None


def main() -> None:
    con = sqlite3.connect(f"file:{_DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    total = cur.execute("SELECT COUNT(*) FROM frame_traces").fetchone()[0]
    cols = [r[1] for r in cur.execute("PRAGMA table_info(frame_traces)").fetchall()]
    print(f"frame_traces rows: {total}")
    print(f"cols: {cols}")

    # Outer MAIN-model turns only: real chat backends, no tool-loop followups.
    rows = cur.execute(
        "SELECT turn_id, parent_turn_id, captured_at, backend, final_messages_json "
        "FROM frame_traces "
        "WHERE backend IN ('cloud','openai') "
        "  AND (parent_turn_id IS NULL OR parent_turn_id = '') "
        "ORDER BY captured_at ASC"
    ).fetchall()
    print(f"outer cloud/openai frames: {len(rows)}")

    pairs = []
    prev_frame = None
    run_len = 0
    for r in rows:
        try:
            msgs = json.loads(r["final_messages_json"] or "[]")
        except Exception:
            continue
        frame = _extract_current_frame(msgs)
        ask = _last_user_ask(msgs)
        if not (frame and ask):
            continue
        run_len = run_len + 1 if frame == prev_frame else 1
        prev_frame = frame
        pairs.append({
            "seq": len(pairs),
            "turn_id": r["turn_id"],
            "captured_at": r["captured_at"],
            "frame_run_len": run_len,   # how many consecutive outer turns this frame has persisted
            "current_frame": frame,
            "user_ask": ask[:500],
        })

    distinct_frames = len({p["current_frame"] for p in pairs})
    max_run = max((p["frame_run_len"] for p in pairs), default=0)
    print(f"usable outer (frame + user_ask) pairs: {len(pairs)}")
    print(f"distinct current_frame values: {distinct_frames}")
    print(f"longest single-frame persistence run: {max_run} outer turns")

    with open(_OUT, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    print(f"-> {_OUT}")

    for p in pairs[-8:]:
        print("=" * 70)
        print(f"  [run x{p['frame_run_len']}] frame: {p['current_frame'][:150]}")
        print(f"             ask  : {p['user_ask'][:150]}")


if __name__ == "__main__":
    main()
