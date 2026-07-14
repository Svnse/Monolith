"""scripts/matrix_sync_probe.py — Verify what sync_once delivers.

Reuses cached credentials. Runs ONE sync (no forever loop), prints every
room and the last 5 events per room. Helps debug why sync_forever isn't
firing RoomMessageText callbacks in the bridge.

Run:
    python scripts/matrix_sync_probe.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nio import AsyncClient, RoomMessageText


async def main() -> int:
    creds_path = Path(
        os.environ.get(
            "MATRIX_CREDENTIALS_PATH",
            str(Path.home() / ".config" / "monolith" / "matrix_credentials.json"),
        )
    ).expanduser()

    if not creds_path.exists():
        sys.stderr.write(f"no cached credentials at {creds_path}\n")
        return 2

    creds = json.loads(creds_path.read_text(encoding="utf-8"))
    homeserver = creds["homeserver"]
    user_id = creds["user_id"]

    client = AsyncClient(homeserver, user_id)
    client.access_token = creds["access_token"]
    client.user_id = user_id
    client.device_id = creds.get("device_id")

    print(f"probing sync as {user_id}", flush=True)

    # First check joined rooms via REST
    joined = await client.joined_rooms()
    rooms = getattr(joined, "rooms", None) or []
    print(f"joined_rooms (REST): {rooms}", flush=True)

    # Register a wildcard-ish callback to count message events seen during sync
    seen = {"count": 0}

    async def on_msg(room, event):
        seen["count"] += 1
        print(
            f"  SYNC EVENT: room={room.room_id[:30]} sender={event.sender} body={getattr(event, 'body', '?')[:80]!r}",
            flush=True,
        )

    client.add_event_callback(on_msg, RoomMessageText)

    # One sync with full state, then exit
    print("calling sync (full_state=True, timeout=10s) ...", flush=True)
    resp = await client.sync(timeout=10000, full_state=True)
    print(f"sync returned: type={type(resp).__name__}", flush=True)
    print(f"  next_batch={getattr(resp, 'next_batch', '?')[:40]}...", flush=True)
    print(f"  joined rooms in sync: {len(getattr(resp.rooms, 'join', {}))}", flush=True)
    for room_id, room in (getattr(resp.rooms, "join", {}) or {}).items():
        timeline_events = getattr(room.timeline, "events", []) or []
        print(f"  {room_id}: timeline={len(timeline_events)} events", flush=True)
        for ev in timeline_events[-5:]:
            ev_type = type(ev).__name__
            body = getattr(ev, "body", None)
            print(f"     {ev_type} sender={getattr(ev, 'sender', '?')} body={body!r}", flush=True)

    print(f"\ncallback fired {seen['count']} times", flush=True)
    await client.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
