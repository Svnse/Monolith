"""scripts/matrix_login_smoke.py — One-shot login test for the Matrix bridge.

Logs in to the homeserver, caches the access_token at
~/.config/monolith/matrix_credentials.json, then exits 0.

Useful for verifying credentials before running the full bridge — doesn't
require Monolith / agent_server.py / CONNECT to be running.

Env vars (same as engine/matrix_bridge.py):
    MATRIX_HOMESERVER   e.g. https://matrix.brainbrigade.xyz
    MATRIX_USER_ID      e.g. @monolith:brainbrigade.xyz
    MATRIX_PASSWORD     bot password (only used here; deleted from memory after)

Run:
    python scripts/matrix_login_smoke.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

# Windows: Python's default console encoding (cp1252) can't print Unicode
# emoji/arrows. Reconfigure so logging works on Windows + UTF-8 elsewhere.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass

# Make repo-root imports work when running this script from anywhere.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nio import AsyncClient, LoginResponse


async def main() -> int:
    homeserver = os.environ.get("MATRIX_HOMESERVER", "").strip()
    user_id = os.environ.get("MATRIX_USER_ID", "").strip()
    password = os.environ.get("MATRIX_PASSWORD", "").strip()

    missing = [
        n
        for n, v in [
            ("MATRIX_HOMESERVER", homeserver),
            ("MATRIX_USER_ID", user_id),
            ("MATRIX_PASSWORD", password),
        ]
        if not v
    ]
    if missing:
        sys.stderr.write(f"missing env vars: {', '.join(missing)}\n")
        return 2

    print(f"login smoke test → {homeserver} as {user_id}", flush=True)

    client = AsyncClient(homeserver, user_id)
    try:
        resp = await client.login(password, device_name="monolith-bridge")
        if not isinstance(resp, LoginResponse):
            sys.stderr.write(f"login failed: {resp}\n")
            return 1
        print(
            f"login OK | user_id={resp.user_id} device_id={resp.device_id}",
            flush=True,
        )

        # Cache creds at the standard path
        creds_path = Path(
            os.environ.get(
                "MATRIX_CREDENTIALS_PATH",
                str(Path.home() / ".config" / "monolith" / "matrix_credentials.json"),
            )
        ).expanduser()
        creds_path.parent.mkdir(parents=True, exist_ok=True)
        creds_path.write_text(
            json.dumps(
                {
                    "homeserver": homeserver,
                    "user_id": resp.user_id,
                    "device_id": resp.device_id,
                    "access_token": resp.access_token,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"credentials cached at {creds_path}", flush=True)

        # Sanity check: fetch joined rooms
        joined = await client.joined_rooms()
        if hasattr(joined, "rooms"):
            rooms = joined.rooms
            print(f"joined rooms: {len(rooms)}", flush=True)
            for r in rooms[:10]:
                print(f"  {r}", flush=True)
        else:
            print(f"joined_rooms response: {joined}", flush=True)

        return 0
    finally:
        await client.close()


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(0)
