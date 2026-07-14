"""Log AgentServer SSE events with repository-relative, configurable defaults.

The listener defaults to the loopback AgentServer and writes under this
checkout's ``artifacts`` directory. Set ``MONOLITH_AGENT_TOKEN`` when the
server requires authentication.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_URL = os.environ.get(
    "MONOLITH_AGENT_EVENTS_URL", "http://127.0.0.1:7821/events"
)
DEFAULT_OUT = ROOT / "artifacts" / "codex_sse_events.log"


def _request(url: str, token: str) -> urllib.request.Request:
    headers = {"Accept": "text/event-stream"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(url, headers=headers, method="GET")


def _append(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()


def listen(url: str, output: Path, token: str, reconnect_delay: float = 2.0) -> None:
    while True:
        try:
            req = _request(url, token)
            with urllib.request.urlopen(req, timeout=300) as response:
                _append(output, f"\n--- connected {dt.datetime.now().isoformat()} ---\n")
                for raw in response:
                    line = raw.decode("utf-8", errors="replace").rstrip("\n")
                    stamp = dt.datetime.now().isoformat()
                    _append(output, f"[{stamp}] {line}\n")
        except KeyboardInterrupt:
            return
        except Exception as exc:  # long-running diagnostic helper
            stamp = dt.datetime.now().isoformat()
            _append(output, f"[{stamp}] reconnect after error: {exc}\n")
            time.sleep(max(0.1, reconnect_delay))


def main() -> int:
    parser = argparse.ArgumentParser(description="Log Monolith AgentServer SSE events")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--reconnect-delay", type=float, default=2.0)
    args = parser.parse_args()
    token = str(os.environ.get("MONOLITH_AGENT_TOKEN", "") or "").strip()
    listen(args.url, args.output.expanduser().resolve(), token, args.reconnect_delay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
