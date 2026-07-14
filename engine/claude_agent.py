"""Small authenticated peer endpoint for connecting Claude to Monolith.

Defaults are loopback-only. Outbound Monolith requests use
``MONOLITH_AGENT_TOKEN``. Inbound ``POST /chat`` requests use the optional
``MONOLITH_PEER_TOKEN``; a non-loopback bind is refused unless that token is
configured.

Usage:
    python engine/claude_agent.py [--port 7822]
        [--monolith http://127.0.0.1:7821]
"""
from __future__ import annotations

import argparse
import hmac
import ipaddress
import json
import os
import queue
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


MONOLITH_BASE = str(
    os.environ.get("MONOLITH_AGENT_URL", "http://127.0.0.1:7821")
).rstrip("/")
MY_NAME = "Claude"
MY_HOST = "127.0.0.1"
MY_ADVERTISE_HOST = "127.0.0.1"
MY_PORT = 7822
MONOLITH_TOKEN = str(os.environ.get("MONOLITH_AGENT_TOKEN", "") or "").strip()
PEER_TOKEN = str(os.environ.get("MONOLITH_PEER_TOKEN", "") or "").strip()
MAX_BODY_BYTES = 1_000_000
MAX_MESSAGE_CHARS = 20_000

_inbox: queue.Queue = queue.Queue()


def _is_loopback_host(host: str) -> bool:
    value = str(host or "").strip().lower().strip("[]")
    if value == "localhost":
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def _validate_bind(host: str, token: str) -> None:
    if not _is_loopback_host(host) and not str(token or "").strip():
        raise ValueError(
            "Refusing a non-loopback Claude helper bind without "
            "MONOLITH_PEER_TOKEN"
        )


def _is_authorized(headers, token: str | None = None) -> bool:
    expected = PEER_TOKEN if token is None else str(token or "").strip()
    if not expected:
        return True
    auth = str(getattr(headers, "get", lambda *_: "")("Authorization", "") or "").strip()
    if auth.lower().startswith("bearer "):
        return hmac.compare_digest(auth[7:].strip(), expected)
    api_key = str(getattr(headers, "get", lambda *_: "")("X-API-Key", "") or "").strip()
    return hmac.compare_digest(api_key, expected)


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _json(self, data: dict, code: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] == "/health":
            self._json({"ok": True, "agent": MY_NAME, "port": MY_PORT})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] != "/chat":
            self._json({"error": "not found"}, 404)
            return
        if not _is_authorized(self.headers):
            self._json({"ok": False, "error": "unauthorized"}, 401)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
        except (TypeError, ValueError):
            self._json({"ok": False, "error": "invalid Content-Length"}, 400)
            return
        if length > MAX_BODY_BYTES:
            self._json({"ok": False, "error": "request body too large"}, 413)
            return
        body = self.rfile.read(length) if length else b""
        try:
            data = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            self._json({"ok": False, "error": "invalid JSON"}, 400)
            return

        message = str(data.get("message") or "").strip()
        if not message:
            self._json({"ok": False, "error": "'message' is required"}, 400)
            return
        if len(message) > MAX_MESSAGE_CHARS:
            self._json({"ok": False, "error": "message too large"}, 413)
            return
        sender = str(data.get("agent") or "unknown")
        history = data.get("history") or []
        if not isinstance(history, list):
            history = []
        _inbox.put({"message": message, "from": sender, "history": history})
        self._json({"ok": True, "response": "[Claude received; responding shortly]"})


def _post(url: str, data: dict, token: str | None = None) -> dict:
    payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    auth_token = MONOLITH_TOKEN if token is None else str(token or "").strip()
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=10) as response:
        return json.loads(response.read())


def join() -> dict:
    return _post(
        f"{MONOLITH_BASE}/join",
        {
            "name": MY_NAME,
            "url": f"http://{MY_ADVERTISE_HOST}:{MY_PORT}",
        },
    )


def send_to_monolith(message: str) -> dict:
    return _post(
        f"{MONOLITH_BASE}/chat",
        {"message": message, "agent": MY_NAME},
    )


def main() -> int:
    global MONOLITH_BASE, MONOLITH_TOKEN, MY_ADVERTISE_HOST
    global MY_HOST, MY_NAME, MY_PORT, PEER_TOKEN

    parser = argparse.ArgumentParser(description="Run the Monolith Claude peer helper")
    parser.add_argument("--port", type=int, default=MY_PORT)
    parser.add_argument("--host", default=MY_HOST)
    parser.add_argument("--advertise-host", default=MY_ADVERTISE_HOST)
    parser.add_argument("--name", default=MY_NAME)
    parser.add_argument("--monolith", default=MONOLITH_BASE)
    args = parser.parse_args()

    MY_PORT = args.port
    MY_HOST = str(args.host or "127.0.0.1").strip()
    MY_ADVERTISE_HOST = str(args.advertise_host or "127.0.0.1").strip()
    MY_NAME = str(args.name or "Claude").strip() or "Claude"
    MONOLITH_BASE = str(args.monolith or MONOLITH_BASE).rstrip("/")
    MONOLITH_TOKEN = str(os.environ.get("MONOLITH_AGENT_TOKEN", "") or "").strip()
    PEER_TOKEN = str(os.environ.get("MONOLITH_PEER_TOKEN", "") or "").strip()
    _validate_bind(MY_HOST, PEER_TOKEN)

    httpd = HTTPServer((MY_HOST, MY_PORT), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    print(f"[claude-agent] /chat listening on {MY_HOST}:{MY_PORT}", flush=True)

    try:
        result = join()
        print(f"[claude-agent] joined: {result}", flush=True)
    except Exception as exc:
        print(f"[claude-agent] join failed ({type(exc).__name__})", flush=True)

    default_log = Path(__file__).resolve().parents[1] / "artifacts" / "claude_inbox.log"
    log_path = Path(
        os.environ.get("MONOLITH_CLAUDE_INBOX_LOG", str(default_log))
    ).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[claude-agent] inbox log: {log_path}", flush=True)

    def _log_inbox() -> None:
        while True:
            try:
                msg = _inbox.get(timeout=1)
            except queue.Empty:
                continue
            line = f"[FROM:{msg['from']}] {msg['message']}"
            if msg.get("history"):
                line += f" [ctx:{len(msg['history'])} msgs]"
            print(line, flush=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    threading.Thread(target=_log_inbox, daemon=True).start()

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[claude-agent] leaving...", flush=True)
        try:
            _post(f"{MONOLITH_BASE}/leave", {"name": MY_NAME})
        except Exception:
            pass
        httpd.shutdown()
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
