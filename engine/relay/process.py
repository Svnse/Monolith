"""RoomProcess — Monolith Relay engine.

Runs the MCP HTTP + SSE servers in a subprocess. The room server has no
GPU/ML work — it just needs subprocess isolation for clean lifecycle
management (start/stop without touching the main app).

IPC ops  (parent → worker)
───────────────────────────
{"op": "load",  "data_dir": str, "http_port": int, "sse_port": int}
{"op": "generate", "config": {"action": str, ...}}
{"op": "stop"}
{"op": "shutdown"}

IPC events  (worker → parent)
───────────────────────────────
{"event": "status",   "status": "loading|ready|error"}
{"event": "message",  "data": {msg dict}}
{"event": "joined",   "participant": {name, color, label, kind}}
{"event": "left",     "name": str}
{"event": "wake",     "name": str, "message": str}
{"event": "guard",    "max_hops": int}
{"event": "trace",    "message": str}
{"event": "error",    "message": str}

generate actions
─────────────────
send         — post a message: {sender, text, reply_to?}
join_loop    — register an internal loop run: {name, color?, label?, run_id}
leave_loop   — deregister: {name}
wake         — write queue file to wake external agent: {agent, message}
who          — emit current participants list
continue     — resume after loop guard pause
clear        — clear message history
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from engine.engine_process import EngineProcess
from core.paths import MONOLITH_ROOT
DATA_DIR = MONOLITH_ROOT


class RoomProcess(EngineProcess):
    """Subprocess-isolated Relay room server."""

    def __init__(self) -> None:
        super().__init__()
        self._data_dir = str(DATA_DIR / "relay")
        self._http_port = 8200
        self._sse_port  = 8201

    # ── EnginePort overrides ───────────────────────────────────────────

    def set_model_path(self, payload: dict) -> None:
        # Relay has no model path — accept silently
        from PySide6.QtCore import QTimer
        from core.state import SystemStatus
        QTimer.singleShot(0, lambda: self.sig_status.emit(SystemStatus.READY))

    def load_model(self) -> None:
        """Start the room server."""
        from core.state import SystemStatus
        if not self._ensure_proc():
            return
        self.sig_status.emit(SystemStatus.LOADING)
        self._send(
            "load",
            data_dir=self._data_dir,
            http_port=self._http_port,
            sse_port=self._sse_port,
        )

    def generate(self, payload: dict) -> None:
        """Dispatch a room action — does NOT flip engine to RUNNING/READY."""
        if not (self._proc and self._proc.is_alive()):
            self.sig_trace.emit("[RELAY] generate: server not running")
            return
        self._send("generate", config=payload)

    def stop_generation(self) -> None:
        """Stop the server subprocess."""
        self._active_gen_id = 0
        if self._proc and self._proc.is_alive():
            self._send("stop")

    # ── Event dispatch ─────────────────────────────────────────────────

    def _dispatch_event(self, event: dict) -> None:
        self.sig_event.emit(event)

        kind = str(event.get("event") or "")

        if kind == "status":
            from core.state import SystemStatus
            _MAP = {
                "loading":  SystemStatus.LOADING,
                "ready":    SystemStatus.READY,
                "error":    SystemStatus.ERROR,
            }
            s = str(event.get("status") or "").lower()
            if s in _MAP:
                self.sig_status.emit(_MAP[s])

        elif kind == "trace":
            self.sig_trace.emit(str(event.get("message") or ""))

        elif kind == "error":
            self.sig_trace.emit(f"[RELAY] ERROR: {event.get('message', '')}")
            from core.state import SystemStatus
            self.sig_status.emit(SystemStatus.ERROR)

        elif kind in ("message", "joined", "left", "wake", "guard", "participants"):
            # Serialise and emit via sig_token so the UI module can parse it
            self.sig_token.emit(json.dumps(event))

    # ── Worker (runs in child process) ─────────────────────────────────

    @staticmethod
    def _worker_fn(to_worker, from_worker) -> None:  # type: ignore[override]
        import sys
        import threading
        import time
        import json

        def _emit(event: dict):
            try:
                from_worker.put_nowait(event)
            except Exception:
                pass

        def _trace(msg: str):
            _emit({"event": "trace", "message": msg})

        # Wait for "load" op first
        op = to_worker.get()
        if op.get("op") != "load":
            _emit({"event": "error", "message": f"Expected 'load' op, got: {op.get('op')}"})
            return

        data_dir  = op.get("data_dir", "./data/relay")
        http_port = int(op.get("http_port", 8200))
        sse_port  = int(op.get("sse_port", 8201))

        _emit({"event": "status", "status": "loading"})

        # ── Setup store, router, mcp_bridge in this subprocess ──────────
        try:
            import engine.relay.store as _store_mod
            import engine.relay.router as _router_mod
            import engine.relay.mcp_bridge as _bridge_mod
        except ImportError as exc:
            _emit({"event": "error", "message": f"Import failed: {exc}"})
            return

        store  = _store_mod.MessageStore(str(data_dir) + "/relay_log.jsonl")
        router = _router_mod.Router(max_hops=4)

        # Register "you" (human) as the default human participant
        router.registry.join("you", color="#e2e8f0", label="You", kind="human")

        # Bridge store + router into mcp_bridge module globals
        _bridge_mod.store  = store
        _bridge_mod.router = router

        # Store callback → emit message events to parent
        def _on_message(msg: dict):
            _emit({"event": "message", "data": msg})
            # Check for @mention wake targets
            targets = router.get_targets(msg.get("sender", ""), msg.get("text", ""))
            if router.is_paused and not router.guard_emitted:
                router.guard_emitted = True
                store.add("system", f"Loop guard: {router.max_hops} agent-to-agent hops reached. Type /continue to resume.", msg_type="system")
                _emit({"event": "guard", "max_hops": router.max_hops})
            for t in targets:
                _emit({"event": "wake", "name": t, "message": f"{msg['sender']}: {msg['text']}"})

        store.on_message(_on_message)

        # ── Start MCP servers ───────────────────────────────────────────
        mcp_http, mcp_sse = _bridge_mod.create_servers(http_port, sse_port)

        mcp_started = [False, False]

        def _run_http():
            try:
                if mcp_http:
                    mcp_http.run(transport="streamable-http")
            except Exception as exc:
                _trace(f"MCP HTTP error: {exc}")
            finally:
                mcp_started[0] = True

        def _run_sse():
            try:
                if mcp_sse:
                    mcp_sse.run(transport="sse")
            except Exception as exc:
                _trace(f"MCP SSE error: {exc}")
            finally:
                mcp_started[1] = True

        if mcp_http:
            threading.Thread(target=_run_http, daemon=True).start()
        if mcp_sse:
            threading.Thread(target=_run_sse, daemon=True).start()

        # Brief wait so servers can bind before emitting ready
        time.sleep(0.8)
        _emit({"event": "status", "status": "ready"})
        _trace(f"[RELAY] MCP HTTP :{http_port}  SSE :{sse_port}  data={data_dir}")

        # Generate launcher scripts
        try:
            from engine.relay.launchers import generate as _gen_launchers
            from pathlib import Path
            _gen_launchers(Path(data_dir) / "launchers")
            _trace(f"[RELAY] Launchers written to {data_dir}/launchers/")
        except Exception as exc:
            _trace(f"[RELAY] Launcher generation skipped: {exc}")

        # ── Op loop ────────────────────────────────────────────────────
        while True:
            try:
                op = to_worker.get(timeout=0.5)
            except Exception:
                continue

            op_name = op.get("op")

            if op_name in ("stop", "shutdown"):
                _trace("[RELAY] shutting down")
                _emit({"event": "status", "status": "ready"})
                break

            elif op_name == "unload":
                _trace("[RELAY] unload requested")
                _emit({"event": "status", "status": "ready"})
                break

            elif op_name == "generate":
                config = op.get("config") or {}
                action = str(config.get("action") or "")

                if action == "send":
                    sender = str(config.get("sender") or "you")
                    text   = str(config.get("text") or "")
                    reply  = config.get("reply_to")
                    if text.strip().lower() == "/continue":
                        router.continue_routing()
                        store.add("system", "Routing resumed.", msg_type="system")
                    elif text.strip().lower() == "/clear":
                        store.clear()
                        _emit({"event": "message", "data": {"type": "clear"}})
                    else:
                        store.add(sender, text, reply_to=reply if isinstance(reply, int) else None)

                elif action == "join_loop":
                    name   = str(config.get("name") or "loop")
                    color  = str(config.get("color") or "")
                    label  = str(config.get("label") or name)
                    run_id = str(config.get("run_id") or "")
                    entry = router.registry.join(name, color=color, label=label, kind="loop")
                    entry["run_id"] = run_id
                    store.add(name, f"{label} joined relay", msg_type="join")
                    _emit({"event": "joined", "participant": entry})

                elif action == "leave_loop":
                    name = str(config.get("name") or "")
                    if name:
                        router.registry.leave(name)
                        store.add(name, f"{name} left relay", msg_type="leave")
                        _emit({"event": "left", "name": name})

                elif action == "wake":
                    # Write queue file so launcher watcher injects wake prompt
                    agent   = str(config.get("agent") or "")
                    message = str(config.get("message") or "")
                    if agent:
                        from pathlib import Path
                        queue_file = Path(data_dir) / f"{agent}_queue.jsonl"
                        try:
                            with open(queue_file, "a", encoding="utf-8") as f:
                                f.write(json.dumps({"text": message, "time": time.strftime("%H:%M:%S")}) + "\n")
                        except Exception as exc:
                            _trace(f"[RELAY] wake write failed: {exc}")

                elif action == "who":
                    participants = router.registry.get_all()
                    _emit({"event": "participants", "data": participants})

                elif action == "continue":
                    router.continue_routing()
                    store.add("system", "Routing resumed.", msg_type="system")

