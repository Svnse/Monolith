import json
import os
import re
import sys
import threading
import traceback
from datetime import datetime
from queue import Empty, Full, Queue

from PySide6.QtCore import QtMsgType, qInstallMessageHandler
from PySide6.QtWidgets import QApplication

from core.event_ledger import EventLedger
from core.theme_config import load_theme_config
from core.theme_engine import ThemeEngine
from core.themes import apply_theme
from core.style import refresh_styles
from core.state import AppState
from engine.bridge import EngineBridge
from monokernel.bridge import MonoBridge
from monokernel.dock import MonoDock
from monokernel.guard import MonoGuard
from ui.addons.builtin import build_builtin_registry
from ui.addons.context import AddonContext
from ui.addons.host import AddonHost
from ui.bridge import UIBridge
from ui.main_window import MonolithUI
from ui.overseer import OverseerWindow


def _install_console_telemetry(app: QApplication, guard: MonoGuard) -> None:
    """Mirror guard/runtime signals to the launching terminal (for .bat sessions)."""
    raw_flag = str(os.getenv("MONOLITH_CONSOLE_LOG", "1")).strip().lower()
    enabled = raw_flag not in {"0", "false", "off", "no"}
    if not enabled:
        return

    def _bool_env(name: str, default: bool = False) -> bool:
        raw = str(os.getenv(name, "1" if default else "0")).strip().lower()
        return raw in {"1", "true", "on", "yes"}

    def _int_env(name: str, default: int) -> int:
        raw = str(os.getenv(name, str(default))).strip()
        try:
            parsed = int(raw)
        except (TypeError, ValueError):
            parsed = default
        return max(64, parsed)

    tag_re = re.compile(r"<[^>]+>")
    write_lock = threading.RLock()
    show_token_events = _bool_env("MONOLITH_CONSOLE_TOKENS", default=False)
    max_str_chars = _int_env("MONOLITH_CONSOLE_MAX_STR", 2000)
    max_queue_items = _int_env("MONOLITH_CONSOLE_QUEUE_MAX", 4096)
    max_list_items = _int_env("MONOLITH_CONSOLE_MAX_LIST_ITEMS", 24)
    max_depth = _int_env("MONOLITH_CONSOLE_MAX_DEPTH", 6)
    queue: Queue[object] = Queue(maxsize=max_queue_items)
    stop_sentinel = object()
    dropped_counter = {"count": 0}

    def _sanitize(text: str) -> str:
        return tag_re.sub("", text or "")

    def _clip_str(text: str) -> str:
        clean = _sanitize(text)
        if len(clean) <= max_str_chars:
            return clean
        remaining = len(clean) - max_str_chars
        return f"{clean[:max_str_chars]}... <truncated {remaining} chars>"

    def _clip_payload(value, *, depth: int = 0):
        if depth >= max_depth:
            return f"<depth_limit:{type(value).__name__}>"
        if isinstance(value, str):
            return _clip_str(value)
        if isinstance(value, dict):
            out: dict[str, object] = {}
            for key, item in value.items():
                key_str = str(key)
                key_lower = key_str.lower()
                if key_lower == "xml" and isinstance(item, str):
                    out[key_str] = f"<omitted xml len={len(item)}>"
                    continue
                out[key_str] = _clip_payload(item, depth=depth + 1)
            return out
        if isinstance(value, (list, tuple)):
            clipped = []
            for item in list(value)[:max_list_items]:
                clipped.append(_clip_payload(item, depth=depth + 1))
            if len(value) > max_list_items:
                clipped.append(f"<truncated {len(value) - max_list_items} items>")
            return clipped
        return value

    def _serialize(payload: dict) -> str:
        return json.dumps(_clip_payload(payload), ensure_ascii=False, default=str)

    def _emit_direct(line: str) -> None:
        with write_lock:
            print(line, flush=True)

    def _writer_loop() -> None:
        while True:
            item = queue.get()
            if item is stop_sentinel:
                return
            _emit_direct(str(item))

    writer = threading.Thread(
        target=_writer_loop,
        name="monolith-console-telemetry",
        daemon=True,
    )
    writer.start()

    def _enqueue_line(line: str) -> None:
        try:
            queue.put_nowait(line)
            return
        except Full:
            dropped_counter["count"] += 1
        if dropped_counter["count"] in {1, 10, 50} or dropped_counter["count"] % 200 == 0:
            warning = (
                f"{datetime.utcnow().isoformat(timespec='milliseconds')}+00:00 "
                f"console_warn "
                f"{_serialize({'message': 'console queue saturated; dropping events', 'dropped': dropped_counter['count']})}"
            )
            try:
                _ = queue.get_nowait()
            except Empty:
                pass
            try:
                queue.put_nowait(warning)
            except Full:
                pass

    def _emit(channel: str, payload: str) -> None:
        stamp = datetime.utcnow().isoformat(timespec="milliseconds") + "+00:00"
        _enqueue_line(f"{stamp} {channel} {payload}")

    def _format_agent_event(event: object) -> dict:
        if not isinstance(event, dict):
            return {"raw": _clip_str(str(event))}
        event_name = str(event.get("event") or "")
        if event_name == "LLM_TOKEN" and not show_token_events:
            return {"_skip": True, "event": event_name}
        if event_name in {"SCRATCHPAD_INITIALIZED", "SCRATCHPAD_UPDATED", "SCRATCHPAD_COMPACTED"}:
            snapshot = event.get("snapshot") if isinstance(event.get("snapshot"), dict) else {}
            slim_snapshot = {
                "run_id": snapshot.get("run_id"),
                "version": snapshot.get("version"),
                "char_count": snapshot.get("char_count"),
                "char_limit": snapshot.get("char_limit"),
                "compacted": snapshot.get("compacted"),
                "updated_at": snapshot.get("updated_at"),
            }
            return {
                "event": event_name,
                "project_id": event.get("project_id"),
                "path": event.get("path"),
                "snapshot": slim_snapshot,
                "timestamp": event.get("timestamp"),
            }
        if event_name == "STEP_LOG_APPEND":
            entry = event.get("entry") if isinstance(event.get("entry"), dict) else {}
            return {
                "event": event_name,
                "version": event.get("version"),
                "entry": {
                    "index": entry.get("index"),
                    "phase": entry.get("phase"),
                    "status": entry.get("status"),
                    "summary": entry.get("summary"),
                    "timestamp": entry.get("timestamp"),
                },
            }
        return event

    def _emit_agent_event(engine_key: str, event: object) -> None:
        formatted = _format_agent_event(event)
        if formatted.get("_skip") is True:
            return
        _emit(
            "agentevent",
            _serialize({"engine_key": engine_key, "event": formatted}),
        )

    guard.sig_trace.connect(
        lambda ek, msg: _emit(
            "guardtrace",
            _serialize({"engine_key": ek, "message": _sanitize(str(msg))}),
        )
    )
    guard.sig_status.connect(
        lambda ek, status: _emit(
            "guardstatus",
            _serialize(
                {"engine_key": ek, "status": getattr(status, "name", str(status))}
            ),
        )
    )
    guard.sig_finished.connect(
        lambda ek, task_id: _emit(
            "guardfinished",
            _serialize({"engine_key": ek, "task_id": task_id}),
        )
    )
    guard.sig_agent_event.connect(_emit_agent_event)

    prev_excepthook = sys.excepthook

    def _sys_excepthook(exc_type, exc_value, exc_tb) -> None:
        _emit(
            "uncaught",
            _serialize(
                {
                    "type": getattr(exc_type, "__name__", str(exc_type)),
                    "error": str(exc_value),
                }
            ),
        )
        _emit(
            "traceback",
            _sanitize(
                "".join(traceback.format_exception(exc_type, exc_value, exc_tb)).rstrip()
            ),
        )
        if prev_excepthook is not None:
            prev_excepthook(exc_type, exc_value, exc_tb)

    sys.excepthook = _sys_excepthook

    if hasattr(threading, "excepthook"):
        prev_thread_excepthook = threading.excepthook

        def _thread_excepthook(args) -> None:
            thread_name = getattr(getattr(args, "thread", None), "name", "unknown")
            _emit(
                "thread_exc",
                _serialize(
                    {
                        "thread": thread_name,
                        "type": getattr(args.exc_type, "__name__", str(args.exc_type)),
                        "error": str(args.exc_value),
                    }
                ),
            )
            _emit(
                "thread_tb",
                _sanitize(
                    "".join(
                        traceback.format_exception(
                            args.exc_type, args.exc_value, args.exc_traceback
                        )
                    ).rstrip()
                ),
            )
            if prev_thread_excepthook is not None:
                prev_thread_excepthook(args)

        threading.excepthook = _thread_excepthook
        app._monolith_prev_thread_excepthook = prev_thread_excepthook

    qt_levels: dict[object, str] = {}
    for attr_name, level_name in (
        ("QtDebugMsg", "qt_debug"),
        ("QtInfoMsg", "qt_info"),
        ("QtWarningMsg", "qt_warn"),
        ("QtCriticalMsg", "qt_critical"),
        ("QtFatalMsg", "qt_fatal"),
    ):
        enum_value = getattr(QtMsgType, attr_name, None)
        if enum_value is not None:
            qt_levels[enum_value] = level_name

    qt_handler_ref: dict[str, object | None] = {"prev": None}

    def _qt_message_handler(mode, context, message) -> None:
        _emit(qt_levels.get(mode, "qt"), _sanitize(str(message)))
        prev = qt_handler_ref.get("prev")
        if prev is not None:
            prev(mode, context, message)

    prev_qt_handler = qInstallMessageHandler(_qt_message_handler)
    qt_handler_ref["prev"] = prev_qt_handler
    app._monolith_prev_qt_handler = prev_qt_handler
    app._monolith_qt_handler = _qt_message_handler
    app.aboutToQuit.connect(
        lambda: (
            queue.put_nowait(stop_sentinel)
            if not queue.full()
            else None
        )
    )
    _emit(
        "console",
        _serialize(
            {
                "enabled": True,
                "MONOLITH_CONSOLE_LOG": raw_flag,
                "MONOLITH_CONSOLE_TOKENS": show_token_events,
                "MONOLITH_CONSOLE_MAX_STR": max_str_chars,
                "MONOLITH_CONSOLE_QUEUE_MAX": max_queue_items,
            }
        ),
    )


def main():
    app = QApplication(sys.argv)
    theme_cfg = load_theme_config()
    apply_theme(theme_cfg.get("theme", "midnight"))
    refresh_styles()
    theme_engine = ThemeEngine()
    theme_engine.apply(app)

    def _repolish_widgets() -> None:
        for widget in app.allWidgets():
            try:
                style = widget.style()
                style.unpolish(widget)
                style.polish(widget)
                widget.update()
            except Exception:
                continue

    _repolish_widgets()

    state = AppState()
    # Vision and Audio engines are registered lazily by their addon factories
    # (sd_factory → VisionProcess, audiogen_factory → AudioProcess).
    guard = MonoGuard(state, {})
    dock = MonoDock(guard)
    bridge = MonoBridge(dock)
    _install_console_telemetry(app, guard)

    ui_bridge = UIBridge()
    ui = MonolithUI(state, ui_bridge)
    overseer = OverseerWindow(guard, ui_bridge)
    ledger = EventLedger(overseer.db, app_version="0.2.2a")

    registry = build_builtin_registry()
    ctx = AddonContext(state=state, guard=guard, bridge=bridge, ui=ui, host=None, ui_bridge=ui_bridge)
    host = AddonHost(registry, ctx)
    ui.attach_host(host)

    ui_bridge.sig_open_overseer.connect(overseer.show)
    ui_bridge.sig_overseer_viz_toggle.connect(guard.enable_viztracer)

    def _apply_theme(theme_name: str) -> None:
        apply_theme(theme_name)
        refresh_styles()
        theme_engine.apply(app)
        _repolish_widgets()

    ui_bridge.sig_theme_changed.connect(_apply_theme)

    # global chrome-only wiring stays here
    guard.sig_status.connect(ui.update_status)
    guard.sig_usage.connect(lambda _ek, used: ui.update_ctx(used))

    # ---- Event Ledger wiretap (Phase 1) ----

    # Engine/kernel status
    guard.sig_status.connect(
        lambda ek, s: ledger.record(
            "guard", "state", "status_changed",
            engine_key=ek,
            payload={"engine_key": ek, "status": s})
    )

    # Task finished
    guard.sig_finished.connect(
        lambda ek, tid: ledger.record(
            "guard", "lifecycle", "task_finished",
            engine_key=ek,
            payload={"engine_key": ek, "task_id": tid},
            correlation_id=tid)
    )

    # Engine ready
    guard.sig_engine_ready.connect(
        lambda ek: ledger.record(
            "guard", "lifecycle", "engine_ready",
            engine_key=ek,
            payload={"engine_key": ek})
    )

    # Traces — FILTERED: only ERROR/WARN/REJECTED
    guard.sig_trace.connect(
        lambda ek, msg: (
            ledger.record(
                "guard", "error", "trace_warning",
                engine_key=ek,
                payload={"engine_key": ek, "message": msg},
                severity=2)
            if any(k in msg.upper() for k in ("ERROR", "WARN", "REJECTED"))
            else None
        )
    )

    # Image generated
    guard.sig_image.connect(
        lambda img: ledger.record(
            "guard", "artifact", "image_generated",
            payload={"type": "image"})
    )

    # Theme changed
    ui_bridge.sig_theme_changed.connect(
        lambda t: ledger.record(
            "ui.bridge", "intent", "theme_changed",
            payload={"theme": t})
    )

    # Operator applied
    ui_bridge.sig_apply_operator.connect(
        lambda d: ledger.record(
            "ui.bridge", "intent", "operator_applied",
            payload=d)
    )

    # Overseer opened
    ui_bridge.sig_open_overseer.connect(
        lambda: ledger.record(
            "ui.bridge", "intent", "open_overseer")
    )

    # Terminal header updated
    ui_bridge.sig_terminal_header.connect(
        lambda a, b, c: ledger.record(
            "ui.bridge", "state", "terminal_header_updated",
            payload={"args": [a, b, c]})
    )

    app.aboutToQuit.connect(ledger.shutdown)
    app.aboutToQuit.connect(guard.stop)
    app.aboutToQuit.connect(overseer.db.close)
    app.aboutToQuit.connect(lambda: guard.enable_viztracer(False) if guard._viztracer is not None else None)
    # Engine subprocesses use daemon=True — they terminate with the main process.

    ui.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
