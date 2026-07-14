import uuid
import weakref

from PySide6.QtWidgets import QWidget

from ui.addons.context import AddonContext
from ui.addons.registry import AddonRegistry


class AddonHost:
    def __init__(self, registry: AddonRegistry, ctx: AddonContext):
        self.registry = registry
        self.ctx = ctx
        self.ctx.host = self
        if hasattr(self.ctx, "services"):
            self.ctx.services["host"] = self
        self._guard_connections: list[tuple[object, object]] = []
        self._engine_event_hooks()

    def _engine_event_hooks(self) -> None:
        guard = getattr(self.ctx, "guard", None)
        if guard is None:
            return
        host_ref = weakref.ref(self)

        def _connect(signal_name: str, event_name: str) -> None:
            signal = getattr(guard, signal_name, None)
            if signal is None:
                return

            def _handler(engine_key, payload):
                host = host_ref()
                if host is None:
                    return
                host._dispatch_engine_event(engine_key, event_name, payload)

            signal.connect(_handler)
            self._guard_connections.append((signal, _handler))

        # Single engine->UI event path for modules. Text backends are expected
        # to emit tokens/status/trace through the kernel so widgets can consume
        # everything via on_engine_event() instead of bespoke signal wiring.
        _connect("sig_token", "token")
        _connect("sig_status", "status")
        _connect("sig_trace", "trace")
        _connect("sig_usage", "usage")
        _connect("sig_finished", "finished")
        _connect("sig_model_capabilities", "model_capabilities")

    def dispose(self) -> None:
        for signal, handler in self._guard_connections:
            try:
                signal.disconnect(handler)
            except Exception:
                pass
        self._guard_connections.clear()
        if getattr(self.ctx, "host", None) is self:
            self.ctx.host = None
        if hasattr(self.ctx, "services") and self.ctx.services.get("host") is self:
            self.ctx.services["host"] = None

    def _dispatch_engine_event(self, engine_key: str, event: str, payload: object) -> None:
        if not self.ctx.ui:
            return
        for i in range(self.ctx.ui.stack.count()):
            widget = self.ctx.ui.stack.widget(i)
            if not widget or not getattr(widget, "_addon_id", None):
                continue
            handler = getattr(widget, "on_engine_event", None)
            if callable(handler):
                try:
                    handler(engine_key, event, payload)
                except Exception as exc:
                    if self.ctx.ui_bridge is not None:
                        self.ctx.ui_bridge.sig_monitor_log.emit(
                            "ERROR",
                            f"[engine_event] {getattr(widget, '_addon_id', '?')} failed on {event}: {exc}",
                        )
                    continue

    def launch_module(self, addon_id: str, *, focus: bool = True) -> str:
        """Mount and start a module. focus=False mounts it headless (widget lives,
        engines/servers it owns run) without switching the UI to it — used by the
        boot-time CONNECT autostart so chat stays the opening surface."""
        if not self.ctx.ui:
            raise RuntimeError("AddonHost requires UI for launching modules")

        try:
            spec = self.registry.get(addon_id)
            if spec.kind != "module":
                raise ValueError(f"Addon '{addon_id}' is not a module")
            instance_id = str(uuid.uuid4())
            widget = spec.factory(self.ctx)
        except Exception as e:
            self.ctx.guard.sig_trace.emit("system", f"<span style='color:red'>ADDON ERROR: {e}</span>")
            if self.ctx.ui_bridge is not None:
                self.ctx.ui_bridge.sig_monitor_log.emit("ERROR", f"Addon launch failed: {addon_id} ({e})")
            return ""

        widget._addon_id = addon_id
        existing_mod_id = str(getattr(widget, "_mod_id", "") or "").strip()
        if not existing_mod_id:
            widget._mod_id = instance_id
        else:
            instance_id = existing_mod_id
        added_stack = False
        added_strip = False
        try:
            self.ctx.ui.stack.addWidget(widget)
            added_stack = True
            self.ctx.ui.register_module(instance_id, addon_id, spec.icon or "?", spec.title)
            added_strip = True
        except Exception:
            if added_strip:
                self.ctx.ui.unregister_module(instance_id)
            if added_stack:
                self.ctx.ui.stack.removeWidget(widget)
            widget.deleteLater()
            if self.ctx.ui_bridge is not None:
                self.ctx.ui_bridge.sig_monitor_log.emit("ERROR", f"Addon mount failed: {addon_id}")
            return ""

        if hasattr(widget, "sig_closed"):
            widget.sig_closed.connect(lambda: self.ctx.ui.close_module(instance_id))
        if hasattr(widget, "sig_finished"):
            widget.sig_finished.connect(lambda: self.ctx.ui.flash_module(instance_id))

        if focus:
            self.ctx.ui.switch_to_module(instance_id)
        if self.ctx.ui_bridge is not None:
            self.ctx.ui_bridge.sig_monitor_log.emit("INFO", f"Module launched: {addon_id} ({instance_id})")
        # capability verification (warn only)
        issues = self.registry.verify_capabilities(spec, widget)
        if issues and self.ctx.ui_bridge is not None:
            for issue in issues:
                self.ctx.ui_bridge.sig_monitor_log.emit("WARNING", f"[capabilities] {addon_id}: {issue}")
        return instance_id

    def request_reload_modules(self) -> None:
        """Notify/recreate running modules that opt into reload."""
        if not self.ctx.ui:
            return
        to_refresh: list[tuple[str, str, dict | None]] = []
        for i in range(self.ctx.ui.stack.count()):
            widget = self.ctx.ui.stack.widget(i)
            addon_id = getattr(widget, "_addon_id", None)
            mod_id = getattr(widget, "_mod_id", None)
            if not addon_id or not mod_id:
                continue
            state = None
            if hasattr(widget, "sig_reload_requested"):
                try:
                    widget.sig_reload_requested.emit()
                except Exception:
                    pass
            on_reload = getattr(widget, "on_reload", None)
            if callable(on_reload):
                try:
                    on_reload()
                except Exception:
                    pass
            on_save = getattr(widget, "on_save_state", None)
            if callable(on_save):
                try:
                    state = on_save()
                except Exception:
                    state = None
            if state is not None:
                to_refresh.append((addon_id, mod_id, state))

        for addon_id, mod_id, state in to_refresh:
            self.ctx.ui.close_module(mod_id)
            new_id = self.launch_module(addon_id)
            if not new_id:
                continue
            new_widget = None
            for i in range(self.ctx.ui.stack.count()):
                w = self.ctx.ui.stack.widget(i)
                if getattr(w, "_mod_id", None) == new_id:
                    new_widget = w
                    break
            if new_widget and hasattr(new_widget, "on_load_state"):
                try:
                    new_widget.on_load_state(state or {})
                except Exception:
                    pass
