from __future__ import annotations

import gc
import weakref

from ui.addons.host import AddonHost
from ui.addons.registry import AddonRegistry


class _FakeSignal:
    def __init__(self) -> None:
        self.handlers: list[object] = []

    def connect(self, handler) -> None:
        self.handlers.append(handler)

    def disconnect(self, handler) -> None:
        self.handlers.remove(handler)

    def emit(self, *args) -> None:
        for handler in list(self.handlers):
            handler(*args)


class _FakeGuard:
    def __init__(self) -> None:
        self.sig_token = _FakeSignal()
        self.sig_status = _FakeSignal()
        self.sig_trace = _FakeSignal()
        self.sig_usage = _FakeSignal()
        self.sig_finished = _FakeSignal()
        self.sig_model_capabilities = _FakeSignal()


class _FakeCtx:
    def __init__(self) -> None:
        self.guard = _FakeGuard()
        self.ui = None
        self.ui_bridge = None
        self.services = {}
        self.host = None


def test_addon_host_dispose_disconnects_guard_hooks() -> None:
    ctx = _FakeCtx()
    host = AddonHost(AddonRegistry(), ctx)

    assert len(ctx.guard.sig_token.handlers) == 1
    assert len(ctx.guard.sig_model_capabilities.handlers) == 1

    host.dispose()

    assert len(ctx.guard.sig_token.handlers) == 0
    assert len(ctx.guard.sig_status.handlers) == 0
    assert len(ctx.guard.sig_trace.handlers) == 0
    assert len(ctx.guard.sig_usage.handlers) == 0
    assert len(ctx.guard.sig_finished.handlers) == 0
    assert len(ctx.guard.sig_model_capabilities.handlers) == 0
    assert ctx.host is None
    assert ctx.services.get("host") is None


def test_addon_host_guard_handlers_do_not_keep_host_alive() -> None:
    ctx = _FakeCtx()
    host = AddonHost(AddonRegistry(), ctx)
    ctx.host = None
    ctx.services["host"] = None

    host_ref = weakref.ref(host)
    del host
    gc.collect()

    assert host_ref() is None
    # Emitting after host GC should be a no-op.
    ctx.guard.sig_token.emit("llm", "tok")
