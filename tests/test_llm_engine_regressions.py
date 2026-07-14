import sys
from types import SimpleNamespace

from PySide6.QtWidgets import QApplication

from core.state import AppState, SystemStatus
from engine.llm import GGUFRuntime, LLMEngine


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class _FakeModel:
    def n_ctx_train(self) -> int:
        return 4096


class _ReloadableLlama:
    def __init__(self, model_path: str, n_ctx: int, n_gpu_layers: int, verbose: bool):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose
        if model_path == "bad.gguf":
            raise RuntimeError("failed to load requested model")
        self._model = _FakeModel()

    def create_chat_completion(self, **kwargs):
        yield {"choices": [{"delta": {"content": "ok"}}]}


def test_gguf_runtime_failed_reload_keeps_previous_model(monkeypatch):
    _app()
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=_ReloadableLlama))

    runtime = GGUFRuntime()
    errors = []
    completions = []
    runtime.load_error.connect(errors.append)
    runtime.done.connect(lambda completed, text: completions.append((completed, text)))

    runtime.load_model("good.gguf", 2048, -1)
    runtime.load_model("bad.gguf", 2048, -1)
    runtime.generate({"messages": [{"role": "user", "content": "hello"}]})

    assert errors
    assert "Load Failed" in errors[-1]
    assert completions[-1] == (True, "ok")


def test_load_error_keeps_engine_ready_when_model_already_loaded():
    _app()
    engine = LLMEngine(AppState())
    try:
        engine.model_loaded = True
        engine.set_status(SystemStatus.LOADING)
        engine._on_load_error("Load Failed: backend down")
        assert engine._status == SystemStatus.READY
    finally:
        engine.shutdown()


def test_load_error_sets_error_when_no_model_loaded():
    _app()
    engine = LLMEngine(AppState())
    try:
        engine.model_loaded = False
        engine.set_status(SystemStatus.LOADING)
        engine._on_load_error("Load Failed: backend down")
        assert engine._status == SystemStatus.ERROR
    finally:
        engine.shutdown()


def test_on_gen_finish_clears_worker_and_stops_timeout_timer():
    _app()
    engine = LLMEngine(AppState())
    try:
        engine.worker = object()
        engine._gen_timeout_timer.start(10_000)
        assert engine._gen_timeout_timer.isActive()
        engine._on_gen_finish(True, "hello")
        assert engine.worker is None
        assert not engine._gen_timeout_timer.isActive()
    finally:
        engine.shutdown()


def test_stop_generation_stops_timeout_timer_even_when_idle():
    _app()
    engine = LLMEngine(AppState())
    try:
        engine._gen_timeout_timer.start(10_000)
        assert engine._gen_timeout_timer.isActive()
        engine.stop_generation()
        assert not engine._gen_timeout_timer.isActive()
    finally:
        engine.shutdown()


def test_ephemeral_followup_message_is_flagged_ephemeral(monkeypatch):
    """When-plane fix #4: on a tool-followup (ephemeral) generation, the
    synthetic tool-result user message must carry ephemeral=True so the
    ephemeral coalescer inserts its (per-minute) temporal block before the
    OUTER user message — preserving the cached prefix through the assistant
    tool-call instead of busting KV-cache mid tool-chain."""
    _app()
    import engine.llm as ellm
    import core.turn_trace as tt
    # Isolate the two collaborators irrelevant to the message-construction change.
    monkeypatch.setattr(ellm, "apply_interceptors", lambda messages, config: messages)
    monkeypatch.setattr(tt, "record_frame", lambda *a, **k: None)

    engine = LLMEngine(AppState())
    try:
        engine.backend = "gguf"
        engine.model_loaded = True
        engine.conversation_history = [
            {"role": "system", "content": "You are Monolith."},
            {"role": "user", "content": "outer user question"},
            {"role": "assistant", "content": "<tool_call>run_command</tool_call>"},
        ]
        captured: list = []
        engine.sig_gguf_generate.connect(lambda payload: captured.append(payload))

        engine.generate({
            "prompt": "[TOOL RESULT:run_command]\nexit_code=0",
            "ephemeral": True,
            "config": {"system_prompt": "You are Monolith."},
        })

        assert captured, "generate() did not dispatch a gguf payload"
        messages = captured[-1]["messages"]
        tool_msg = next(
            m for m in messages
            if str(m.get("content", "")).startswith("[TOOL RESULT:run_command]")
        )
        assert tool_msg.get("ephemeral") is True
    finally:
        engine.shutdown()


def test_reset_conversation_clears_context_refresh_marker():
    """When-plane fix #6: reset_conversation must clear the process-global
    context_refresh marker. Otherwise a new conversation is suppressed from
    refreshing until its message count passes the prior conversation's
    high-water mark (cross-conversation state leak)."""
    _app()
    import core.context_refresh as cr
    engine = LLMEngine(AppState())
    try:
        cr._last_context_refresh = {"message_count": 80, "triggered": True}
        engine.reset_conversation("You are Monolith.")
        assert cr.get_last_context_refresh() == {}
    finally:
        cr._last_context_refresh = {}
        engine.shutdown()
