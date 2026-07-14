import sys
from types import SimpleNamespace

from PySide6.QtWidgets import QApplication

from engine.llm import GGUFRuntime


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class _FakeModel:
    def n_ctx_train(self) -> int:
        return 4096


class _FakeLlama:
    def __init__(self, model_path: str, n_ctx: int, n_gpu_layers: int, verbose: bool):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose
        self._model = _FakeModel()

    def create_chat_completion(self, **kwargs):
        yield {"choices": [{"delta": {"reasoning_content": "thinking"}}]}
        yield {"choices": [{"delta": {"content": "hello"}}]}


def test_gguf_runtime_loads_and_streams_tokens(monkeypatch):
    _app()
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=_FakeLlama))

    runtime = GGUFRuntime()
    loaded = []
    tokens = []
    completions = []

    runtime.loaded.connect(loaded.append)
    runtime.token.connect(tokens.append)
    runtime.done.connect(lambda completed, text: completions.append((completed, text)))

    runtime.load_model("fake.gguf", 2048, -1)
    runtime.generate(
        {
            "messages": [{"role": "user", "content": "Hi"}],
            "temp": 0.7,
            "top_p": 0.9,
            "max_tokens": 32,
            "thinking_enabled": True,
            "sampling": {},
        }
    )

    assert loaded == [4096]
    assert tokens == ["<think>", "thinking", "</think>", "hello"]
    assert completions == [(True, "<think>thinking</think>hello")]
