from PySide6.QtCore import QObject, QThread, Signal, QTimer
from core.state import AppState, SystemStatus
from core.llm_config import load_config

class ModelLoader(QThread):
    trace = Signal(str)
    finished = Signal(object, int)
    error = Signal(str)

    def __init__(self, path, n_ctx=8192, n_gpu_layers=-1):
        super().__init__()
        self.path = path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers

    def run(self):
        try:
            try:
                from llama_cpp import Llama
            except ImportError as exc:
                raise RuntimeError(
                    "llama-cpp-python is not installed. Install it to use the local LLM engine."
                ) from exc
            self.trace.emit(f"→ init backend: {self.path}")
            llm_instance = Llama(
                model_path=self.path,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False
            )
            model_ctx_length = llm_instance._model.n_ctx_train()
            self.finished.emit(llm_instance, model_ctx_length)
        except Exception as e:
            self.error.emit(f"Load Failed: {str(e)}")

class GeneratorWorker(QThread):
    token = Signal(str)
    trace = Signal(str)
    done = Signal()
    usage = Signal(int)

    def __init__(self, llm, messages, temp, top_p, max_tokens):
        super().__init__()
        self.llm = llm
        self.messages = messages
        self.temp = temp
        self.top_p = top_p
        self.max_tokens = max_tokens

    def run(self):
        self.trace.emit("→ inference started")
        try:
            if self.isInterruptionRequested():
                return

            stream = self.llm.create_chat_completion(
                messages=self.messages,
                temperature=self.temp,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                stream=True
            )

            total_generated = 0
            for chunk in stream:
                if self.isInterruptionRequested():
                    self.trace.emit("→ inference aborted")
                    break

                if "content" in chunk["choices"][0]["delta"]:
                    text = chunk["choices"][0]["delta"]["content"]
                    self.token.emit(text)
                    total_generated += 1
                    self.usage.emit(total_generated)
            
            self.trace.emit("→ inference complete")
        except Exception as e:
            self.trace.emit(f"<span style='color:red'>ERROR: {e}</span>")
        finally:
            self.done.emit()

class LLMEngine(QObject):
    sig_token = Signal(str)
    sig_trace = Signal(str)
    sig_status = Signal(SystemStatus)
    sig_finished = Signal()
    sig_usage = Signal(int)
    sig_image = Signal(object)
    sig_model_capabilities = Signal(dict)

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.llm = None
        self.loader = None
        self.worker = None
        self.model_path: str | None = None
        self._load_cancel_requested: bool = False
        self._shutdown_requested: bool = False
        self._status: SystemStatus = SystemStatus.READY
        self.state.model_ctx_length = None
        self.state.sig_model_capabilities = self.sig_model_capabilities

    def set_model_path(self, path: str) -> None:
        self.model_path = path
        self.state.gguf_path = path
        QTimer.singleShot(0, lambda: self.set_status(SystemStatus.READY))

    def load_model(self):
        if self._status == SystemStatus.LOADING:
            self.sig_trace.emit("ERROR: Load already in progress.")
            self.set_status(SystemStatus.ERROR)
            return
        
        model_path = self.model_path or self.state.gguf_path
        if not model_path:
            self.sig_trace.emit("ERROR: No GGUF selected.")
            self.set_status(SystemStatus.ERROR)
            return

        self.set_status(SystemStatus.LOADING)
        self._load_cancel_requested = False
        # Keep reference to loader to prevent GC
        n_ctx = (
            min(self.state.ctx_limit, self.state.model_ctx_length)
            if self.state.model_ctx_length
            else self.state.ctx_limit
        )
        self.loader = ModelLoader(model_path, n_ctx)
        self.loader.trace.connect(self.sig_trace)
        self.loader.error.connect(self._on_load_error)
        self.loader.finished.connect(self._on_load_success)
        self.loader.finished.connect(self._cleanup_loader)
        self.loader.error.connect(self._cleanup_loader)
        self.loader.start()

    def _on_load_success(self, llm_instance, model_ctx_length):
        if self._shutdown_requested:
            del llm_instance
            self.set_status(SystemStatus.READY)
            return

        if self._load_cancel_requested:
            del llm_instance
            self.llm = None
            self.state.model_loaded = False
            self.set_status(SystemStatus.READY)
            self.sig_trace.emit("→ load cancelled")
            self.loader = None
            return

        self.llm = llm_instance
        self.state.model_ctx_length = int(model_ctx_length)
        self.state.ctx_limit = min(self.state.ctx_limit, self.state.model_ctx_length)
        self.sig_model_capabilities.emit(
            {
                "model_ctx_length": self.state.model_ctx_length,
                "ctx_limit": self.state.ctx_limit,
            }
        )
        self.state.model_loaded = True
        self.set_status(SystemStatus.READY)
        self.sig_trace.emit("→ system online")
        self.loader = None

    def _on_load_error(self, err_msg):
        self.sig_trace.emit(f"<span style='color:red'>{err_msg}</span>")
        if self._shutdown_requested:
            self.set_status(SystemStatus.READY)
        else:
            self.set_status(SystemStatus.ERROR)
        self.loader = None

    def _cleanup_loader(self, *args, **kwargs):
        self.loader = None

    def unload_model(self):
        if self._status == SystemStatus.LOADING and self.loader and self.loader.isRunning():
            self._load_cancel_requested = True
            self.sig_trace.emit("→ unload requested during load; will cancel when init completes")
            return

        if self._status == SystemStatus.RUNNING:
            self.sig_trace.emit("ERROR: Cannot unload while generating.")
            return

        if self.llm:
            self.set_status(SystemStatus.UNLOADING)
            del self.llm
            self.llm = None
        self.state.model_loaded = False
        config = load_config()
        self.state.ctx_limit = int(config.get("ctx_limit", self.state.ctx_limit))
        self.state.model_ctx_length = None
        QTimer.singleShot(0, lambda: self.set_status(SystemStatus.READY))
        self.sig_trace.emit("→ model unloaded")

    def generate(self, payload: dict):
        if not self.state.model_loaded:
            self.sig_trace.emit("ERROR: Model offline.")
            self.set_status(SystemStatus.ERROR)
            return

        if self._status == SystemStatus.RUNNING:
            self.sig_trace.emit("ERROR: Busy. Wait for completion.")
            self.set_status(SystemStatus.ERROR)
            return

        self.set_status(SystemStatus.RUNNING)

        prompt = payload.get("prompt", "")
        config = payload.get("config")
        if config is None:
            config = load_config()

        system_prompt = config.get("system_prompt", "You are Monolith. Be precise.")
        context_injection = config.get("context_injection", "")
        temp = float(config.get("temp", 0.7))
        top_p = float(config.get("top_p", 0.9))
        max_tokens = int(config.get("max_tokens", 2048))

        messages = [{"role": "system", "content": system_prompt}]
        if context_injection:
            messages.append({"role": "system", "content": f"CONTEXT: {context_injection}"})
        messages.append({"role": "user", "content": prompt})

        self.worker = GeneratorWorker(
            self.llm, messages, temp,
            top_p, max_tokens
        )
        self.worker.token.connect(self.sig_token)
        self.worker.trace.connect(self.sig_trace)
        self.worker.usage.connect(self._on_usage_update)
        self.worker.done.connect(self._on_gen_finish)
        self.worker.start()

    def stop_generation(self):
        if self._status == SystemStatus.LOADING and self.loader and self.loader.isRunning():
            self._load_cancel_requested = True
            self.sig_trace.emit("→ load cancel requested; will stop after initialization completes")
            return

        if self.worker and self.worker.isRunning():
            self.worker.requestInterruption()

    def _on_usage_update(self, count):
        self.sig_usage.emit(count)

    def _on_gen_finish(self):
        self.sig_token.emit("\n")
        self.set_status(SystemStatus.READY)
        self.sig_finished.emit()

    def set_status(self, s):
        self._status = s
        self.sig_status.emit(s)

    def shutdown(self):
        self._shutdown_requested = True
        self.stop_generation()

        if self.worker:
            self.worker.requestInterruption()
            self.worker.wait(1500)
            self.worker = None

        if self.loader and self.loader.isRunning():
            self._load_cancel_requested = True
            self.loader.wait(150)
