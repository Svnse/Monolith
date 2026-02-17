from threading import Lock

from PySide6.QtCore import QObject, QThread, Signal, QTimer

from core.llm_config import AGENT_PROMPT, MASTER_PROMPT, load_config
from core.state import AppState, SystemStatus
from engine.agent_runtime import AgentRuntime
from engine.tools import set_workspace_root


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
                verbose=False,
            )
            model_ctx_length = llm_instance._model.n_ctx_train()
            self.finished.emit(llm_instance, model_ctx_length)
        except Exception as e:
            self.error.emit(f"Load Failed: {str(e)}")


class GeneratorWorker(QThread):
    token = Signal(str)
    trace = Signal(str)
    done = Signal(bool, str, list)
    usage = Signal(int)
    event = Signal(dict)

    def __init__(
        self,
        llm,
        messages,
        temp,
        top_p,
        max_tokens,
        runtime: AgentRuntime | None = None,
        agent_mode=False,
    ):
        super().__init__()
        self.llm = llm
        self.messages = list(messages)
        self.temp = temp
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.agent_mode = bool(agent_mode)
        self.runtime = runtime

    def _extract_text(self, response: dict) -> str:
        choices = response.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks = [part.get("text", "") for part in content if isinstance(part, dict)]
            return "".join(chunks)
        return ""

    def _try_chat_completion(self, messages):
        kwargs = {
            "messages": messages,
            "temperature": self.temp,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        return self.llm.create_chat_completion(**kwargs)

    def _chat_once_text(self, messages: list[dict]) -> str:
        response = self._try_chat_completion(messages)
        return self._extract_text(response)

    def _emit_runtime_event(self, payload: dict) -> None:
        self.event.emit(payload)
        event_name = payload.get("event")
        if event_name == "LLM_TOKEN":
            token = payload.get("data", "")
            if isinstance(token, str) and token:
                self.token.emit(token)
                self.usage.emit(len(token))
        elif event_name == "FINAL_OUTPUT":
            token = payload.get("data", "")
            if isinstance(token, str) and token:
                self.token.emit(token)


    def run(self):
        self.trace.emit(
            f"[WORKER] started: msgs={len(self.messages)}, temp={self.temp}, max_tokens={self.max_tokens}, agent_mode={self.agent_mode}"
        )
        completed = False
        assistant_text = ""
        loop_history = list(self.messages)

        try:
            if self.agent_mode and self.runtime is not None:
                self.runtime._llm_call = self._chat_once_text
                self.runtime._should_stop = self.isInterruptionRequested
                self.runtime._emit_event = self._emit_runtime_event
                completed, assistant_text, loop_history = self.runtime.run(self.messages)
            else:
                assistant_text = self._chat_once_text(self.messages)
                if assistant_text:
                    self.token.emit(assistant_text)
                    self.usage.emit(len(assistant_text))
                completed = not self.isInterruptionRequested()

            if completed:
                self.trace.emit("→ inference complete")
        except Exception as e:
            self.trace.emit(f"[WORKER] EXCEPTION: {e}")
            self.trace.emit(f"<span style='color:red'>ERROR: {e}</span>")
        finally:
            self.trace.emit(f"[WORKER] finished: completed={completed}, text_len={len(assistant_text)}")
            self.done.emit(completed, assistant_text, loop_history)


class LLMEngine(QObject):
    sig_token = Signal(str)
    sig_trace = Signal(str)
    sig_status = Signal(SystemStatus)
    sig_finished = Signal()
    sig_usage = Signal(int)
    sig_image = Signal(object)
    sig_model_capabilities = Signal(dict)
    sig_agent_event = Signal(dict)

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.llm = None
        self.loader = None
        self.worker = None
        self.model_path: str | None = None
        self.conversation_history: list[dict] = []
        self._pending_user_index: int | None = None
        self._load_cancel_requested: bool = False
        self._shutdown_requested: bool = False
        self._status: SystemStatus = SystemStatus.READY
        self._ephemeral_generation: bool = False
        self.model_loaded: bool = False
        self.model_ctx_length: int | None = None
        self.ctx_limit: int = int(getattr(self.state, "ctx_limit", 8192))
        self.gguf_path: str | None = None
        self._worker_seed_count: int = 0
        self._worker_agent_mode: bool = False
        self._runtime_lock = Lock()
        self._runtime = AgentRuntime(llm_call=lambda _m: "", emit_event=self.sig_agent_event.emit)

    def set_ctx_limit(self, payload: dict) -> None:
        value = payload.get("ctx_limit") if isinstance(payload, dict) else None
        if value is None:
            return
        try:
            self.ctx_limit = int(value)
        except (TypeError, ValueError):
            return

    def set_model_path(self, payload: dict) -> None:
        path = payload.get("path") if isinstance(payload, dict) else None
        self.model_path = path
        self.gguf_path = path
        QTimer.singleShot(0, lambda: self.set_status(SystemStatus.READY))

    def load_model(self):
        if self._status == SystemStatus.LOADING:
            self.sig_trace.emit("ERROR: Load already in progress.")
            self.set_status(SystemStatus.ERROR)
            return

        model_path = self.model_path or self.gguf_path
        if not model_path:
            self.sig_trace.emit("ERROR: No GGUF selected.")
            self.set_status(SystemStatus.ERROR)
            return

        self.set_status(SystemStatus.LOADING)
        self._load_cancel_requested = False
        n_ctx = min(self.ctx_limit, self.model_ctx_length) if self.model_ctx_length else self.ctx_limit
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
            self.model_loaded = False
            self.set_status(SystemStatus.READY)
            self.sig_trace.emit("→ load cancelled")
            self.loader = None
            return

        self.llm = llm_instance
        self.model_ctx_length = int(model_ctx_length)
        self.ctx_limit = min(self.ctx_limit, self.model_ctx_length)
        self.sig_model_capabilities.emit(
            {
                "model_ctx_length": self.model_ctx_length,
                "ctx_limit": self.ctx_limit,
            }
        )
        self.model_loaded = True
        self.set_status(SystemStatus.READY)
        self.reset_conversation(MASTER_PROMPT)
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
        self.model_loaded = False
        self.model_ctx_length = None
        self.reset_conversation(MASTER_PROMPT)
        QTimer.singleShot(0, lambda: self.set_status(SystemStatus.READY))
        self.sig_trace.emit("→ model unloaded")

    def reset_conversation(self, system_prompt):
        self.conversation_history = [{"role": "system", "content": system_prompt}]
        self._pending_user_index = None

    def set_history(self, payload: dict):
        history = payload.get("history", []) if isinstance(payload, dict) else []
        if not isinstance(history, list):
            return
        self.conversation_history = [h for h in history if isinstance(h, dict)]
        self._pending_user_index = None

    def _compile_system_prompt(self, config, agent_mode=False):
        base_prompt = AGENT_PROMPT if agent_mode else MASTER_PROMPT
        tags = config.get("behavior_tags", [])
        cleaned = [tag.strip() for tag in tags if isinstance(tag, str) and tag.strip()]
        if not cleaned:
            return base_prompt
        return f"{base_prompt}\n\n[BEHAVIOR TAGS]\n" + "\n".join(cleaned)

    def generate(self, payload: dict):
        if not self.model_loaded:
            self.sig_trace.emit("ERROR: Model offline.")
            self.set_status(SystemStatus.ERROR)
            return

        if isinstance(payload, dict) and "ctx_limit" in payload:
            try:
                self.ctx_limit = int(payload.get("ctx_limit", self.ctx_limit))
            except (TypeError, ValueError):
                pass

        if self._status == SystemStatus.RUNNING:
            self.sig_trace.emit("ERROR: Busy. Wait for completion.")
            self.set_status(SystemStatus.ERROR)
            return

        self.set_status(SystemStatus.RUNNING)

        prompt = payload.get("prompt", "")
        self.sig_trace.emit(
            f"[ENGINE] generate: history_len={len(self.conversation_history)}, prompt={repr(prompt[:80])}, model_loaded={self.model_loaded}"
        )
        config = payload.get("config")
        if config is None:
            config = load_config()

        request_agent_mode = bool(payload.get("agent_mode", False))

        system_prompt = self._compile_system_prompt(config, agent_mode=request_agent_mode)
        temp = float(config.get("temp", 0.7))
        top_p = float(config.get("top_p", 0.9))
        max_tokens = int(config.get("max_tokens", 2048))

        self._ephemeral_generation = bool(payload.get("ephemeral", False))
        thinking_mode = bool(payload.get("thinking_mode", False))

        if not self.conversation_history:
            self.reset_conversation(MASTER_PROMPT)

        system_entry = {"role": "system", "content": system_prompt}
        if self.conversation_history[0].get("role") != "system":
            self.conversation_history.insert(0, system_entry)
        else:
            self.conversation_history[0] = system_entry

        is_update = prompt.startswith("You were interrupted mid-generation.")
        if not self._ephemeral_generation and not is_update:
            self.conversation_history.append({"role": "user", "content": prompt})
            self._pending_user_index = len(self.conversation_history) - 1
            messages = list(self.conversation_history)
        else:
            messages = list(self.conversation_history)
            if not is_update:
                messages.append({"role": "user", "content": prompt})
            self._pending_user_index = None

        if thinking_mode and not self._ephemeral_generation:
            messages = list(messages)
            messages.append(
                {
                    "role": "system",
                    "content": "Use private reasoning to think step-by-step, then provide a concise final answer.",
                }
            )

        set_workspace_root()
        self._worker_seed_count = len(messages)
        self._worker_agent_mode = request_agent_mode

        self.worker = GeneratorWorker(
            self.llm,
            messages,
            temp,
            top_p,
            max_tokens,
            runtime=self._runtime if self._worker_agent_mode else None,
            agent_mode=self._worker_agent_mode,
        )
        self.worker.token.connect(self.sig_token)
        self.worker.trace.connect(self.sig_trace)
        self.worker.usage.connect(self._on_usage_update)
        self.worker.event.connect(self.sig_agent_event)
        self.worker.done.connect(self._on_gen_finish)
        self.worker.start()


    def runtime_command(self, command: str, payload: dict | None = None) -> dict:
        request = payload if isinstance(payload, dict) else {}
        result = self._runtime.runtime_command(command, request)
        self.sig_agent_event.emit({"event": "RUNTIME_COMMAND_RESULT", "request": request, "result": result})
        return result

    def stop_generation(self):
        if self._status == SystemStatus.LOADING and self.loader and self.loader.isRunning():
            self._load_cancel_requested = True
            self.sig_trace.emit("→ load cancel requested; will stop after initialization completes")
            return

        self._ephemeral_generation = False
        if self.worker and self.worker.isRunning():
            self.worker.requestInterruption()

    def _on_usage_update(self, count):
        self.sig_usage.emit(count)

    def _on_gen_finish(self, completed, assistant_text, loop_history):
        self.sig_trace.emit(f"[ENGINE] _on_gen_finish: completed={completed}, text_len={len(assistant_text)}")
        if completed and not self._ephemeral_generation:
            if self._worker_agent_mode and isinstance(loop_history, list):
                delta = loop_history[self._worker_seed_count :]
                for msg in delta:
                    if isinstance(msg, dict):
                        self.conversation_history.append(msg)
                if assistant_text:
                    self.conversation_history.append(
                        {"role": "assistant", "content": assistant_text}
                    )
            else:
                self.conversation_history.append({"role": "assistant", "content": assistant_text})
        self._pending_user_index = None
        self._ephemeral_generation = False
        self.sig_token.emit("\n")
        self.sig_finished.emit()
        self.set_status(SystemStatus.READY)

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
