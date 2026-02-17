import json
import re
import time

from PySide6.QtCore import QObject, QThread, Signal, QTimer
from core.state import AppState, SystemStatus
from core.llm_config import load_config, MASTER_PROMPT, AGENT_PROMPT
from engine.tools import TOOL_REGISTRY, set_workspace_root

MAX_AGENT_STEPS = 25
MAX_AGENT_TIMEOUT = 120

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"},
                    "offset": {"type": "integer", "description": "Line offset to start from"},
                    "limit": {"type": "integer", "description": "Max lines to return"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file, creating directories if needed",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List directory contents",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path"},
                    "pattern": {"type": "string", "description": "Glob filter pattern"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_search",
            "description": "Search file contents with regex",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search"},
                    "path": {"type": "string", "description": "File or directory to search in"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_cmd",
            "description": "Execute a shell command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": "Find and replace text in a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File to patch"},
                    "old": {"type": "string", "description": "Text to find"},
                    "new": {"type": "string", "description": "Replacement text"},
                },
                "required": ["path", "old", "new"],
            },
        },
    },
]


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

    def __init__(self, llm, messages, temp, top_p, max_tokens, agent_mode=False):
        super().__init__()
        self.llm = llm
        self.messages = list(messages)
        self.temp = temp
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.agent_mode = bool(agent_mode)

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

    def _extract_native_tool_calls(self, response: dict) -> list[dict]:
        choices = response.get("choices", [])
        if not choices:
            return []
        message = choices[0].get("message", {})
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            return []
        parsed = []
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            fn = call.get("function", {})
            name = fn.get("name")
            raw_args = fn.get("arguments", "{}")
            if not isinstance(name, str) or not name:
                continue
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except Exception:
                args = {}
            if not isinstance(args, dict):
                args = {}
            parsed.append({"name": name, "args": args})
        return parsed

    def _extract_prompt_tool_calls(self, text: str) -> list[dict]:
        if not text:
            return []
        blocks = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, flags=re.DOTALL)
        out = []
        for block in blocks:
            try:
                obj = json.loads(block)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            name = obj.get("name")
            args = obj.get("args", {})
            if isinstance(name, str) and isinstance(args, dict):
                out.append({"name": name, "args": args})
        return out

    def _try_chat_completion(self, messages, tools=None):
        kwargs = {
            "messages": messages,
            "temperature": self.temp,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        return self.llm.create_chat_completion(**kwargs)

    def _chat_once(self, messages):
        if not self.agent_mode:
            return self._try_chat_completion(messages), False

        try:
            return self._try_chat_completion(messages, tools=TOOL_SCHEMAS), True
        except Exception:
            self.trace.emit("[WORKER] native tool calling unavailable; using prompt tool blocks")
            return self._try_chat_completion(messages), False

    def _execute_tool_call(self, call: dict) -> dict:
        name = call.get("name")
        args = call.get("args", {})
        if not isinstance(name, str) or name not in TOOL_REGISTRY:
            return {"ok": False, "content": "", "error": f"tool not found: {name}"}
        if not isinstance(args, dict):
            return {"ok": False, "content": "", "error": "tool args must be an object"}
        try:
            result = TOOL_REGISTRY[name](args)
            return result.to_message()
        except Exception as exc:
            return {"ok": False, "content": "", "error": f"tool execution failed: {exc}"}

    def run(self):
        self.trace.emit(f"[WORKER] started: msgs={len(self.messages)}, temp={self.temp}, max_tokens={self.max_tokens}, agent_mode={self.agent_mode}")
        assistant_chunks = []
        completed = False
        total_generated = 0
        loop_messages = list(self.messages)
        step_count = 0
        loop_start = time.monotonic()
        last_assistant_text = ""

        try:
            while True:
                if self.isInterruptionRequested():
                    self.trace.emit("→ inference aborted")
                    break

                if self.agent_mode and (time.monotonic() - loop_start > MAX_AGENT_TIMEOUT):
                    self.trace.emit(f"[WORKER] agent timeout reached ({MAX_AGENT_TIMEOUT}s), forcing termination")
                    if last_assistant_text:
                        assistant_chunks = [last_assistant_text]
                    completed = True
                    break

                self.trace.emit("→ inference started")
                response, native_tool_mode = self._chat_once(loop_messages)
                assistant_text = self._extract_text(response)
                last_assistant_text = assistant_text or last_assistant_text

                tool_calls = self._extract_native_tool_calls(response)
                if not tool_calls:
                    tool_calls = self._extract_prompt_tool_calls(assistant_text)

                if self.agent_mode and tool_calls:
                    loop_messages.append({"role": "assistant", "content": assistant_text})
                    for call in tool_calls:
                        if self.isInterruptionRequested():
                            self.trace.emit("→ inference aborted")
                            break
                        name = call.get("name", "unknown")
                        self.token.emit(f"\n[agent] running {name}...\n")
                        result = self._execute_tool_call(call)
                        if native_tool_mode:
                            loop_messages.append(
                                {
                                    "role": "tool",
                                    "content": json.dumps(result, ensure_ascii=False),
                                    "name": str(name),
                                }
                            )
                        else:
                            loop_messages.append(
                                {
                                    "role": "user",
                                    "content": f"[Tool Result: {name}]\n{json.dumps(result, ensure_ascii=False)}",
                                }
                            )
                        self.trace.emit(f"[WORKER] tool={name} ok={result.get('ok')}")
                    if self.isInterruptionRequested():
                        break

                    step_count += 1
                    if step_count >= MAX_AGENT_STEPS:
                        self.trace.emit(f"[WORKER] agent step limit reached ({MAX_AGENT_STEPS}), forcing termination")
                        if last_assistant_text:
                            assistant_chunks = [last_assistant_text]
                        completed = True
                        break

                    if native_tool_mode:
                        self.trace.emit("[WORKER] tool loop iteration complete (native tools)")
                    else:
                        self.trace.emit("[WORKER] tool loop iteration complete (prompt blocks)")
                    continue

                if assistant_text:
                    assistant_chunks.append(assistant_text)
                    self.token.emit(assistant_text)
                    total_generated += len(assistant_text)
                    self.usage.emit(total_generated)

                completed = not self.isInterruptionRequested()
                if completed:
                    self.trace.emit("→ inference complete")
                break

        except Exception as e:
            self.trace.emit(f"[WORKER] EXCEPTION: {e}")
            self.trace.emit(f"<span style='color:red'>ERROR: {e}</span>")
        finally:
            self.trace.emit(f"[WORKER] finished: completed={completed}, chunks={len(assistant_chunks)}")
            self.done.emit(completed, "".join(assistant_chunks), loop_messages)


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

    def _compile_system_prompt(self, config):
        base_prompt = AGENT_PROMPT if bool(config.get("agent_mode", False)) else MASTER_PROMPT
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

        system_prompt = self._compile_system_prompt(config)
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

        set_workspace_root(payload.get("workspace_root") if isinstance(payload, dict) else None)
        self._worker_seed_count = len(messages)
        self._worker_agent_mode = bool(config.get("agent_mode", False))

        self.worker = GeneratorWorker(
            self.llm,
            messages,
            temp,
            top_p,
            max_tokens,
            agent_mode=self._worker_agent_mode,
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
