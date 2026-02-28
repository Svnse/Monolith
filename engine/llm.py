import hashlib
import json
import os
import re
import time
from threading import Lock
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal, QTimer

from core.llm_config import (
    MASTER_PROMPT,
)
from core.state import AppState, SystemStatus
from engine.llm_modes.chat import ChatModeStrategy
from engine.tools import set_workspace_root, stop_active_process_groups

MAX_AGENT_STEPS = 25


class AgentRuntime:
    pass


ContractFactory = None
ProtocolAdapter = None
_AGENT_STACK_AVAILABLE = False


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class AgentMessage:
    role: str
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None


def get_profile(_model_profile_id: str):
    raise RuntimeError("agent stack disabled")


def _find_first_tool_json_end(*_args, **_kwargs):
    return None


_TASK_QUERY_BATCH_OPEN_RE = re.compile(r"<task_query_batch>", re.IGNORECASE)
_TASK_QUERY_BATCH_CLOSE_RE = re.compile(r"</task_query_batch>", re.IGNORECASE)

_BARE_JSON_DECODER = json.JSONDecoder()


def _find_first_bare_json_object_end(text: str) -> int | None:
    """
    Return the end offset of the first complete top-level JSON object in text,
    regardless of its keys. Used by the observe/commit streaming guard to stop
    as soon as the model finishes its review JSON, before it starts emitting
    trailing \\n tokens (Qwen2.5 / models that decode <|im_end|> as a newline).
    """
    if not isinstance(text, str):
        return None
    cursor = 0
    length = len(text)
    while cursor < length:
        start = text.find("{", cursor)
        if start < 0:
            break
        try:
            parsed, end = _BARE_JSON_DECODER.raw_decode(text, start)
        except json.JSONDecodeError:
            cursor = start + 1
            continue
        if isinstance(parsed, dict) and parsed:
            return end
        cursor = max(start + 1, end)
    return None


def _find_first_task_query_batch_end(text: str) -> int | None:
    """
    Return end index of the first complete <task_query_batch>...</task_query_batch>
    block when payload is valid JSON. Used to stop streaming decode early in
    required task-query routes and avoid long speculative trailing prose.
    """
    if not isinstance(text, str) or not text:
        return None
    open_match = _TASK_QUERY_BATCH_OPEN_RE.search(text)
    if open_match is None:
        return None
    close_match = _TASK_QUERY_BATCH_CLOSE_RE.search(text, open_match.end())
    if close_match is None:
        return None
    payload = text[open_match.end():close_match.start()].strip()
    if not payload:
        return None
    try:
        parsed = json.loads(payload)
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    return close_match.end()


def normalize_openai_response(raw: dict) -> AgentMessage:
    """Legacy normalizer — kept for backward compatibility with non-adapter paths."""
    choices = raw.get("choices", []) if isinstance(raw, dict) else []
    message = choices[0].get("message", {}) if choices else {}
    role = message.get("role", "assistant")
    if role not in {"system", "user", "assistant", "tool"}:
        role = "assistant"

    content = message.get("content")
    normalized_content: str | None
    if isinstance(content, str):
        normalized_content = content
    elif isinstance(content, list):
        normalized_content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    else:
        normalized_content = None

    tool_calls_raw = message.get("tool_calls")
    normalized_calls: list[ToolCall] = []
    if isinstance(tool_calls_raw, list):
        for idx, item in enumerate(tool_calls_raw):
            if not isinstance(item, dict):
                continue
            fn = item.get("function", {}) if isinstance(item.get("function"), dict) else {}
            name = fn.get("name")
            if not isinstance(name, str) or not name:
                continue
            arguments = fn.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    parsed = json.loads(arguments)
                    arguments = parsed if isinstance(parsed, dict) else {}
                except Exception:
                    arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}
            normalized_calls.append(ToolCall(id=str(item.get("id") or f"call_{idx}"), name=name, arguments=arguments))

    return AgentMessage(
        role=role,
        content=normalized_content,
        tool_calls=normalized_calls or None,
        tool_call_id=message.get("tool_call_id") if isinstance(message.get("tool_call_id"), str) else None,
        name=message.get("name") if isinstance(message.get("name"), str) else None,
    )


class ModelLoader(QThread):
    trace = Signal(str)
    loaded = Signal(object, int)   # renamed from 'finished' to avoid shadowing QThread.finished
    error = Signal(str)

    def __init__(self, path, n_ctx=8192, n_gpu_layers=-1):
        super().__init__()
        self.path = path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers

    # Minimum context size before giving up on fallback retries
    _MIN_CTX = 512

    def run(self):
        try:
            try:
                from llama_cpp import Llama
            except ImportError as exc:
                raise RuntimeError(
                    "llama-cpp-python is not installed. Install it to use the local LLM engine."
                ) from exc

            n_ctx = self.n_ctx
            llm_instance = None

            while n_ctx >= self._MIN_CTX:
                self.trace.emit(f"→ init backend: {self.path} (n_ctx={n_ctx})")
                try:
                    llm_instance = Llama(
                        model_path=self.path,
                        n_ctx=n_ctx,
                        n_gpu_layers=self.n_gpu_layers,
                        verbose=False,
                    )
                    break  # success
                except Exception as ctx_err:
                    err_lower = str(ctx_err).lower()
                    # Only retry on context-allocation failures
                    if "llama_context" in err_lower or "kv cache" in err_lower or "memory" in err_lower:
                        prev = n_ctx
                        n_ctx = max(n_ctx // 2, self._MIN_CTX) if n_ctx > self._MIN_CTX else 0
                        if n_ctx > 0:
                            self.trace.emit(f"→ ctx alloc failed at n_ctx={prev}, retrying with n_ctx={n_ctx}")
                            continue
                    raise  # non-recoverable or all retries exhausted

            if llm_instance is None:
                raise RuntimeError(
                    f"Failed to create llama_context: n_ctx={self.n_ctx} is too large. "
                    f"Tried down to {self._MIN_CTX}. Reduce ctx_limit or free VRAM."
                )

            model_ctx_length = llm_instance._model.n_ctx_train()
            self.loaded.emit(llm_instance, model_ctx_length)
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
        mode_runner=None,
    ):
        super().__init__()
        self.llm = llm
        self.messages = list(messages)
        self.temp = temp
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.agent_mode = bool(agent_mode)
        self.runtime = runtime
        self._mode_runner = mode_runner
        self._first_infer_complete = False
        try:
            self._first_infer_cap = max(0, int(os.environ.get("MONOLITH_FIRST_INFER_MAX_TOKENS", "4096")))
        except Exception:
            self._first_infer_cap = 4096
        stream_mode = str(os.environ.get("MONOLITH_STREAM_MODE", "char")).strip().lower()
        self._stream_mode = stream_mode if stream_mode in {"char", "chunk"} else "char"
        self._stream_token_buffer: list[str] = []
        self._stream_buffer_chars: int = 0
        try:
            self._stream_emit_chars = max(16, int(os.environ.get("MONOLITH_STREAM_EMIT_CHARS", "64")))
        except Exception:
            self._stream_emit_chars = 64
        try:
            self._stream_emit_interval_ms = max(20, int(os.environ.get("MONOLITH_STREAM_EMIT_MS", "60")))
        except Exception:
            self._stream_emit_interval_ms = 60
        self._stream_last_emit_ts = time.monotonic()

    def _emit_text_stream(self, text: str) -> None:
        if not isinstance(text, str) or not text:
            return
        if self._stream_mode == "char":
            for char in text:
                self.token.emit(char)
            self.usage.emit(len(text))
            return
        self.token.emit(text)
        self.usage.emit(len(text))

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

    # Stop sequences to prevent model from emitting prompt scaffolding tags
    _STOP_SEQUENCES = ["</response>", "</answer>", "</output>", "<|end|>", "<|im_end|>"]

    def _try_chat_completion(self, messages, tools=None, stream: bool = False):
        effective_max_tokens = self.max_tokens
        using_first_infer_cap = False
        if self.agent_mode and not self._first_infer_complete and self._first_infer_cap > 0:
            if effective_max_tokens > self._first_infer_cap:
                effective_max_tokens = self._first_infer_cap
                using_first_infer_cap = True

        kwargs = {
            "messages": messages,
            "temperature": self.temp,
            "top_p": self.top_p,
            "max_tokens": effective_max_tokens,
            "stream": stream,
            "stop": self._STOP_SEQUENCES,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        # Phase 5: thread grammar spec from contract if available
        contract = getattr(self, "_contract", None)
        if contract is not None and getattr(contract, "grammar_profile", None):
            try:
                # Legacy protocol adapter was removed from the active stack.
                # Keep grammar wiring fail-safe by treating profile lookup as unavailable.
                gp = None
                if gp is not None and gp.grammar_spec and gp.grammar_type == "bnf":
                    from llama_cpp import LlamaGrammar
                    kwargs["grammar"] = LlamaGrammar.from_string(gp.grammar_spec)
            except Exception:
                pass  # graceful fallback if grammar not supported

        if using_first_infer_cap:
            self.trace.emit(
                f"[WORKER] first infer token cap active: requested={self.max_tokens}, capped={effective_max_tokens}"
            )

        try:
            return self.llm.create_chat_completion(**kwargs)
        finally:
            if self.agent_mode and not self._first_infer_complete:
                self._first_infer_complete = True

    def _chat_once_text(self, messages: list[dict]) -> str:
        response = self._try_chat_completion(messages, stream=False)
        return self._extract_text(response)

    def _chat_once_agent(self, messages: list[dict], tools: list[dict]) -> AgentMessage:
        """Legacy agent path — returns normalized AgentMessage (no adapter)."""
        response = self._try_chat_completion(messages, tools=tools, stream=False)
        return normalize_openai_response(response)

    def _merge_stream_tool_calls(self, merged: list[dict], delta_calls: list) -> None:
        for item in delta_calls:
            if not isinstance(item, dict):
                continue
            idx = item.get("index")
            if not isinstance(idx, int) or idx < 0:
                idx = len(merged)

            while len(merged) <= idx:
                merged.append(
                    {
                        "id": f"call_{len(merged)}",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                )

            slot = merged[idx]
            if isinstance(item.get("id"), str) and item["id"]:
                slot["id"] = item["id"]
            if isinstance(item.get("type"), str) and item["type"]:
                slot["type"] = item["type"]

            fn_delta = item.get("function")
            if isinstance(fn_delta, dict):
                fn_slot = slot.setdefault("function", {})
                fn_name = fn_delta.get("name")
                if isinstance(fn_name, str) and fn_name:
                    fn_slot["name"] = fn_name
                fn_args = fn_delta.get("arguments")
                if isinstance(fn_args, str) and fn_args:
                    prev = fn_slot.get("arguments", "")
                    fn_slot["arguments"] = f"{prev}{fn_args}"

    def _chat_once_agent_streaming_raw(self, messages: list[dict], tools: list[dict]) -> dict:
        stream = self._try_chat_completion(messages, tools=tools, stream=True)
        merged_tool_calls: list[dict] = []
        merged_content_parts: list[str] = []
        merged_role = "assistant"
        finish_reason = None
        usage_payload = None
        allowed_tool_names = set()
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            fn = tool.get("function")
            if not isinstance(fn, dict):
                continue
            name = fn.get("name")
            if isinstance(name, str) and name:
                allowed_tool_names.add(name)

        for chunk in stream:
            if self.isInterruptionRequested():
                raise RuntimeError("generation interrupted")

            if not isinstance(chunk, dict):
                continue

            usage = chunk.get("usage")
            if isinstance(usage, dict):
                usage_payload = usage

            choices = chunk.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            choice = choices[0] if isinstance(choices[0], dict) else {}
            delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}

            role = delta.get("role")
            if isinstance(role, str) and role:
                merged_role = role

            token = delta.get("content")
            if isinstance(token, str) and token:
                merged_content_parts.append(token)
                if self.runtime is not None and getattr(self.runtime, "_debug_enabled", False) and getattr(self.runtime, "_debug_emit_tokens", False):
                    self.event.emit(
                        {
                            "event": "LLM_TOKEN",
                            "data": token,
                            "timestamp": time.time(),
                            "channel": "raw_infer",
                        }
                    )
                # Step-2 stop guard for raw JSON tool-call emission:
                # once a valid bare tool-call JSON is complete, stop this infer
                # stream before trailing speculative prose is generated.
                joined = "".join(merged_content_parts)
                task_batch_end = _find_first_task_query_batch_end(joined)
                if task_batch_end is not None:
                    merged_content_parts = [joined[:task_batch_end]]
                    finish_reason = finish_reason or "task_query_batch_recovered"
                    break
                tool_json_end = _find_first_tool_json_end(
                    joined,
                    allowed_tool_names=allowed_tool_names if allowed_tool_names else None,
                )
                if tool_json_end is not None:
                    merged_content_parts = [joined[:tool_json_end]]
                    finish_reason = finish_reason or "raw_tool_call_recovered"
                    break
                # Guard 3: no-tools path (observe/commit review JSON).
                # Models like Qwen2.5 decode <|im_end|> as a newline token that
                # passes through streaming content, causing hundreds of \n tokens
                # after the closing }. Cut the stream as soon as a complete JSON
                # object is present and no tools are expected.
                if not allowed_tool_names:
                    bare_json_end = _find_first_bare_json_object_end(joined)
                    if bare_json_end is not None:
                        merged_content_parts = [joined[:bare_json_end]]
                        finish_reason = finish_reason or "bare_json_complete"
                        break

            delta_tool_calls = delta.get("tool_calls")
            if isinstance(delta_tool_calls, list):
                self._merge_stream_tool_calls(merged_tool_calls, delta_tool_calls)

            chunk_finish = choice.get("finish_reason")
            if isinstance(chunk_finish, str):
                finish_reason = chunk_finish

        message: dict = {
            "role": merged_role,
            "content": "".join(merged_content_parts) if merged_content_parts else None,
        }
        if merged_tool_calls:
            message["tool_calls"] = merged_tool_calls

        response: dict = {
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ]
        }
        if isinstance(usage_payload, dict):
            response["usage"] = usage_payload
        return response

    def _chat_once_agent_raw(self, messages: list[dict], tools: list[dict]) -> dict:
        """Adapter-aware agent path — returns raw dict for ProtocolAdapter."""
        try:
            return self._chat_once_agent_streaming_raw(messages, tools)
        except Exception as exc:
            if self.isInterruptionRequested():
                raise
            self.trace.emit(f"[WORKER] stream fallback: {exc}")
            return self._try_chat_completion(messages, tools=tools, stream=False)

    def _emit_runtime_event(self, payload: dict) -> None:
        self.event.emit(payload)
        event_name = payload.get("event")
        if event_name == "LLM_TOKEN":
            token = payload.get("data", "")
            if isinstance(token, str) and token:
                if self._stream_mode == "char":
                    for char in token:
                        self.token.emit(char)
                    self.usage.emit(len(token))
                    return
                self._stream_token_buffer.append(token)
                self._stream_buffer_chars += len(token)
                now = time.monotonic()
                elapsed_ms = (now - self._stream_last_emit_ts) * 1000.0
                if (
                    self._stream_buffer_chars >= self._stream_emit_chars
                    or elapsed_ms >= self._stream_emit_interval_ms
                ):
                    self._flush_stream_token_buffer()

    def _flush_stream_token_buffer(self, *, force: bool = False) -> None:
        if not self._stream_token_buffer:
            return
        if not force and self._stream_buffer_chars <= 0:
            return
        chunk = "".join(self._stream_token_buffer)
        self._stream_token_buffer.clear()
        self._stream_buffer_chars = 0
        self._stream_last_emit_ts = time.monotonic()
        if chunk:
            self.token.emit(chunk)
            self.usage.emit(len(chunk))


    def run(self):
        self.trace.emit(
            f"[WORKER] started: msgs={len(self.messages)}, temp={self.temp}, max_tokens={self.max_tokens}, agent_mode={self.agent_mode}"
        )
        completed = False
        assistant_text = ""
        loop_history = list(self.messages)

        try:
            mode_runner = self._mode_runner
            if mode_runner is None:
                from engine.llm_modes.chat_worker import ChatWorkerExecutionMode
                mode_runner = ChatWorkerExecutionMode()
            completed, assistant_text, loop_history = mode_runner.run(self)

            if completed:
                self.trace.emit("→ inference complete")
        except Exception as e:
            self.trace.emit(f"[WORKER] EXCEPTION: {e}")
            import traceback
            self.trace.emit(f"<span style='color:red'>ERROR: {e}\n{traceback.format_exc()}</span>")
        finally:
            self._flush_stream_token_buffer(force=True)
            self.trace.emit(f"[WORKER] finished: completed={completed}, text_len={len(assistant_text)}")
            self.done.emit(completed, assistant_text, loop_history)
            # Note: Worker cleanup handled by LLMEngine _on_gen_finish


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
        self._runtime = None
        self._mode_chat = ChatModeStrategy()

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

        # Clean up any lingering loader from a previous load cycle
        if self.loader is not None:
            if self.loader.isRunning():
                self.loader.wait(2000)
            self._disconnect_loader(self.loader)
            self.loader = None

        self.set_status(SystemStatus.LOADING)
        self._load_cancel_requested = False
        n_ctx = min(self.ctx_limit, self.model_ctx_length) if self.model_ctx_length else self.ctx_limit
        self.loader = ModelLoader(model_path, n_ctx)
        self.loader.trace.connect(self.sig_trace)
        self.loader.error.connect(self._on_load_error)
        self.loader.loaded.connect(self._on_load_success)
        # Connect cleanup to QThread's built-in finished signal — fires AFTER run() returns.
        # This prevents "QThread: Destroyed while thread is still running" by ensuring
        # self.loader = None only happens after the thread has fully stopped.
        self.loader.finished.connect(self._cleanup_loader)
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
            # Note: self.loader cleanup deferred to _cleanup_loader via QThread.finished
            return

        self.llm = llm_instance
        self.model_ctx_length = int(model_ctx_length)
        # Clamp to the actual allocated context size (may be smaller after fallback)
        try:
            actual_ctx = int(llm_instance.n_ctx())
        except Exception:
            actual_ctx = self.model_ctx_length
        self.ctx_limit = min(self.ctx_limit, self.model_ctx_length, actual_ctx)

        # Phase 5: compute model fingerprint
        try:
            model_path = self.model_path or ""
            file_size = os.path.getsize(model_path) if model_path and os.path.exists(model_path) else 0
            self._model_fingerprint = hashlib.sha256(
                f"{model_path}:{file_size}".encode()
            ).hexdigest()
        except Exception:
            self._model_fingerprint = ""

        self.sig_model_capabilities.emit(
            {
                "model_ctx_length": self.model_ctx_length,
                "ctx_limit": self.ctx_limit,
                "actual_ctx": actual_ctx,
            }
        )
        self.model_loaded = True
        self.set_status(SystemStatus.READY)
        self.reset_conversation(MASTER_PROMPT)

        self.sig_trace.emit("→ system online")
        # Note: self.loader cleanup deferred to _cleanup_loader via QThread.finished

    def _on_load_error(self, err_msg):
        self.sig_trace.emit(f"<span style='color:red'>{err_msg}</span>")
        if self._shutdown_requested:
            self.set_status(SystemStatus.READY)
        else:
            self.set_status(SystemStatus.ERROR)
        # Note: self.loader cleanup deferred to _cleanup_loader via QThread.finished

    def _cleanup_loader(self):
        """Clean up loader thread reference.

        Connected to QThread.finished (not our custom loaded/error signals),
        so this fires AFTER run() has fully returned. We still defer via
        QTimer.singleShot(0) to ensure we're back on the main event loop
        before destroying the QThread object.
        """
        def _deferred_cleanup():
            loader = self.loader
            if loader is not None:
                self._disconnect_loader(loader)
                self.loader = None
        QTimer.singleShot(0, _deferred_cleanup)

    def _disconnect_loader(self, loader: ModelLoader) -> None:
        """Safely disconnect all signals from a loader instance."""
        for sig_name in ("trace", "loaded", "error", "finished"):
            sig = getattr(loader, sig_name, None)
            if sig is not None:
                try:
                    sig.disconnect()
                except (TypeError, RuntimeError):
                    pass

    def unload_model(self):
        if self._status == SystemStatus.LOADING and self.loader and self.loader.isRunning():
            self._load_cancel_requested = True
            self.sig_trace.emit("→ unload requested during load; will cancel when init completes")
            return

        if self._status == SystemStatus.RUNNING:
            self.sig_trace.emit("ERROR: Cannot unload while generating.")
            return

        self.set_status(SystemStatus.UNLOADING)
        if self.llm is not None:
            try:
                if hasattr(self.llm, 'close'):
                    self.llm.close()
            except Exception:
                pass
            self.llm = None
        import gc
        gc.collect()
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

    def _compile_system_prompt(
        self,
        config,
    ):
        base_prompt = MASTER_PROMPT
        tags = config.get("behavior_tags", [])
        cleaned = [tag.strip() for tag in tags if isinstance(tag, str) and tag.strip()]
        if not cleaned:
            return base_prompt
        return f"{base_prompt}\n\n[BEHAVIOR TAGS]\n" + "\n".join(cleaned)

    def _create_generator_worker(self, messages, temp, top_p, max_tokens):
        return GeneratorWorker(
            self.llm,
            messages,
            temp,
            top_p,
            max_tokens,
            runtime=None,
            agent_mode=False,
            mode_runner=self._mode_chat.create_worker_mode(),
        )

    def _build_execution_contract(
        self,
        *,
        request_agent_mode: bool,
        payload: dict,
        prompt: str,
        model_profile_id: str,
    ):
        return None

    def generate(self, payload: dict):
        # Clean up any previous worker that has finished
        if self.worker is not None:
            if not self.worker.isRunning():
                # Worker finished but not cleaned up - safe to delete reference
                self.worker = None
            else:
                # Worker still running - this shouldn't happen, but handle it
                self.sig_trace.emit("[ENGINE] WARNING: previous worker still running, stopping it")
                self.worker.requestInterruption()
                self.worker.wait(1000)
                if self.worker.isRunning():
                    self.worker.terminate()
                    self.worker.wait(500)
                self.worker = None
        
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
        spec = self._mode_chat.prepare_generation(self, payload if isinstance(payload, dict) else {})
        messages = spec.messages
        model_profile_id = spec.model_profile_id
        temp = spec.temp
        top_p = spec.top_p
        max_tokens = spec.max_tokens

        # Clean up any existing worker before starting a new one (Bug 2 fix)
        if self.worker is not None and self.worker.isRunning():
            self.worker.requestInterruption()
            self.worker.wait(3000)
        if self.loader is not None and self.loader.isRunning():
            self.loader.wait(3000)

        set_workspace_root()
        self._worker_seed_count = len(messages)
        self._worker_agent_mode = False

        contract = None

        self.worker = self._create_generator_worker(messages, temp, top_p, max_tokens)
        # Pass profile and contract to worker for adapter setup
        self.worker._model_profile_id = model_profile_id
        self.worker._contract = contract

        self.worker.token.connect(self.sig_token)
        self.worker.trace.connect(self.sig_trace)
        self.worker.usage.connect(self._on_usage_update)
        self.worker.event.connect(self.sig_agent_event)
        self.worker.done.connect(self._on_gen_finish)
        self.worker.start()


    def runtime_command(self, command: str, payload: dict | None = None) -> dict:
        request = payload if isinstance(payload, dict) else {}
        result = {
            "ok": False,
            "error": "runtime_command is only available on AgentLLMEngine",
            "command": command,
            "request": request,
        }
        self.sig_trace.emit(f"[ENGINE] runtime_command ignored: {command}")
        return result

    def stop_generation(self):
        if self._status == SystemStatus.LOADING and self.loader and self.loader.isRunning():
            self._load_cancel_requested = True
            self.sig_trace.emit("-> load cancel requested; will stop after initialization completes")
            return

        self._ephemeral_generation = False
        if self.worker and self.worker.isRunning():
            self.worker.requestInterruption()

        try:
            stop_summary = stop_active_process_groups(grace_timeout_s=1.0)
            if int(stop_summary.get("active_before", 0) or 0) > 0:
                self.sig_trace.emit(
                    "[ENGINE] stop: terminated subprocess groups "
                    f"(terminated={stop_summary.get('terminated', 0)}, "
                    f"force_killed={stop_summary.get('force_killed', 0)})"
                )
        except Exception as exc:
            self.sig_trace.emit(f"[ENGINE] stop: subprocess cleanup warning: {exc}")

    def force_stop(self):
        """Force terminate running generation thread."""
        self.stop_generation()
        if self.worker and self.worker.isRunning():
            self.worker.requestInterruption()
            if hasattr(self.worker, 'runtime') and self.worker.runtime:
                self.worker.runtime._force_terminate_after_next_inference = True
            if not self.worker.wait(2000):
                self.worker.terminate()
                self.worker.wait(1000)
        self.sig_status.emit(SystemStatus.READY)

    def _on_usage_update(self, count):
        self.sig_usage.emit(count)

    def _on_gen_finish(self, completed, assistant_text, loop_history):
        self.sig_trace.emit(f"[ENGINE] _on_gen_finish: completed={completed}, text_len={len(assistant_text)}")
        self._mode_chat.on_generation_finished(self, completed, assistant_text, loop_history)
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
            if not self.worker.wait(3000):  # Wait up to 3 seconds
                self.worker.terminate()
                self.worker.wait(1000)
            # Explicitly disconnect signals before deleting reference
            try:
                self.worker.token.disconnect()
                self.worker.trace.disconnect()
                self.worker.usage.disconnect()
                self.worker.event.disconnect()
                self.worker.done.disconnect()
            except Exception:
                pass
            self.worker = None

        if self.loader:
            if self.loader.isRunning():
                self._load_cancel_requested = True
                if not self.loader.wait(1000):
                    self.loader.terminate()
                    self.loader.wait(500)
            self._disconnect_loader(self.loader)
            self.loader = None
