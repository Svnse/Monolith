import json
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone

from PySide6.QtCore import QObject, QThread, Signal, QTimer, Slot
from core.state import AppState, SystemStatus
from core.llm_config import load_config
from core.llm_prompt import load_master_prompt
from core.message_interceptors import apply_interceptors
from core.context_profiles import build_profiled_tool_catalog
from core.ctx_window import resolve_cloud_window

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


def build_tool_call_grammar() -> str:
    """Return a GBNF grammar string that constrains output to a single <tool_call> JSON block.

    Use this for forced-tool-call contexts (e.g. llm_call skill requesting a
    structured response).  Not suitable for free-text + tool-call mixed output.
    """
    return r"""
root        ::= "<tool_call>" ws json-obj ws "</tool_call>"
ws          ::= ([ \t\n\r])*
json-obj    ::= "{" ws (json-pair ("," ws json-pair)*)? ws "}"
json-pair   ::= json-str ws ":" ws json-val
json-val    ::= json-str | json-num | json-obj | json-arr | "true" | "false" | "null"
json-str    ::= "\"" ([^"\\] | "\\" ["\\/bfnrt] | "\\u" [0-9a-fA-F]{4})* "\""
json-num    ::= "-"? ([0-9] | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
json-arr    ::= "[" ws (json-val ("," ws json-val)*)? ws "]"
""".strip()


def _normalize_base_url(base_url: str) -> str:
    return (base_url or "").strip().rstrip("/")


def _chat_completions_url(base_url: str) -> str:
    base = _normalize_base_url(base_url)
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _models_url(base_url: str) -> str:
    base = _normalize_base_url(base_url)
    if base.endswith("/models"):
        return base
    if base.endswith("/v1"):
        return f"{base}/models"
    return f"{base}/v1/models"


def _extract_ctx_length_from_model_item(item: dict) -> int | None:
    """Return a conservative effective context length from a /models item."""
    if not isinstance(item, dict):
        return None
    meta = item.get("meta", {}) if isinstance(item.get("meta"), dict) else {}
    candidates: list[int] = []
    for source in (meta, item):
        for key in ("n_ctx", "context_length", "max_context_length", "n_ctx_train"):
            raw = source.get(key)
            if raw is None:
                continue
            try:
                value = int(raw)
            except (TypeError, ValueError):
                continue
            if value > 0:
                candidates.append(value)
    if not candidates:
        return None
    return min(candidates)


class OpenAICompatLLM:
    def __init__(self, base_url: str, api_key: str | None, model: str, timeout: int = 300):
        self.base_url = _normalize_base_url(base_url)
        self.api_key = api_key or ""
        self.model = model
        self.timeout = timeout
        # Active streaming response. Held so cancel() can close the socket
        # and break the worker's `for raw_line in resp:` loop -- without
        # this, requestInterruption() only sets a flag and the worker
        # stays blocked inside urllib until the next chunk arrives.
        self._active_response = None

    def cancel(self) -> None:
        """Close the active streaming response so the worker can exit."""
        resp = self._active_response
        self._active_response = None
        if resp is None:
            return
        try:
            resp.close()
        except Exception:
            # urllib may raise on close after partial read; we don't care --
            # the goal is to unblock the worker iteration.
            pass

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def list_models(self) -> dict:
        url = _models_url(self.base_url)
        req = urllib.request.Request(url, headers=self._headers(), method="GET")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = resp.read()
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {}

    def _open_request(self, payload: dict) -> urllib.response.addinfourl:
        url = _chat_completions_url(self.base_url)
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=self._headers(), method="POST")
        return urllib.request.urlopen(req, timeout=self.timeout)

    def create_chat_completion(self, **kwargs):
        payload = dict(kwargs)
        payload["model"] = self.model

        try:
            resp = self._open_request(payload)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            if "enable_thinking" in body:
                payload.pop("enable_thinking", None)
                resp = self._open_request(payload)
            else:
                raise RuntimeError(f"HTTP {exc.code}: {body or exc.reason}") from exc

        self._active_response = resp
        stream = bool(payload.get("stream", False))
        if not stream:
            body = resp.read()
            resp.close()
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError:
                parsed = {}
            content = (
                parsed.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            if content:
                yield {"choices": [{"delta": {"content": content}}]}
            return

        try:
            buffered_lines: list[bytes] = []
            saw_stream_chunk = False
            decode_failures = 0
            data_buf = ""
            try:
                line_iter = iter(resp)
            except Exception:
                line_iter = iter([])
            while True:
                try:
                    raw_line = next(line_iter)
                except StopIteration:
                    break
                except (urllib.error.URLError, ValueError, OSError, AttributeError):
                    # Response was closed externally (cancel()) or the socket
                    # died mid-iteration. urllib raises AttributeError
                    # ('NoneType' has no attribute 'peek') when fp is yanked
                    # from under it; treat any of these as graceful end of
                    # stream rather than a hard exception.
                    break
                buffered_lines.append(raw_line)
                line = raw_line.decode("utf-8", errors="ignore").rstrip("\r\n")
                # Empty line marks end of an SSE event. Process whatever we
                # have accumulated for this event before moving on.
                if not line:
                    if data_buf:
                        if data_buf == "[DONE]":
                            data_buf = ""
                            break
                        try:
                            chunk = json.loads(data_buf)
                            saw_stream_chunk = True
                            yield chunk
                        except json.JSONDecodeError:
                            # Don't drop in silence -- record so the user can
                            # see in the trace if this is causing missing
                            # tokens. The body is truncated to keep the log
                            # readable.
                            decode_failures += 1
                            preview = data_buf[:120].replace("\n", " ")
                            yield {
                                "_trace": (
                                    f"[SSE] dropped malformed chunk "
                                    f"({decode_failures} so far): {preview!r}"
                                )
                            }
                        data_buf = ""
                    continue
                # Multi-line `data:` per SSE spec is joined with newlines.
                if line.startswith("data:"):
                    payload_part = line[len("data:"):].lstrip()
                    data_buf = payload_part if not data_buf else f"{data_buf}\n{payload_part}"
                # Non-data lines (event:, id:, retry:, comments) are ignored.
            # Trailing event without a terminating blank line.
            if data_buf and data_buf != "[DONE]":
                try:
                    chunk = json.loads(data_buf)
                    saw_stream_chunk = True
                    yield chunk
                except json.JSONDecodeError:
                    decode_failures += 1
                    preview = data_buf[:120].replace("\n", " ")
                    yield {
                        "_trace": (
                            f"[SSE] dropped trailing malformed chunk: {preview!r}"
                        )
                    }
            if not saw_stream_chunk and buffered_lines:
                try:
                    parsed = json.loads(b"".join(buffered_lines))
                except json.JSONDecodeError:
                    parsed = {}
                content = (
                    parsed.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                if content:
                    yield {"choices": [{"delta": {"content": content}}]}
        finally:
            try:
                resp.close()
            except Exception:
                pass
            # Allow cancel() to be called again without hitting a closed socket.
            self._active_response = None


def make_cloud_llm(base_url: str, api_key: str | None, model: str, timeout: int = 300):
    """Pick the cloud client by api_base URL: the native Anthropic adapter when the URL is an
    Anthropic host, else the OpenAI-compatible client (today's path, byte-identical). Any
    detection/import failure falls back to OpenAICompatLLM so this branch can never break the
    cloud path. Both clients expose the identical interface, so callers are unchanged."""
    try:
        from engine.anthropic_llm import AnthropicLLM, is_anthropic_url
        if is_anthropic_url(base_url):
            return AnthropicLLM(base_url, api_key, model, timeout=timeout)
    except Exception:
        pass
    return OpenAICompatLLM(base_url, api_key, model, timeout=timeout)


class HttpModelLoader(QThread):
    trace = Signal(str)
    finished = Signal(object, object)
    error = Signal(str)

    def __init__(self, base_url: str, api_key: str | None, model: str, provider: str = "openai"):
        super().__init__()
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.provider = provider

    def run(self):
        try:
            if not self.base_url:
                raise RuntimeError("Missing API base URL.")
            if not self.model:
                raise RuntimeError("Missing model name.")
            self.trace.emit(f"→ init backend: {self.provider} ({self.base_url})")
            llm_instance = make_cloud_llm(self.base_url, self.api_key, self.model)
            ctx_length = None
            try:
                models_data = llm_instance.list_models()
                # Native llama-server returns ctx length in model metadata
                for item in (models_data.get("data") or []):
                    if isinstance(item, dict):
                        ctx_length = _extract_ctx_length_from_model_item(item)
                        if ctx_length:
                            break
            except Exception:
                pass
            self.finished.emit(llm_instance, ctx_length)
        except Exception as e:
            self.error.emit(f"Load Failed: {str(e)}")


class GGUFRuntime(QObject):
    trace = Signal(str)
    loaded = Signal(int)
    load_error = Signal(str)
    token = Signal(str)
    done = Signal(bool, str)
    usage = Signal(int)
    unloaded = Signal()

    def __init__(self):
        super().__init__()
        self._llm = None
        self._interrupt_requested = False

    @Slot(str, int, int)
    def load_model(self, path: str, n_ctx: int, n_gpu_layers: int) -> None:
        try:
            try:
                from llama_cpp import Llama
            except ImportError as exc:
                raise RuntimeError(
                    "llama-cpp-python is not installed. Install it to use the local LLM engine."
                ) from exc
            self.trace.emit(f"-> init backend: {path}")
            self._interrupt_requested = False
            # Build the new backend instance before swapping so a failed load
            # does not drop the currently loaded model.
            next_llm = Llama(
                model_path=path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
            )
            model_ctx_length = int(next_llm._model.n_ctx_train())
            prev_llm = self._llm
            self._llm = next_llm
            if prev_llm is not None:
                del prev_llm
            self.loaded.emit(model_ctx_length)
        except Exception as exc:
            self.load_error.emit(f"Load Failed: {exc}")

    @staticmethod
    def _coerce_mapping(payload):
        from core.llm_parser import coerce_mapping
        return coerce_mapping(payload)

    @staticmethod
    def _content_to_text(content) -> str:
        from core.llm_parser import content_to_text
        return content_to_text(content)

    @staticmethod
    def _extract_chunk_parts(chunk) -> tuple[str, str]:
        from core.llm_parser import extract_chunk_parts
        return extract_chunk_parts(chunk)

    @Slot(object)
    def generate(self, payload: dict) -> None:
        if self._llm is None:
            self.trace.emit("ERROR: Model offline.")
            self.done.emit(False, "")
            return

        messages = payload.get("messages", [])
        temp = float(payload.get("temp", 1.0))
        top_p = float(payload.get("top_p", 0.95))
        # max_tokens parked per-call (smart-spec re-make 2026-05-09):
        # no per-call ceiling is sent to llama-cpp; the backend uses its
        # own default. Re-add via payload key when /effort lands.
        thinking_enabled = payload.get("thinking_enabled")
        sampling = payload.get("sampling") or {}
        grammar_gbnf = payload.get("grammar")

        self.trace.emit(
            f"[WORKER] started: msgs={len(messages)}, temp={temp}, top_p={top_p}"
            + (" grammar=yes" if grammar_gbnf else "")
        )
        self.trace.emit("-> inference started")

        assistant_chunks = []
        completed = False
        self._interrupt_requested = False
        try:
            kwargs = {
                "messages": messages,
                "temperature": temp,
                "top_p": top_p,
                "stream": True,
            }
            kwargs.update(sampling)
            if thinking_enabled is not None:
                kwargs["enable_thinking"] = bool(thinking_enabled)
            if grammar_gbnf:
                try:
                    from llama_cpp import LlamaGrammar
                    kwargs["grammar"] = LlamaGrammar.from_string(grammar_gbnf, verbose=False)
                    self.trace.emit("[WORKER] grammar constraint active")
                except Exception as exc:
                    self.trace.emit(f"[WORKER] grammar load failed, ignoring: {exc}")

            try:
                stream = self._llm.create_chat_completion(**kwargs)
            except TypeError as exc:
                if "enable_thinking" in str(exc):
                    kwargs.pop("enable_thinking", None)
                    self.trace.emit("[WORKER] enable_thinking unsupported by backend; ignoring")
                    stream = self._llm.create_chat_completion(**kwargs)
                else:
                    raise

            total_generated = 0
            in_reasoning = False
            last_finish_reason: str | None = None
            for chunk in stream:
                if self._interrupt_requested:
                    self.trace.emit("-> inference aborted")
                    break

                # Synthetic trace chunks injected by the SSE parser when a
                # malformed event was dropped. Surface to the trace panel
                # so missing tokens don't go unnoticed.
                if isinstance(chunk, dict) and "_trace" in chunk and len(chunk) == 1:
                    self.trace.emit(str(chunk["_trace"]))
                    continue

                _payload = self._coerce_mapping(chunk)
                _choices = _payload.get("choices")
                if isinstance(_choices, list) and _choices:
                    _fr = self._coerce_mapping(_choices[0]).get("finish_reason")
                    if isinstance(_fr, str) and _fr:
                        last_finish_reason = _fr

                text, reasoning_text = self._extract_chunk_parts(chunk)

                if reasoning_text:
                    if not in_reasoning:
                        in_reasoning = True
                        self.token.emit("<think>")
                        assistant_chunks.append("<think>")
                    self.token.emit(reasoning_text)
                    assistant_chunks.append(reasoning_text)
                    total_generated += 1
                    self.usage.emit(total_generated)
                    continue

                if text:
                    if in_reasoning:
                        in_reasoning = False
                        self.token.emit("</think>")
                        assistant_chunks.append("</think>")
                    assistant_chunks.append(text)
                    self.token.emit(text)
                    total_generated += 1
                    self.usage.emit(total_generated)

            if in_reasoning:
                self.token.emit("</think>")
                assistant_chunks.append("</think>")
            if not self._interrupt_requested:
                completed = True
                self.trace.emit("-> inference complete")
        except Exception as exc:
            self.trace.emit(f"[WORKER] EXCEPTION: {exc}")
            self.trace.emit(f"<span style='color:red'>ERROR: {exc}</span>")
        finally:
            fr = last_finish_reason if "last_finish_reason" in locals() else None
            if fr and fr != "stop":
                self.trace.emit(
                    f"<span style='color:orange'>[WORKER] generation ended early: "
                    f"finish_reason={fr}</span>"
                )
            elif completed and fr is None and assistant_chunks:
                self.trace.emit(
                    "[WORKER] stream ended without finish_reason — "
                    "may be truncated by network/provider"
                )
            self.trace.emit(
                f"[WORKER] finished: completed={completed}, "
                f"chunks={len(assistant_chunks)}, finish_reason={fr or 'unset'}"
            )
            self.done.emit(completed, "".join(assistant_chunks))

    @Slot()
    def stop_generation(self) -> None:
        self._interrupt_requested = True

    @Slot()
    def unload_model(self) -> None:
        self._interrupt_requested = True
        if self._llm is not None:
            close_fn = getattr(self._llm, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass
            self._llm = None
        self.unloaded.emit()

    @Slot()
    def shutdown(self) -> None:
        self.unload_model()

class GeneratorWorker(QThread):
    token = Signal(str)
    trace = Signal(str)
    done = Signal(bool, str)
    usage = Signal(int)

    def __init__(
        self,
        llm,
        messages,
        temp,
        top_p,
        thinking_enabled: bool | None,
        sampling: dict | None = None,
        grammar_gbnf: str | None = None,
    ):
        super().__init__()
        self.llm = llm
        self.messages = messages
        self.temp = temp
        self.top_p = top_p
        # max_tokens parameter parked (smart-spec re-make 2026-05-09):
        # GeneratorWorker no longer accepts a per-call ceiling. The
        # backend resolves its own default. Re-add when /effort lands.
        self._thinking_enabled = thinking_enabled
        self.sampling = sampling or {}
        self._grammar_gbnf = grammar_gbnf

    @staticmethod
    def _coerce_mapping(payload):
        from core.llm_parser import coerce_mapping
        return coerce_mapping(payload)

    @staticmethod
    def _content_to_text(content) -> str:
        from core.llm_parser import content_to_text
        return content_to_text(content)

    @staticmethod
    def _extract_chunk_parts(chunk) -> tuple[str, str]:
        from core.llm_parser import extract_chunk_parts
        return extract_chunk_parts(chunk)

    def run(self):
        self.trace.emit(
            f"[WORKER] started: msgs={len(self.messages)}, temp={self.temp}, top_p={self.top_p}"
        )
        self.trace.emit("→ inference started")
        assistant_chunks = []
        completed = False
        try:
            if self.isInterruptionRequested():
                self.trace.emit("[WORKER] interrupted before inference")
                return

            kwargs = {
                "messages": self.messages,
                "temperature": self.temp,
                "top_p": self.top_p,
                "stream": True,
            }
            kwargs.update(self.sampling)
            # Pass an explicit high max_tokens so the cloud provider's small
            # default (DeepSeek/OpenAI typically default to ~4096) doesn't cap
            # generation at ~3k. The user wanted "no max" -- in practice we
            # send a ceiling well above what any model emits in one turn.
            # Providers that don't accept the param raise TypeError and the
            # retry below strips it.
            kwargs.setdefault("max_tokens", 65536)
            if self._thinking_enabled is not None:
                kwargs["enable_thinking"] = bool(self._thinking_enabled)
            if self._grammar_gbnf:
                try:
                    from llama_cpp import LlamaGrammar
                    kwargs["grammar"] = LlamaGrammar.from_string(self._grammar_gbnf, verbose=False)
                    self.trace.emit("[WORKER] grammar constraint active")
                except Exception as exc:
                    self.trace.emit(f"[WORKER] grammar load failed, ignoring: {exc}")

            def _open_stream(call_kwargs: dict):
                """Try the call; strip params the backend rejects and retry."""
                try:
                    return self.llm.create_chat_completion(**call_kwargs)
                except TypeError as exc:
                    msg = str(exc)
                    if "enable_thinking" in msg and "enable_thinking" in call_kwargs:
                        call_kwargs.pop("enable_thinking", None)
                        self.trace.emit("[WORKER] enable_thinking unsupported; ignoring")
                        return self.llm.create_chat_completion(**call_kwargs)
                    if "max_tokens" in msg and "max_tokens" in call_kwargs:
                        call_kwargs.pop("max_tokens", None)
                        self.trace.emit("[WORKER] max_tokens unsupported; ignoring")
                        return self.llm.create_chat_completion(**call_kwargs)
                    raise

            stream = _open_stream(kwargs)

            total_generated = 0
            in_reasoning = False
            last_finish_reason: str | None = None
            for chunk in stream:
                if self.isInterruptionRequested():
                    self.trace.emit("→ inference aborted")
                    break

                # Capture finish_reason if the chunk carries it. Most
                # OpenAI-compat providers populate this on the final chunk;
                # surfacing it tells the user whether the stream ended
                # cleanly ("stop"), hit a token cap ("length"), got
                # filtered ("content_filter"), or cut for unknown reasons.
                _payload = self._coerce_mapping(chunk)
                _choices = _payload.get("choices")
                if isinstance(_choices, list) and _choices:
                    _fr = self._coerce_mapping(_choices[0]).get("finish_reason")
                    if isinstance(_fr, str) and _fr:
                        last_finish_reason = _fr

                text, reasoning_text = self._extract_chunk_parts(chunk)

                if reasoning_text:
                    if not in_reasoning:
                        in_reasoning = True
                        self.token.emit("<think>")
                        assistant_chunks.append("<think>")
                    self.token.emit(reasoning_text)
                    assistant_chunks.append(reasoning_text)
                    total_generated += 1
                    self.usage.emit(total_generated)
                    continue

                if text:
                    if in_reasoning:
                        in_reasoning = False
                        self.token.emit("</think>")
                        assistant_chunks.append("</think>")
                    assistant_chunks.append(text)
                    self.token.emit(text)
                    total_generated += 1
                    self.usage.emit(total_generated)

            if in_reasoning:
                self.token.emit("</think>")
                assistant_chunks.append("</think>")
            if not self.isInterruptionRequested():
                completed = True
                self.trace.emit("→ inference complete")
        except Exception as e:
            self.trace.emit(f"[WORKER] EXCEPTION: {e}")
            self.trace.emit(f"<span style='color:red'>ERROR: {e}</span>")
        finally:
            # Surface finish_reason so a "length"/"content_filter"/missing
            # reason reaches the user as a hint that the message was cut.
            fr = last_finish_reason if "last_finish_reason" in locals() else None
            if fr and fr != "stop":
                self.trace.emit(
                    f"<span style='color:orange'>[WORKER] generation ended early: "
                    f"finish_reason={fr}</span>"
                )
            elif completed and fr is None and assistant_chunks:
                self.trace.emit(
                    "[WORKER] stream ended without finish_reason — "
                    "may be truncated"
                )
            self.trace.emit(
                f"[WORKER] finished: completed={completed}, "
                f"chunks={len(assistant_chunks)}, finish_reason={fr or 'unset'}"
            )
            self.done.emit(completed, "".join(assistant_chunks))

class LLMEngine(QObject):
    sig_token = Signal(str)
    sig_trace = Signal(str)
    sig_status = Signal(SystemStatus)
    sig_finished = Signal()
    sig_usage = Signal(int)
    sig_image = Signal(object)
    sig_model_capabilities = Signal(dict)
    sig_gguf_load = Signal(str, int, int)
    sig_gguf_generate = Signal(object)
    sig_gguf_stop = Signal()
    sig_gguf_unload = Signal()
    sig_gguf_shutdown = Signal()

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.llm = None
        self.loader = None
        self.worker = None
        self._gguf_thread = QThread(self)
        self._gguf_runtime = GGUFRuntime()
        self._gguf_runtime.moveToThread(self._gguf_thread)
        self._gguf_thread.finished.connect(self._gguf_runtime.deleteLater)
        self.sig_gguf_load.connect(self._gguf_runtime.load_model)
        self.sig_gguf_generate.connect(self._gguf_runtime.generate)
        self.sig_gguf_stop.connect(self._gguf_runtime.stop_generation)
        self.sig_gguf_unload.connect(self._gguf_runtime.unload_model)
        self.sig_gguf_shutdown.connect(self._gguf_runtime.shutdown)
        self._gguf_runtime.trace.connect(self._on_runtime_trace)
        self._gguf_runtime.loaded.connect(self._on_gguf_load_success)
        self._gguf_runtime.load_error.connect(self._on_load_error)
        self._gguf_runtime.token.connect(self.sig_token)
        self._gguf_runtime.usage.connect(self._on_usage_update)
        self._gguf_runtime.done.connect(self._on_gen_finish)
        self._gguf_runtime.unloaded.connect(self._on_gguf_unloaded)
        self._gguf_thread.start()
        self.model_path: str | None = None
        self.conversation_history: list[dict] = []
        self._pending_user_index: int | None = None
        self._load_cancel_requested: bool = False
        self._shutdown_requested: bool = False
        self._status: SystemStatus = SystemStatus.READY
        self._ephemeral_generation: bool = False
        self.model_loaded: bool = False
        self.model_ctx_length: int | None = None
        self.ctx_limit: int = int(getattr(self.state, "ctx_limit", 0) or 0)
        self.gguf_path: str | None = None
        self.backend: str = "gguf_api"
        self.api_provider: str = "openai"
        self.api_base: str = ""
        self.api_key: str = ""
        self.api_model: str = ""
        self._current_task_id: str = ""
        # Turn-trace: most recent OUTER turn task_id. Tool followups
        # (ephemeral generations) point parent_turn_id at this so the
        # inspector can group the followup chain.
        self._last_outer_turn_id: str = ""
        # Readable monotonic turn-count for the current dispatch (bearing age
        # render). Set on outer turns when MONOLITH_TURN_COUNTER_V1 is on;
        # inner/tool-followup turns reuse it so the render is constant within a
        # turn (KV-cache safe). 0 = feature off / not yet stamped.
        self._last_turn_n: int = 0
        # Canonical per-turn wall-clock instant (dark sub-flag
        # MONOLITH_TURN_CLOCK_V1). Captured once on outer turns at the same
        # boundary as the counter; inner/tool-followup turns reuse it so every
        # downstream reader derives from ONE instant. "" = feature off / not
        # yet stamped (flag-off byte-identical).
        self._last_now_iso: str = ""
        # 0 = no wall-clock cap. The previous 120s default cut long cloud
        # generations (especially DeepSeek reasoner thinking) in the middle.
        # Use set_timeout(seconds) to re-enable a per-engine ceiling.
        self._gen_timeout_sec: int = 0
        self._gen_timeout_timer = QTimer(self)
        self._gen_timeout_timer.setSingleShot(True)
        self._gen_timeout_timer.timeout.connect(self._on_gen_timeout)
        self._last_generation_error: str | None = None
        self._last_prompt_breakdown: dict | None = None
        # Estimated tokens for the assembled prompt (system + history + user +
        # ephemeral interceptor injections + KV-cache prefix). Seeded once per
        # generate() call; the per-chunk worker emit is added on top so the
        # vitals footer reflects total context-window occupancy, not just
        # output chunks streamed.
        self._prompt_token_baseline: int = 0

    @staticmethod
    def _estimate_prompt_tokens(messages: list[dict]) -> int:
        """Heuristic token estimate for the assembled message list.

        Char count divided by an English-text average of 4 chars/token, plus a
        small per-message overhead for chat-template role boundaries. Path-
        agnostic (works for gguf and cloud); within ~10-20% of real tokenizer
        counts for the model families this app targets. Refine later by
        plumbing through tiktoken/llama-cpp tokenize when accuracy matters.
        """
        _CHARS_PER_TOKEN = 4
        _PER_MESSAGE_OVERHEAD = 4
        total = 0
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = str(msg.get("content", "") or "")
            total += max(1, len(content) // _CHARS_PER_TOKEN) + _PER_MESSAGE_OVERHEAD
        return total

    def _on_runtime_trace(self, message: str) -> None:
        text = str(message or "")
        self._capture_generation_error(text)
        self.sig_trace.emit(text)

    def _capture_generation_error(self, message: str) -> None:
        text = str(message or "").strip()
        if not text:
            return
        marker = "[WORKER] EXCEPTION:"
        if marker in text:
            self._last_generation_error = text.split(marker, 1)[1].strip()
            return
        plain = re.sub(r"<[^>]+>", "", text).strip()
        if plain.startswith("ERROR:"):
            self._last_generation_error = plain[len("ERROR:"):].strip()

    def _format_generation_error(self, raw_error: str) -> str:
        text = str(raw_error or "").strip()
        lowered = text.lower()
        if "exceed_context_size_error" in lowered or "exceeds the available context size" in lowered:
            prompt_tokens = ""
            context_tokens = ""
            m_prompt = re.search(r"n_prompt_tokens\"?\s*[:=]\s*(\d+)", text)
            if not m_prompt:
                m_prompt = re.search(r"request\s*\((\d+)\s*tokens\)", text, re.IGNORECASE)
            if m_prompt:
                prompt_tokens = m_prompt.group(1)
            m_ctx = re.search(r"n_ctx\"?\s*[:=]\s*(\d+)", text)
            if not m_ctx:
                m_ctx = re.search(r"available context size\s*\((\d+)\s*tokens\)", text, re.IGNORECASE)
            if m_ctx:
                context_tokens = m_ctx.group(1)
            if prompt_tokens and context_tokens:
                return (
                    f"Context limit exceeded: request used {prompt_tokens} tokens but this model allows "
                    f"{context_tokens}. Reduce history/context or switch to a larger-context model."
                )
            return (
                "Context limit exceeded for this model. Reduce history/context or switch to a "
                "larger-context model."
            )
        return f"Generation failed: {text}"

    def set_ctx_limit(self, payload: dict) -> None:
        value = payload.get("ctx_limit") if isinstance(payload, dict) else None
        if value is None:
            return
        try:
            self.ctx_limit = int(value)
        except (TypeError, ValueError):
            return

    def set_model_path(self, payload: dict) -> None:
        if isinstance(payload, dict):
            path = payload.get("path")
            if path is None:
                path = payload.get("gguf_path")
            if path is not None:
                self.model_path = path
                self.gguf_path = path
            backend = payload.get("backend")
            if backend:
                self.backend = str(backend)
            api_provider = payload.get("api_provider")
            if api_provider:
                self.api_provider = str(api_provider)
            api_base = payload.get("api_base")
            if api_base is not None:
                self.api_base = str(api_base)
            api_key = payload.get("api_key")
            if api_key is not None:
                self.api_key = str(api_key)
            api_model = payload.get("api_model")
            if api_model is not None:
                self.api_model = str(api_model)
        else:
            self.model_path = payload
            self.gguf_path = payload

    def _on_load_error(self, err_msg):
        self.sig_trace.emit(f"<span style='color:red'>{err_msg}</span>")
        if self._shutdown_requested:
            self.set_status(SystemStatus.READY)
        elif self.model_loaded:
            # Keep serving with the previous model when a reload fails.
            self.set_status(SystemStatus.READY)
            self.sig_trace.emit("-> keeping previous model online after load failure")
        else:
            self.set_status(SystemStatus.ERROR)
        self.loader = None

    def _cleanup_loader(self, *args, **kwargs):
        self.loader = None

    def reset_conversation(self, system_prompt):
        self.conversation_history = [{"role": "system", "content": system_prompt}]
        self._pending_user_index = None
        # Clear the per-conversation context_refresh high-water mark so the new
        # conversation's refresh cadence starts fresh instead of inheriting the
        # prior conversation's message count (when-plane fix #6).
        from core import context_refresh
        context_refresh.reset_refresh_state()

    def set_history(self, payload: dict):
        history = payload.get("history", []) if isinstance(payload, dict) else []
        if not isinstance(history, list):
            return
        self.conversation_history = [h for h in history if isinstance(h, dict)]
        self._pending_user_index = None

    def _compile_system_prompt(self, config):
        from core.llm_config import build_system_prompt, inject_working_memory_into_prompt
        template_prompt = (config or {}).get("system_prompt") or ""
        base_prompt = template_prompt
        # If the config prompt already has the catalog substituted, use it.
        # Otherwise build from the master prompt file.
        if "{skills_catalog}" in base_prompt or not base_prompt.strip():
            base_prompt = build_system_prompt(config)
        # WORKING MEMORY injection — fires per-turn against the (possibly cached)
        # base_prompt. Idempotent, so safe even if build_system_prompt already
        # injected it (e.g., on first-build-from-master path).
        base_prompt = inject_working_memory_into_prompt(base_prompt)
        catalog = build_profiled_tool_catalog()
        # [WORLD STATE] block parked — see docs (smart-spec re-make 2026-05-09).
        # Re-enable when async tool dispatch lands and a decision surface needs
        # to act on engine/task/resource deltas. WorldStateStore itself is
        # untouched; this just stops baking it into every system prompt.
        world_state = getattr(self.state, "world_state", None)
        final_prompt = base_prompt
        self._last_prompt_breakdown = {
            "template_chars": len(template_prompt),
            "placeholder_in_template": "{skills_catalog}" in template_prompt,
            "catalog_chars": len(catalog),
            "base_chars": len(base_prompt),
            "world_ctx_chars": 0,
            "final_chars": len(final_prompt),
            "world_snapshot_present": world_state is not None,
        }
        return final_prompt

    def _trace_prompt_breakdown(self) -> None:
        info = self._last_prompt_breakdown
        if not isinstance(info, dict):
            return
        self.sig_trace.emit(
            (
                "[PROMPT] recipe: template={template_chars} "
                "placeholder={placeholder_in_template} catalog={catalog_chars} "
                "base={base_chars} world_ctx={world_ctx_chars} final={final_chars} "
                "world_snapshot={world_snapshot_present}"
            ).format(**info)
        )

    @staticmethod
    def _summarize_messages(messages: list[dict]) -> dict[str, int]:
        totals = {"all": 0, "system": 0, "user": 0, "assistant": 0, "tool": 0, "other": 0}
        for msg in messages:
            role = str(msg.get("role", "other")).lower()
            content = str(msg.get("content", "") or "")
            size = len(content)
            totals["all"] += size
            if role in totals:
                totals[role] += size
            else:
                totals["other"] += size
        return totals

    def set_timeout(self, sec: int) -> None:
        """Configure the generation timeout wall (0 = disabled)."""
        self._gen_timeout_sec = max(0, int(sec))

    def _on_gen_timeout(self) -> None:
        self.sig_trace.emit(
            f"[ENGINE] generation timeout after {self._gen_timeout_sec}s — forcing stop"
        )
        self.stop_generation()

    def _on_usage_update(self, count):
        self.sig_usage.emit(self._prompt_token_baseline + count)

    def _on_gen_finish(self, completed, assistant_text):
        self._gen_timeout_timer.stop()
        self.sig_trace.emit(f"[ENGINE] _on_gen_finish: completed={completed}, text_len={len(assistant_text)}")
        emitted_error_token = False
        assistant_for_history = re.sub(
            r"</?think\s*>", "", str(assistant_text or ""),
            flags=re.IGNORECASE,
        )
        if not completed and not assistant_text and self._last_generation_error:
            # Surface hard failures in the assistant stream so users see what happened.
            friendly_error = self._format_generation_error(self._last_generation_error)
            self.sig_token.emit(friendly_error)
            emitted_error_token = True
            assistant_for_history = friendly_error
        # Fault detection — observe-only; must never break generation.
        # Runs on completed turns with content; error tokens are not analysed.
        if completed and assistant_text and not emitted_error_token:
            try:
                from core.fault_response import run_all_detectors, emit_fault
                _turn_id = str(self._current_task_id or "")
                if _turn_id:
                    # Build a minimal frame_traces context so the regen-mismatch
                    # detector can count tool-result messages in history.
                    _frame_msgs = [
                        {"role": m.get("role", ""), "content": m.get("content", "")}
                        for m in self.conversation_history
                        if isinstance(m, dict)
                    ]
                    _ctx = {"frame_traces": _frame_msgs}
                    _faults = run_all_detectors(assistant_text, _turn_id, _ctx)
                    for _r in _faults:
                        emit_fault(
                            _r.turn_id,
                            _r.fault_kind,
                            _r.detector_name,
                            _r.evidence,
                            _r.metadata if _r.metadata else None,
                        )
            except Exception:
                pass

        # Append the assistant response to history — even for ephemeral
        # (tool followup) generations.  The ephemeral flag means the *user prompt*
        # was not persisted (it's a synthetic tool-result injection), but the
        # assistant's answer must stay in context so the LLM can reference prior
        # tool results on multi-hop chains.
        # Error tokens are NOT persisted — they're for the user, not the model.
        # Feeding "Generation failed: ..." as a prior assistant turn causes the
        # model to apologize for or reference an error it didn't produce.
        if not emitted_error_token and (completed or assistant_for_history.strip()):
            self.conversation_history.append(
                {"role": "assistant", "content": assistant_for_history}
            )
        self._pending_user_index = None
        self._ephemeral_generation = False
        # Avoid emitting synthetic tokens on failed/empty generations.
        if completed and assistant_text:
            self.sig_token.emit("\n")
        elif emitted_error_token:
            self.sig_token.emit("\n")
        self.worker = None
        self.sig_finished.emit()
        self.set_status(SystemStatus.READY)
        self._last_generation_error = None

    def set_status(self, s):
        self._status = s
        self.sig_status.emit(s)

    def load_model(self):
        if self._status == SystemStatus.LOADING:
            self.sig_trace.emit("ERROR: Load already in progress.")
            self.set_status(SystemStatus.ERROR)
            return

        if self.backend == "gguf":
            model_path = self.model_path or self.gguf_path
            if not model_path:
                self.sig_trace.emit("ERROR: No GGUF selected.")
                self.set_status(SystemStatus.ERROR)
                return
        else:
            if not self.api_base:
                self.sig_trace.emit("ERROR: Missing API base URL.")
                self.set_status(SystemStatus.ERROR)
                return
            if not self.api_model:
                self.sig_trace.emit("ERROR: Missing API model.")
                self.set_status(SystemStatus.ERROR)
                return

        self.set_status(SystemStatus.LOADING)
        self._load_cancel_requested = False
        if self.backend == "gguf":
            n_ctx = (
                min(self.ctx_limit, self.model_ctx_length)
                if self.model_ctx_length
                else self.ctx_limit
            )
            self.sig_gguf_load.emit(str(model_path), int(n_ctx), -1)
            return

        self.loader = HttpModelLoader(
            self.api_base,
            self.api_key,
            self.api_model,
            provider=self.api_provider or "openai",
        )
        self.loader.trace.connect(self.sig_trace)
        self.loader.error.connect(self._on_load_error)
        self.loader.finished.connect(self._on_load_success)
        self.loader.finished.connect(self._cleanup_loader)
        self.loader.error.connect(self._cleanup_loader)
        self.loader.start()

    def _on_load_success(self, llm_instance, model_ctx_length):
        self._finish_load(model_ctx_length, llm_instance)

    def _on_gguf_load_success(self, model_ctx_length):
        self._finish_load(model_ctx_length, None)

    def _finish_load(self, model_ctx_length, llm_instance):
        if self._shutdown_requested:
            if llm_instance is not None:
                del llm_instance
            self.set_status(SystemStatus.READY)
            return

        if self._load_cancel_requested:
            if llm_instance is not None:
                del llm_instance
            self.llm = None
            self.model_loaded = False
            self.set_status(SystemStatus.READY)
            if self.backend == "gguf":
                self.sig_gguf_unload.emit()
            self.sig_trace.emit("-> load cancelled")
            self.loader = None
            return

        self.llm = llm_instance
        self.model_ctx_length = int(model_ctx_length) if model_ctx_length else None

        # Cloud-load window resolution. The HTTP loader rarely returns a real
        # n_ctx_train, so for cloud backends we ask /v1/models, then fall back
        # to a small DeepSeek-aware inference table. Local (gguf) models report
        # n_ctx_train directly via the loader and skip this path.
        if self.backend != "gguf" and self.model_ctx_length is None:
            resolved, source = resolve_cloud_window(
                api_base=self.api_base,
                api_key=self.api_key,
                api_provider=self.api_provider,
                api_model=self.api_model,
            )
            if resolved:
                self.model_ctx_length = int(resolved)
                self.sig_trace.emit(
                    f"[ctx] cloud window={int(resolved):,} src={source} "
                    f"model={self.api_model or '<unknown>'}"
                )

        # Priority: model capability is ground truth. The persisted ctx_limit
        # acts as a USER-EXPLICIT cost cap and applies only when it would
        # *lower* the resolved ceiling -- a stale 8192 default must not
        # impersonate ground truth above a 1M model window.
        #
        # Stale-cap heuristic: a saved ctx_limit that's <1/8 of the resolved
        # window almost certainly came from an old config that predates the
        # current model. Ignore it with a trace line so the user sees what
        # happened. Caps written by the user via the slider tend to be the
        # same order of magnitude as the model's window (e.g., 64k cap on a
        # 128k model), so the 8x threshold is well clear of legitimate use.
        _STALE_CAP_RATIO = 8
        if self.model_ctx_length:
            user_cap = self.ctx_limit if self.ctx_limit > 0 else None
            if (
                user_cap
                and self.model_ctx_length // _STALE_CAP_RATIO > user_cap
            ):
                self.sig_trace.emit(
                    f"[ctx] ignoring stale cap {user_cap} "
                    f"(model window {self.model_ctx_length}); "
                    "drag the Context Limit slider to set a real cap"
                )
                self.ctx_limit = int(self.model_ctx_length)
            elif user_cap and user_cap < self.model_ctx_length:
                self.sig_trace.emit(
                    f"[ctx] user cap lowering ceiling "
                    f"{self.model_ctx_length} -> {user_cap}"
                )
                self.ctx_limit = int(user_cap)
            else:
                self.ctx_limit = int(self.model_ctx_length)

        # Mirror onto shared state so the vitals footer (which reads
        # state.ctx_limit / state.gguf_path / state.api_model directly)
        # reflects the resolved window and active model name without waiting
        # for the next signal pump. Without this, the footer would show
        # "no-model" even with a model loaded because state never updates.
        try:
            self.state.ctx_limit = int(self.ctx_limit)
            if self.backend == "gguf":
                self.state.gguf_path = self.gguf_path or self.model_path
                self.state.api_model = ""
            else:
                self.state.gguf_path = None
                self.state.api_model = self.api_model or ""
        except Exception:
            pass

        self.sig_model_capabilities.emit(
            {
                "model_ctx_length": self.model_ctx_length,
                "ctx_limit": self.ctx_limit,
            }
        )
        self.model_loaded = True
        self.set_status(SystemStatus.READY)
        self.reset_conversation(load_master_prompt())
        self.sig_trace.emit("-> system online")
        self.loader = None

    def unload_model(self):
        if self._status == SystemStatus.LOADING and self.backend == "gguf":
            self._load_cancel_requested = True
            self.sig_trace.emit("-> unload requested during load; will cancel when init completes")
            return

        if self._status == SystemStatus.LOADING and self.loader and self.loader.isRunning():
            self._load_cancel_requested = True
            self.sig_trace.emit("-> unload requested during load; will cancel when init completes")
            return

        if self._status == SystemStatus.RUNNING:
            self.sig_trace.emit("ERROR: Cannot unload while generating.")
            return

        if self.backend == "gguf" and self.model_loaded:
            self.set_status(SystemStatus.UNLOADING)
            self.sig_gguf_unload.emit()
            return

        if self.llm:
            self.set_status(SystemStatus.UNLOADING)
            close_fn = getattr(self.llm, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass
            self.llm = None
        self._finalize_unload()

    def _on_gguf_unloaded(self) -> None:
        if self.backend != "gguf":
            return
        self._finalize_unload()

    def _finalize_unload(self) -> None:
        self.model_loaded = False
        self.model_ctx_length = None
        # Clear shared state so the footer doesn't keep showing the
        # previous model name after unload.
        try:
            self.state.gguf_path = None
            self.state.api_model = ""
            self.state.ctx_limit = 0
        except Exception:
            pass
        self.reset_conversation(load_master_prompt())
        QTimer.singleShot(0, lambda: self.set_status(SystemStatus.READY))
        self.sig_trace.emit("-> model unloaded")

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
        self._last_generation_error = None

        prompt = payload.get("prompt", "")
        self.sig_trace.emit(
            f"[ENGINE] generate: history_len={len(self.conversation_history)}, prompt={repr(prompt[:80])}, model_loaded={self.model_loaded}"
        )
        config = payload.get("config")
        if config is None:
            config = load_config()
        thinking_enabled = payload.get("thinking")  # None = not set → don't pass enable_thinking

        system_prompt = self._compile_system_prompt(config)
        self._trace_prompt_breakdown()
        # [THINKING] suffix parked — see docs (smart-spec re-make 2026-05-09).
        # Suppression of <think> output is handled at the API level via the
        # enable_thinking parameter (with TypeError/HTTPError retry-on-failure
        # in OpenAICompatLLM.create_chat_completion and GeneratorWorker.run).
        # The unconditional prompt-side suffix was redundant for backends that
        # respect the API param. Re-add as a backend-capability-gated injection
        # only if a backend is found that ignores enable_thinking entirely.
        if self.backend != "gguf":
            preview = system_prompt.replace("\n", " ")[:220]
            self.sig_trace.emit(f"[ENGINE] system_prompt(len={len(system_prompt)}): {preview}")
        temp = float(config.get("temp", 1.0))
        top_p = float(config.get("top_p", 0.95))
        top_k = int(config.get("top_k", 20))
        min_p = float(config.get("min_p", 0.0))
        presence_penalty = float(config.get("presence_penalty", 1.5))
        repetition_penalty = float(config.get("repetition_penalty", 1.0))
        # max_tokens parked per-call (smart-spec re-make 2026-05-09):
        # no per-call ceiling is sent. Backends use their own default
        # (context-bound for llama-cpp; server default for OpenAI-compat).
        # The deterministic /effort surface will own this when it lands.
        sampling: dict[str, int | float] = {
            "presence_penalty": presence_penalty,
        }
        if self.backend in ("gguf", "gguf_api"):
            sampling.update(
                {
                    "top_k": top_k,
                    "min_p": min_p,
                    "repeat_penalty": repetition_penalty,
                }
            )

        self._ephemeral_generation = bool(payload.get("ephemeral", False))

        if not self.conversation_history:
            self.reset_conversation(load_master_prompt())

        system_entry = {"role": "system", "content": system_prompt}
        self.conversation_history = [
            m for m in self.conversation_history
            if not (isinstance(m, dict) and m.get("role") == "system")
        ]
        self.conversation_history.insert(0, system_entry)

        is_update = prompt.startswith("You were interrupted mid-generation.")
        if not self._ephemeral_generation and not is_update:
            # Skip if the last non-system message is an identical user prompt
            # (double-click send, UI retry without edit). Compare tag-stripped
            # bodies — the stored turn carries a [CHANNEL: ...] header the raw
            # prompt lacks.
            from core.channel_tag import build_channel_tag, strip_leading_channel_tag
            last_msg = next(
                (m for m in reversed(self.conversation_history)
                 if isinstance(m, dict) and m.get("role") != "system"),
                None,
            )
            is_dup = (
                last_msg is not None
                and last_msg.get("role") == "user"
                and strip_leading_channel_tag(str(last_msg.get("content") or ""))[0] == prompt
            )
            if not is_dup:
                # Tag the generating turn so the model can tell the user's
                # words apart from the ephemeral lanes inserted before it —
                # cloud backends merge consecutive user-role messages, so the
                # header is the only boundary that survives. Agent-server
                # turns arrive pre-tagged ([CHANNEL: connect/...]); skip those.
                content = prompt
                if not prompt.lstrip().startswith("[CHANNEL:"):
                    content = f"{build_channel_tag('USER', include_modes=True)}\n\n{prompt}"
                self.conversation_history.append({"role": "user", "content": content})
            self._pending_user_index = len(self.conversation_history) - 1
            messages = list(self.conversation_history)
        else:
            messages = list(self.conversation_history)
            if not is_update:
                # Tool-followup (ephemeral generation): mark the synthetic
                # tool-result user message ephemeral so the coalescer inserts its
                # per-minute temporal block before the OUTER user message, not
                # before this followup — preserving the cached prefix through the
                # assistant tool-call instead of busting KV-cache (when-plane fix #4).
                messages.append({"role": "user", "content": prompt, "ephemeral": True})
            self._pending_user_index = None

        task_id = payload.get("task_id", "") or ""
        if not task_id:
            # Upstream callers don't always populate task_id (raw-string
            # prompts, edit-rewrites, etc.). Without a turn_id the turn_trace
            # stage/frame writers skip silently — the root cause behind the
            # 0-row stage_traces/frame_traces tables observed 2026-05-11.
            # Mint a UUID4 hex here so trace writes always fire.
            import uuid as _uuid
            task_id = _uuid.uuid4().hex
        self._current_task_id = task_id

        # System-side turn classification. Runs once per turn before LLM
        # dispatch — replaces the LLM-emit → world_state → next-turn-inject
        # feedback loop that caused the lag bug. Pure function of (messages,
        # config); never reads the LLM's response. The TurnShape is stashed
        # on config so interceptors (effort.py Layer 4, lag_watch) read the
        # same value and the frame_trace captures it for /trace.
        try:
            from core.turn_classifier import classify as _classify_turn
            turn_shape = _classify_turn(messages, config)
            if isinstance(config, dict):
                config["_turn_shape"] = turn_shape
        except Exception:
            turn_shape = None

        # M2 — effort governs generation depth (execution-plane sync).
        # The classifier emits an effort_tier every turn; resolve it and, for
        # hard turns, monotonically upgrade the backend reasoning mode. This is
        # the one lever the 2026-05-09 smart-spec reserved for /effort:
        # enable_thinking is decode-side (KV-stable) and yields a real
        # behavioral delta, unlike a max_tokens cap (inert-or-truncating).
        # Non-performative: no prompt instruction. An explicit manual
        # 'thinking on' always wins; effort never forces thinking off. Resolving
        # also stamps config["_resolved_effort_tier"] (the key the frame trace
        # below already reads). Flag-gated (MONOLITH_EFFORT_V1, default on);
        # failures isolated so effort never breaks a turn.
        try:
            from core.effort_resolver import (
                effort_governance_enabled,
                resolve_effort_tier,
                resolve_thinking,
            )
            if effort_governance_enabled():
                _effort_tier = resolve_effort_tier(config)
                thinking_enabled = resolve_thinking(
                    manual=thinking_enabled, tier=_effort_tier
                )
        except Exception:
            pass

        # Turn-trace turn_id propagation (Q1: config injection). For tool
        # followups (ephemeral generations), parent_turn_id points to the
        # last outer-turn task_id so the inspector can group the followup
        # chain. Fresh user turns set themselves as the new outer turn.
        is_outer_turn = not (self._ephemeral_generation or is_update)
        if is_outer_turn:
            self._last_outer_turn_id = task_id
        parent_turn_id = self._last_outer_turn_id if not is_outer_turn else None
        config["_turn_id"] = task_id
        # Readable monotonic turn-count (dark sub-flag MONOLITH_TURN_COUNTER_V1).
        # resolve_turn_n owns the decision (unit-tested): on+outer → increment +
        # persist; on+inner/followup → reuse (so the bearing "N turns ago" render
        # is constant within a turn / KV-safe); off → 0 (flag-off byte-identical,
        # even on a mid-session toggle). Best-effort; never raises into the turn.
        try:
            from core import turn_counter as _tc
            self._last_turn_n = _tc.resolve_turn_n(self._last_turn_n, is_outer_turn)
        except Exception:
            pass
        if self._last_turn_n:
            config["_turn_n"] = self._last_turn_n
        # Canonical per-turn wall-clock instant (dark sub-flag
        # MONOLITH_TURN_CLOCK_V1). resolve_turn_now owns the decision (unit-tested):
        # on+outer → fresh capture; on+inner/followup → reuse (so 'now' is constant
        # within the turn, coherent with the counter, KV-safe); off → "" (so
        # config['_now_iso'] is never written — prompt byte-identical). Captured here
        # alongside the counter and BEFORE apply_interceptors so the temporal lanes
        # derive from one instant. Best-effort; never raises into the turn.
        try:
            from core import turn_clock as _tk
            self._last_now_iso = _tk.resolve_turn_now(self._last_now_iso, is_outer_turn)
        except Exception:
            pass
        if self._last_now_iso:
            config["_now_iso"] = self._last_now_iso
        if parent_turn_id:
            config["_parent_turn_id"] = parent_turn_id
        else:
            config.pop("_parent_turn_id", None)

        from core.history_compactor import compact_for_dispatch
        messages = compact_for_dispatch(messages)

        pre_interceptor = self._summarize_messages(messages)
        self.sig_trace.emit(
            (
                "[PROMPT] history pre-interceptors chars: total={all} "
                "system={system} user={user} assistant={assistant} tool={tool} other={other}"
            ).format(**pre_interceptor)
        )
        messages = apply_interceptors(messages, config)
        post_interceptor = self._summarize_messages(messages)
        self.sig_trace.emit(
            (
                "[PROMPT] history post-interceptors chars: total={all} "
                "system={system} user={user} assistant={assistant} tool={tool} other={other}"
            ).format(**post_interceptor)
        )

        # Seed the context-window counter with the assembled prompt's estimated
        # token weight (system + history + user + ephemerals + KV-cache prefix).
        # The streaming worker's per-chunk emits are added on top inside
        # _on_usage_update so the footer reflects total occupancy, not just
        # output chunks streamed.
        self._prompt_token_baseline = self._estimate_prompt_tokens(messages)
        self.sig_usage.emit(self._prompt_token_baseline)
        self.sig_trace.emit(
            f"[PROMPT] estimated tokens: prompt={self._prompt_token_baseline} "
            f"(chars/token=4 heuristic, +4/msg template overhead)"
        )

        # Layer B: persist the final assembled frame snapshot before backend
        # dispatch. Best-effort — failures log to stderr (Q7), never break.
        if task_id:
            try:
                from core import turn_trace as _tt
                system_chars = 0
                user_chars = 0
                total_chars = 0
                fmsgs: list[_tt.FrameMessage] = []
                for m in messages:
                    if not isinstance(m, dict):
                        continue
                    fm = _tt.FrameMessage.from_message(m)
                    fmsgs.append(fm)
                    total_chars += fm.content_chars
                    if fm.role == "system":
                        system_chars += fm.content_chars
                    elif fm.role == "user" and not fm.ephemeral:
                        user_chars = fm.content_chars  # last non-ephemeral wins
                # Resolved effort tier — set by effort_interceptor as a side
                # channel on config. Falls back to force_effort_tier (testing
                # override) or None when effort layer is disabled.
                resolved_tier: str | None = None
                if isinstance(config, dict):
                    raw_tier = config.get("_resolved_effort_tier") or config.get("force_effort_tier")
                    if isinstance(raw_tier, str) and raw_tier.strip():
                        resolved_tier = raw_tier.strip().lower()
                # Resolved reasoning mode — legacy field from reasoning plane.
                # Kept for backward compat; new code uses prompts_applied +
                # monothink_active instead.
                resolved_reasoning: str | None = None
                if isinstance(config, dict):
                    raw_reasoning = config.get("_resolved_reasoning_mode") or config.get("force_reasoning_mode")
                    if isinstance(raw_reasoning, str) and raw_reasoning.strip():
                        resolved_reasoning = raw_reasoning.strip().lower()
                # New /prompt system fields
                resolved_prompts: list[str] | None = None
                resolved_monothink: bool = False
                if isinstance(config, dict):
                    rp = config.get("_resolved_prompts")
                    if isinstance(rp, list) and rp:
                        resolved_prompts = rp
                    resolved_monothink = bool(config.get("_resolved_monothink", False))
                # max_tokens is parked per-call (smart-spec 2026-05-09 — see
                # comment near line 1325). Read from config dict instead of
                # a local that no longer exists; default to 0 = "not set"
                # so the trace records the config's claimed ceiling without
                # implying we sent it.
                _cfg_max_tokens = (
                    int(config.get("max_tokens", 0)) if isinstance(config, dict) else 0
                )
                config_snapshot = {
                    "max_tokens": _cfg_max_tokens,
                    "temp": float(temp),
                    "top_p": float(top_p),
                    "thinking_enabled": thinking_enabled,
                    "effort_tier": resolved_tier or "",
                    "ephemeral": self._ephemeral_generation,
                    "is_update": is_update,
                }
                # System-side classification (TurnShape). Captured before
                # LLM dispatch by the call earlier in this method.
                classification_payload: dict | None = None
                if turn_shape is not None and hasattr(turn_shape, "to_dict"):
                    try:
                        classification_payload = turn_shape.to_dict()
                    except Exception:
                        classification_payload = None
                _tt.record_frame(_tt.FrameTraceRecord(
                    turn_id=task_id,
                    parent_turn_id=parent_turn_id,
                    captured_at=datetime.now(timezone.utc).isoformat(),
                    backend=self.backend,
                    engine_key=str(getattr(self, "_engine_key", "") or "llm"),
                    gen_id=int(getattr(self, "_current_task_id_seq", 0) or 0),
                    effort_tier=resolved_tier,
                    reasoning_mode=resolved_reasoning,
                    prompts_applied=resolved_prompts,
                    monothink_active=resolved_monothink,
                    classification=classification_payload,
                    final_messages=tuple(fmsgs),
                    system_prompt_chars=system_chars,
                    user_prompt_chars=user_chars,
                    total_chars=total_chars,
                    config_snapshot=config_snapshot,
                    metadata={
                        "model_loaded": self.model_loaded,
                        "ctx_limit": self.ctx_limit,
                    },
                ))
            except Exception as _frame_exc:
                # Surface the failure via the engine trace channel — the bare
                # except that lived here was hiding Layer B construction
                # errors (stage_traces wrote, frame_traces didn't) and made
                # the /trace/* HTTP surface return empty. Logged here so the
                # next turn's debug trace shows why the write didn't land.
                try:
                    self.sig_trace.emit(
                        f"[FRAME_TRACE] record_frame failed: {type(_frame_exc).__name__}: {_frame_exc}"
                    )
                except Exception:
                    pass

        if self.backend == "gguf":
            self.sig_gguf_generate.emit(
                {
                    "messages": messages,
                    "temp": temp,
                    "top_p": top_p,
                    "thinking_enabled": thinking_enabled,
                    "sampling": sampling,
                }
            )
        else:
            self.worker = GeneratorWorker(
                self.llm,
                messages,
                temp,
                top_p,
                thinking_enabled,
                sampling=sampling,
            )
            self.worker._task_id = task_id
            self.worker.token.connect(self.sig_token)
            self.worker.trace.connect(self._on_runtime_trace)
            self.worker.usage.connect(self._on_usage_update)
            self.worker.done.connect(self._on_gen_finish)
            self.worker.start()
        if self._gen_timeout_sec > 0:
            self._gen_timeout_timer.start(self._gen_timeout_sec * 1000)

    def stop_generation(self):
        try:
            self._gen_timeout_timer.stop()
        except Exception:
            pass
        if self._status == SystemStatus.LOADING and self.backend == "gguf":
            self._load_cancel_requested = True
            self.sig_trace.emit("-> load cancel requested; will stop after initialization completes")
            return

        if self._status == SystemStatus.LOADING and self.loader and self.loader.isRunning():
            self._load_cancel_requested = True
            self.sig_trace.emit("-> load cancel requested; will stop after initialization completes")
            return

        self._ephemeral_generation = False
        if self.backend == "gguf":
            try:
                self.sig_gguf_stop.emit()
            except Exception as exc:
                self.sig_trace.emit(f"[ENGINE] stop emit failed: {exc!r}")
            return

        # Cloud / OpenAI-compat path. Two-step cancel:
        #   1. Close the underlying HTTP response so the worker's
        #      `for raw_line in resp:` loop unblocks immediately. Without
        #      this, requestInterruption alone leaves the worker stuck
        #      reading from the socket, and the next click can race with
        #      teardown -- the path that was crashing.
        #   2. Then signal the QThread to interrupt cleanly.
        try:
            llm_client = getattr(self, "llm", None)
            if llm_client is not None and hasattr(llm_client, "cancel"):
                llm_client.cancel()
        except Exception as exc:
            self.sig_trace.emit(f"[ENGINE] cloud cancel failed: {exc!r}")
        try:
            worker = self.worker
            if worker is not None and worker.isRunning():
                worker.requestInterruption()
        except RuntimeError:
            # Worker QThread already deleted (race with finish/teardown).
            self.sig_trace.emit("[ENGINE] stop: worker already gone")
        except Exception as exc:
            self.sig_trace.emit(f"[ENGINE] stop request failed: {exc!r}")

    def shutdown(self):
        self._shutdown_requested = True
        self._gen_timeout_timer.stop()
        self.stop_generation()

        if self.worker:
            self.worker.requestInterruption()
            self.worker.wait(1500)
            self.worker = None

        if self.loader and self.loader.isRunning():
            self._load_cancel_requested = True
            self.loader.wait(150)

        if self._gguf_thread.isRunning():
            self.sig_gguf_shutdown.emit()
            self._gguf_thread.quit()
            self._gguf_thread.wait(1500)
