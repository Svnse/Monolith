import hashlib
import json
import os
from threading import Lock

from PySide6.QtCore import QObject, QThread, Signal, QTimer

from core.llm_config import MASTER_PROMPT, get_agent_prompt, load_config
from core.state import AppState, SystemStatus
from engine.agent_runtime import MAX_AGENT_STEPS, AgentMessage, AgentRuntime, ToolCall
from engine.contract import AgentOutcome, ContractFactory, ToolPolicy
from engine.protocol_adapter import ProtocolAdapter, get_profile
from engine.tools import set_workspace_root


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

    # Stop sequences to prevent model from emitting prompt scaffolding tags
    _STOP_SEQUENCES = ["</response>", "</answer>", "</output>", "<|end|>", "<|im_end|>"]

    def _try_chat_completion(self, messages, tools=None):
        kwargs = {
            "messages": messages,
            "temperature": self.temp,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stream": False,
            "stop": self._STOP_SEQUENCES,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        # Phase 5: thread grammar spec from contract if available
        contract = getattr(self, "_contract", None)
        if contract is not None and getattr(contract, "grammar_profile", None):
            try:
                from engine.protocol_adapter import get_grammar_profile
                gp = get_grammar_profile(
                    contract.model_profile_id,
                    contract.tool_policy.value if hasattr(contract.tool_policy, "value") else str(contract.tool_policy),
                )
                if gp is not None and gp.grammar_spec and gp.grammar_type == "bnf":
                    from llama_cpp import LlamaGrammar
                    kwargs["grammar"] = LlamaGrammar.from_string(gp.grammar_spec)
            except Exception:
                pass  # graceful fallback if grammar not supported

        return self.llm.create_chat_completion(**kwargs)

    def _chat_once_text(self, messages: list[dict]) -> str:
        response = self._try_chat_completion(messages)
        return self._extract_text(response)

    def _chat_once_agent(self, messages: list[dict], tools: list[dict]) -> AgentMessage:
        """Legacy agent path — returns normalized AgentMessage (no adapter)."""
        response = self._try_chat_completion(messages, tools=tools)
        return normalize_openai_response(response)

    def _chat_once_agent_raw(self, messages: list[dict], tools: list[dict]) -> dict:
        """Adapter-aware agent path — returns raw dict for ProtocolAdapter."""
        response = self._try_chat_completion(messages, tools=tools)
        return response

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
                self.usage.emit(len(token))


    def run(self):
        self.trace.emit(
            f"[WORKER] started: msgs={len(self.messages)}, temp={self.temp}, max_tokens={self.max_tokens}, agent_mode={self.agent_mode}"
        )
        completed = False
        assistant_text = ""
        loop_history = list(self.messages)

        try:
            if self.agent_mode and self.runtime is not None:
                # --- Set up Protocol Adapter ---
                model_profile_id = getattr(self, "_model_profile_id", "local_xml")
                profile = get_profile(model_profile_id)
                adapter = ProtocolAdapter(profile)
                self.runtime._protocol_adapter = adapter

                # --- Set up Execution Contract ---
                contract = getattr(self, "_contract", None)
                self.runtime._contract = contract

                # --- Wire adapter-aware LLM call ---
                self.runtime._llm_call = self._chat_once_agent_raw
                self.runtime._should_stop = self.isInterruptionRequested
                self.runtime._emit_event = self._emit_runtime_event

                self.trace.emit(
                    f"[WORKER] adapter: profile={profile.profile_id}, "
                    f"format={profile.tool_call_format.value}, "
                    f"strict={profile.strict_mode}"
                )
                if contract is not None:
                    self.trace.emit(
                        f"[WORKER] contract: policy={contract.tool_policy.value}, "
                        f"max_inferences={contract.max_inferences}, "
                        f"retries={contract.max_format_retries}"
                    )

                # --- STEP-WISE EXECUTION (Option B) ---
                # Use StepwiseAgentRuntime to yield between steps
                # This allows Qt signals to flush to UI in real-time
                from engine.agent_runtime_stepper import StepwiseAgentRuntime
                
                stepper = StepwiseAgentRuntime(
                    runtime=self.runtime,
                    messages=self.messages,
                    emit_event=self._emit_runtime_event,
                )
                
                # Initialize
                init_events = stepper.initialize()
                for ev in init_events:
                    self._emit_runtime_event(ev)
                
                # Store stepper on self so it can be resumed from WAIT_ACK
                self._stepper = stepper

                # Execute steps — worker thread yields between steps
                step_count = 0
                awaiting_ack = False
                self.trace.emit(f"[WORKER] starting step loop, should_continue={stepper.should_continue()}")
                
                while stepper.should_continue():
                    step_count += 1
                    self.trace.emit(f"[WORKER] beginning step {step_count}, state={stepper._current_state.value if stepper._current_state else 'None'}")

                    # Check for interruption
                    if self.isInterruptionRequested():
                        self.trace.emit("[WORKER] interruption requested, stopping")
                        break

                    # Execute one step
                    self.trace.emit(f"[WORKER] calling stepper.step() for step {step_count}")
                    step_result = stepper.step()
                    self.trace.emit(f"[WORKER] stepper.step() returned for step {step_count}")

                    # Emit trace for state transition visibility
                    self.trace.emit(
                        f"[WORKER] step {step_count}: state={step_result.state.value}, "
                        f"continue={step_result.should_continue}"
                    )

                    if not step_result.should_continue:
                        if step_result.awaiting_ack:
                            # WAIT_ACK: worker thread dies cleanly.
                            # Stepper stays in memory. MonoGuard will spawn
                            # a new worker via resume_agent() after user action.
                            self.trace.emit("[WORKER] entering WAIT_ACK — thread will terminate")
                            awaiting_ack = True
                        break

                if awaiting_ack:
                    # Don't emit done — the stepper is parked in WAIT_ACK.
                    # The resume path will create a new worker to finish.
                    # Note: Worker will be cleaned up by LLMEngine when replaced
                    return

                # Get final result
                result = stepper.get_result()
                if result is not None:
                    self._last_run_result = result
                    completed = result.success
                    assistant_text = result.output
                    loop_history = result.history
                else:
                    completed = False
                    
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
            import traceback
            self.trace.emit(f"<span style='color:red'>ERROR: {e}\n{traceback.format_exc()}</span>")
        finally:
            self.trace.emit(f"[WORKER] finished: completed={completed}, text_len={len(assistant_text)}")
            self.done.emit(completed, assistant_text, loop_history)
            # Note: Worker cleanup handled by LLMEngine _on_gen_finish


class _ResumeWorker(QThread):
    """
    Worker thread that resumes a StepwiseAgentRuntime from WAIT_ACK.

    Created by LLMEngine.resume_agent() after user approves/denies.
    """
    token = Signal(str)
    trace = Signal(str)
    done = Signal(bool, str, list)
    usage = Signal(int)
    event = Signal(dict)

    def __init__(self, stepper, action: str):
        super().__init__()
        self._stepper = stepper
        self._action = action

    def _emit_runtime_event(self, payload: dict) -> None:
        self.event.emit(payload)
        event_name = payload.get("event")
        if event_name in ("LLM_TOKEN", "FINAL_OUTPUT"):
            token_data = payload.get("data", "")
            if isinstance(token_data, str) and token_data:
                self.token.emit(token_data)
                self.usage.emit(len(token_data))

    def run(self):
        completed = False
        assistant_text = ""
        loop_history = []

        try:
            stepper = self._stepper

            # Rewire the event emitter to this worker's signals
            stepper._emit_event = self._emit_runtime_event
            stepper._runtime._emit_event = self._emit_runtime_event
            stepper._runtime._should_stop = self.isInterruptionRequested

            # Resume from ACK
            self.trace.emit(f"[RESUME] action={self._action}")
            stepper.resume_from_ack(self._action)

            # Continue stepping
            step_count = 0
            awaiting_ack = False
            while stepper.should_continue():
                step_count += 1

                if self.isInterruptionRequested():
                    self.trace.emit("[RESUME] interruption requested, stopping")
                    break

                step_result = stepper.step()
                self.trace.emit(
                    f"[RESUME] step {step_count}: state={step_result.state.value}, "
                    f"continue={step_result.should_continue}"
                )

                if not step_result.should_continue:
                    if step_result.awaiting_ack:
                        self.trace.emit("[RESUME] re-entering WAIT_ACK — thread will terminate")
                        awaiting_ack = True
                    break

            if awaiting_ack:
                return

            result = stepper.get_result()
            if result is not None:
                completed = result.success
                assistant_text = result.output
                loop_history = result.history
            else:
                completed = False

        except Exception as e:
            self.trace.emit(f"[RESUME] EXCEPTION: {e}")
            import traceback
            self.trace.emit(f"<span style='color:red'>RESUME ERROR: {e}\n{traceback.format_exc()}</span>")
        finally:
            self.trace.emit(f"[RESUME] finished: completed={completed}, text_len={len(assistant_text)}")
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
        self._runtime = AgentRuntime(llm_call=lambda _m, _t: AgentMessage(role="assistant", content=""), emit_event=self.sig_agent_event.emit)

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

        # Eager-load agent runtime dependencies to avoid first-call delay
        # These modules have regex patterns that compile on first use
        try:
            from engine import contract, protocol_adapter
            # Touch intent classification to pre-compile regexes if needed
            _ = contract.classify_intent("warmup", "code")
            _ = protocol_adapter.get_profile("local_xml")
            self.sig_trace.emit("→ agent runtime warmed up")
        except Exception as e:
            self.sig_trace.emit(f"[WARN] agent runtime warmup failed: {e}")

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

    def _compile_system_prompt(self, config, agent_mode=False, model_profile_id="local_xml"):
        if agent_mode:
            base_prompt = get_agent_prompt(model_profile_id)
        else:
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
            runtime=self._runtime if self._worker_agent_mode else None,
            agent_mode=self._worker_agent_mode,
        )

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
        config = payload.get("config")
        if config is None:
            config = load_config()

        request_agent_mode = bool(payload.get("agent_mode", False))

        # Model profile for protocol adapter alignment
        model_profile_id = str(payload.get("model_profile_id", "local_xml"))

        system_prompt = self._compile_system_prompt(config, agent_mode=request_agent_mode, model_profile_id=model_profile_id)
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

        # Clean up any existing worker before starting a new one (Bug 2 fix)
        if self.worker is not None and self.worker.isRunning():
            self.worker.requestInterruption()
            self.worker.wait(3000)
        if self.loader is not None and self.loader.isRunning():
            self.loader.wait(3000)

        set_workspace_root()
        self._worker_seed_count = len(messages)
        self._worker_agent_mode = request_agent_mode

        # --- Create ExecutionContract for agent mode ---
        contract = None
        if request_agent_mode:
            source_page = str(payload.get("source_page", "code"))
            allowed_tools = self._runtime._capability_manager.allowed_tools() if self._runtime else None
            contract_factory = ContractFactory(
                default_profile_id=model_profile_id,
                default_max_inferences=MAX_AGENT_STEPS if not hasattr(self, '_runtime') else 25,
                default_ctx_limit=self.ctx_limit,
            )
            contract = contract_factory.create(
                prompt=prompt,
                source_page=source_page,
                allowed_tools=allowed_tools,
                model_profile_id=model_profile_id,
                ctx_limit=self.ctx_limit,
                model_fingerprint=getattr(self, "_model_fingerprint", ""),
            )
            self.sig_trace.emit(
                f"[ENGINE] contract: id={contract.contract_id[:8]}..., "
                f"policy={contract.tool_policy.value}, "
                f"profile={contract.model_profile_id}"
            )

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


    def resume_agent(self, action: str = "approve") -> bool:
        """
        Resume an agent parked in WAIT_ACK state.

        Creates a new GeneratorWorker that picks up the stepper from memory.
        action: "approve" | "deny" | "timeout"

        Returns True if resume was initiated, False if no stepper to resume.
        """
        # Find the parked stepper from the last worker
        stepper = None
        if self.worker and hasattr(self.worker, "_stepper"):
            stepper = self.worker._stepper

        if stepper is None:
            self.sig_trace.emit("[ENGINE] resume_agent: no parked stepper found")
            return False

        from engine.contract import RuntimeState as RS
        if stepper._current_state != RS.WAIT_ACK:
            self.sig_trace.emit(f"[ENGINE] resume_agent: stepper not in WAIT_ACK (state={stepper._current_state})")
            return False

        self.sig_trace.emit(f"[ENGINE] resume_agent: action={action}")
        self.set_status(SystemStatus.RUNNING)

        # Clean up old worker if it exists but is finished
        old_worker = self.worker
        if old_worker is not None:
            # Only cleanup if finished - if still running, something is wrong
            if not old_worker.isRunning():
                # Worker finished - safe to disconnect and replace
                try:
                    old_worker.token.disconnect()
                    old_worker.trace.disconnect()
                    old_worker.usage.disconnect()
                    old_worker.event.disconnect()
                    old_worker.done.disconnect()
                except Exception:
                    pass
                self.worker = None
            else:
                # Worker still running - this shouldn't happen in WAIT_ACK flow
                self.sig_trace.emit("[ENGINE] WARNING: worker still running during resume, waiting...")
                old_worker.requestInterruption()
                if not old_worker.wait(1000):
                    old_worker.terminate()
                    old_worker.wait(500)
                try:
                    old_worker.token.disconnect()
                    old_worker.trace.disconnect()
                    old_worker.usage.disconnect()
                    old_worker.event.disconnect()
                    old_worker.done.disconnect()
                except Exception:
                    pass
                self.worker = None

        # Create a resume worker that runs the stepper from WAIT_ACK
        resume_worker = _ResumeWorker(stepper, action)
        resume_worker.token.connect(self.sig_token)
        resume_worker.trace.connect(self.sig_trace)
        resume_worker.usage.connect(self._on_usage_update)
        resume_worker.event.connect(self.sig_agent_event)
        resume_worker.done.connect(self._on_gen_finish)

        self.worker = resume_worker
        self._worker_agent_mode = True
        resume_worker.start()
        return True

    def runtime_command(self, command: str, payload: dict | None = None) -> dict:
        request = payload if isinstance(payload, dict) else {}
        
        # OFAC v0.2: Handle ack_decision to resume from WAIT_ACK
        if command == "ack_decision":
            decision = request.get("decision", "deny")
            success = self.resume_agent(action=decision)
            result = {"ok": success, "action": decision}
            self.sig_agent_event.emit({"event": "ACK_DECISION_HANDLED", "decision": decision, "success": success})
            return result
        
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
