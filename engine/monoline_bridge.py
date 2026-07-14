"""Lazy, in-process Monoline integration. Imported ONLY on an actual Monoline run
(never at Genesis startup -- INV-#0). Binds Monolith's Phase-A atom as the per-block
'monolith' provider, a gated L3 tool surface, conservative egress, and records the
run skeleton into the single store (turn_trace)."""
from __future__ import annotations

import os
import sys
import threading
import time
import uuid
from pathlib import Path

from core.llm_config import load_config
from core.run_model import BlockFinished, RunBlockSpec, RunFinished, RunStarted

_MONOLINE_CACHE: dict | None = None
_LOAD_LOCK = threading.Lock()

CHAT_ALLOWED_LLM_PROVIDERS = frozenset({"local", "monolith", "api", "airllm"})

# Per-block output is persisted (truncated) into the monoline_block payload so the
# historical run browser matches the live card. PAYLOAD ONLY -- never evidence/fault_kind
# (spec 2026-06-14 §4: keeps model-generated text out of the model-facing fault readers).
_OUTPUT_PREVIEW_CHARS = 1024

# Monolith-provider LLM blocks want a FULL answer, not run_subagent's L3 leaf default
# (1024). A DeepSeek-class thinking model charges its reasoning AGAINST max_tokens, so a
# small cap truncates before the final answer (finish_reason=length, empty content) -> the
# block silently returns "" and the flow dies. Floor the per-block budget to a value proven
# to finish (8192 -> finish_reason=stop on the heavy synthesis blocks), and regenerate on an
# empty result before failing the block. See .workshop_empty_output_rootcause.md.
_MONOLINE_LLM_MIN_TOKENS = 8192
# ...but a per-block OUTPUT budget must also stay UNDER the provider's output cap. load_config()
# carries the GLOBAL runtime max_tokens, which on a long-context cloud profile is the CONTEXT
# window (e.g. 1,000,000) -- far above what a cloud provider accepts as max_tokens (DeepSeek
# HTTP-400s anything >393216). The streaming worker dodges this by capping cloud calls at 65536
# (engine/llm.py setdefault); mirror that ceiling here so the monolith-provider sync path is
# consistent and never sends a context-window-sized max_tokens to the wire.
_MONOLINE_LLM_MAX_TOKENS = 65536
_MONOLINE_EMPTY_REGEN_MAX = 3


def _projection_enabled() -> bool:
    """Slices 1-2 (detector + verifier projection into the block payload). Default OFF.
    Independent of MONOLITH_TURN_TRACE_V1 (which gates the record_* writes themselves)."""
    return os.environ.get("MONOLINE_PROJECTION", "").strip().lower() in ("1", "true", "yes", "on")


def _is_monoline_checkout(path: Path) -> bool:
    return path.is_dir() and (path / "core" / "monoline_headless.py").is_file()


def _resolve_monoline_root() -> Path:
    """Resolve an optional Monoline checkout without a maintainer-specific path."""
    configured = str(os.environ.get("MONOLITH_MONOLINE_ROOT", "") or "").strip()
    if configured:
        root = Path(configured).expanduser()
        if _is_monoline_checkout(root):
            return root.resolve()
        raise RuntimeError(
            "Configured Monoline directory is unavailable. "
            "Check MONOLITH_MONOLINE_ROOT."
        )

    repo_root = Path(__file__).resolve().parents[1]
    candidates = (
        repo_root / "Monoline",
        repo_root.parent / "Monoline",
        repo_root.parent / "Project" / "Monoline",
    )
    for candidate in candidates:
        if _is_monoline_checkout(candidate):
            return candidate.resolve()
    raise RuntimeError(
        "Monoline is not configured. Set MONOLITH_MONOLINE_ROOT to the "
        "Monoline repository directory."
    )


def ensure_monoline_on_path() -> Path:
    """Validate the Monoline plugin dir and pin Monoline's data tree to Monolith's, so
    saved worlds land under MONOLITH_ROOT (matching the registry's WORKFLOWS_DIR). Monoline's
    core/paths.py reads MONOLITH_ROOT AT IMPORT TIME, so this must run BEFORE the swap window.
    Does NOT permanently mutate sys.path -- load_monoline() owns the import window so Monoline's
    top-level ui/ never lingers to shadow Monolith's ui/ on a later uncached import."""
    root = _resolve_monoline_root()
    from core.paths import MONOLITH_ROOT  # `core` == Monolith here (pre-swap)
    os.environ.setdefault("MONOLITH_ROOT", str(MONOLITH_ROOT))
    return root


def load_monoline() -> dict:
    """Lazy + cached + thread-safe. Imports Monoline's modules in-process DESPITE the two-repo
    top-level `core/` collision (Monolith core = namespace pkg; Monoline core = REGULAR pkg;
    both ship a core/paths.py). A bare shared `core` cannot resolve both repos, so the bare
    `import monoline` shim approach FAILS in-process (verified). We do a ONE-TIME guarded
    sys.modules swap instead:
        snapshot Monolith's core.* -> evict them -> put Monoline first on sys.path ->
        import EVERY core.monoline_* (so all lazy `from core.monoline_X import ...` targets are
        cached; run_workflow/build_preset import core.monoline_engine/_tools/_runtime/etc. lazily
        at call time) -> restore Monolith's shared core.* -> KEEP Monoline's core.monoline_*
        resident (names Monolith never uses -> zero collision) -> drop Monoline off sys.path.
    Monoline's modules then run via these cached objects (their globals are bound at import);
    Monolith's core is byte-for-byte restored. PROVEN by a swap round-trip (run_workflow returns
    'echo:hi'; core.paths/core.subagent stay Monolith's; sys.path clean).
    CONCURRENCY: the swap mutates global sys.modules. The lock serializes two cold loads, but it
    does NOT protect a concurrent uncached `import core.X` on ANY OTHER live thread (Qt/background)
    during the one-time swap window -- that would briefly see Monoline's `core`. So WARM THIS ON THE
    MAIN THREAD at a quiescent point (e.g. before spawning the Task-5 pipeline worker) so the window
    runs with no concurrent importers and every later caller hits the cache."""
    global _MONOLINE_CACHE
    if _MONOLINE_CACHE is not None:
        return _MONOLINE_CACHE
    with _LOAD_LOCK:
        if _MONOLINE_CACHE is not None:  # double-checked under lock
            return _MONOLINE_CACHE
        import importlib
        root = ensure_monoline_on_path()  # pins MONOLITH_ROOT env while `core` is still Monolith's
        root_str = str(root)
        saved = {n: m for n, m in sys.modules.items()
                 if n == "core" or n.startswith("core.")}
        monoline_submods: dict = {}
        cache: dict = {}
        # Insert UNCONDITIONALLY at index 0 (not gated on "already present") so resolution in the
        # window is deterministic -- Monoline's regular `core` pkg wins regardless of any pre-existing
        # sys.path entry -- and so the finally always removes exactly the entry we added, restoring
        # sys.path to its pre-call state (no lingering Monoline root to shadow Monolith's top-level ui/).
        sys.path.insert(0, root_str)
        try:
            for n in list(saved):
                del sys.modules[n]
            for f in sorted((root / "core").glob("monoline_*.py")):
                importlib.import_module(f"core.{f.stem}")
            cache = {
                "headless": sys.modules["core.monoline_headless"],
                "store": sys.modules["core.monoline_store"],
                "model": sys.modules["core.monoline_model"],
                "blueprint": sys.modules["core.monoline_blueprint"],
                "engine": sys.modules["core.monoline_engine"],
            }
            monoline_submods = {n: m for n, m in sys.modules.items()
                                if n.startswith("core.monoline_")}
        finally:
            for n in [k for k in list(sys.modules)
                      if k == "core" or k.startswith("core.")]:
                del sys.modules[n]
            sys.modules.update(saved)             # Monolith's shared core.* restored
            sys.modules.update(monoline_submods)  # Monoline's core.monoline_* stay resident
            try:
                sys.path.remove(root_str)  # removes the entry we inserted at 0 (first occurrence)
            except ValueError:
                pass
        _MONOLINE_CACHE = cache
        return _MONOLINE_CACHE


def make_engine_func(*, parent_turn_id, spawn_budget, should_cancel, is_busy,
                     allow_egress: bool = False, busy_retries: int = 3,
                     busy_backoff: float = 0.25):
    """The injected engine_func. Serializes its whole body through a per-run lock so
    the 8 pool threads never contend; the native branch additionally takes the global
    generation_lock (blocking, vs Genesis/expedition); the 'monolith' branch calls the
    atom with bounded retry on a busy refusal, then raises (fails the block, no hang)."""
    import threading
    run_lock = threading.Lock()  # per-run: serialize the pool threads
    from core.generation import generation_lock
    from core.subagent import run_subagent, SubagentResult  # noqa: F401
    native_engine_call = load_monoline()["engine"].engine_call

    def _engine(messages, config):
        provider = str(config.get("provider", "local") or "local").strip().lower()
        with run_lock:  # one block at a time within THIS run
            if provider == "monolith":
                # run_subagent -> generate_sync_parts_from_config is API-only (needs
                # api_base/api_model). Hand it MONOLITH's runtime config (the loaded model's
                # endpoint), NOT the Monoline block config -- which carries no api_base, so the
                # generator raised "Missing api_base or api_model" and the block yielded EMPTY
                # output. Matches skill_runtime's spawn path (load_config()). The block config is
                # still used for the label + generation budget below.
                sub_config = load_config()
                label = str(config.get("label", "llm"))
                # Budget: the monolith branch previously passed NO llm_config, so run_subagent's
                # 1024-token LEAF default applied -- which truncates a DeepSeek-class thinking
                # model's reasoning BEFORE the final answer (reasoning is charged against
                # max_tokens; finish_reason=length, empty content). Honor the block's max_tokens,
                # floored to a runtime budget proven to leave room for the answer.
                # Floor to _MONOLINE_LLM_MIN_TOKENS for thinking headroom and honor a deliberately
                # larger BLOCK budget, but do NOT inherit sub_config's (load_config) max_tokens --
                # that is the global runtime ceiling (context-window-sized on cloud profiles) and
                # overflows the provider output cap. Clamp to _MONOLINE_LLM_MAX_TOKENS.
                gen_budget = min(max(int(config.get("max_tokens") or 0),
                                     _MONOLINE_LLM_MIN_TOKENS),
                                 _MONOLINE_LLM_MAX_TOKENS)
                gen_temp = float(config.get("temperature", sub_config.get("temp", 0.4)))

                def _one_inference(budget):
                    # ONE block inference, preserving the busy refuse-not-hang retry.
                    busy_attempts = 0
                    while True:
                        res = run_subagent(
                            messages, sub_config, level=3, frame=f"monoline:{label}",
                            parent_turn_id=parent_turn_id, allowed_tools=frozenset(),
                            should_cancel=should_cancel, max_followups=0,
                            llm_config={"max_tokens": int(budget), "temp": gen_temp},
                            spawn_budget=spawn_budget, is_busy=is_busy)
                        if res.halt_reason != "busy":
                            return res
                        if busy_attempts >= busy_retries:
                            raise RuntimeError(
                                "monolith provider: generator busy after "
                                f"{busy_retries} retries (block failed, did not hang)")
                        busy_attempts += 1
                        if busy_backoff:
                            time.sleep(busy_backoff * busy_attempts)

                # Empty-output regen: a thinking model can spend its whole budget on reasoning
                # and emit no final answer (non-deterministic, budget-driven). Regenerate with an
                # ESCALATING budget up to _MONOLINE_EMPTY_REGEN_MAX times; a user-cancel is NOT an
                # empty-to-retry. On exhaustion FAIL the block (halt the run) with a diagnostic.
                res = None
                _max_attempt = min(gen_budget * _MONOLINE_EMPTY_REGEN_MAX, _MONOLINE_LLM_MAX_TOKENS)
                for regen in range(_MONOLINE_EMPTY_REGEN_MAX):
                    # Escalate the budget on empty regen, but never past the provider output cap.
                    res = _one_inference(min(gen_budget * (regen + 1), _MONOLINE_LLM_MAX_TOKENS))
                    if res.text.strip() or res.halt_reason == "cancelled":
                        return res.text
                raise RuntimeError(
                    f"monoline block {label!r}: no final answer after "
                    f"{_MONOLINE_EMPTY_REGEN_MAX} attempts (max_tokens up to "
                    f"{_max_attempt}). The thinking model spent its "
                    "token budget on reasoning and never emitted content -- raise the block's "
                    "max_tokens or shorten the prompt. Last trace: "
                    f"{((res.thinking if res else '') or '(none)')[:200]}")
            if provider in ("api", "airllm") and not allow_egress:
                raise RuntimeError(
                    f"egress provider {provider!r} requires explicit opt-in "
                    "(allow_egress); chat-active flows default to local/monolith only")
            with generation_lock:  # native gen serialized vs Genesis/expedition
                return native_engine_call(messages, config)

    return _engine


def make_tool_func(*, parent_turn_id, should_cancel):
    """The injected tool_func. Adapts (tool_id, payload, config) to a fresh L3
    ToolExecutionContext via derive_child_context and routes through the Phase-A gate.
    Any tool outside L3_LEAF_TOOLS is denied by the gate's own capability check."""
    from core.paths import LOG_DIR
    from core.skill_runtime import (ToolExecutionContext, L1_PRINCIPAL_TOOLS,
                                     L3_LEAF_TOOLS, derive_child_context,
                                     execute_tool_call)

    root = ToolExecutionContext(
        archive_dir=LOG_DIR, level=1, allowed_tools=L1_PRINCIPAL_TOOLS,
        parent_turn_id=str(parent_turn_id or ""),
        should_cancel=should_cancel)
    leaf = derive_child_context(root, 3, label="monoline-tool")  # allowed_tools <= L3 floor

    def _tool(tool_id: str, payload: str, config: dict) -> str:
        # Map the single-string payload onto the tool's named arg. Pass ONLY that arg -- a
        # strict tool schema (e.g. calculate wants 'expr') rejects an extra generic 'input'
        # field ("unknown field(s): input"). 'input' is the fallback for unmapped tools only.
        cmd = {"tool": str(tool_id)}
        if tool_id in ("read_file", "list_files"):
            cmd["path"] = str(payload)
        elif tool_id in ("grep", "find_files"):
            cmd["pattern"] = str(payload)
        elif tool_id == "calculate":
            cmd["expr"] = str(payload)
        else:
            cmd["input"] = str(payload)
        return execute_tool_call(cmd, leaf)

    return _tool


def _iter_preset_blocks(preset):
    for blk in getattr(preset, "blocks", []):
        yield blk
    for composite in getattr(preset, "composites", []):
        for blk in getattr(composite, "blocks", []):
            yield blk


def _block_label(block) -> str:
    label = str(getattr(block, "label", "") or "").strip()
    return label or str(getattr(block, "id", "") or "block")


def validate_chat_workflow(workflow, *, allow_egress: bool = False) -> str | None:
    """Return a user-visible problem string for chat-active flows, or None.

    Monolith-launched chat flows support Monolith-backed blocks and valid local GGUF
    blocks. Remote/air providers remain explicit opt-in, and blank local model
    configs are rejected before they can turn into an empty successful run.
    """
    m = load_monoline()
    preset = m["headless"].load_workflow(str(workflow.source_path))
    for block in _iter_preset_blocks(preset):
        if str(getattr(block, "kind", "") or "").strip().lower() != "llm":
            continue
        config = getattr(block, "config", {}) or {}
        provider = str(config.get("provider", "local") or "local").strip().lower()
        label = _block_label(block)
        if provider not in CHAT_ALLOWED_LLM_PROVIDERS:
            return f"{label}: unknown LLM provider {provider!r}"
        if provider == "local" and not str(config.get("model_path", "") or "").strip():
            return (
                f"{label}: local provider requires a model_path. "
                "Choose a GGUF model or set this block's provider to monolith."
            )
        if provider in ("api", "airllm") and not allow_egress:
            return (
                f"{label}: egress provider {provider!r} requires explicit opt-in; "
                "chat-active flows default to local/monolith only."
            )
    return None


def summarize_run_failure(run) -> str | None:
    result = getattr(run, "result", None)
    top_error = str(getattr(result, "error", "") or "").strip()
    if top_error:
        return top_error
    for sr in list(getattr(result, "step_log", None) or []):
        error = str(getattr(sr, "error", "") or "").strip()
        if error:
            label = str(getattr(sr, "block_label", "") or "block")
            kind = str(getattr(sr, "step_kind", "") or "step")
            return f"{label} [{kind}]: {error}"
    statuses = getattr(result, "block_status", None) or {}
    if isinstance(statuses, dict):
        failed = [str(k) for k, v in statuses.items() if str(v).strip().lower() == "error"]
        if failed:
            return "workflow reported error block(s): " + ", ".join(failed[:5])
    return None


def run_monoline_world(workflow, *, user_input, parent_turn_id, spawn_budget,
                       should_cancel, is_busy, on_step, should_stop,
                       allow_egress: bool = False, max_steps: int = 60,
                       on_event=None):
    """Bind the two callables, mint a run-root turn_id, record the run-root frame
    (parent = the live L1 turn), then run off the caller's thread. Per-block frames
    (the 'monolith' atom writes its own) and a fault overlay tree UNDER the run root.
    NOTE (decision 1): the flow's own state_store.json (plugin scratch) is EXEMPT from
    the single-store rule -- run_workflow uses Monoline's per-flow store; we do NOT
    route it through turn_trace.

    on_event (spec 2026-06-14): optional sink for the normalized RunEvent stream
    (RunStarted -> BlockFinished* -> RunFinished) the unified RunView consumes. Best-effort;
    a raising sink never breaks the run. The native Monoline-v2 organ layer emits the same
    stream later (drop-in)."""
    import core.turn_trace as tt
    m = load_monoline()
    run_root = uuid.uuid4().hex

    def _emit(event) -> None:
        if on_event is not None:
            try:
                on_event(event)
            except Exception:
                pass

    # Precompute the set of block ids whose provider is "monolith" so that _on_step can
    # skip writing a bridge frame for those blocks (the atom already writes its own frame
    # via run_subagent).  This is immutable and shared read-only across pool threads, so
    # it is race-safe even when ActivationRuntime dispatches llm blocks concurrently via
    # ThreadPoolExecutor.  We walk both top-level blocks and composite sub-blocks,
    # mirroring _apply_llm_overrides, so nested monolith-provider blocks are also covered.
    def _collect_atom_block_ids(preset) -> set[str]:
        ids: set[str] = set()
        for blk in getattr(preset, "blocks", []):
            if str(getattr(blk, "config", {}).get("provider", "")).strip().lower() == "monolith":
                ids.add(str(blk.id))
        for composite in getattr(preset, "composites", []):
            for blk in getattr(composite, "blocks", []):
                if str(getattr(blk, "config", {}).get("provider", "")).strip().lower() == "monolith":
                    ids.add(str(blk.id))
        return ids

    # INVARIANT: this parse sees NO llm_overrides, and run_monoline_world passes none to
    # run_workflow below.  If the bridge is ever changed to pass llm_overrides that rewrite a
    # block's `provider` (see Monoline _apply_llm_overrides), this precomputed set will desync
    # from the executed preset (causing a double-frame or a missed skip) -- recompute it from
    # the same override-applied preset in that case.
    try:
        _preset = m["headless"].load_workflow(str(workflow.source_path))
        atom_block_ids: set[str] = _collect_atom_block_ids(_preset)
    except Exception:
        atom_block_ids = set()

    # graph + wires from the on-disk .monoline (the registry's stable contract). Stored in the
    # run-root frame metadata so the historical browser rehydrates the RunModel from turn_trace
    # ALONE -- surviving a later flow deletion (spec §9) -- and reused for the run_started event.
    graph_dicts: list = []
    wires: list = []
    try:
        import json as _json
        _doc = _json.loads(Path(workflow.source_path).read_text(encoding="utf-8"))
        for _b in (_doc.get("blocks") or []):
            _bid = str(_b.get("id", "") or "")
            _label = str((_b.get("config") or {}).get("label", "") or _b.get("label", "") or _bid)
            graph_dicts.append({"id": _bid, "label": _label, "kind": str(_b.get("kind", "") or "")})
        for _pair in (_doc.get("connections") or []):
            try:
                wires.append(f"{_pair[0]} -> {_pair[1]}")
            except Exception:
                pass
    except Exception:
        pass

    # (1) run-root frame: parent = the live L1 turn, so the run trees under the turn.
    try:
        tt.record_frame(tt.FrameTraceRecord(
            turn_id=run_root, captured_at=_now(), backend="monoline",
            engine_key=f"monoline:{workflow.name}", gen_id=0,
            final_messages=tuple(), system_prompt_chars=0, user_prompt_chars=len(user_input or ""),
            total_chars=len(user_input or ""),
            # None (NULL) when parentless (dry-run) -> run-root is its own governance root,
            # found via the monoline_block fault-union; a real L1 turn id when chat-launched
            # -> the run trees UNDER that turn (found via the frame-parent edge).
            parent_turn_id=(str(parent_turn_id) if parent_turn_id else None),
            # user_input + graph + wires (the run's shape) so the browser rehydrates from
            # turn_trace alone. user_input is the USER's own text (not model-generated) and
            # graph/wires are structure -> no provenance concern (spec §4).
            metadata={"kind": "workflow", "flow": workflow.id, "name": workflow.name,
                      "user_input": str(user_input or "")[:_OUTPUT_PREVIEW_CHARS],
                      "graph": graph_dicts, "wires": wires}))
    except Exception:
        pass

    _emit(RunStarted(
        run_id=run_root, flow_id=str(workflow.id), name=str(workflow.name),
        user_input=str(user_input or ""),
        graph=[RunBlockSpec(id=g["id"], label=g["label"], kind=g["kind"]) for g in graph_dicts],
        wires=list(wires)))

    # (3) per-block fault overlay: one record_fault per StepResult (the Workshop read).
    def _on_step(sr):
        ok = not bool(getattr(sr, "error", None))
        _started = float(getattr(sr, "started_at", 0.0) or 0.0)
        _completed = float(getattr(sr, "completed_at", 0.0) or 0.0)
        _outputs = {str(k): str(v) for k, v in (getattr(sr, "outputs", {}) or {}).items()}
        payload = {"block_id": str(getattr(sr, "step_id", "") or ""),
                   "block_label": getattr(sr, "block_label", ""),
                   "step_kind": getattr(sr, "step_kind", ""),
                   "ok": ok, "error": str(getattr(sr, "error", "") or ""),
                   # spec §4: per-port outputs (truncated) + block_id + precise timing -> PAYLOAD
                   # ONLY (never evidence/fault_kind -> invisible to model-facing fault readers).
                   "outputs": {k: v[:_OUTPUT_PREVIEW_CHARS] for k, v in _outputs.items()},
                   "started_at": _started, "completed_at": _completed}
        # --- projection hook (Tasks 4-5) ---
        # Task 4: run fault detectors on llm block output; fold findings into payload.
        # Only when MONOLINE_PROJECTION is ON; findings go into payload only (never emit_fault).
        # tool_no_fire is filtered — it false-fires on Monoline's separate-block tool model.
        if _projection_enabled() and str(getattr(sr, "step_kind", "")).lower() == "call_llm":
            import core.fault_response as _fr
            outs = getattr(sr, "outputs", {}) or {}
            text = outs.get("response") or outs.get("text") or " ".join(
                str(v) for v in outs.values() if isinstance(v, str))
            if text:
                try:
                    findings = [f for f in _fr.run_all_detectors(text, run_root, {})
                                if getattr(f, "fault_kind", "") != "tool_no_fire"]
                    if findings:
                        payload["detectors"] = [
                            {"kind": getattr(f, "fault_kind", ""),
                             "name": getattr(f, "detector_name", ""),
                             "evidence": (getattr(f, "evidence", "") or "")[:200]}
                            for f in findings]
                except Exception:
                    pass
                try:
                    from core.response_verifier import verify_response
                    vr = verify_response(raw_answer=text, public_answer=text,
                                         tools_used=(), task_type="chat")
                    payload["verdict"] = {
                        "verdict": str(getattr(vr, "verdict", "") or ""),
                        "reasons": [str(getattr(f, "message", "")) for f in
                                    list(getattr(vr, "findings", None) or [])[:5]]
                    }
                except Exception:
                    pass
        try:
            tt.record_fault(tt.FaultTraceRecord(
                turn_id=run_root, parent_turn_id=run_root, seq=0, emitted_at=_now(),
                event_kind="monoline_block",
                source_kind="kernel",               # was "producer" — out of get_tool_usage; "observation" is authority_tier vocab
                source_name="monoline_bridge",
                authority_tier="observation",
                fault_kind=None,                    # ALWAYS None — out of the fault readers
                severity=None,                      # must match fault_kind=None; error detail lives in payload
                payload=payload))
        except Exception:
            pass
        # (2) per-LLM-block frame for NON-atom providers (the 'monolith' atom writes
        # its own via run_subagent). Only llm blocks are inferences -> frame-worthy.
        # atom_block_ids is a precomputed, immutable set keyed on block id (step_id ==
        # block.id == step.id in ActivationRuntime).  It is race-safe across pool threads.
        if (str(getattr(sr, "step_kind", "")).lower() == "call_llm"
                and str(getattr(sr, "step_id", "")) not in atom_block_ids):
            out_chars = sum(len(str(v)) for v in (getattr(sr, "outputs", {}) or {}).values())
            try:
                tt.record_frame(tt.FrameTraceRecord(
                    turn_id=uuid.uuid4().hex, captured_at=_now(), backend="monoline",
                    engine_key=f"monoline:{getattr(sr, 'block_label', 'llm')}", gen_id=0,
                    final_messages=tuple(), system_prompt_chars=0, user_prompt_chars=0,
                    total_chars=out_chars, parent_turn_id=run_root,
                    metadata={"kind": "monoline_block",
                              "block_label": getattr(sr, "block_label", ""),
                              "step_kind": getattr(sr, "step_kind", "")}))
            except Exception:
                pass
        # RunEvent: the unified RunView's per-block update (live inline card + run browser).
        # A block that finished with NO output while a stop is in flight was interrupted mid-stream
        # -> report a clean "stopped" state (not "done") so the RunView renders it neutrally.
        block_status = "error" if not ok else "done"
        _has_output = any(str(v).strip() for v in _outputs.values())
        if block_status == "done" and not _has_output:
            try:
                if should_cancel and should_cancel():
                    block_status = "stopped"
            except Exception:
                pass
        _emit(BlockFinished(
            run_id=run_root, block_id=str(getattr(sr, "step_id", "") or ""),
            label=str(getattr(sr, "block_label", "") or ""),
            kind=str(getattr(sr, "step_kind", "") or ""),
            outputs=dict(_outputs),
            started_at=_started, completed_at=_completed,
            status=block_status,
            error=str(getattr(sr, "error", "") or ""),
            verdict=payload.get("verdict"), detectors=payload.get("detectors")))
        if on_step is not None:
            try:
                on_step(sr)
            except Exception:
                pass

    engine_func = make_engine_func(parent_turn_id=run_root, spawn_budget=spawn_budget,
                                   should_cancel=should_cancel, is_busy=is_busy,
                                   allow_egress=allow_egress)
    tool_func = make_tool_func(parent_turn_id=run_root, should_cancel=should_cancel)
    run = m["headless"].run_workflow(
        str(workflow.source_path), user_input=user_input,
        session_messages=[],               # decision A: empty by default; no Genesis voice leak
        engine_func=engine_func, tool_func=tool_func,
        activation_mode="on_activate",     # chat flows single-shot (reject daemon modes)
        on_step=_on_step, should_stop=should_stop,
        max_steps=max_steps)               # conservative chat ceiling (< Monoline standalone 200)
    try:
        _result = getattr(run, "result", None)
        _err = str(getattr(_result, "error", "") or "")
        # A user STOP halts via should_stop; the runtime sets result.error to its stop sentinel
        # ("Activation stopped."). Report it as a clean stop (stopped=True, no error text) so the
        # RunView shows "stopped" rather than a red failure.
        _stopped = False
        try:
            if should_stop and should_stop():
                _stopped = True
        except Exception:
            pass
        if not _stopped and _err.strip().lower().startswith("activation stopped"):
            _stopped = True
        _emit(RunFinished(run_id=run_root,
                          output=str(getattr(_result, "output", "") or ""),
                          error=("" if _stopped else _err),
                          stopped=_stopped))
    except Exception:
        pass
    return run


def open_create_canvas(parent, *, world_id=None):
    """Launch Monoline's standalone canvas (app.py) as a SEPARATE PROCESS. In-process is
    impossible: Monoline's top-level `ui`/`core` collide with Monolith's and a live interactive
    window keeps re-resolving them. The subprocess inherits MONOLITH_ROOT (pinned below), so saved
    worlds land in the shared WORLD_DIR the registry globs. world_id=None -> blank Create canvas;
    a world id -> Edit (app.py's additive `--world <id>` preloads that flow)."""
    import subprocess
    import sys as _sys
    root = ensure_monoline_on_path()      # validates the dir + pins MONOLITH_ROOT in os.environ
    app_py = root / "app.py"
    if not app_py.exists():
        raise RuntimeError(f"Monoline entry point not found: {app_py}")
    args = [_sys.executable, str(app_py)]
    if world_id is not None:
        args += ["--world", str(world_id)]   # Edit: preload this flow; Create (None) -> blank canvas
    env = dict(os.environ)
    env.setdefault("MONOLINE_DEFAULT_LLM_PROVIDER", "monolith")
    proc = subprocess.Popen(args, cwd=str(root), env=env)  # inherits MONOLITH_ROOT + bridge defaults
    refs = getattr(parent, "_monoline_canvas_procs", None)
    if refs is None:
        refs = []
        try:
            parent._monoline_canvas_procs = refs
        except Exception:
            pass
    refs.append(proc)                     # keep a handle; don't orphan the launch
    return proc


def _now() -> str:
    import datetime as _dt
    return _dt.datetime.now(_dt.timezone.utc).isoformat()
