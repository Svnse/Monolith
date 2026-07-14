"""Turn Pipeline — kernel-tier event bus between LLM producers and consumers.

Companion to MonoBridge / MonoDock / MonoGuard. The bus model:

    Producer (engine/llm.py local OR engine/agent_server.py remote — via
    adapters that depend on this module, not the other way around) calls
    pipeline.run_turn(turn_id, producer_adapter). The pipeline drives the
    stream, publishes events at every observable point, and dispatches each
    event to registered policies in topo-sorted order.

Authority model:
  - OBSERVATION policies may emit events. They cannot mutate the stream
    or force loop continuation.
  - MUTATION policies may modify chunk content (e.g., output_sanitizer)
    and must declare a kill_switch_env_flag.
  - DISPATCH policies may force or suppress loop continuation, inject
    hints, or short-circuit retries. They declare retry_budget.

Persistence: every published event is written to fault_traces in the
existing turn_trace.sqlite3. No new persistence path.

Independence: this module MUST NOT import from engine/llm.py,
engine/agent_server.py, or any acu_*/acatalepsy code. Producers are
adapter objects passed in; identity/ACU work belongs to MonoBase.

Phase 1 scope: bus mechanics + run_turn skeleton + boot validator. No
real stream consumption logic yet (no producer adapter calls). Phases 2+
add adapters, policies, UI.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from core import pipeline_registry as _registry
from core import turn_trace as _tt
from core.turn_pipeline_events import (
    AuthorityTier,
    FaultDetectedEvent,
    PipelineEvent,
    ProducerKind,
    TurnCompleteEvent,
    TurnStreamStartedEvent,
    stamp_event,
)


# ── producer adapter contract ───────────────────────────────────────


@dataclass(frozen=True)
class ProducerChunk:
    """A unit of streamed output from a producer."""
    text: str
    meta: dict[str, Any] = field(default_factory=dict)


class ProducerAdapter(Protocol):
    """Source-agnostic stream producer.

    Both the local llama.cpp engine and the remote CONNECT bridge implement
    this protocol via thin adapters that live in engine/adapters/* (Phase 2).
    The pipeline does not import engines directly — adapters depend on the
    pipeline, never the other way around.
    """
    producer_kind: str  # one of ProducerKind values

    def stream(self, turn_id: str, context: dict[str, Any]) -> Iterable[ProducerChunk]:
        """Yield producer chunks for this turn. Synchronous iterable.

        Async streams may be wrapped via an adapter that yields chunks as they
        arrive on the event loop's queue.
        """
        ...

    def supports_continuation(self) -> bool:
        """Whether continue_with_tool_result is implemented."""
        ...

    def continue_with_tool_result(
        self,
        turn_id: str,
        tool_result_summary: dict[str, Any],
        hint: str | None,
    ) -> Iterable[ProducerChunk]:
        """Re-invoke the producer with the latest tool result appended."""
        ...


# ── kernel-internal turn context ────────────────────────────────────


@dataclass
class TurnContext:
    """Per-turn accumulator. Lives only for the duration of run_turn().

    NOT a persistence surface. Anything durable must be a FaultTraceRecord
    in fault_traces. See spec: 'replayable' discipline.

    Callers (chat.py) may construct a fresh TurnContext on every emit for
    the same turn_id; the kernel hydrates each fresh ctx from canonical
    per-turn state at publish() time so dispatch-tier accumulators stay
    coherent across emits.
    """
    turn_id: str
    parent_turn_id: str | None = None
    started_at: float = 0.0
    seq_counter: int = 0
    fault_count: int = 0
    mutation_count: int = 0
    requeue_count: int = 0
    retry_budget_used: dict[str, int] = field(default_factory=dict)
    suppressed_continuation: bool = False
    # MUTATION-tier output: when a terminal-mutation policy (output_sanitizer)
    # corrects the public answer, it writes the corrected text here. The
    # publisher (chat.py finalize) reads it back after publish() and re-commits.
    # Survives publish() — the canonical write-back only touches the four
    # scalars above, never this field. Single-emit (TurnReadyEvent), so it is
    # intentionally not part of _CanonicalTurnState cross-emit accounting.
    sanitized_text: str | None = None

    def next_seq(self) -> int:
        v = self.seq_counter
        self.seq_counter += 1
        return v


@dataclass
class _CanonicalTurnState:
    """Kernel-owned per-turn accumulator. One per turn_id, lives in the
    pipeline singleton across all publish() calls for that turn.

    Solves the per-emit ctx reset problem: chat.py constructs a fresh
    TurnContext for each emit site (TurnStreamEnded, TurnReady, …). Without
    this canonical store, dispatch-tier state (retry_budget_used,
    suppressed_continuation) would reset between emits and budget thresholds
    would never trip. publish() hydrates the caller's ctx from canonical
    before dispatch and writes back deltas after, so handlers see real
    cross-emit continuity.

    The dict field (retry_budget_used) is aliased into the caller's ctx so
    handler writes go straight to canonical. Scalar fields use delta
    accounting to remain correct under reentrant publish() (handler emits a
    nested event whose handlers also mutate the canonical state).
    """
    fault_count: int = 0
    mutation_count: int = 0
    requeue_count: int = 0
    retry_budget_used: dict[str, int] = field(default_factory=dict)
    suppressed_continuation: bool = False


# ── the bus ─────────────────────────────────────────────────────────


_Handler = Callable[[PipelineEvent, "TurnContext"], None]


class TurnPipeline:
    """Event bus + run_turn driver.

    Singleton in practice (constructed once by bootstrap_pipeline) but the
    class is parameterizable for tests that need an isolated instance with
    a stubbed registry.
    """

    def __init__(self) -> None:
        # event_kind_name -> ordered list of (registration, handler)
        self._subscribers: dict[str, list[tuple[_registry.PolicyRegistration, _Handler]]] = {}
        self._registered_names: set[str] = set()
        self._lock = threading.RLock()
        self._frozen: bool = False
        # Turn-scoped seq counter — survives across multiple TurnContext
        # instances for the same turn_id, so callers (chat.py) can emit
        # without holding a single ctx for the full turn lifetime.
        self._seq_by_turn: dict[str, int] = {}
        self._seq_lock = threading.Lock()
        # Canonical per-turn accumulator state. Parallel to _seq_by_turn:
        # the kernel owns the cross-emit truth so dispatch-tier policies see
        # continuity even when callers build fresh TurnContext per publish.
        # RLock for reentrant publish() (handler emits a nested event).
        self._state_by_turn: dict[str, _CanonicalTurnState] = {}
        self._state_lock = threading.RLock()
        # Thread-local set of ctx ids currently in publish() chains. Used to
        # detect nested-publish-on-same-ctx (the standard policy pattern
        # via _publish_via_kernel) so the nested frame skips hydrate/sync —
        # otherwise it would (a) wipe scalar mutations the outer handler
        # made mid-dispatch by re-reading stale canonical and (b)
        # double-count via its own delta sync. The outer frame owns
        # hydrate/sync; nested frames just stamp/persist/dispatch.
        self._active_ctx_ids = threading.local()

    # ── registration ────────────────────────────────────────────────

    def register(self, reg: _registry.PolicyRegistration, handler: _Handler) -> None:
        """Attach a handler to every event the policy subscribes to.

        Order within an event is unsorted until finalize_registration() runs.
        """
        with self._lock:
            if self._frozen:
                raise RuntimeError(
                    f"cannot register {reg.name!r}: pipeline registration is frozen"
                )
            if reg.name in self._registered_names:
                raise RuntimeError(f"policy {reg.name!r} already registered")
            self._registered_names.add(reg.name)
            for ev in reg.subscribes_to:
                self._subscribers.setdefault(ev, []).append((reg, handler))

    def finalize_registration(self) -> None:
        """Apply topo-sort across each event's subscriber list and freeze."""
        with self._lock:
            per_event = _registry.topo_sort_per_event()
            # per_event holds the declared policies in dependency order. Map
            # them onto the (reg, handler) tuples we actually have, preserving
            # only registered handlers (a policy declared but not register()'d
            # is a registry mismatch caught by validate at boot, not here).
            new_subs: dict[str, list[tuple[_registry.PolicyRegistration, _Handler]]] = {}
            for ev, ordered_regs in per_event.items():
                handlers_for_ev = {r.name: (r, h) for (r, h) in self._subscribers.get(ev, [])}
                resolved: list[tuple[_registry.PolicyRegistration, _Handler]] = []
                for r in ordered_regs:
                    if r.name in handlers_for_ev:
                        resolved.append(handlers_for_ev[r.name])
                if resolved:
                    new_subs[ev] = resolved
            self._subscribers = new_subs
            self._frozen = True

    def _next_seq_for_turn(self, turn_id: str) -> int:
        """Atomic kernel-scoped seq for a turn_id."""
        with self._seq_lock:
            cur = self._seq_by_turn.get(turn_id, 0)
            self._seq_by_turn[turn_id] = cur + 1
            return cur

    def _hydrate_ctx_from_canonical(self, ctx: TurnContext) -> _CanonicalTurnState:
        """Sync canonical per-turn state into the caller's ctx.

        First publish for a turn_id seeds the canonical store from ctx so
        callers can pre-populate (tests do this). Subsequent publishes
        overwrite ctx scalars from canonical and alias the mutable dict so
        handler writes flow straight back to canonical.

        Must be called under self._state_lock by the caller.
        """
        canonical = self._state_by_turn.get(ctx.turn_id)
        if canonical is None:
            canonical = _CanonicalTurnState(
                fault_count=ctx.fault_count,
                mutation_count=ctx.mutation_count,
                requeue_count=ctx.requeue_count,
                retry_budget_used=dict(ctx.retry_budget_used),
                suppressed_continuation=ctx.suppressed_continuation,
            )
            self._state_by_turn[ctx.turn_id] = canonical
        # Scalar fields: overwrite ctx from canonical.
        ctx.fault_count = canonical.fault_count
        ctx.mutation_count = canonical.mutation_count
        ctx.requeue_count = canonical.requeue_count
        ctx.suppressed_continuation = canonical.suppressed_continuation
        # Mutable dict: alias so handler writes hit canonical directly.
        ctx.retry_budget_used = canonical.retry_budget_used
        return canonical

    def cleanup_turn(self, turn_id: str) -> None:
        """Drop kernel-side per-turn state. Optional. Callers that know a
        turn is fully done may invoke this to bound memory across long
        sessions. Safe to call multiple times or on unknown turn_ids.

        Note: _seq_by_turn is also dropped so a future turn with the same id
        (replay, deterministic test) starts clean.
        """
        with self._state_lock:
            self._state_by_turn.pop(turn_id, None)
        with self._seq_lock:
            self._seq_by_turn.pop(turn_id, None)

    # ── kill switches ───────────────────────────────────────────────

    @staticmethod
    def _policy_enabled(reg: _registry.PolicyRegistration) -> bool:
        if reg.authority_tier == AuthorityTier.OBSERVATION:
            return True  # observation tier has no kill switch
        raw = os.environ.get(reg.kill_switch_env_flag, "1").strip().lower()
        return raw in {"1", "true", "yes", "on"}

    # ── publish ─────────────────────────────────────────────────────

    def publish(
        self,
        event: PipelineEvent,
        ctx: TurnContext,
        *,
        source_kind: str,
        source_name: str,
    ) -> PipelineEvent:
        """Stamp identity, record, dispatch to subscribers. Returns stamped event.

        Best-effort persistence (matches turn_trace's Q7 default — failures
        log to stderr, never raise). Subscriber failures are isolated: one
        misbehaving policy must not break the turn for the others.

        State discipline: hydrates the caller's ctx from canonical per-turn
        state, dispatches, then writes ctx scalar deltas back into canonical
        additively (reentrant-safe — a handler that publishes a nested event
        sees the same canonical state, applies its own deltas, and the outer
        sync still preserves both contributions). retry_budget_used is
        aliased into the caller's ctx so handler writes flow directly to
        canonical without a delta pass. suppressed_continuation uses
        sticky-OR.
        """
        # Detect nested-publish-on-same-ctx (the standard policy pattern via
        # _publish_via_kernel). The outer frame owns hydrate/sync of the
        # canonical state; nested frames just stamp/persist/dispatch using
        # the ctx the outer is already managing.
        active = getattr(self._active_ctx_ids, "ids", None)
        if active is None:
            active = set()
            self._active_ctx_ids.ids = active
        is_nested_same_ctx = id(ctx) in active

        # Hold state lock across hydrate → dispatch → sync. RLock allows
        # reentrant publish() from inside a handler.
        with self._state_lock:
            if is_nested_same_ctx:
                # Outer owns canonical. Just stamp/persist/dispatch; do not
                # re-hydrate (would wipe outer's mid-dispatch mutations) or
                # sync (would double-count via outer's delta).
                canonical = None
                snap_fault = snap_mut = snap_req = 0  # unused
            else:
                canonical = self._hydrate_ctx_from_canonical(ctx)
                # Snapshot scalars at hydrate time so we can compute and
                # apply deltas additively after dispatch (preserves nested-
                # publish contributions via a *different* ctx that advanced
                # canonical mid-dispatch).
                snap_fault = canonical.fault_count
                snap_mut = canonical.mutation_count
                snap_req = canonical.requeue_count

            active.add(id(ctx))
            try:
                # seq is kernel-scoped by turn_id, not per-ctx. Multiple ctx
                # instances in the same turn (e.g., chat.py emits each one
                # fresh) still get monotonic seq across the turn.
                seq = self._next_seq_for_turn(ctx.turn_id)
                # Mirror to ctx so its local counter stays in sync (some
                # callers may read ctx.seq_counter directly).
                ctx.seq_counter = seq + 1
                stamped = stamp_event(
                    event,
                    turn_id=ctx.turn_id,
                    parent_turn_id=ctx.parent_turn_id,
                    seq=seq,
                    emitted_at=_now_iso(),
                    source_kind=source_kind,
                    source_name=source_name,
                )
                # Track aggregate counters before dispatch so subscribers
                # can read them.
                if isinstance(stamped, FaultDetectedEvent):
                    ctx.fault_count += 1
                # Mutation/requeue counters are bumped by policy handlers
                # (output_sanitizer for mutation_count, tool_loop_continuation
                # / parse_retry for requeue_count). Those writes land on ctx
                # and propagate to canonical via the outer frame's delta sync.

                # Persist.
                _record_event_to_fault_traces(stamped)

                # Dispatch.
                for reg, handler in list(self._subscribers.get(stamped.kind, ())):
                    if not self._policy_enabled(reg):
                        continue
                    try:
                        handler(stamped, ctx)
                    except Exception as exc:  # noqa: BLE001 — isolation, by design
                        _trace_failure(
                            f"policy {reg.name!r} raised on {stamped.kind}: "
                            f"{type(exc).__name__}: {exc}"
                        )
            finally:
                if not is_nested_same_ctx:
                    active.discard(id(ctx))

            if not is_nested_same_ctx:
                # Write deltas additively into canonical. A nested publish()
                # on a *different* ctx that advanced canonical during
                # dispatch is preserved because we compute deltas against
                # our own pre-dispatch snapshot, not canonical's current
                # value.
                assert canonical is not None  # outer frame always has it
                canonical.fault_count += (ctx.fault_count - snap_fault)
                canonical.mutation_count += (ctx.mutation_count - snap_mut)
                canonical.requeue_count += (ctx.requeue_count - snap_req)
                # Sticky-OR: once any handler (this turn) flips suppression
                # on, it stays on for the rest of the turn.
                canonical.suppressed_continuation = (
                    canonical.suppressed_continuation or ctx.suppressed_continuation
                )
                # Re-hydrate ctx scalars so the caller observes the final
                # canonical values after publish returns (preserves
                # existing assertions like `ctx.fault_count == 1`).
                ctx.fault_count = canonical.fault_count
                ctx.mutation_count = canonical.mutation_count
                ctx.requeue_count = canonical.requeue_count
                ctx.suppressed_continuation = canonical.suppressed_continuation

            return stamped

    # ── run_turn (Phase 1 skeleton) ────────────────────────────────

    def run_turn(
        self,
        turn_id: str,
        producer: ProducerAdapter | None = None,
        *,
        parent_turn_id: str | None = None,
    ) -> dict[str, Any]:
        """Drive a turn end-to-end. Phase 1: skeleton only.

        Emits TurnStreamStartedEvent and TurnCompleteEvent, returns a summary.
        Real stream consumption, tool dispatch, and continuation loop arrive
        in Phase 2 alongside the first producer adapter.

        Phase 1 callers should pass producer=None; the pipeline still emits
        bracketing lifecycle events so the substrate is exercisable.
        """
        ctx = TurnContext(
            turn_id=turn_id,
            parent_turn_id=parent_turn_id,
            started_at=time.perf_counter(),
        )
        kind = (
            getattr(producer, "producer_kind", ProducerKind.OTHER.value)
            if producer is not None
            else ProducerKind.OTHER.value
        )
        self.publish(
            TurnStreamStartedEvent(producer_kind=kind),
            ctx,
            source_kind="kernel",
            source_name="turn_pipeline",
        )
        # Phase 2 inserts the stream loop here.
        duration_ms = (time.perf_counter() - ctx.started_at) * 1000.0
        self.publish(
            TurnCompleteEvent(
                outcome="ok",
                fault_count=ctx.fault_count,
                mutation_count=ctx.mutation_count,
                requeue_count=ctx.requeue_count,
                duration_ms=duration_ms,
            ),
            ctx,
            source_kind="kernel",
            source_name="turn_pipeline",
        )
        return {
            "turn_id": turn_id,
            "outcome": "ok",
            "fault_count": ctx.fault_count,
            "mutation_count": ctx.mutation_count,
            "requeue_count": ctx.requeue_count,
            "duration_ms": duration_ms,
        }


# ── module-level singleton + bootstrap ──────────────────────────────


_pipeline: TurnPipeline | None = None
_bootstrap_lock = threading.Lock()


def get_pipeline() -> TurnPipeline:
    """Return the bootstrapped singleton. Bootstraps lazily if needed."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    with _bootstrap_lock:
        if _pipeline is None:
            _pipeline = bootstrap_pipeline()
    return _pipeline


def bootstrap_pipeline(*, validate_filesystem: bool = True) -> TurnPipeline:
    """Construct the pipeline, validate the registry, finalize subscribers.

    Called from bootstrap.py at startup, after monokernel/bridge.py,
    dock.py, guard.py are constructed and before any engine registers.

    If validate_filesystem is True (default), every .py file under
    core/pipeline_policies/ must appear in pipeline_registry.POLICIES, or
    boot fails fast with a clear error. Tests may pass False.
    """
    pipeline = TurnPipeline()

    if validate_filesystem:
        policies_dir = _policies_dir()
        _registry.validate_against_filesystem(policies_dir)

    # Each declared policy registers itself via its module's register_with
    # function. Importing the module is sufficient to trigger registration —
    # by convention each policy file calls
    # pipeline.register(reg, handler) inside register_with(pipeline).
    for reg in _registry.iter_policies():
        try:
            module = _import_module(reg.module_path)
        except Exception as exc:  # noqa: BLE001
            _trace_failure(
                f"failed to import policy {reg.module_path!r}: "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        register_with = getattr(module, "register_with", None)
        if register_with is None:
            _trace_failure(
                f"policy {reg.name!r} has no register_with(pipeline) — skipping"
            )
            continue
        try:
            register_with(pipeline)
        except Exception as exc:  # noqa: BLE001
            _trace_failure(
                f"policy {reg.name!r} register_with raised: "
                f"{type(exc).__name__}: {exc}"
            )

    pipeline.finalize_registration()

    global _pipeline
    _pipeline = pipeline
    return pipeline


def reset_for_tests() -> None:
    """Drop the singleton. Tests that need a fresh instance call this."""
    global _pipeline
    with _bootstrap_lock:
        _pipeline = None


# ── helpers ─────────────────────────────────────────────────────────


def _policies_dir() -> Path:
    """Locate core/pipeline_policies/ from this file's known position."""
    here = Path(__file__).resolve()
    return here.parent.parent / "core" / "pipeline_policies"


def _import_module(dotted: str):
    """Lazy importlib wrapper kept local so the pipeline module's import
    graph stays minimal at module load."""
    import importlib
    return importlib.import_module(dotted)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _record_event_to_fault_traces(event: PipelineEvent) -> None:
    """Best-effort persistence of an event to the fault_traces table.

    Delegates to turn_trace.record_fault — keeps all sqlite handling in one
    module. Failures are swallowed (logged to stderr by turn_trace).
    """
    payload = event.payload_fields()
    # If the event is a FaultDetectedEvent, lift fault_kind / severity to
    # their own columns; otherwise leave them null.
    fault_kind: str | None = None
    severity: str | None = None
    if isinstance(event, FaultDetectedEvent):
        fault_kind = event.fault_kind
        severity = event.severity

    # authority_tier on the row is the policy's tier when source_kind="policy",
    # null for producer/kernel events. We don't know the tier here without
    # looking up the policy — leave to the policy's handler to enrich payload
    # if needed. authority_tier column is nullable for that reason.
    record = _tt.FaultTraceRecord(
        turn_id=event.turn_id,
        parent_turn_id=event.parent_turn_id,
        seq=event.seq,
        emitted_at=event.emitted_at,
        event_kind=event.kind,
        source_kind=event.source_kind,
        source_name=event.source_name,
        authority_tier=None,
        fault_kind=fault_kind,
        severity=severity,
        payload=payload,
    )
    _tt.record_fault(record)


def _trace_failure(msg: str) -> None:
    """Mirror turn_trace's stderr logging discipline."""
    try:
        import sys
        sys.stderr.write(f"[turn_pipeline] {msg}\n")
    except Exception:
        pass
