# RUNTIME_AWARENESS_SPEC.md

## Overview

RuntimeAwareness is a **zero-LLM** instrumentation layer that sits between
`LoopEngine` and `LoopRuntime`. It observes the existing event stream, performs
arithmetic on latency/budget/truncation signals, and either adjusts runtime
parameters or gates runs that are not viable.

It does not plan. It does not infer. It multiplies and divides.

**Module location**: `engine/loop/awareness.py`

**Depends on**: `contracts.py` (RunPolicy, RunContext, PreflightResult, Pad)

**Depended on by**: `loop_engine.py` (LoopEngine), `runtime.py` (LoopRuntime via
`on_event`)

---

## Design Constraints

1. **No LLM calls.** Every computation is deterministic arithmetic or lookup.
2. **No new threads.** Runs synchronously inside existing event callbacks.
3. **No mutation of contracts.** Produces new `RunPolicy` / config dicts; never
   patches live objects.
4. **Stateless within a run.** All per-run state lives in a `RunProfile` dataclass
   that is created fresh per run and discarded after.
5. **Persistent across runs.** A `HardwareProfile` accumulates latency observations
   across runs and is saved/loaded from the config directory alongside
   `code_loop_config.json`.
6. **Opt-in.** The layer can be disabled entirely. When disabled, LoopEngine and
   LoopRuntime behave exactly as they do today.

---

## Data Structures

### HardwareProfile (persistent, JSON-serialized)

```python
@dataclass
class HardwareProfile:
    """Accumulated observations about the inference backend.
    
    Persisted to: {config_dir}/hardware_profile.json
    Updated after every completed or wall-killed run.
    """

    # Inference latency (seconds per LLM call).
    # Stored as exponential moving average (EMA) for stability.
    latency_ema: float = 0.0          # current EMA of seconds-per-call
    latency_samples: int = 0          # total observations
    latency_last: float = 0.0         # most recent observation
    latency_max_seen: float = 0.0     # worst case observed

    # Token throughput.
    chars_per_token_ema: float = 3.5   # EMA of output_chars / max_tokens
    throughput_samples: int = 0

    # Truncation rate.
    truncation_total: int = 0          # total calls observed
    truncation_hits: int = 0           # calls where output hit ceiling

    # Model fingerprint — reset profile if model changes.
    model_path_hash: str = ""
    n_ctx: int = 0

    EMA_ALPHA: float = 0.3            # weight of new observation
```

### RunProfile (per-run, ephemeral)

```python
@dataclass
class RunProfile:
    """Live telemetry for the current run. Created at run start, discarded at end."""

    run_id: str = ""

    # Clock.
    wall_start: float = 0.0           # time.time() at true session start (before preflight)
    preflight_start: float = 0.0
    preflight_end: float = 0.0
    runtime_start: float = 0.0        # when LoopRuntime.run() begins

    # Per-call observations (appended in order).
    call_latencies: list[float] = field(default_factory=list)
    call_output_chars: list[int] = field(default_factory=list)
    call_truncated: list[bool] = field(default_factory=list)

    # Derived (recomputed after each call).
    avg_latency: float = 0.0
    estimated_remaining_cycles: int = 0
    pace_ratio: float = 1.0           # >1 = ahead of schedule, <1 = behind
    truncation_streak: int = 0        # consecutive truncated calls
    budget_viable: bool = True
```

### AwarenessVerdict (output of pre-run gate)

```python
@dataclass(frozen=True)
class AwarenessVerdict:
    """Result of pre-run viability check."""

    viable: bool
    adjusted_policy: RunPolicy | None    # None = no changes needed
    adjusted_infer: dict[str, Any] | None
    warnings: list[str]                  # human-readable, emitted to trace
    estimated_max_cycles: int            # how many cycles this hardware can afford
    estimated_tokens_needed: int         # heuristic from preflight complexity
    reason: str                          # "ok" | "token_budget_insufficient" | "time_budget_insufficient" | ...
```

---

## Components

### 1. PreRunGate

**When**: Called in `LoopEngine.prompt_run_start()` after `_infer_config_from_payload`
and `_policy_from_payload` resolve, before the `_LoopWorker` is created.

**Inputs**: `RunPolicy`, infer config dict, `HardwareProfile`, `PreflightResult | None`,
user prompt string.

**Outputs**: `AwarenessVerdict`.

**Logic** (all arithmetic):

```
estimated_latency = hardware.latency_ema or 30.0  # fallback for first run
preflight_cost   = estimated_latency * 1           # one call
usable_time      = policy.max_elapsed_sec - preflight_cost
max_affordable   = floor(usable_time / estimated_latency)

# Token budget heuristic.
# The Step JSON envelope is ~200-400 tokens of overhead.
# write_file content is the dominant payload.
ENVELOPE_OVERHEAD = 350  # tokens
usable_content_tokens = infer.max_tokens - ENVELOPE_OVERHEAD
complexity_hint = preflight.execution_weight if preflight else 3

# A write_file for a simple script needs ~800-2000 tokens of content.
# Scale by complexity.
TOKENS_PER_COMPLEXITY = {1: 400, 2: 800, 3: 1200, 4: 2000, 5: 3000}
estimated_needed = TOKENS_PER_COMPLEXITY.get(complexity_hint, 1200)

warnings = []
adjustments_policy = {}
adjustments_infer = {}

# Check 1: Can we afford enough cycles?
todo_count = len(preflight.todo) if preflight else 3
# Each todo item needs ~1-2 cycles (action + possible verification).
min_cycles_needed = todo_count * 2
if max_affordable < min_cycles_needed:
    warnings.append(
        f"Hardware can afford ~{max_affordable} cycles at {estimated_latency:.0f}s/call, "
        f"but plan needs ~{min_cycles_needed}. "
        f"Recommend increasing max_elapsed_sec to {int(min_cycles_needed * estimated_latency * 1.3)}s."
    )
    # Auto-adjust: extend time budget to fit, capped at 15 minutes.
    adjusted_time = min(900.0, min_cycles_needed * estimated_latency * 1.3)
    adjustments_policy["max_elapsed_sec"] = adjusted_time

# Check 2: Will the token budget truncate write_file content?
if usable_content_tokens < estimated_needed:
    floor_tokens = estimated_needed + ENVELOPE_OVERHEAD
    # Clamp to what the model context can handle (heuristic: max_tokens ≤ n_ctx / 3).
    max_sane = (hardware.n_ctx // 3) if hardware.n_ctx > 0 else 8192
    adjusted_tokens = min(max_sane, max(floor_tokens, 2048))
    warnings.append(
        f"max_tokens={infer['max_tokens']} leaves ~{usable_content_tokens} for content, "
        f"but task complexity suggests ~{estimated_needed} needed. "
        f"Adjusting to {adjusted_tokens}."
    )
    adjustments_infer["max_tokens"] = adjusted_tokens

# Check 3: Historical truncation rate is high.
if hardware.truncation_total >= 5:
    rate = hardware.truncation_hits / hardware.truncation_total
    if rate > 0.3 and "max_tokens" not in adjustments_infer:
        bump = int(infer["max_tokens"] * 1.5)
        max_sane = (hardware.n_ctx // 3) if hardware.n_ctx > 0 else 8192
        adjustments_infer["max_tokens"] = min(max_sane, bump)
        warnings.append(
            f"Historical truncation rate is {rate:.0%}. Bumping max_tokens to {adjustments_infer['max_tokens']}."
        )

# Check 4: max_cycles vs max_elapsed_sec coherence.
effective_time = adjustments_policy.get("max_elapsed_sec", policy.max_elapsed_sec)
realistic_max_cycles = int(effective_time / max(1.0, estimated_latency))
effective_max_cycles = adjustments_policy.get("max_cycles", policy.max_cycles)
if effective_max_cycles > realistic_max_cycles * 1.5:
    adjustments_policy["max_cycles"] = realistic_max_cycles
    warnings.append(
        f"max_cycles={effective_max_cycles} is unreachable at {estimated_latency:.0f}s/call "
        f"within {effective_time:.0f}s. Clamping to {realistic_max_cycles}."
    )

# Build verdict.
viable = True
reason = "ok"
if not adjustments_policy and not adjustments_infer:
    adjusted_policy = None
    adjusted_infer = None
else:
    # Construct adjusted objects.
    ...

return AwarenessVerdict(viable=viable, ..., warnings=warnings)
```

**Integration point in `loop_engine.py`**:

```python
# In prompt_run_start(), after infer_cfg and policy are resolved:
if self._awareness_enabled:
    verdict = self._awareness.pre_run_gate(
        policy=policy,
        infer_config=infer_cfg,
        hardware=self._hardware_profile,
        preflight=None,  # not yet available; use complexity hint from prompt length
        user_prompt=user_prompt,
    )
    for w in verdict.warnings:
        self.sig_trace.emit(f"[AWARENESS] {w}")
    if verdict.adjusted_policy is not None:
        policy = verdict.adjusted_policy
    if verdict.adjusted_infer is not None:
        infer_cfg = {**infer_cfg, **verdict.adjusted_infer}
```

---

### 2. MidRunObserver

**When**: Registered as the `on_event` callback on `LoopRuntime`. Wraps the existing
trace emitter — does not replace it.

**Observes these event kinds**:

| Event kind | What it extracts |
|---|---|
| `llm_call` | `call_index`, `ok`, `output` (length), timestamps (derived from consecutive events) |
| `llm_input` | Timestamp of call start (paired with next `llm_call` for latency) |
| `wall_check_detail` | `elapsed_sec`, `max_elapsed_sec` for pace computation |
| `step_parsed` | `actions_count` for truncation-vs-intent discrimination |
| `retry` | Retry triggered — check if cause was truncation |
| `cycle_start` | Increment cycle counter for pace tracking |

**Computations** (run after each `llm_call` event):

```python
def on_llm_call(self, profile: RunProfile, data: dict) -> list[str]:
    """Returns list of advisory strings to inject into trace. No mutations."""
    advisories = []
    output_chars = data.get("response_chars", 0)
    max_tokens = self._current_max_tokens
    
    # --- Latency tracking ---
    if profile.call_latencies:
        latest = profile.call_latencies[-1]
        profile.avg_latency = sum(profile.call_latencies) / len(profile.call_latencies)
    
    # --- Truncation detection ---
    # Heuristic: if output chars ≥ 90% of (max_tokens * chars_per_token_ema),
    # the output likely hit the ceiling.
    chars_per_token = self._hardware.chars_per_token_ema or 3.5
    ceiling = max_tokens * chars_per_token * 0.90
    truncated = output_chars >= ceiling and not data.get("ok", True) is False
    profile.call_truncated.append(truncated)
    
    if truncated:
        profile.truncation_streak += 1
        if profile.truncation_streak >= 2:
            advisories.append(
                f"TRUNCATION: {profile.truncation_streak} consecutive outputs hit token ceiling. "
                f"Current max_tokens={max_tokens}, output_chars={output_chars}."
            )
    else:
        profile.truncation_streak = 0
    
    # --- Pace tracking ---
    elapsed = time.time() - profile.runtime_start
    remaining_time = self._effective_max_elapsed - elapsed
    if profile.avg_latency > 0:
        remaining_cycles = int(remaining_time / profile.avg_latency)
        profile.estimated_remaining_cycles = max(0, remaining_cycles)
        
        # How many cycles do we still need? (open todos * ~1.5)
        open_todos = self._open_todo_count  # updated from pad_snapshot events
        needed = max(1, int(open_todos * 1.5))
        profile.pace_ratio = remaining_cycles / max(1, needed)
        
        if profile.pace_ratio < 0.5 and remaining_cycles < 3:
            advisories.append(
                f"PACE: ~{remaining_cycles} cycles left, ~{needed} needed. "
                f"Run will likely hit max_elapsed wall."
            )
    
    return advisories
```

**Integration point in `runtime.py`**:

The MidRunObserver is **not** wired into the runtime's decision-making in v1.
It only emits trace advisories via the existing `on_event` → trace path.

This is intentional. The observer builds confidence in its signals before they
drive automated responses. The trace advisories surface in the debug JSONL, letting
you validate accuracy before promoting any signal to an automated adjustment.

**Future v2 hooks** (documented but not implemented):

- `should_stop` integration: trip the streaming stop flag when
  `elapsed + avg_latency > max_elapsed_sec` (prevents the overshoot bug).
- Dynamic `max_tokens` bump mid-run: if `truncation_streak >= 2`, inject an
  adjusted `max_tokens` into the next `LLMEngineInferAdapter` call.
- Early termination advisory: if `pace_ratio < 0.3` and no critical todos are
  crystallized, suggest the run yield with a partial summary rather than
  wall-killing silently.

---

### 3. PostRunRecorder

**When**: Called after `_on_worker_done` or `_on_worker_failed` in `LoopEngine`.

**Inputs**: Completed `RunProfile`, `RunResult`.

**Logic**:

```python
def record(self, profile: RunProfile, result: RunResult, hw: HardwareProfile) -> HardwareProfile:
    """Update hardware profile with observations from this run. Returns new profile."""
    alpha = hw.EMA_ALPHA

    # Update latency EMA from all calls in this run.
    for latency in profile.call_latencies:
        if hw.latency_samples == 0:
            hw.latency_ema = latency
        else:
            hw.latency_ema = alpha * latency + (1 - alpha) * hw.latency_ema
        hw.latency_samples += 1
        hw.latency_last = latency
        hw.latency_max_seen = max(hw.latency_max_seen, latency)

    # Update truncation stats.
    hw.truncation_total += len(profile.call_truncated)
    hw.truncation_hits += sum(1 for t in profile.call_truncated if t)

    # Update chars-per-token EMA.
    for chars in profile.call_output_chars:
        if chars > 0 and self._max_tokens > 0:
            ratio = chars / self._max_tokens
            if hw.throughput_samples == 0:
                hw.chars_per_token_ema = ratio
            else:
                hw.chars_per_token_ema = alpha * ratio + (1 - alpha) * hw.chars_per_token_ema
            hw.throughput_samples += 1

    return hw
```

**Persistence**: `HardwareProfile` is serialized to
`{config_dir}/hardware_profile.json` after every run. On load, if
`model_path_hash` doesn't match the currently loaded model, the profile is
reset to defaults (new model = new latency characteristics).

---

### 4. StreamingWallGuard (the should_stop fix)

This is a small, standalone piece that fixes the wall-clock overshoot bug
independently of the rest of RuntimeAwareness.

**Current problem**: `LLMEngineInferAdapter._consume_stream` checks
`self.should_stop()`, but that lambda only reads `self._stop_requested` (user
stop). Budget exhaustion is never signaled during inference.

**Fix**: Add a second predicate to the should_stop chain.

```python
# In LoopEngine.prompt_run_start(), when constructing the infer_fn:

wall_deadline = time.time() + float(policy.max_elapsed_sec) + 30.0  # 30s grace for preflight

def _should_stop_with_wall():
    if self._stop_requested:
        return True
    # Leave 5s margin so the runtime can emit clean wall_hit event.
    if time.time() > wall_deadline - 5.0:
        return True
    return False

infer_fn = LLMEngineInferAdapter(
    self._llm,
    ...
    should_stop=_should_stop_with_wall,
)
```

This is **safe to ship immediately** without the rest of RuntimeAwareness.
It bounds the worst-case overshoot to ~5 seconds regardless of inference
latency.

---

## Integration Summary

### What changes in existing files

**`loop_engine.py`** (LoopEngine class):

```python
# New imports
from engine.loop.awareness import (
    RuntimeAwareness,
    HardwareProfile,
    load_hardware_profile,
    save_hardware_profile,
)

# In __init__:
self._awareness = RuntimeAwareness()
self._hardware_profile = load_hardware_profile(config_dir)
self._awareness_enabled = True  # or read from config

# In prompt_run_start():
# 1. Pre-run gate (after policy/infer resolution, before worker creation)
# 2. Wall deadline for should_stop
# 3. Pass awareness observer as event wrapper

# In _on_worker_done() / _on_worker_failed():
# Post-run recording
```

**`runtime.py`** — **No changes.** The observer wraps the existing `on_event`
callback externally. The runtime remains framework-agnostic.

**`contracts.py`** — **No changes.** No new fields on RunPolicy, RunContext, or
any existing dataclass.

**`walls.py`** — **No changes.** Wall logic is untouched. The StreamingWallGuard
prevents overshoot from the *inference* side, not the wall-check side.

### New files

| File | Purpose | Lines (est.) |
|---|---|---|
| `engine/loop/awareness.py` | RuntimeAwareness, HardwareProfile, RunProfile, AwarenessVerdict, PreRunGate, MidRunObserver, PostRunRecorder | ~300 |
| `{config_dir}/hardware_profile.json` | Persisted HardwareProfile | ~15 (auto-generated) |

---

## Event Kinds Emitted

RuntimeAwareness emits through the existing `sig_trace` / event system.
No new event infrastructure.

| Trace prefix | Example | When |
|---|---|---|
| `[AWARENESS] gate:` | `gate: max_tokens adjusted 1076 → 3200` | Pre-run |
| `[AWARENESS] gate:` | `gate: max_elapsed_sec adjusted 300 → 520` | Pre-run |
| `[AWARENESS] pace:` | `pace: ~2 cycles left, ~4 needed` | Mid-run (after each llm_call) |
| `[AWARENESS] truncation:` | `truncation: 2 consecutive ceiling hits` | Mid-run |
| `[AWARENESS] recorded:` | `recorded: latency_ema=162s, truncation_rate=40%` | Post-run |
| `[AWARENESS] wall_guard:` | `wall_guard: streaming stopped (deadline - 5s)` | Mid-inference |

---

## What This Would Have Done for the Pong Session

Using the exact numbers from `code_e0ed5b03ad`:

**Pre-run gate** (first run, no HardwareProfile yet — uses fallback 30s):
- `usable_content_tokens = 1076 - 350 = 726`
- `estimated_needed = 1200` (complexity_class=short_multi, execution_weight=3)
- **Verdict**: `max_tokens` adjusted `1076 → 1550` minimum.
- With HardwareProfile populated: latency ~160s → `max_affordable = floor((300 - 160) / 160) = 0`.
  Gate warns: "Hardware can afford ~0 cycles. Recommend max_elapsed_sec=1248s."
  Auto-adjusts to 900s cap. Adjusts `max_cycles` from 34 to 5 (realistic).

**After LLM call #1** (cycle 1, latency=31s — fast because small output):
- HardwareProfile updates: `latency_ema = 31s`
- Pace: 5 open todos × 1.5 = ~8 cycles needed, `(300 - 31) / 31 = ~8` affordable. Ratio ≈ 1.0. OK.

**After LLM call #2** (cycle 2, latency=168s, output=4058 chars, truncated):
- `ceiling = 1076 × 3.5 × 0.90 = 3389`. Output 4058 > 3389. **Truncated = true.**
- `truncation_streak = 1`
- HardwareProfile: `latency_ema = 0.3 × 168 + 0.7 × 31 = 72s`
- Pace: `(300 - 199) / 72 = ~1.4 cycles left`, `~6 needed`. Ratio = 0.23. **PACE warning fires.**

**After LLM call #3** (cycle 2 retry, latency=73s, output=1782, not truncated):
- `truncation_streak = 0` (reset)
- `latency_ema = 0.3 × 73 + 0.7 × 72 = 72.3s`

**StreamingWallGuard** (cycle 3 inference start at ~272s):
- `wall_deadline = start + 300 + 30 = 330s` (with 30s preflight grace).
- At `325s` into streaming: `time.time() > 330 - 5 = 325`. **Stop triggered.**
- Inference aborts cleanly. Runtime emits `wall_hit` within seconds of budget, not 133s after.

**Post-run**: HardwareProfile saved. Next run starts with `latency_ema ≈ 72s` and
`truncation_rate = 33%`. Gate adjusts accordingly before cycle 1.

---

## Non-Goals (Explicit)

- **No prompt rewriting.** Awareness does not touch the Step schema, system prompt,
  or `build_messages`.
- **No plan adjustment.** If the plan has 5 todos and the budget affords 3 cycles,
  Awareness warns but does not prune the plan. That's the agent's job.
- **No model selection.** Awareness doesn't suggest switching models. Out of scope.
- **No UI changes.** All output goes through existing `sig_trace`. A future UI
  could read `[AWARENESS]` prefixed traces to show a budget gauge, but that's
  not part of this spec.
- **No changes to `contracts.py`.** The awareness layer is purely additive.

---

## Implementation Order

| Phase | What | Depends on | Standalone value |
|---|---|---|---|
| **0** | `StreamingWallGuard` — the `should_stop` fix | Nothing | Fixes wall overshoot immediately |
| **1** | `HardwareProfile` + `PostRunRecorder` | Phase 0 | Starts collecting latency/truncation data |
| **2** | `PreRunGate` | Phase 1 | Prevents impossible runs, adjusts token budget |
| **3** | `MidRunObserver` (trace-only) | Phase 1 | Surfaces pace/truncation warnings in debug log |
| **4** | Future: observer-driven `should_stop` + dynamic `max_tokens` | Phase 3 validated | Closes the loop from observation to action |

Each phase is independently shippable and testable. Phase 0 is a ~15-line change
to `loop_engine.py` with no new files.
