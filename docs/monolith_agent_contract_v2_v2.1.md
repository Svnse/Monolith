# Monolith Agent Contract V2 + V2.1

## Status
- **Version:** V2 (with V2.1 hardening overlays)
- **Scope:** End-to-end agent execution path from UI request submission to runtime termination semantics
- **Target:** Local-first Monolith runtime (GGUF/provider-agnostic)

---

## 1) Purpose
This contract defines the deterministic execution model for Monolith agents.

It is designed to eliminate:
- protocol mismatch between prompt/inference/runtime,
- ambiguous terminal states,
- silent degradation ("talking instead of acting"),
- non-auditable execution outcomes.

---

## 2) Design Goals
1. No ambiguous terminal states.
2. No protocol mismatch across inference boundary.
3. No silent success for required-tool tasks.
4. Deterministic, auditable state transitions.
5. Local-first budgets (tokens, time, context) are first-class constraints.

---

## 3) Global Invariants

### Invariant A — Canonical Boundary Only
Runtime executes only canonical `AgentMessage` objects. Provider-native raw payloads are not consumed directly by runtime execution logic.

### Invariant B — Contract Immutability
`ExecutionContract` is immutable after first inference.

### Invariant C — Required-Tool Proof
If `tool_policy=required`, terminal success is impossible unless at least one validated tool invocation executed.

### Invariant D — State-Derived UI
UI render state must derive from runtime state machine + typed outcome, not event ordering alone.

### Invariant E — Immutable Transcript
Every state transition appends an immutable transcript entry with `contract_hash`, `step_id`, `state`, and boundary metadata.

### Invariant F (V2.1) — Token Gate for Required Tool Mode
If `tool_policy=required` and `token_gate=true`, no transition may emit terminal free-form text before a complete canonical tool invocation is structurally validated.

---

## 4) Core Contract Objects

## 4.1 `ExecutionContract`
```yaml
contract_id: string
contract_hash: string
parent_contract_hash: string | null

# policy
tool_policy: required | optional | forbidden
allowed_tools: list[string] | null
strict_mode: bool

# budgets
max_inferences: int
max_tokens_consumed: int
max_format_retries: int          # model-profile tuned; strict profiles SHOULD set 0 or 1
step_timeout_ms: int
total_timeout_ms: int

# context budget
context_budget:
  context_window: int
  reserved_system: int
  reserved_synthesis: int
  force_synthesis_at_ratio: float

# tool output budget (V2.1)
tool_output_budget:
  max_bytes_per_call: int
  truncation_marker: string
  summarizer_model: string | null

# boundary/runtime config
adapter_version: string
model_profile_id: string
grammar_profile: string | null
grammar: string | null            # compiled grammar payload if applicable
logits_mask: list[float] | null   # optional token gating mask
token_gate: bool                  # V2.1
speculative_exec: bool            # V2.1 (optional)

# anti-cycle controls (V2.1)
cycle_forbid: list[[string,string]]
```

### `tool_policy` semantics
- `required`: at least one validated+executed tool invocation is required for terminal success.
- `optional`: tool invocation is allowed but not required; terminal prose success is valid.
- `forbidden`: any tool invocation is a contract violation.

### `cycle_forbid` semantics
`cycle_forbid` is a list of forbidden ordered tool-name pairs `[from_tool, to_tool]`.
If runtime observes a transition matching any forbidden pair within a run (e.g., `write_file -> write_file`, `apply_patch -> revert_patch`), it must terminate with `FAILED_CONTRACT_VIOLATION` unless an explicit exception rule is defined in contract metadata.

## 4.2 `ProtocolAdapterResult`
```yaml
status: native | rejected | recovered
canonical_message: AgentMessage | null
failure_code: string | null
raw_hash: string
adapter_version: string
```

> V2.1 strict recommendation: for `strict_mode=true`, allow only `native|rejected` at runtime; `recovered` is legacy-compat mode only.

## 4.3 `AgentMessage` (canonical)
```yaml
role: system | user | assistant | tool
content: string
tool_calls: list[ToolInvocation]
```

## 4.4 `ToolInvocation`
```yaml
id: string
name: string
arguments: map
source_model: string
raw_payload: map | null
```

## 4.5 `AgentOutcome`
```text
COMPLETED_WITH_TOOLS
COMPLETED_CHAT_ONLY
FAILED_PREFLIGHT
FAILED_PROTOCOL_NO_TOOLS
FAILED_PROTOCOL_MALFORMED
FAILED_VALIDATION
FAILED_BUDGET_EXHAUSTED
FAILED_TIMEOUT
FAILED_CONTRACT_VIOLATION
INTERRUPTED
```

---

## 5) Runtime State Machine (authoritative)

```text
PRECHECK
  -> INFER
  -> VALIDATE_CALLS
  -> EXECUTE
  -> OBSERVE
  -> COMMIT
  -> (INFER | TERMINATE)
```

### State definitions
- **PRECHECK:** Validate immutable contract feasibility before first inference.
- **INFER:** Run model call with contract-scoped decoding constraints.
- **VALIDATE_CALLS:** Validate canonical tool call structure, policy, and budget impact.
- **EXECUTE:** Execute authorized tools with timeout and envelope validation.
- **OBSERVE:** Transform tool results into canonical observations.
- **COMMIT:** Update counters/hashes/transcript and evaluate transition predicates.
- **TERMINATE:** Emit typed `AgentOutcome` only from finalized machine state.

No early-return path may bypass state transition accounting.

## COMMIT transition predicates (authoritative)
At `COMMIT`, runtime evaluates transitions in deterministic order:

1. **Interrupt/timeout checks**
   - interrupted -> `TERMINATE(INTERRUPTED)`
   - `step_timeout_ms`/`total_timeout_ms` exceeded -> `TERMINATE(FAILED_TIMEOUT)`
2. **Budget checks**
   - `max_inferences` or `max_tokens_consumed` exceeded -> `TERMINATE(FAILED_BUDGET_EXHAUSTED)`
3. **Contract policy violations**
   - `tool_policy=forbidden` and any tool call executed -> `TERMINATE(FAILED_CONTRACT_VIOLATION)`
   - `tool_policy=required` and model produced terminal no-call response -> `TERMINATE(FAILED_PROTOCOL_NO_TOOLS)`
4. **Context threshold check**
   - if `force_synthesis_at_ratio` exceeded -> force synthesis terminal path:
     - if any tool executed -> `TERMINATE(COMPLETED_WITH_TOOLS)`
     - else -> `TERMINATE(COMPLETED_CHAT_ONLY)` for `optional`, otherwise policy failure
5. **Normal success checks**
   - if model emits no further tool calls and policy satisfied:
     - `required` + tools used -> `TERMINATE(COMPLETED_WITH_TOOLS)`
     - `optional` + tools used -> `TERMINATE(COMPLETED_WITH_TOOLS)`
     - `optional` + no tools used -> `TERMINATE(COMPLETED_CHAT_ONLY)`
6. **Continue loop**
   - otherwise -> `INFER`.

---

## 6) Phase Roadmap

## Phase 1 — Baseline Contract

### Purpose
Document current behavior and establish baseline metrics.

### Baseline behavior
- Agent execution loop may terminate success on no tool calls.
- Prompt/protocol mismatch may exist between prompt instructions and parsed tool-call format.
- Return semantics may remain boolean/ambiguous.

### Exit criteria
- Baseline metrics captured on representative corpus:
  - no-tool completion rate on required-action prompts,
  - parse failure rate,
  - avg/95p inference count,
  - avg/95p runtime duration,
  - max-step exits.

---

## Phase 2a — Deterministic Protocol Boundary

### Deliverables
1. ProtocolAdapter as strict boundary (canonical message or explicit failure).
2. Prompt/protocol alignment rule per `model_profile_id`.
3. Immutable transcript introduced as first-class artifact.
4. Adapter conformance tests per model profile.

### Rules
- No silent coercion of malformed structures.
- `strict_mode=true`: recovery disabled; rejection is explicit.
- Adapter behavior versioned and deterministic by input hash.

### Exit criteria
- No silent parse fallback paths.
- Protocol violations surfaced as explicit failure signals.
- Transcript emitted for every adapter decision.

---

## Phase 2b — Execution Contract Enforcement

### Deliverables
1. Thread immutable `ExecutionContract` through UI/addon/guard/engine/worker/runtime.
2. Preflight validation in `PRECHECK`.
3. Required/forbidden tool-policy enforcement.
4. Retry ladder enforcement (`max_format_retries`, hard bounded).
5. ContractFactory/IntentPolicyResolver introduced for explicit `tool_policy` assignment.

### Preflight checks
- Contract structural validity.
- Tool availability/capability consistency with `allowed_tools`.
- Model profile compatibility with policy/grammar profile.
- Context feasibility (`reserved_system + reserved_synthesis + minimum_loop_margin <= context_window`).
- Runtime backend responsiveness.

### Contract creation boundary (required)
`ExecutionContract` must be created by a deterministic factory layer before runtime starts:
- `ContractFactory` (or `IntentPolicyResolver`) consumes page context, user request, model profile, and capability manifest.
- It emits immutable contract with explicit `tool_policy`, budget defaults, and allowed tool scope.
- Runtime must reject contracts missing provenance metadata (`contract_id`, `contract_hash`, `model_profile_id`).

Minimum recommended deterministic defaults:
- chat page -> `tool_policy=forbidden`
- code page task with explicit file/command/workspace intent -> `tool_policy=required`
- code page conceptual/explanatory prompt -> `tool_policy=optional`

### Retry ladder (deterministic)
- Trigger: adapter returns `rejected` for malformed structured output.
- Retry budget decremented; retry consumes inference budget.
- Max retries bounded by contract and profile policy (recommended defaults: 0-1 strict, 1-3 legacy profiles).
- Retry exhaustion => `FAILED_PROTOCOL_MALFORMED`.

### Exit criteria
- `tool_policy=required` cannot end in terminal success with zero executed tool calls.
- Contract violations emit typed failure outcomes.
- No contract mutation after first inference.
- ContractFactory policy decisions are traceable in transcript metadata.

---

## Phase 3 — Typed Outcomes + Full FSM Semantics

### Deliverables
1. Replace boolean completion with typed `AgentOutcome`.
2. Enforce authoritative FSM transitions.
3. Unified budget accounting:
   - inferences,
   - token consumption,
   - tool output budget,
   - step/total timeouts.
4. UI state contract based on state digest + outcome.

### Tool execution semantics
- Default execution: sequential deterministic order.
- Any malformed tool result envelope => `FAILED_VALIDATION`.
- Partial execution must be explicitly recorded in transcript.

### Commit-time checks
- Invariant checks,
- budget threshold checks,
- forced-synthesis transition when context ratio exceeded.

### Exit criteria
- All terminal paths map to disjoint typed outcomes.
- Replay of recorded run reproduces same outcome class under same versions.
- UI and runtime state digests remain consistent.

---

## Phase 4 — Open Foundation (Metric-Gated)

### Purpose
Harden portability, replay, and ecosystem openness.

### Deliverables
1. Deterministic hash-chain transcript:
   - `H_n = hash(H_n-1, contract_hash, state, action_hash, result_hash, adapter_version, model_profile_id, model_fingerprint)`
2. Grammar-constrained decoding profiles by policy.
3. Provider/model conformance suite.
4. Versioned migration policy for contract and adapter formats.

### Optional (only if justified by metrics)
- Split internal modules into Decision/Execution/Policy/Trace engines.
- Parallel execution profiles for explicitly safe tools.
- Capability escrow or SAT-style prechecks.

### Exit criteria
- Conformance suite passes across supported model profiles.
- Replay divergence is detectable and attributable by hash-chain checkpoint.
- Required-tool contract violations are structurally unreachable in strict profile.

---

## 7) V2.1 Hardening Overlays (Applied to V2)

1. **Transcript moved to Phase 2a** (not deferred).
2. **Grammar ownership clarified:** runtime derives semantics/constraints, adapter applies provider-specific boundary handling.
3. **Tool output budget added** to avoid context blow-ups from oversized tool envelopes.
4. **`FAILED_PREFLIGHT` added** to separate setup faults from runtime execution faults.
5. **`strict_mode` made exhaustive:** affects recovery, retries, and gating behavior.
6. **Retry path deterministic and bounded (`<=1`)**.
7. **Token-gating invariant added (Invariant F)** for required-tool mode.
8. **Phase 4 split remains optional and metric-gated** (avoid premature architecture fragmentation).

---

## 8) Keep / Modify / Remove

### Keep
- Single-loop runtime skeleton.
- Capability authorization and tool schema validation.
- Event emission as observability channel.

### Modify
- Adapter/result semantics (deterministic boundary).
- Termination semantics (typed outcomes only).
- Budget model (add tool output + timeout clarity).
- UI contract (state-first rendering).

### Remove
- Ambiguous success semantics.
- Any unbounded retry behavior.
- Any hidden contract mutation at runtime.

---

## 9) Minimal Conformance Corpus (Required)

Run for each model profile:
1. required-tool prompt with valid canonical tool call.
2. required-tool prompt with malformed structured output.
3. required-tool prompt with narration-only response.
4. forbidden-tool prompt with attempted tool call.
5. oversized tool result envelope.
6. timeout-inducing tool call.

Pass criteria: outcome class and transcript invariants match expected contract behavior.

---

## 10) Non-Goals (for this contract version)
- Full planner/executor split by default.
- Mandatory transactional rollback DAGs for all tools.
- Advanced escrow/SAT mechanisms before baseline deterministic behavior is stable.

---

## 11) Final Contract Rule
If behavior requires runtime interpretation rather than contract/state predicates, the contract is incomplete and must be tightened before release.
