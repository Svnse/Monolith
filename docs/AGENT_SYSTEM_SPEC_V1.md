# Monolith Agent System Spec (V1)

Status: Implemented baseline + structural cleanup complete  
Scope: `engine/loop/*`, `engine/loop_engine.py`, `ui/pages/code.py`, tests under `tests/engine/*`

## 1. Product Intent
- Build a chat-first coding buddy, not a batch task runner.
- Preserve runtime safety walls, policy gating, and approvals.
- Support interrupt + redirect behavior mid-run.
- Keep debug visibility available without making raw machine traces the default UX.

## 2. Layered Architecture
1. Conversation Surface (UI)
- Primary timeline with user/agent/tool/approval/system cards.
- Inline approvals and cycle grouping/folding.
- Composer-first interaction model (`MESSAGE` + `SEND`).

2. Control Plane (Run/Turn)
- Explicit control events and run metadata.
- Run lifecycle transitions with reason codes.
- Redirect/stop orchestration and approval command channel.

3. Execution Runtime
- Single loop runtime over a `Pad`.
- Tool execution, wall enforcement, parse/retry behavior.
- Per-step structured output (`StepPacket`) parsing.

4. Context Memory
- In-run continuity via `Pad`.
- Cross-turn carry-forward context (summary + evidence + touched files + self-checks).

## 3. Core Data Contracts
Defined in `engine/loop/contracts.py`.

1. `RunPolicy`
- `max_cycles`, `max_tool_calls`, `max_elapsed_sec`, retry and anti-loop windows.
- Scope-based `auto_approve` and `require_approval`.

2. `ToolSpec`
- Tool name, description, parameters, scope.

3. `Evidence`
- Normalized tool output record per cycle.

4. `Pad`
- Working memory artifact for a run.
- Includes `plan`, `steps`, `evidence`, `progress`, `artifacts`, `next_intent`, `open_questions`.
- Includes verification memory: `recent_checks` with bounded retention.
- Includes mission and recovery memory (`mission_refs`, `mission_gaps`, tool-failure recovery hints).

5. `Step`
- Parsed model step:
  - `response`, `intent`, `reasoning`, `actions`
  - `expected_outcome`, `verification`, `contract_refs`
  - `step_ok`, `self_check`
  - `pad_patch`, `finish`, `finish_summary`

6. `RunContext` / `RunResult`
- Mutable run state and terminal summary payload.

## 4. Prompt/Step Contract
Defined in `engine/loop/prompt.py` and parsed in `engine/loop/runtime.py`.

1. StepPacket requirements
- JSON-only response.
- Conversational `response` each cycle.
- Verification fields (`step_ok`, `self_check`) requested each cycle.

2. Cycle input context
- Goal + current pad snapshot.
- Tool catalog and budget.
- Tool usage contracts (`when_to_use`, `when_not_to_use`, `required_args`, `failure_recovery`, example calls).
- Recent evidence and recent self-checks rendered into prompt context.

## 5. Runtime Behavior
Implemented in `engine/loop/runtime.py`.

1. Loop flow
- `cycle_start` -> infer/parse -> wall check -> execute -> update pad -> finish/continue.

2. Stop behavior
- Cooperative stop request checked at cycle boundaries.

3. Wall enforcement
- Max cycles/tool calls/time and anti-stall/repetition checks.

4. Step parsing
- Supports fenced JSON and object extraction fallback.
- Validates required shape and action format.

5. Pad update discipline
- Applies `pad_patch`.
- Records evidence.
- Records step self-check signals.
- Adds continuity pressure:
  - failed self-checks append bounded `open_questions`.
  - repeated failures nudge `next_intent` toward strategy change if not provided.
  - tool failures append recovery guidance derived from tool contracts.

6. Finish summary composition
- Uses `finish_summary` if provided.
- Falls back to `self_check` / `verification` / `expected_outcome` when needed.

## 6. Policy Kernel
Implemented in `engine/loop/policy.py`.

1. Decision actions
- `allow`, `needs_approval`, `deny`

2. Reason codes (examples)
- `empty_tool_name`, `unknown_tool`
- `requires_approval_scope`, `policy_scope_conflict`
- `auto_approve_scope`, `default_allow_scope`

3. Runtime integration
- Runtime emits policy decisions before tool execution.
- Approval-required decisions trigger approval events/flow.

## 7. Run/Turn Control Plane Contract
Primary engine wrapper: `engine/loop_engine.py`  
Shared constants: `engine/loop/events.py`  
Lifecycle reducer helper: `engine/loop/lifecycle.py`

1. Metadata on emitted events (`_meta`)
- `spec_version`, `seq`, `ts`
- `session_id`, `turn_id`, `parent_run_id`, `run_id`

2. Control events
- `turn_started`
- `redirect_started`
- `turn_stop_requested`
- `approval_response`
- `turn_completed`
- `turn_failed`

3. Lifecycle state model
- States: `created`, `running`, `waiting_approval`, `stopping`, `redirected`, `completed`, `stopped`, `failed`
- Shared transition legality map in `engine/loop/events.py`.
- Transition application helper in `engine/loop/lifecycle.py`.

4. Run identity
- `run_id` pre-bound at turn start and passed into `LoopRuntime`.
- Ensures all lifecycle/control events are run-bound from first emission.

## 8. Control Inbox Commands
Shared command constants in `engine/loop/commands.py`.

Supported commands:
- `stop`
- `set_tools`
- `set_infer_config`
- `set_infer_backend`
- `get_last_result`
- `get_run_journal`
- `approval_response`

## 9. Effect Journal (V1)
Implemented in `engine/loop_engine.py`.

1. Storage
- In-memory per-run append-only journal with bounded retention.

2. Coverage
- Control and loop events needed for lifecycle/policy/tool/approval tracing.

3. Entry schema
- Top-level fields: `event`, `kind`, `entry_type`, `run_id`, `seq`, `ts`, `session_id`, `turn_id`
- Retains convenient flattened fields (tool/status/state/reason/etc.)
- Includes normalized payload snapshot: `payload = {event, kind, data}`

4. Query API (`get_run_journal`)
- Optional `run_id` (defaults to last/active run).
- Supports:
  - `kind` filter (`str` or `list[str]`)
  - `limit`
  - `reverse`
  - `include_summary`

## 10. UI Surface (CODE Page)
Implemented primarily in `ui/pages/code.py`.

1. Timeline-first UX
- Message cards for user/agent/thinking/tool/approval/system.
- Tool/details panels are collapsible.
- Cycle headers with fold/unfold.
- Older cycles auto-collapse on new cycle.

2. Inline approvals
- Approval prompts shown in timeline with inline Approve/Deny.
- Fallback approval panel retained (de-emphasized).

3. Redirect behavior
- Sending while active run triggers stop + queued redirect.
- Redirect restarts on READY and keeps context.

4. Carry-forward context
- On follow-up/redirect, model receives structured previous-run context:
  - status/reason, summary, touched files, recent evidence, recent self-checks.

5. Debug events panel
- Raw JSON replaced by structured event-feed lines:
  - sequence id
  - arrowed event type
  - cycle-based indentation
  - key event fields

## 11. Safety and Enforcement
- Runtime walls enforced in loop runtime.
- Scope policy admission via policy kernel.
- Approval gate remains runtime-controlled.
- Stop requests propagate to worker/runtime and pending approval gates fail closed.

## 12. Testing Contract
Primary test suites:
- `tests/engine/test_loop.py`
- `tests/engine/test_loop_engine.py`
- `tests/engine/test_loop_lifecycle.py`
- plus other engine suites in `tests/engine/`

Current verification command:
```powershell
$env:PYTHONPATH='.'; pytest -q tests\engine
```

## 13. Non-Goals (V1)
- CLI parity as primary UX.
- Full persistent cross-session memory.
- Multi-surface orchestrator unification beyond current control/event contracts.

## 14. Known Structural Follow-Ups
1. Reduce dual-journal drift by moving more UI-derived journal shaping to engine projections.
2. Continue slimming `ui/pages/code.py` into helper modules (timeline formatter/carry-forward builder).
3. Expand integration tests for redirect/approval race paths at UI boundary.

## 15. Intent + Mission Stack (Implemented)
1. Intent compiler
- Module: `engine/loop/intent_compiler.py`
- Compiles raw turn prompt into:
  - `turn_intent` (typed intent envelope)
  - `mission_contract` (objective, constraints, success criteria, artifacts)
- Loop engine prepends a rendered mission block to the run goal.

2. Mission-aware step schema
- `Step` supports `contract_refs` (criteria IDs advanced by the cycle).
- Prompt contract instructs model to emit `contract_refs` when mission criteria exist.

3. Mission adjudicator
- Module: `engine/loop/adjudicator.py`
- Runtime checks mission completion when `finish=true`.
- If required criteria are missing:
  - finish is denied for that cycle
  - gaps are written into `Pad`
  - run continues instead of falsely completing.

4. Pad mission memory
- `Pad` now carries:
  - `mission_origin`
  - `turn_intent`
  - `mission_contract`
  - `mission_refs`
  - `mission_gaps`

## 16. Structured Tool Intelligence (Implemented)
1. Deterministic outcome classifier
- Module: `engine/loop/tool_intelligence.py`
- Classifies failed tool outcomes into stable classes:
  - `missing_dependency`, `permission`, `path_not_found`, `syntax_error`, `timeout`, `runtime_error`, `unknown`
- Runtime emits these via `tool_failure_guidance` fields.

2. Semantic action signatures
- Module: `engine/loop/tool_intelligence.py`
- Normalizes action identity (e.g., `run_cmd` based on command string, ignoring exec flags).
- Enables retry accounting on semantic action, not literal arg dict.

3. Circuit breaker
- Runtime blocks repeated failing semantic actions (threshold=2 prior failures).
- Emits `circuit_breaker` and injects deterministic pivot directives into Pad.

4. Discovery no-op blocker
- Runtime blocks consecutive identical discovery actions (`read/list/search/grep`) with `noop_blocked`.
- Preserves planning autonomy while preventing redundant execution.

5. Verifier policy
- Module: `engine/loop/verifier_policy.py`
- Exposes deterministic verification mode hints from recent classified failures (e.g., static-preferred on missing dependency).

6. Mission decomposition hardening
- Module: `engine/loop/mission_decomposer.py`
- Validates mission criteria schema, dedupes IDs, and enforces minimum required criterion.

7. Gap-state projection (new planning substrate)
- Module: `engine/loop/gap_state.py`
- Runtime computes per-cycle:
  - `completed_requirements`
  - `open_requirements`
  - `blocked_requirements`
  - `last_failure_class`
  - `available_strategies`
- Injected into prompt as `GAP STATE` so the planner reasons over explicit deltas, not implicit history parsing.

8. Hard execution routing for dependency-blocked retries
- For `run_cmd`, if the same semantic command failed with `missing_dependency`, retries are blocked (`routing_blocked`) until an install attempt for that dependency is observed.

9. Budget semantics
- Pseudo-actions blocked by runtime guards (`noop_blocked`, `circuit_breaker`, `routing_blocked`) do not consume tool-call budget.

## 17. Platform Portability Baseline (Implemented)
1. Cross-platform root resolution
- Module: `core/paths.py`
- `MONOLITH_ROOT` resolution order:
  - `MONOLITH_ROOT` env override
  - Windows: `%APPDATA%/Monolith` fallback to `~/AppData/Roaming/Monolith`
  - Linux/macOS:
    - preserves legacy `~/Monolith` when present
    - else uses XDG data root (`$XDG_DATA_HOME/Monolith` or `~/.local/share/Monolith`)

2. Workspace default normalization
- Module: `core/config.py`
- `DEFAULT_WORKSPACE_ROOT` now derives from `MONOLITH_ROOT / "workspace"` (unless `MONOLITH_WORKSPACE` is set), removing Windows-only path assumptions.

3. Databank base path portability
- Module: `ui/pages/databank.py`
- Default knowledge base path no longer hardcodes `C:\\Models\\knowledge_base`.
- Resolution order:
  - `MONOLITH_KNOWLEDGE_BASE` env override
  - `DEFAULT_WORKSPACE_ROOT / "knowledge_base"`

4. PTY snapshot portability hardening
- Module: `engine/pty_runtime.py`
- PTY environment snapshot parsing no longer depends on GNU-specific `base64 -w0`.
- Uses `env -0` parsing directly for broader Linux/macOS compatibility.
