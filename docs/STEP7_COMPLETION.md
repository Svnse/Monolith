# Step 7 Completion: Ledger Type Enforcement & OFAC Payloads

## Summary
Completed wiring of OFAC v0.2 structured payloads (FIT_CONTRACT, PLAN_SNAPSHOT, MUTATION_RECORD) into the AgentRuntime and StepwiseAgentRuntime.

## Changes Made

### 1. `engine/agent_runtime_stepper.py` - OFAC Payload Emission

#### Initialization (`initialize()` method)
- Added `ENV_SNAPSHOT` emission via `_runtime._emit_env_snapshot()`
- Added `FIT_CONTRACT` emission via `_runtime._emit_fit_contract()`
- Goal extracted from first user message (first 100 chars)
- Events emitted before first step execution

#### EXECUTE State (`_step_execute()` method)
- Added `_runtime._snapshot_workspace_before()` call before first tool execution
- Only snapshots on first tool (when `_current_tool_index == 0`)
- Uses ManifestOracle for fast filesystem stat snapshots

#### OBSERVE State (`_step_observe()` method)
- Added `_runtime._snapshot_workspace_after_and_emit()` call
- Emits `MUTATION_RECORD` with filesystem diff (added/modified/deleted files)
- Hash-based comparison, not stdout parsing

### 2. Hash Boundary Verification

All OFAC payloads comply with "No floats/timestamps inside hash boundary":

| Payload Class | Fields | Hash-Safe |
|--------------|--------|-----------|
| `EnvSnapshot` | strings, bool | ✓ No floats/timestamps |
| `FitContract` | strings, tuples | ✓ No floats/timestamps |
| `PlanSnapshot` | int, tuple of PlanStep | ✓ No floats/timestamps |
| `PlanStep` | str, bool | ✓ No floats/timestamps |
| `MutationRecord` | list[str], hash strings | ✓ No floats/timestamps |

### 3. Hash Functions Used

- `canonical_json()` - Deterministic JSON with `sort_keys=True`, compact separators
- `canonical_hash()` - SHA-256 of canonical JSON
- All payload `to_dict()` methods return hash-safe structures

## E2E Test Checklist

Before marking complete, verify:

- [ ] **Windows STOP handling**: ProcessGroupController kills child PIDs on STOP
- [ ] **UI ACK modal**: WAIT_ACK shows approval dialog, blocks until user response
- [ ] **Large repos**: ManifestOracle uses stat cache, doesn't freeze on 10k+ files
- [ ] **Signal propagation**: Step-wise execution yields between steps, UI updates real-time

## Payload Flow

```
initialize()
  ├── ENV_SNAPSHOT (cached, computed once)
  └── FIT_CONTRACT (goal from first user message)
  
step() → EXECUTE
  └── _snapshot_workspace_before() (first tool only)
  
step() → OBSERVE
  └── _snapshot_workspace_after_and_emit()
      └── MUTATION_RECORD (if mutations detected)
```

## Backward Compatibility

- All new payload emissions are additive
- Existing events (STEP_START, TOOL_CALL_START, etc.) unchanged
- Non-agent mode unaffected
- Ledger schema extensible (new event types)
