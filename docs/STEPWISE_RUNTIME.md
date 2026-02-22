# Step-wise Agent Runtime (Option B Implementation)

## Problem
The original `AgentRuntime.run()` was a blocking synchronous loop that executed all agent steps (multiple LLM calls) before returning. This caused:
- UI updates to batch at the end
- No real-time visibility into step progression
- Assistant responses appearing all at once

## Solution
Refactored to `StepwiseAgentRuntime` that yields control between each LLM call, allowing Qt signals to propagate to the UI in real-time.

## Architecture Changes

### 1. New File: `engine/agent_runtime_stepper.py`
A resumable state machine wrapper around `AgentRuntime`:

```
StepwiseAgentRuntime
├── initialize()           # Setup history, capabilities, contract
├── step() → StepResult   # Execute ONE state transition
│   ├── _step_precheck()
│   ├── _step_infer()      # LLM call (blocking, but isolated)
│   ├── _step_validate_calls()
│   ├── _step_execute()    # Tool execution (one per call)
│   ├── _step_observe()
│   └── _step_commit()     # Evaluate continue/terminate
├── should_continue()      # Check if more steps needed
└── get_result()          # Final AgentRunResult
```

### 2. Modified: `engine/llm.py` - `GeneratorWorker.run()`
Changed from:
```python
result = self.runtime.run(self.messages)  # Blocking, all steps
```

To:
```python
stepper = StepwiseAgentRuntime(...)
stepper.initialize()

while stepper.should_continue():
    result = stepper.step()    # One step
    self.msleep(10)            # Yield for signal propagation
    
result = stepper.get_result()
```

## State Machine Flow

```
PRECHECK → INFER → VALIDATE_CALLS → EXECUTE* → OBSERVE → COMMIT
              ↑___________________________________________|
              (loop back for multi-step agents)
```

*EXECUTE iterates one tool at a time, yielding between each

## Signal Propagation

Each `step()` call:
1. Emits events (STEP_START, LLM_TOKEN, TOOL_CALL_START, etc.)
2. Returns to `GeneratorWorker`
3. `GeneratorWorker.msleep(10)` processes Qt event loop
4. Signals flush to UI
5. Next step executes

## UI Benefits

1. **Real-time step visibility**: Each LLM call appears as it happens
2. **Streaming tokens**: Assistant content streams chunk-by-chunk
3. **Tool execution visibility**: Each tool call shown individually
4. **Interruptible**: STOP request checked between every step

## Performance Characteristics

- **Same total latency**: LLM calls still blocking
- **Slightly more overhead**: ~10ms yield per step
- **Better perceived responsiveness**: UI updates during execution
- **Memory identical**: No additional buffering

## Backward Compatibility

- Original `AgentRuntime.run()` still exists
- Non-agent mode unchanged (single LLM call)
- All existing events emitted with same format
- `AgentRunResult` return type unchanged

## Configuration

No configuration needed. Automatically active for all agent mode executions.

To disable (revert to blocking), modify `GeneratorWorker.run()`:
```python
# Replace step-wise execution with:
result = self.runtime.run(self.messages)
```
