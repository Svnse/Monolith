# WAIT_ACK Flow Implementation (OFAC v0.2)

## Overview
When the agent attempts to execute destructive tools (EXEC/WRITE scope), it enters WAIT_ACK state and pauses for user approval.

## Flow Diagram

```
Agent Step Execution
        ↓
VALIDATE_CALLS — checks if tools need ACK
        ↓
    ┌─────────────────┐
    │ Tool requires   │──No──→ EXECUTE (normal flow)
    │ ACK?            │
    └─────────────────┘
          │Yes
          ↓
   WAIT_ACK_ENTER event emitted
          ↓
   UI shows ACK modal dialog
          ↓
   Worker thread terminates cleanly
          ↓
   ┌──────────────────────────────┐
   │  USER DECISION               │
   │  [Approve] [Deny]            │
   └──────────────────────────────┘
          │
    ┌─────┴─────┐
    ↓           ↓
 Approve      Deny
    ↓           ↓
RESUME        TERMINATE
    ↓
_ResumeWorker spawns
    ↓
EXECUTE continues
```

## Code Flow

### 1. VALIDATE_CALLS → WAIT_ACK
**File**: `engine/agent_runtime_stepper.py:560-581`

```python
if needs_ack:
    self._current_state = RuntimeState.WAIT_ACK
    self._emit({"event": "WAIT_ACK_ENTER", "tools": ack_summaries})
    return StepResult(awaiting_ack=True, should_continue=False)
```

### 2. UI Shows Modal
**File**: `ui/pages/code.py:964-1010`

```python
def _show_ack_modal(self, tools: list[dict]):
    # Shows QMessageBox with tool list
    # User clicks Yes/No
    # Emits sig_runtime_command with ack_decision
```

### 3. Worker Terminates
**File**: `engine/llm.py:273-285`

```python
if step_result.awaiting_ack:
    self.trace.emit("[WORKER] entering WAIT_ACK — thread will terminate")
    return  # Don't emit done! Stepper parked in memory.
```

### 4. User Decision → Resume
**File**: `ui/pages/code.py:1007-1010`

```python
if reply == QMessageBox.Yes:
    self.sig_runtime_command.emit({"action": "ack_decision", "decision": "approve"})
else:
    self.sig_runtime_command.emit({"action": "ack_decision", "decision": "deny"})
```

### 5. Runtime Command Handler
**File**: `engine/llm.py:746-758`

```python
def runtime_command(self, command: str, payload: dict):
    if command == "ack_decision":
        decision = request.get("decision", "deny")
        success = self.resume_agent(action=decision)
```

### 6. Resume Agent
**File**: `engine/llm.py:707-744`

```python
def resume_agent(self, action: str) -> bool:
    # Finds parked stepper from previous worker
    stepper = self.worker._stepper
    
    # Creates _ResumeWorker
    resume_worker = _ResumeWorker(stepper, action)
    resume_worker.start()
```

### 7. Resume Worker Runs Stepper
**File**: `engine/llm.py:341-394`

```python
class _ResumeWorker(QThread):
    def run(self):
        stepper.resume_from_ack(self._action)  # TRANSITION to EXECUTE or TERMINATE
        while stepper.should_continue():
            step_result = stepper.step()
            # Continue execution...
```

## Events Emitted

| Event | Direction | Description |
|-------|-----------|-------------|
| `WAIT_ACK_ENTER` | Runtime → UI | Agent entering WAIT_ACK, show modal |
| `ACK_APPROVED` | Runtime → UI | User approved, resuming execution |
| `ACK_REJECTED` | Runtime → UI | User denied or timeout, terminating |
| `ACK_DECISION_HANDLED` | Engine → UI | Command processed |

## State Transitions

### Approve Path
```
WAIT_ACK → EXECUTE → OBSERVE → COMMIT → ...
```

### Deny Path
```
WAIT_ACK → TERMINATE (outcome=INTERRUPTED, reason=user_denied)
```

## Thread Lifecycle

1. **Initial Worker**: Dies cleanly at WAIT_ACK (no `done` signal)
2. **UI Thread**: Shows modal, waits for user
3. **Resume Worker**: New thread spawned after decision, continues stepper
4. **No Thread Leaks**: Each worker terminates cleanly, stepper stays in memory

## Testing

### Manual Test Steps
1. Open CODE addon
2. Request something requiring file write (e.g., "create a file")
3. **Expected**: Modal appears listing tools to execute
4. Click **Approve**
5. **Expected**: Execution continues, file created
6. Request another file write
7. Click **Deny**
8. **Expected**: Agent terminates with "user_denied" reason

### Edge Cases
- **Double-click**: Rapid approve/deny should be idempotent
- **STOP during WAIT_ACK**: Should terminate immediately
- **Close window during WAIT_ACK**: Should timeout/deny after TTL
