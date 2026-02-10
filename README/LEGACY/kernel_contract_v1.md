MONOLITH KERNEL CONTRACT — v1 (FROZEN)
0. Purpose (Non-Negotiable)

The Monolith kernel (MonoGuard) is the sole authority between UI and execution engines.

Its role is arbitration, not computation.

All future functionality must route around the kernel, not into it.

1. Authority Rules

1.1 Single Ingress
All user-initiated commands that affect execution MUST pass through MonoGuard.

UI must not call engine methods directly.

Addons must not call engine methods directly.

1.2 Single Egress
All execution state, tokens, traces, and usage metrics MUST pass through MonoGuard.

UI must not subscribe to engine signals directly.

Engines must never emit directly to UI.

2. Kernel Scope

The kernel MAY:

Route commands

Gate execution by system state

Preempt execution via STOP

Queue at most one pending command

Re-emit engine signals verbatim

Observe system state transitions

The kernel MUST NOT:

Execute business logic

Perform blocking operations

Sleep, wait, or poll

Contain UI logic

Contain engine logic

Know what “chat”, “LLM”, or “RAG” is

Accumulate feature-specific state

3. STOP Semantics (Hard Law)

3.1 STOP Always Wins

When STOP is issued:

Current execution is interrupted immediately (non-blocking)

Any pending command is cleared

Control returns to the UI instantly

3.2 Truthful State

SystemStatus.READY MUST only be emitted when:

No execution is running

No engine work is active

No pending command is executing

The kernel must never emit READY prematurely.

4. Pending Command Rule

4.1 Single Pending Slot

The kernel may hold at most one pending command.

Pending commands exist only to resume after STOP-based preemption.

4.2 Replay on READY

A pending command may execute once when the system transitions to READY.

Pending commands are discarded if STOP is explicitly invoked.

No scheduling, prioritization, or batching exists in v1.

5. Engine Isolation

The engine:

Is execution-only

Knows nothing about UI

Knows nothing about kernel rules

Knows nothing about addons

Accepts commands and emits signals only

The kernel adapts the engine; the engine never adapts to the kernel.

6. UI Restrictions

The UI:

May emit commands freely

Must not assume commands will execute

Must not block waiting for execution

Must treat kernel signals as authoritative truth

UI correctness depends on kernel truth, not intent.

7. Stability Guarantee

Any future change that violates:

Single ingress

Single egress

STOP dominance

Non-blocking kernel behavior

breaks the kernel contract and must be treated as a major architectural change.

8. Extension Rule

If a feature:

Can be removed without breaking the kernel
→ it does not belong in the kernel.

If a feature:

Requires changing kernel behavior
→ it is a kernel version bump, not a feature patch.

9. Freeze Declaration

This kernel contract is considered frozen as of this state.

Future development must build above this boundary unless explicitly redesigning the kernel.

One-Line Summary

The kernel decides when things may happen — never what happens.
