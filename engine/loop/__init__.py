"""
Loop — goal-seeking autonomous agent runtime.

No fixed FSM. One loop, one Pad, hard walls.
The agent chooses its own cognitive mode each cycle.
The runtime only enforces boundaries and validity.
"""

from engine.loop.contracts import (
    Evidence,
    Pad,
    PreflightResult,
    RunContext,
    RunPolicy,
    RunResult,
    Step,
    ToolSpec,
)
from engine.loop.runtime import LoopRuntime
from engine.loop.monolith_adapter import (
    LLMEngineInferAdapter,
    build_monolith_tool_specs,
    execute_monolith_tool,
    make_loop_trace_emitter,
)

__all__ = [
    "Evidence",
    "LoopRuntime",
    "LLMEngineInferAdapter",
    "Pad",
    "PreflightResult",
    "RunContext",
    "RunPolicy",
    "RunResult",
    "Step",
    "ToolSpec",
    "build_monolith_tool_specs",
    "execute_monolith_tool",
    "make_loop_trace_emitter",
]
