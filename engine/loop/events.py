from __future__ import annotations

# Shared run/turn control-plane event contract constants (v1).

CONTROL_EVENT_SPEC_VERSION = "runturn.v1"

# Top-level event channels.
EVENT_CONTROL = "control"
EVENT_LOOP = "loop"

# Control event kinds.
CTRL_TURN_STARTED = "turn_started"
CTRL_REDIRECT_STARTED = "redirect_started"
CTRL_TURN_STOP_REQUESTED = "turn_stop_requested"
CTRL_TURN_COMPLETED = "turn_completed"
CTRL_TURN_FAILED = "turn_failed"
CTRL_APPROVAL_RESPONSE = "approval_response"

# Lifecycle states.
RUN_STATE_CREATED = "created"
RUN_STATE_RUNNING = "running"
RUN_STATE_WAITING_APPROVAL = "waiting_approval"
RUN_STATE_STOPPING = "stopping"
RUN_STATE_REDIRECTED = "redirected"
RUN_STATE_COMPLETED = "completed"
RUN_STATE_STOPPED = "stopped"
RUN_STATE_FAILED = "failed"

RUN_STATES = frozenset({
    RUN_STATE_CREATED,
    RUN_STATE_RUNNING,
    RUN_STATE_WAITING_APPROVAL,
    RUN_STATE_STOPPING,
    RUN_STATE_REDIRECTED,
    RUN_STATE_COMPLETED,
    RUN_STATE_STOPPED,
    RUN_STATE_FAILED,
})

# Lifecycle reason codes.
REASON_TURN_STARTED = "turn_started"
REASON_CYCLE_START = "cycle_start"
REASON_APPROVAL_PROMPT = "approval_prompt"
REASON_APPROVAL_GRANTED = "approval_granted"
REASON_APPROVAL_DENIED = "approval_denied"
REASON_STOP_REQUESTED = "stop_requested"
REASON_USER_REDIRECT = "user_redirect"
REASON_COMPLETED = "completed"
REASON_WALL_HIT = "wall_hit"
REASON_ERROR = "error"

RUN_REASON_CODES = frozenset({
    REASON_TURN_STARTED,
    REASON_CYCLE_START,
    REASON_APPROVAL_PROMPT,
    REASON_APPROVAL_GRANTED,
    REASON_APPROVAL_DENIED,
    REASON_STOP_REQUESTED,
    REASON_USER_REDIRECT,
    REASON_COMPLETED,
    REASON_WALL_HIT,
    REASON_ERROR,
})

# Legal lifecycle transitions (v1 UI/runtime contract).
RUN_STATE_TRANSITIONS: dict[str, frozenset[str]] = {
    RUN_STATE_CREATED: frozenset({RUN_STATE_RUNNING, RUN_STATE_FAILED}),
    RUN_STATE_RUNNING: frozenset({
        RUN_STATE_WAITING_APPROVAL,
        RUN_STATE_STOPPING,
        RUN_STATE_REDIRECTED,
        RUN_STATE_COMPLETED,
        RUN_STATE_STOPPED,
        RUN_STATE_FAILED,
    }),
    RUN_STATE_WAITING_APPROVAL: frozenset({
        RUN_STATE_RUNNING,
        RUN_STATE_STOPPING,
        RUN_STATE_REDIRECTED,
        RUN_STATE_COMPLETED,
        RUN_STATE_STOPPED,
        RUN_STATE_FAILED,
    }),
    RUN_STATE_STOPPING: frozenset({
        RUN_STATE_REDIRECTED,
        RUN_STATE_COMPLETED,
        RUN_STATE_STOPPED,
        RUN_STATE_FAILED,
    }),
    RUN_STATE_REDIRECTED: frozenset({
        RUN_STATE_RUNNING,
        RUN_STATE_COMPLETED,
        RUN_STATE_STOPPED,
        RUN_STATE_FAILED,
    }),
    RUN_STATE_COMPLETED: frozenset(),
    RUN_STATE_STOPPED: frozenset(),
    RUN_STATE_FAILED: frozenset(),
}
