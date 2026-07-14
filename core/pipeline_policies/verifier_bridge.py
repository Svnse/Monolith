"""Verifier bridge — wraps core/response_verifier.verify_response on TurnReadyEvent.

Authority tier: OBSERVATION. No kill switch declared (observation tier).
The wrapped verifier itself has its own kill switch via the existing
MONOLITH_VERIFIER_V1 env flag (core/response_verifier.py:58), which this
bridge respects implicitly by calling verifier_enabled() before running.

Subscribes to TurnReadyEvent. Emits VerifierVerdictEvent and, for each
non-pass finding, a FaultDetectedEvent with fault_kind="verifier:<code>".

The render-time / direct-call path to response_verifier in chat.py stays
intact in Phase 2 (defense in depth — the bridge gives turn_trace
visibility without removing the existing call). Phase 4 considers whether
the direct call can be removed.

Independence: imports core/response_verifier.py — that's an in-core, non-
engine, non-ACU module. Boundary preserved.
"""
from __future__ import annotations

from core import response_verifier as _rv
from core.pipeline_registry import PolicyRegistration
from core.turn_pipeline_events import (
    AuthorityTier,
    FaultDetectedEvent,
    PipelineEvent,
    TurnReadyEvent,
    VerifierVerdictEvent,
)


NAME = "verifier_bridge"


REGISTRATION = PolicyRegistration(
    name=NAME,
    module_path="core.pipeline_policies.verifier_bridge",
    subscribes_to=("TurnReadyEvent",),
    depends_on=("output_sanitizer",),  # sanitizer must run first on TurnReadyEvent
    authority_tier=AuthorityTier.OBSERVATION,
    kill_switch_env_flag="",  # observation tier — no per-policy switch
)


def register_with(pipeline) -> None:
    pipeline.register(REGISTRATION, _handle)


def _handle(event: PipelineEvent, ctx) -> None:
    if not isinstance(event, TurnReadyEvent):
        return
    if not _rv.verifier_enabled():
        return

    result = _rv.verify_response(
        raw_answer=event.raw_answer,
        public_answer=event.public_answer,
        tools_used=list(event.tools_used),
    )

    from monokernel.turn_pipeline import get_pipeline
    pipeline = get_pipeline()
    findings_payload = tuple(f.to_payload() for f in result.findings)
    pipeline.publish(
        VerifierVerdictEvent(
            verdict=result.verdict,
            findings=findings_payload,
        ),
        ctx,
        source_kind="policy",
        source_name=NAME,
    )

    for finding in result.findings:
        severity = "hard" if finding.severity == _rv.SEVERITY_HARD_FAIL else "warn"
        pipeline.publish(
            FaultDetectedEvent(
                fault_kind=f"verifier:{finding.code}",
                severity=severity,
                source_event_seq=event.seq,
                detail={
                    "message": finding.message,
                    "verifier_code": finding.code,
                    "verifier_detail": dict(finding.detail or {}),
                },
            ),
            ctx,
            source_kind="policy",
            source_name=NAME,
        )
