"""intent_predict_bridge — the predict beat as a TurnReadyEvent policy.

Mirrors core/pipeline_policies/verifier_bridge.py. Subscribes to TurnReadyEvent,
which carries the CLEAN final public_answer (never the raw tool-loop intermediate
the message-interceptor view would expose) and is stamped once per outer turn.
On each event it freezes a prediction from the answer's staked content via
friction_organ.on_turn_ready (floor + optional card), recording it to Layer F.

DARK: this file lives in core/ (NOT core/pipeline_policies/) on purpose — the
kernel's filesystem validator (monokernel/turn_pipeline.validate_against_filesystem)
requires every file under pipeline_policies/ to be in pipeline_registry.POLICIES,
so an unregistered policy file there would FAIL boot. Keeping it here = no boot
impact, registration stays a documented activation diff (FRICTION_RECALIBRATE.md):
move this file into core/pipeline_policies/, set module_path accordingly, and add
NAME to pipeline_registry.POLICIES.

Authority tier: OBSERVATION (emits no events, mutates no stream). Self-gates on
MONOLITH_FRICTION_V1 (friction_store.flag_enabled) so flag-off = byte-identical.

created_at = event.emitted_at (kernel UTC-ISO publish instant). The settle
interceptor compares it against config['_now_iso'] (TurnClock, frozen per outer
turn): a prediction from a PRIOR turn settles; this turn's own (a tool-loop
followup) is skipped. abandon_open() in on_turn_ready keeps one prediction open.
"""
from __future__ import annotations

from core import friction_store as _fs
from core.turn_pipeline_events import PipelineEvent, TurnReadyEvent

NAME = "intent_predict_bridge"

# REGISTRATION is built lazily (not at import) because PolicyRegistration rejects
# any module_path not under core.pipeline_policies — and while this file is dark
# it lives in core/. On activation (see FRICTION_RECALIBRATE.md): move this module
# into core/pipeline_policies/intent_predict_bridge.py, add NAME to
# pipeline_registry.POLICIES, then build_registration()/register_with resolve.


def build_registration():
    """Construct the PolicyRegistration. Only valid once this module has been
    moved under core/pipeline_policies/ (activation step)."""
    from core.pipeline_registry import PolicyRegistration
    from core.turn_pipeline_events import AuthorityTier
    return PolicyRegistration(
        name=NAME,
        module_path="core.pipeline_policies.intent_predict_bridge",
        subscribes_to=("TurnReadyEvent",),
        depends_on=(),
        authority_tier=AuthorityTier.OBSERVATION,
        kill_switch_env_flag="",  # observation tier — self-gates on the friction flag
    )


def register_with(pipeline) -> None:
    pipeline.register(build_registration(), _handle)


def _handle(event: PipelineEvent, ctx) -> None:
    if not isinstance(event, TurnReadyEvent):
        return
    if not _fs.flag_enabled():
        return
    try:
        from core import friction_organ
        friction_organ.on_turn_ready(
            public_answer=event.public_answer,
            last_user_msg="",
            turn_id=str(event.turn_id or ""),
            turn_n=0,
            now_iso=str(event.emitted_at or ""),
        )
    except Exception:
        # predict is best-effort; never break the pipeline
        pass
