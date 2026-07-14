"""Subordinate-clause premise detector — embedded false-premise refusal trigger.

Authority tier: OBSERVATION. No kill switch (observation tier — pattern from
core/pipeline_policies/verifier_bridge.py:41).

Detects compliance with subordinate-clause premises about runtime properties
(locality, statefulness, memory, backend) in the model's public answer. On a
positive hit, publishes IdentityRetryRequestedEvent — the kernel consumes
this (separate ticket) to dispatch a retry with the original messages +
describe_self() facts injected, the original output discarded.

Doctrinal anchor: cites describe_self.claim_scope.embedded_premise (added in
core/self_description.py during the 2026-05-14 hardening) as the canonical
text the retry context surfaces. The detector only flags; the retry path
enforces the facts-not-verdicts protection (see project memory
project_verifier_retry_facts_not_verdicts.md).

Independence: no engine/* or acu/acatalepsy imports. Pure on the model's
public answer text + event metadata.
"""
from __future__ import annotations

import re
from typing import Any

from core.pipeline_registry import PolicyRegistration
from core.turn_pipeline_events import (
    AuthorityTier,
    IdentityRetryRequestedEvent,
    PipelineEvent,
    TurnReadyEvent,
)


NAME = "subordinate_clause_detector"


# Grammar patterns that signal subordinate-clause framing AROUND a runtime-
# property keyword (locality, statefulness, memory, backend). Each entry is
# (regex, label). The regex matches a compliance frame — i.e., the model
# speaking AS-IF it had the property. Refusal frames are filtered by the
# negating-prefix check, not by these patterns directly.
_COMPLIANCE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Adverbial: "as a local-first system..."
    (
        re.compile(
            r"\bas\s+(?:a|an)\s+local[\w\-]*?\s+"
            r"(?:system|runtime|workstation|app|setup)",
            re.IGNORECASE,
        ),
        "adverbial_local_system",
    ),
    # Adverbial: "as a <noun> with persistent memory..."
    (
        re.compile(
            r"\bas\s+(?:a|an)\s+\w+(?:\s+\w+){0,3}?\s+with\s+persistent\s+memory",
            re.IGNORECASE,
        ),
        "adverbial_persistent_memory",
    ),
    # Participial: "having persistent local storage..."
    (
        re.compile(
            r"\bhaving\s+(?:persistent|local|stateful)\s+\w+"
            r"(?:\s+\w+)?\s+(?:storage|memory|state|process)",
            re.IGNORECASE,
        ),
        "participial_persistence",
    ),
    # Possessive: "my local memory..."
    (
        re.compile(
            r"\b(?:my|our)\s+(?:local|persistent|stateful)\s+"
            r"(?:memory|state|storage|backend|persistence|process)",
            re.IGNORECASE,
        ),
        "possessive_runtime",
    ),
    # Conditional: "since I'm a local..."
    (
        re.compile(
            r"\bsince\s+(?:i[\'’]?m|i\s+am|i\s+operate|i\s+run)\s+"
            r"(?:a\s+)?(?:local|persistent)",
            re.IGNORECASE,
        ),
        "conditional_local",
    ),
    # Conditional: "given that I operate locally..."
    (
        re.compile(
            r"\bgiven\s+(?:that\s+)?i\s+(?:operate|run|live|exist|am)\s+"
            r"(?:local|with\s+persistent)",
            re.IGNORECASE,
        ),
        "conditional_operate_local",
    ),
    # Appositive: "Monolith, being a local system..."
    (
        re.compile(
            r"\bbeing\s+(?:a\s+)?local\s+\w+",
            re.IGNORECASE,
        ),
        "appositive_local",
    ),
    # Relative: "the local persistence that I possess..."
    (
        re.compile(
            r"\bthe\s+(?:local|persistent)\s+\w+"
            r"(?:\s+\w+){0,2}?\s+that\s+i\s+(?:possess|maintain|have)",
            re.IGNORECASE,
        ),
        "relative_runtime",
    ),
)


# Negating frames that, if they appear in the ~120 chars BEFORE a compliance
# match (and within the same sentence — i.e., no intervening .!?\n), defuse
# the detection. The model saying "I'm not a local-first system..." is a
# refusal, not compliance, even though the phrase "a local-first system"
# appears.
_NEGATING_PREFIXES: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:i[\'’]?m\s+not|i\s+am\s+not|am\s+not|are\s+not|"
        r"not\s+actually|not\s+a)\b[^.!?\n]{0,120}$",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:doesn[\'’]?t\s+match|does\s+not\s+match|"
        r"is\s+(?:wrong|incorrect|false)|refus(?:e|ing)|reject(?:ing)?|"
        r"correction|wrong\s+premise)\b[^.!?\n]{0,80}$",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bcontradict[s]?\b[^.!?\n]{0,80}$",
        re.IGNORECASE,
    ),
)


# Source describe_self fact keys to surface on retry, per the mini_spec.
# These are stable keys the detector publishes so the kernel-side retry
# dispatcher can read them out of describe_self() and inject only the
# named scopes. claim_scope.embedded_premise is the doctrinal anchor.
_RETRY_FACT_KEYS: tuple[str, ...] = (
    "identity_material",
    "current_model_execution",
    "claim_scope.embedded_premise",
)


REGISTRATION = PolicyRegistration(
    name=NAME,
    module_path="core.pipeline_policies.subordinate_clause_detector",
    subscribes_to=("TurnReadyEvent",),
    depends_on=("output_sanitizer",),
    authority_tier=AuthorityTier.OBSERVATION,
    kill_switch_env_flag="",  # observation tier — no per-policy switch
)


def register_with(pipeline) -> None:
    pipeline.register(REGISTRATION, _handle)


def detect_subordinate_clause_premise(text: str) -> tuple[bool, tuple[str, ...]]:
    """Detect subordinate-clause premise compliance about runtime properties.

    Returns (detected, matched_labels). Fires when a compliance phrase appears
    without a negating frame in the ~120 chars before the match.
    """
    if not text:
        return False, ()
    matches: list[str] = []
    for pattern, label in _COMPLIANCE_PATTERNS:
        for m in pattern.finditer(text):
            start = max(0, m.start() - 120)
            preceding = text[start:m.start()]
            negated = any(neg.search(preceding) for neg in _NEGATING_PREFIXES)
            if not negated:
                matches.append(label)
                break  # at most one match per pattern
    return (len(matches) > 0, tuple(matches))


def _handle(event: PipelineEvent, ctx) -> None:
    if not isinstance(event, TurnReadyEvent):
        return
    detected, labels = detect_subordinate_clause_premise(event.public_answer)
    if not detected:
        return
    _publish_via_kernel(
        IdentityRetryRequestedEvent(
            fixture_hint="embedded_premise",
            premise_trigger=labels[0] if labels else "",
            source_fact_keys=_RETRY_FACT_KEYS,
            public_answer_chars=len(event.public_answer),
        ),
        ctx,
    )


def _publish_via_kernel(event: PipelineEvent, ctx) -> None:
    """Publish an event back onto the bus as this policy's source."""
    from monokernel.turn_pipeline import get_pipeline
    get_pipeline().publish(
        event,
        ctx,
        source_kind="policy",
        source_name=NAME,
    )
