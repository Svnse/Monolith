"""Versioned canonical_log `kind` enum.

Extracted from legacy Monolith's `core/auditor.py:_KNOWN_INTENTS` to make
the intent vocabulary a first-class versioned surface. The original
legacy auditor mixed structural-veto logic with intent enumeration;
this module separates the enum so v1 producer code can validate kinds
without inheriting the legacy auditor's veto pipeline.

DESIGN PRINCIPLE — declarative, not mutable
============================================
This module is purely declarative. There is NO runtime `add_kind()`
function. To add a new canonical_log kind:

  1. Edit this file: add the kind string to ``KNOWN_KINDS`` literal
  2. Bump ``KIND_VERSION`` by 1
  3. Commit the change

Audit trail is git diff. Runtime can ONLY read or assert. This is the
GPT-suggested guard against silent intent drift: a versioned enum that
arbitrary app code can mutate isn't versioned — it's just a set.

Use ``assert_valid_kind(kind)`` at write sites to enforce the contract.
"""
from __future__ import annotations


KIND_VERSION: int = 12


# Ported verbatim from legacy auditor.py _KNOWN_INTENTS (legacy Monolith
# in the legacy Monolith auditor). Comments preserved
# from the legacy source to keep the provenance trail explicit.
KNOWN_KINDS: frozenset[str] = frozenset({
    # acus
    "entry_insert", "entry_reinforce", "entry_contradict",
    "entry_reinforce_rejected", "entry_cluster_set", "entry_merge",
    "entry_soft_delete", "entry_touch", "entry_promote", "entry_demote",
    "entry_crystallize",

    # canonical_log (general kinds — the v1 producer loop uses these)
    "user_msg", "assistant_msg", "tool_msg", "message",
    "mono_verify_alignment", "mono_verify_contradiction",

    # path-recorder
    "path_traversal", "path_acceptance",

    # OFAC v0.2 lifecycle (forward-compat — not yet wired in Restore)
    "ofac_route", "ofac_env", "ofac_fit",
    "ofac_plan", "ofac_replan", "ofac_result",
    "ofac_acceptance",

    # L1 Comparison Pass (Acatalepsy spec § L1 COMPARISON PASS)
    "l1_match_reinforce", "l1_partial_edge_induced", "l1_novel_survive",

    # KIND_VERSION 5: Truth branch — the Verifier records a grounded verdict
    # (confirmed/contradicted/unverifiable/contested) on a world-fact/causal ACU.
    "truth_verdict",

    # Identity-seed repair (legacy v2/v3 migrations)
    "identity_seed_repaired", "identity_seed_repair_complete",
    "identity_seeded",

    # Imagination store (Acatalepsy spec § Imagination vs Belief)
    "imagination_insert",
    "imagination_promote_user",
    "imagination_promote_tool",
    "imagination_promote_session_bridge",
    "imagination_retract",

    # Veracity Engine (Acatalepsy spec § VERACITY ENGINE)
    "veracity_update_below_floor",

    # Promotion gate (legacy Week 4 #18)
    "verdict_recorded",

    # relations
    "relation_upsert", "relation_insert", "relation_update",

    # clusters
    "cluster_created",

    # consolidation
    "consolidation_run_complete",

    # decay cycle
    "decay_cycle_summary",
    "relation_soft_delete",

    # Cognitive scaffold fire-rate (legacy D3 from ULTIMATE_PLAN)
    "scaffold_fire",

    # Phase 3 reflector (legacy)
    "reflector_cycle_started",
    "reflector_cycle_complete",
    "reflector_cadence_changed",
    "reflector_cap_fired",
    "reflector_circuit_breaker_open",
    "equivalence_candidate_queued",
    "stale_claim_aged",

    # Memory governance proposals/actions
    "governance_action_proposed",
    "governance_action_applied",
    "governance_action_rejected",
    "governance_ambiguity_marked",
    "governance_ambiguity_cleared",

    # Subject backfill (legacy #38)
    "subject_backfilled",

    # Phase 4 extractor (forward-compat slots — Restore v1 fills these)
    "prefetch_dispatched",
    "extractor_called",
    "extractor_failed",
    "extractor_quota_exceeded",

    # Phase 1 frame_compiler refusal (legacy)
    "frame_compiler_refusal",
    "frame_compiler_integration_rollback",

    # Input Inspector / turn_trace
    "stage_trace",
    "turn_trace_start",
    "turn_trace_end",

    # ── Acatalepsy v1 producer loop additions ───────────────────────
    # These are NEW for Restore v1 (not in legacy). The auditor + decision
    # surface emits these. Adding here is the migration boundary — bump
    # KIND_VERSION when this block grows.

    # Chat dispatch — every user message + assistant turn writes one
    "user_message",       # canonical chat-side user input
    "assistant_message",  # canonical chat-side assistant output

    # Auditor lifecycle
    "auditor_run_started",
    "auditor_run_complete",
    "auditor_run_failed",
    "auditor_atomicity_reject",     # candidate rejected by deterministic gate
    # KIND_VERSION 3: pre-atomicity extraction-quality filter rejects
    # conversational fragments (questions, "want me to..." etc.) before
    # they become canonical_log noise. See core/acatalepsy/extraction_quality.py.
    "auditor_extraction_filter_reject",
    "auditor_cursor_advance",       # last_processed_event_id moved
    # KIND_VERSION 2: live LLM-call visibility so the MonoBase panel can
    # show "calling LLM..." vs "parsing response..." vs "saving candidates"
    # during the long synchronous LLM wait instead of a silent black box.
    "auditor_llm_call_started",     # prompt sent to LLM
    "auditor_llm_call_returned",    # LLM response received (or errored)

    # Candidate lifecycle (decisions land in acu_decisions, kind here is
    # the corresponding canonical_log event)
    "candidate_emitted",
    "candidate_accepted",
    "candidate_rejected",
    "candidate_edited",
    "candidate_deferred",

    # Session bookkeeping
    "session_open",
    "session_close",

    # KIND_VERSION 4: Observer V0 turn-boundary event. Emitted once per
    # assistant turn so the Auditor can read Observer output as auditable
    # material and eventually extract claims about advisory usefulness.
    "observer_fired",

    # KIND_VERSION 6: M2 Identity Evolution (docs/specs/M2_IDENTITY_EVOLUTION_SPEC.md).
    # The emergence detector and identity_review skill emit these; the apply +
    # milestone-snapshot kinds are forward-compat slots (V1 apply path / ledger).
    "identity_emergence_detected",
    "identity_amendment_proposed",
    "identity_amendment_applied",
    "identity_milestone_snapshot",

    # KIND_VERSION 7: M3 Curiosity (docs/specs/M3_CURIOSITY_SPEC.md). The curiosity
    # detector surfaces fresh, identity-aligned "pulls" (the not-yet-stable
    # disposition of the identity signal).
    "curiosity_pull_detected",

    # KIND_VERSION 8: M1 Planner (docs/specs/M1_PLANNER_SPEC.md). Propose-only goal
    # decomposition into a tracked step DAG; steps execute via the existing
    # gated tool loop (the planner never auto-drives execution).
    "plan_proposed",
    "plan_step_marked",
    "plan_status_changed",

    # KIND_VERSION 9: M3.1 curiosity kill-actuator. The SAFE half of closing the
    # curiosity loop — Monolith may RETIRE a pull it judges as noise (reversible,
    # audited here); promotion-into-identity stays human-gated.
    "curiosity_pull_killed",

    # KIND_VERSION 10: reasoning branch tree (session_tree chokepoint)
    "branch_forked",
    "branch_pruned",
    "branch_switched",

    # KIND_VERSION 11: MonoExplore expedition tick telemetry. ExpeditionRunner.
    # _log_tick emits one per tick (tools_run/grounded/ingested); it was raising
    # UnknownKind (swallowed) every tick until registered here.
    "expedition_tick",

    # KIND_VERSION 12: MonoNote explicit note-read provenance. Passive note
    # indexing/search/canvas display is not ACU evidence; this event is emitted
    # only when a user-authorized read/send path exposes note text to the model.
    "mononote_note_read",
})


class UnknownKind(ValueError):
    """Raised when an unknown canonical_log kind is asserted.

    The presence of this exception type is itself a forward-compat
    signal: if you see it raised, either the kind is a typo or the
    declarative enum needs to be extended (see module docstring).
    """


def is_valid_kind(kind: str) -> bool:
    """Read-only validity check. Returns True if kind is in the
    versioned enum, False otherwise. Use at read sites where unknown
    kinds should be skipped rather than reject."""
    return kind in KNOWN_KINDS


def assert_valid_kind(kind: str) -> None:
    """Raises UnknownKind if kind is not in the versioned enum.

    Use at write sites (canonical_log.append, candidate.insert with
    kind metadata, etc.) to enforce the versioned contract.
    """
    if kind not in KNOWN_KINDS:
        raise UnknownKind(
            f"unknown canonical_log kind: {kind!r} "
            f"(KIND_VERSION={KIND_VERSION}, "
            f"{len(KNOWN_KINDS)} known kinds). "
            f"To add: edit core/acatalepsy/canonical_log_kinds.py + bump KIND_VERSION."
        )
