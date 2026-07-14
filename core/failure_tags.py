"""MonoThink failure-tag vocabulary — the closed, controlled language a rating uses to
steer scaffold evolution.

This module is the *entire corruption surface* of the autonomous monothink training loop
(SP1). A rating no longer carries free text into the evolution decider; it carries a list
of `failure_tags` drawn from this closed enum. The decider (the model bound in the UI) is
fed each tag's canonical, **descriptive-only** gloss — never rater-authored prose. Holistic
feedback rides a separate `surface_note` that this module never touches.

Design rulings (see docs/superpowers/specs/2026-06-03-monothink-training-loop-sp1-rating-contract-design.md):
  * D2  — the decider receives `tag_id` + a canonical gloss looked up here. The rater never
          writes the gloss. Glosses are DESCRIPTIVE ONLY: they state *what failure occurred*,
          not what edit to make (prescriptive) and not why it is bad (evaluative).
  * Enum — 21 tags, advisor scope-tested against monothink.md's Scope boundary so every tag
          names a failure monothink's pruning audit can actually edit (no layer leakage).
          The original 14 were ratified in SP1; 6 more (audit_preflight,
          generic_reasoning_not_applied, fabricated_cite, no_ground_laundering,
          conflict_unannotated, conflict_self_resolved) close invariants in §Audit/Threshold,
          §Grounding cite, and §Conflict Resolution that had no covering tag. The 21st,
          capitulation_under_pressure, was added 2026-06-25 after a live audit found the
          premise training overfit into folding sound/normative positions under authority
          pressure — the audit can edit hold-vs-fold discipline (name the category error,
          hold), so it is in-scope.
  * Unknown tags are dropped on normalization; a rating that normalizes to zero tags does not
          drive evolution (the triviality gate — a bare "hi" cannot mutate the scaffold).

This module is pure: no IO, no deps, deterministic.
"""
from __future__ import annotations

# ── the closed vocabulary ────────────────────────────────────────────────────
#
# tag_id -> canonical descriptive-only gloss. Insertion order is the canonical order
# (grouped by failure family). Edit this dict to evolve the vocabulary; it is the one
# reviewable artifact that caps what monothink can learn.
FAILURE_TAGS: dict[str, str] = {
    # load-bearing / pruning — the Audit's core job (monothink.md §Audit)
    "non_load_bearing_step_kept":
        "a step was kept whose removal would not change the conclusion.",
    "restatement_unpruned":
        "a step that only restated or reframed a prior step, adding no new information, "
        "was kept.",
    "audit_became_ritual":
        "the audit produced verbose step-by-step enumeration without deleting anything.",
    "over_pruned_load_bearing":
        "the audit deleted a step whose removal does change the conclusion.",
    "audit_preflight":
        "the turn opened by classifying the trace, or ran the audit as a pre-flight "
        "checklist, rather than reasoning the question and letting the audit apply to the "
        "trace that resulted.",
    # branch / alternatives
    "missing_branch_pressure":
        "the trace branched (alternatives or competing premises existed) but they were not "
        "enumerated or pruned.",
    "premature_convergence":
        "the trace settled on a conclusion before competing premises were tested.",
    "generic_reasoning_not_applied":
        "the trace listed factors, procedures, or intentions without a step that applied "
        "analysis to the specific case at hand.",
    # evidence / grounding
    "assertion_without_argument":
        "a conclusion was stated with no inference connecting it to anything.",
    "foundation_unaudited":
        "a premise the rest of the answer rests on was not itself examined.",
    "premise_unchecked":
        "a premise was used without being compared against the evidence present in the turn.",
    # epistemic limit
    "overresolved_unverifiable":
        "a single definite answer was stated where the available information supports more "
        "than one, or none.",
    # cost / consequence
    "unexamined_tradeoff_cost":
        "an action or mitigation was proposed without addressing its cost or downside.",
    "decision_rationalized":
        "support was assembled for a predetermined conclusion, rather than derived.",
    # context fit
    "context_mismatch":
        "the reasoning addressed a system, architecture, or context different from the "
        "actual one.",
    # grounding cite (monothink.md §Grounding cite)
    "fabricated_cite":
        "a grounding cite named a recall handle or tool result that was not shown this "
        "turn, or invented a ground that resolves to nothing.",
    "no_ground_laundering":
        "a conclusion resting on an unchecked premise was relabeled as no-ground, rather "
        "than flagged as an unchecked premise.",
    # conflict resolution (monothink.md §Conflict Resolution)
    "contract_step_suppressed":
        "a step required by another layer (identity refusal, tool-truth, user-specified "
        "shape, effort contract) was pruned.",
    "conflict_unannotated":
        "a step retained because another layer required it was kept silently, with no "
        "annotation recording the load-bearing judgment that was overridden.",
    "conflict_self_resolved":
        "a conflict between the pruning judgment and a structural contract was resolved "
        "unilaterally, rather than surfaced as tension for the contract author.",
    # conviction / independence (a sound position abandoned under pressure)
    "capitulation_under_pressure":
        "a sound or normative position was retracted under social or authority pressure "
        "rather than on its merits.",
}

# Display/selection grouping for the rater (UI /rating, the SP4 skill). Every tag appears
# in exactly one group; the union is FAILURE_TAGS.
TAG_GROUPS: dict[str, list[str]] = {
    "load-bearing / pruning": [
        "non_load_bearing_step_kept",
        "restatement_unpruned",
        "audit_became_ritual",
        "over_pruned_load_bearing",
        "audit_preflight",
    ],
    "branch / alternatives": [
        "missing_branch_pressure",
        "premature_convergence",
        "generic_reasoning_not_applied",
    ],
    "evidence / grounding": [
        "assertion_without_argument",
        "foundation_unaudited",
        "premise_unchecked",
    ],
    "epistemic limit": [
        "overresolved_unverifiable",
    ],
    "cost / consequence": [
        "unexamined_tradeoff_cost",
        "decision_rationalized",
    ],
    "context fit": [
        "context_mismatch",
    ],
    "grounding cite": [
        "fabricated_cite",
        "no_ground_laundering",
    ],
    "conflict resolution": [
        "contract_step_suppressed",
        "conflict_unannotated",
        "conflict_self_resolved",
    ],
    "conviction": [
        "capitulation_under_pressure",
    ],
}


# ── helpers ──────────────────────────────────────────────────────────────────


def is_valid_tag(tag: str) -> bool:
    """True iff *tag* is a member of the closed vocabulary."""
    return isinstance(tag, str) and tag in FAILURE_TAGS


def normalize_tags(tags: list[str]) -> list[str]:
    """Validate + dedupe a rater-supplied tag list, preserving first-seen order.

    Unknown tags are dropped (never coerced to a valid one — D3 validation rule). The
    result is the authoritative tag set; an empty result means the rating carries no
    directional signal and must not drive evolution (the triviality gate).
    """
    seen: set[str] = set()
    out: list[str] = []
    for tag in tags or []:
        if is_valid_tag(tag) and tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out


def compose_monothink_signal(tags: list[str]) -> str:
    """Build the monothink-VISIBLE directional text for the evolution decider.

    One line per normalized tag: ``tag_id: <canonical descriptive gloss>``. This is the
    only failure language the decider reads. Returns "" when no valid tag remains (the
    decider is then never invoked).
    """
    valid = normalize_tags(tags)
    if not valid:
        return ""
    return "\n".join(f"{tag}: {FAILURE_TAGS[tag]}" for tag in valid)


def compose_reasoning_why(tags: list[str]) -> str:
    """Build the human-readable echo stored in the outcome `reason` column.

    Auto-composed from the tags (never a rater input). Read by telemetry/stats/UI; NOT the
    channel the decider reads (that is :func:`compose_monothink_signal`). Returns "" when no
    valid tag remains.
    """
    valid = normalize_tags(tags)
    if not valid:
        return ""
    return "Reasoning-failure(s) flagged — " + " ".join(
        f"[{tag}] {FAILURE_TAGS[tag]}" for tag in valid
    )
