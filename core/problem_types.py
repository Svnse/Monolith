"""Closed problem-type vocabulary for BRANCH frame-selection (the coordinate system).

Mirrors core/failure_tags.py in spirit: a small CLOSED enum is the entire
corruption surface of frame-selection. A classifier may only emit a type from
this list; an unrecognized label normalizes away (earning no cell) rather than
silently minting a new coordinate. Free-text the model wants to add accumulates
as `other:<free>` evidence for a future ratified enum revision — it never
becomes a cell on its own.

Glosses are DESCRIPTIVE ONLY — they describe what the type IS, never what anyone
should do. Banned gloss words (enforced by test): should, must, fix, instead,
wrong, bad. Seeded 2026-06-12 from the golden-probe classes
(docs/superpowers/specs/2026-06-12-branch-golden-probes-enum-seed-draft.md §6);
`net_rate_extrapolation` is the logged pending split (folded into
aggregate_ratio_composition until staked runs earn it its own cell).

Pure module: no I/O, no deps.
"""
from __future__ import annotations

PROBLEM_TYPES: dict[str, str] = {
    "central_tendency_estimation":
        "The asked quantity is a mean, expected value, or typical value of a distribution or dataset.",
    "order_statistic_estimation":
        "The asked quantity is an extreme or rank-position value (minimum, maximum, k-th) across multiple random draws or parallel components.",
    "threshold_probability_estimation":
        "The asked quantity is the probability that an outcome meets a stated bound or deadline.",
    "conditional_probability_inversion":
        "Conditional rates are given in one direction together with base rates, and the asked quantity is the conditional probability in the reverse direction.",
    "aggregate_ratio_composition":
        "The asked quantity is an overall rate or ratio formed from totals of numerator and denominator accumulated across heterogeneous segments.",
    "worst_case_bound":
        "The asked quantity holds under every admissible arrangement or under the most adverse outcome (guarantees, minimax values).",
    "aggregate_cost_minimization":
        "The asked quantity is the value of a decision variable that minimizes a summed cost over all instances.",
    "constrained_schedule_construction":
        "The asked quantity is a completion time governed by indivisible tasks, limited parallel capacity, and/or precedence structure.",
    "event_driven_accumulation":
        "The asked quantity is a threshold crossing or extremum of a stock that evolves under continuous flows plus discrete scheduled events.",
    "stratified_group_comparison":
        "A comparison between groups conditioned on the specific subpopulation named in the question.",
    "pooled_group_comparison":
        "A comparison between groups over their entire mixed populations.",
    "eliminative_deduction":
        "Identification of the unique candidate consistent with every given observation or constraint.",
    "change_correlation_attribution":
        "Attribution of an observed effect to the most recently changed or most temporally proximate element.",
    "deterministic_computation":
        "A single fully determined result obtained from a closed specification by direct calculation or procedure tracing.",
}

# Display groups (partition the enum); for prompt readability + future UI.
TYPE_GROUPS: dict[str, tuple[str, ...]] = {
    "DISTRIBUTIONAL": (
        "central_tendency_estimation", "order_statistic_estimation",
        "threshold_probability_estimation", "conditional_probability_inversion",
    ),
    "AGGREGATION": (
        "aggregate_ratio_composition", "aggregate_cost_minimization",
    ),
    "GUARANTEE/STRUCTURE": (
        "worst_case_bound", "constrained_schedule_construction", "event_driven_accumulation",
    ),
    "COMPARISON": (
        "stratified_group_comparison", "pooled_group_comparison",
    ),
    "DEDUCTION": (
        "eliminative_deduction", "change_correlation_attribution",
    ),
    "BASELINE": (
        "deterministic_computation",
    ),
}

# Operational steer per type — the GENERAL method for solving a problem of this
# type, plus the common wrong-method it warns off. Distinct from the gloss (the
# descriptive coordinate): the gloss says what the type IS, the approach says
# how to reason under it. The e2e value test showed a bare gloss is too weak a
# frame to override a salient in-task wrong-attractor; the approach is the
# stronger steer. Kept GENERAL (method-level, not probe-specific) so it helps
# any task of the type, not just the probe set.
APPROACHES: dict[str, str] = {
    "central_tendency_estimation":
        "Compute the probability- or frequency-weighted average (mean / expected value) over all outcomes; do not substitute the single most-likely outcome.",
    "order_statistic_estimation":
        "Work with the distribution of the extreme (max / min / k-th) across the draws; the expected extreme is not the average of one draw.",
    "threshold_probability_estimation":
        "Compute the probability mass on the stated side of the bound; do not report a central or expected value.",
    "conditional_probability_inversion":
        "Apply Bayes: weight each forward rate by its base rate and renormalize; the reverse conditional is not any single stated forward rate.",
    "aggregate_ratio_composition":
        "Form the ratio of summed numerator to summed denominator across the segments; do not average the per-segment ratios.",
    "worst_case_bound":
        "Reason over the most adversarial arrangement and find the count/value that holds no matter how things fall (pigeonhole / minimax); do not compute the likely or average case.",
    "aggregate_cost_minimization":
        "Choose the value that minimizes the summed cost over all instances (for sums of absolute distances, the median); not the extreme, the midpoint, or the mean.",
    "constrained_schedule_construction":
        "Construct and validate an actual schedule honoring indivisibility and precedence; the answer is the makespan or critical path, which can exceed total-work divided by capacity.",
    "event_driven_accumulation":
        "Step the stock through each discrete event over time; the threshold is first crossed at a peak or trough between events, not by extrapolating one net rate.",
    "stratified_group_comparison":
        "Compare only within the subpopulation the question fixes; do not pool across strata.",
    "pooled_group_comparison":
        "Compare the combined totals across all subpopulations; do not stop at a single stratum.",
    "eliminative_deduction":
        "List what each observation rules in or out; a candidate on any passing or observed-good path is exonerated; the answer is the unique one consistent with ALL observations — how recently something changed is irrelevant.",
    "change_correlation_attribution":
        "Identify the element whose change most proximately precedes the observed effect.",
    "deterministic_computation":
        "Carry out the single determined calculation, or trace the specified procedure step by step.",
}

_OTHER_PREFIX = "other:"


def get_approach(type_id: str) -> str:
    return APPROACHES.get(type_id, "")


def is_valid_type(type_id: str) -> bool:
    return type_id in PROBLEM_TYPES


def normalize_type(raw: str | None) -> str | None:
    """Map a raw classifier emission to a known type_id, an ``other:<free>``
    marker, or None.

    - exact known id -> the id
    - a known id appearing as a token in the text -> that id (single match only)
    - anything else, non-empty -> ``other:<slug>`` (earns no cell; accumulates)
    - empty/None -> None
    """
    if not raw:
        return None
    s = raw.strip().lower()
    # strip a leading "type:" label if present
    if s.startswith("type:"):
        s = s[5:].strip()
    s = s.strip("`'\" .")
    if s in PROBLEM_TYPES:
        return s
    hits = [t for t in PROBLEM_TYPES if t in s]
    if len(hits) == 1:
        return hits[0]
    if not s:
        return None
    # unknown but non-empty: an other:<free> marker, never a cell
    slug = "_".join(s.split())[:48]
    return f"{_OTHER_PREFIX}{slug}"


def is_other(type_id: str | None) -> bool:
    return bool(type_id) and type_id.startswith(_OTHER_PREFIX)


def compose_type_menu() -> str:
    """The closed list as the classifier sees it: ``type_id: gloss`` per line,
    grouped. The only type language the classifier is given."""
    lines: list[str] = []
    for group, ids in TYPE_GROUPS.items():
        lines.append(f"[{group}]")
        for tid in ids:
            lines.append(f"  {tid}: {PROBLEM_TYPES[tid]}")
    return "\n".join(lines)
