"""System-wide closed invariant taxonomy.

Tag existence is system-wide; tag VISIBILITY is layer-local. This registry names invariant
breaks across Monolith's layers; :data:`core.failure_tags.FAILURE_TAGS` is the scoped
PROJECTION of it that the live monothink decider (the bound model) is allowed to read.

Contract (do not weaken):
  * A tag is ``monothink_decider_visible=True`` ONLY if ``owner_layer == "monothink"`` AND
    ``"monothink_pruning_audit" in editable_by`` — i.e. the pruning audit has a concrete
    edit move for it. The field is named for the ONE decider it gates (the live MonoThink
    pruning decider), not "any decider": Bearings / ACU / tool deciders get their own
    visibility fields, never this one. Visibility is not granted because a tag "sounds like"
    monothink. This is enforced at construction (:meth:`TagSpec.__post_init__`), so an
    over-scoped spec is unbuildable.
  * Canonical names are the live strings. Prefixes (``op_`` / ``frame_`` / ``branch_``) are
    a ``family`` LINT signal, not authority: ``family`` is explicit, but a known prefix must
    match it (:data:`KNOWN_FAMILY_PREFIXES`) unless the spec is ``legacy=True``. A persisted
    name is never renamed; legacy names migrate via :data:`ALIASES` (ingestion / read only —
    never exported to the decider, never written to new journal rows, never a member of
    ``FAILURE_TAGS``).
  * ``failure_tags.py`` stays hand-authored and authoritative for live behavior; this
    module is an INDEPENDENTLY-authored superset. Equality is enforced by tests against a
    frozen snapshot, NOT by deriving one module from the other.

Commit 1 (framework): registers ONLY the current live monothink slice; :data:`ALIASES` is
empty (the rail is built + tested via :func:`resolve_in`; families land in commits 2-5).
Pure module, no IO, deterministic.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# A descriptive-only gloss states WHAT failure occurred — never the fix ("should"/"must"/
# "fix") nor a value judgment ("instead"/"wrong"/"bad"). Mirrors the rule already enforced
# on FAILURE_TAGS so a promoted tag can never smuggle a prescriptive gloss to the decider.
_BANNED_GLOSS = ("should", "must", " fix", "instead", "wrong", " bad")
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_MONOTHINK = "monothink"
_PRUNE_AUDIT = "monothink_pruning_audit"

# Known family prefixes — a LINT signal, not identity. If a canonical name begins with
# ``<prefix>_`` and <prefix> is a key here, its explicit ``family`` MUST equal the mapped
# value (enforced in __post_init__) unless the spec is ``legacy=True``. Prefixless live
# names (e.g. ``premise_unchecked``) never trip it. Each family commit (2-5) adds its prefix
# here as it lands; a family token need NOT be a prefix (e.g. ``pruning``/``conflict``).
KNOWN_FAMILY_PREFIXES: dict[str, str] = {
    "request": "request",
    "frame": "frame",
    "bearing": "bearing",
    "claim": "claim",
    "evidence": "evidence",
    "op": "op",
    "branch": "branch",
    "contract": "contract",
    "tool": "tool",
    "memory": "memory",
    "authority": "authority",
    "acu": "acu",
    "temporal": "temporal",
    "output": "output",
    "governance": "governance",
    "telemetry": "telemetry",
    "safety": "safety",
}


@dataclass(frozen=True)
class TagSpec:
    """One invariant-break tag. ``kind`` is always ``"failure"`` (dimensions live in
    :mod:`core.invariants.schema`). All scope fields are required — there is no default
    owner_layer / editable_by / monothink_decider_visible, so a spec cannot be built
    underspecified.
    """

    name: str
    kind: str
    owner_layer: str
    editable_by: frozenset[str]
    monothink_decider_visible: bool
    definition: str
    family: str
    canonical: bool = True
    legacy: bool = False
    aliases: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not _NAME_RE.match(self.name or ""):
            raise ValueError(f"tag name not lower_snake_case: {self.name!r}")
        if self.kind != "failure":
            raise ValueError(
                f"{self.name}: only kind='failure' tags belong in the taxonomy; "
                f"dimensions (intent/state_change/verdict/source) live in schema.py"
            )
        if not self.owner_layer:
            raise ValueError(f"{self.name}: owner_layer required")
        if not self.editable_by:
            raise ValueError(f"{self.name}: editable_by required")
        if not (self.definition or "").strip():
            raise ValueError(f"{self.name}: definition required")
        if not (self.family or "").strip():
            raise ValueError(f"{self.name}: family required")
        low = self.definition.lower()
        for word in _BANNED_GLOSS:
            if word in low:
                raise ValueError(
                    f"{self.name}: definition leaks non-descriptive {word.strip()!r}"
                )
        if self.monothink_decider_visible and not (
            self.owner_layer == _MONOTHINK and _PRUNE_AUDIT in self.editable_by
        ):
            raise ValueError(
                f"{self.name}: monothink_decider_visible requires owner_layer=="
                f"'{_MONOTHINK}' and '{_PRUNE_AUDIT}' in editable_by (the pruning audit "
                f"must be able to edit it)"
            )
        # Prefix is a lint signal, not authority: a known prefix must agree with the
        # explicit family. legacy=True exempts old/live names that predate the convention.
        if not self.legacy:
            prefix = self.name.split("_", 1)[0]
            expected = KNOWN_FAMILY_PREFIXES.get(prefix)
            if expected is not None and self.family != expected:
                raise ValueError(
                    f"{self.name}: name prefix '{prefix}_' implies family "
                    f"'{expected}' but family is '{self.family}' (set family='{expected}' "
                    f"or mark legacy=True)"
                )


def _mt(name: str, family: str, definition: str) -> TagSpec:
    """A monothink-owned, decider-visible failure tag. Glosses are hand-copied from
    ``FAILURE_TAGS`` (the gloss-non-fork test asserts they still match — a typo here fails
    a test rather than silently forking the decider's description of the tag)."""
    return TagSpec(
        name=name,
        kind="failure",
        owner_layer=_MONOTHINK,
        editable_by=frozenset({_PRUNE_AUDIT}),
        monothink_decider_visible=True,
        definition=definition,
        family=family,
        canonical=True,
        # the live 20 predate the prefix convention (e.g. contract_step_suppressed lives in
        # the conflict-resolution audit family); exempt them from the prefix↔family guard.
        legacy=True,
    )


# ── the live monothink slice, mirrored (commit 1 registers ONLY these) ────────
#
# family tokens are organizational (the §Audit families of monothink.md), NOT identity.
_SPECS: tuple[TagSpec, ...] = (
    # pruning / load-bearing — the Audit's core job
    _mt("non_load_bearing_step_kept", "pruning",
        "a step was kept whose removal would not change the conclusion."),
    _mt("restatement_unpruned", "pruning",
        "a step that only restated or reframed a prior step, adding no new information, "
        "was kept."),
    _mt("audit_became_ritual", "pruning",
        "the audit produced verbose step-by-step enumeration without deleting anything."),
    _mt("over_pruned_load_bearing", "pruning",
        "the audit deleted a step whose removal does change the conclusion."),
    _mt("audit_preflight", "pruning",
        "the turn opened by classifying the trace, or ran the audit as a pre-flight "
        "checklist, rather than reasoning the question and letting the audit apply to the "
        "trace that resulted."),
    # branch / alternatives
    _mt("missing_branch_pressure", "branch",
        "the trace branched (alternatives or competing premises existed) but they were not "
        "enumerated or pruned."),
    _mt("premature_convergence", "branch",
        "the trace settled on a conclusion before competing premises were tested."),
    _mt("generic_reasoning_not_applied", "branch",
        "the trace listed factors, procedures, or intentions without a step that applied "
        "analysis to the specific case at hand."),
    # evidence / grounding
    _mt("assertion_without_argument", "evidence",
        "a conclusion was stated with no inference connecting it to anything."),
    _mt("foundation_unaudited", "evidence",
        "a premise the rest of the answer rests on was not itself examined."),
    _mt("premise_unchecked", "evidence",
        "a premise was used without being compared against the evidence present in the "
        "turn."),
    # epistemic limit
    _mt("overresolved_unverifiable", "epistemic",
        "a single definite answer was stated where the available information supports more "
        "than one, or none."),
    # cost / consequence
    _mt("unexamined_tradeoff_cost", "cost",
        "an action or mitigation was proposed without addressing its cost or downside."),
    _mt("decision_rationalized", "cost",
        "support was assembled for a predetermined conclusion, rather than derived."),
    # context fit
    _mt("context_mismatch", "context",
        "the reasoning addressed a system, architecture, or context different from the "
        "actual one."),
    # grounding cite
    _mt("fabricated_cite", "grounding_cite",
        "a grounding cite named a recall handle or tool result that was not shown this "
        "turn, or invented a ground that resolves to nothing."),
    _mt("no_ground_laundering", "grounding_cite",
        "a conclusion resting on an unchecked premise was relabeled as no-ground, rather "
        "than flagged as an unchecked premise."),
    # conflict resolution
    _mt("contract_step_suppressed", "conflict",
        "a step required by another layer (identity refusal, tool-truth, user-specified "
        "shape, effort contract) was pruned."),
    _mt("conflict_unannotated", "conflict",
        "a step retained because another layer required it was kept silently, with no "
        "annotation recording the load-bearing judgment that was overridden."),
    _mt("conflict_self_resolved", "conflict",
        "a conflict between the pruning judgment and a structural contract was resolved "
        "unilaterally, rather than surfaced as tension for the contract author."),
    # conviction / independence
    _mt("capitulation_under_pressure", "conviction",
        "a sound or normative position was retracted under social or authority pressure "
        "rather than on its merits."),
)

# ── build the closed registry + the (empty, for now) migration map ────────────

REGISTRY: dict[str, TagSpec] = {}
# alias -> canonical. Ingestion / read only: an alias never enters REGISTRY, FAILURE_TAGS,
# the decider export, or a new journal row. Populated in family commits 2-5 as legacy names
# (e.g. "stale_frame_used" -> "frame_stale_carried") are folded in.
ALIASES: dict[str, str] = {}


def validate_aliases(registry: dict[str, TagSpec], aliases: dict[str, str]) -> None:
    """Enforce the alias invariants (pure over the passed maps, so family commits can test
    a bad fixture before populating the real :data:`ALIASES`):

      * an alias cannot equal a canonical tag name (no shadowing);
      * an alias must target a canonical that exists (no dangling migration);
      * an alias points to exactly ONE canonical (the dict shape guarantees this — a second
        mapping for the same key is a Python-level overwrite, surfaced by this check reading
        the final value).

    Visibility invariants ("an alias is never decider-exported / never written to a new
    journal row / resolves only on ingestion/read/migration") are properties of the
    *consumers*: :func:`resolve_in` is the only read path, it returns a canonical, and an
    alias is never placed in :data:`REGISTRY` or exported. The taxonomy tests assert the
    structural half here; the export/journal half is asserted where those paths live.
    """
    for alias, canonical in aliases.items():
        if alias in registry:
            raise ValueError(f"alias collides with a canonical name: {alias!r}")
        if canonical not in registry:
            raise ValueError(f"alias {alias!r} targets unknown canonical {canonical!r}")


def _build() -> None:
    for spec in _SPECS:
        if spec.name in REGISTRY:
            raise ValueError(f"duplicate canonical tag name: {spec.name}")
        REGISTRY[spec.name] = spec
    validate_aliases(REGISTRY, ALIASES)


_build()


# ── projection + query helpers ────────────────────────────────────────────────


def decider_visible_tags(owner_layer: str = _MONOTHINK) -> frozenset[str]:
    """The scoped projection: names a layer's decider is allowed to read. For monothink
    this equals :data:`core.failure_tags.FAILURE_TAGS` (enforced by the snapshot test)."""
    return frozenset(
        name
        for name, spec in REGISTRY.items()
        if spec.monothink_decider_visible and spec.owner_layer == owner_layer
    )


def is_decider_tag(name: str) -> bool:
    """True iff *name* is a tag the monothink decider may read. An alias never qualifies."""
    spec = REGISTRY.get(name)
    return (
        spec is not None
        and spec.monothink_decider_visible
        and spec.owner_layer == _MONOTHINK
    )


def is_registered(name: str) -> bool:
    """True iff *name* is a canonical tag (aliases are not canonical)."""
    return name in REGISTRY


def all_tag_names() -> frozenset[str]:
    """Every canonical tag name in the registry."""
    return frozenset(REGISTRY)


def tags_by_family() -> dict[str, list[str]]:
    """Canonical names grouped by ``family`` (each name in exactly one family)."""
    out: dict[str, list[str]] = {}
    for name, spec in REGISTRY.items():
        out.setdefault(spec.family, []).append(name)
    return out


def resolve_in(name: str, registry: dict[str, TagSpec], aliases: dict[str, str]) -> str | None:
    """Map *name* (canonical or legacy alias) to its canonical form; ``None`` if unknown.
    Pure over the passed registry/aliases — the migration rail family commits exercise."""
    if name in registry:
        return name
    return aliases.get(name)


def resolve(name: str) -> str | None:
    """Module-level :func:`resolve_in` over :data:`REGISTRY` + :data:`ALIASES`."""
    return resolve_in(name, REGISTRY, ALIASES)
