"""Commit 1 of the system-wide invariant taxonomy — FRAMEWORK + live-slice mirror only.

E's landing rule (2026-06-25): "prove the migration rail, not expand the vocabulary."
This commit adds core/invariants/{taxonomy,schema}.py and mirrors ONLY the current live
monothink decider slice as TagSpecs. No new semantic tags, no live decider vocabulary
change, no populated aliases (the rail is built + tested; families land in commits 2-5).

The keystone is the FROZEN SNAPSHOT below: a hand-authored copy of today's live decider
vocabulary. failure_tags.py and the taxonomy projection are each compared *against this
snapshot* — neither is derived from the other, so they cannot "agree by construction."
"""
from __future__ import annotations

import inspect
import re

import pytest

import core.failure_tags as ft
from core.invariants import schema as sch
from core.invariants import taxonomy as tax
from core.invariants.taxonomy import TagSpec

# Frozen snapshot of TODAY's live monothink decider vocabulary (the independent anchor).
# If failure_tags.py OR the taxonomy projection drifts from this set, a test fails.
LEGACY_MONOTHINK_FAILURE_TAGS_V1 = {
    # pruning / load-bearing
    "non_load_bearing_step_kept",
    "restatement_unpruned",
    "audit_became_ritual",
    "over_pruned_load_bearing",
    "audit_preflight",
    # branch / alternatives
    "missing_branch_pressure",
    "premature_convergence",
    "generic_reasoning_not_applied",
    # evidence / grounding
    "assertion_without_argument",
    "foundation_unaudited",
    "premise_unchecked",
    # epistemic limit
    "overresolved_unverifiable",
    # cost / consequence
    "unexamined_tradeoff_cost",
    "decision_rationalized",
    # context fit
    "context_mismatch",
    # grounding cite
    "fabricated_cite",
    "no_ground_laundering",
    # conflict resolution
    "contract_step_suppressed",
    "conflict_unannotated",
    "conflict_self_resolved",
    # conviction
    "capitulation_under_pressure",
}

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


# ── the live surface is untouched, and the projection equals it ───────────────


def test_existing_failure_tags_preserved_exactly():
    assert set(ft.FAILURE_TAGS) == LEGACY_MONOTHINK_FAILURE_TAGS_V1


def test_taxonomy_monothink_projection_equals_legacy_snapshot():
    assert tax.decider_visible_tags("monothink") == LEGACY_MONOTHINK_FAILURE_TAGS_V1


def test_no_new_canonical_tags_in_framework_commit():
    # commit-1 guard: registry canonical names are EXACTLY today's live slice; no family
    # tags merged yet (they land in commits 2-5).
    assert tax.all_tag_names() == LEGACY_MONOTHINK_FAILURE_TAGS_V1


def test_journal_adjudicated_tags_validate_against_legacy_slice():
    accepted = {n for n in tax.all_tag_names() if tax.is_decider_tag(n)}
    assert accepted == LEGACY_MONOTHINK_FAILURE_TAGS_V1


def test_failure_tags_not_generated_from_broad_taxonomy_yet():
    # failure_tags stays hand-authored + authoritative for live behavior. Guard against a
    # future refactor silently deriving it from the (broader) taxonomy: it must not import
    # the taxonomy. (Inspect import lines only, so prose mentioning "invariants" is fine.)
    src = inspect.getsource(ft)
    import_lines = " ".join(
        ln for ln in src.splitlines() if ln.strip().startswith(("import ", "from "))
    )
    assert "taxonomy" not in import_lines
    assert "invariants" not in import_lines


def test_gloss_not_forked_for_shared_slice():
    # the taxonomy definition and the FAILURE_TAGS gloss for the same tag must agree.
    for name in LEGACY_MONOTHINK_FAILURE_TAGS_V1:
        assert tax.REGISTRY[name].definition == ft.FAILURE_TAGS[name], name


# ── visibility is layer-local: only monothink-editable tags reach the decider ──


def test_monothink_decider_visible_requires_monothink_owner_and_editable_by_monothink():
    for name, spec in tax.REGISTRY.items():
        if spec.monothink_decider_visible:
            assert spec.owner_layer == "monothink", name
            assert "monothink_pruning_audit" in spec.editable_by, name


def test_no_non_monothink_tag_is_decider_visible():
    # generalized exclusion — future-proof vs enumerating bearing/memory/governance/...
    for name, spec in tax.REGISTRY.items():
        if spec.owner_layer != "monothink":
            assert not spec.monothink_decider_visible, name


def test_field_is_monothink_scoped_not_a_generic_decider_flag():
    # E's rename rationale: the field gates the ONE live MonoThink pruning decider, not
    # "any decider". A generic `decider_visible` would collide with future Bearings/ACU/tool
    # deciders. Guard the name so a rename back to the generic form fails a test.
    assert hasattr(TagSpec, "__dataclass_fields__")
    assert "monothink_decider_visible" in TagSpec.__dataclass_fields__
    assert "decider_visible" not in TagSpec.__dataclass_fields__


# ── construction-time invariants (a malformed/over-scoped TagSpec is unbuildable) ─


def test_tag_requires_owner_layer_kind_editable_by_visibility():
    with pytest.raises((TypeError, ValueError)):
        TagSpec(name="x_missing_fields")  # required fields absent
    with pytest.raises(ValueError):
        TagSpec(  # name not lower_snake_case
            name="Bad Name", kind="failure", owner_layer="monothink",
            editable_by=frozenset({"monothink_pruning_audit"}),
            monothink_decider_visible=False,
            definition="a description.", family="pruning",
        )
    with pytest.raises(ValueError):
        TagSpec(  # monothink_decider_visible without monothink scope is unconstructable
            name="leaky_tag", kind="failure", owner_layer="bearings",
            editable_by=frozenset({"bearing_updater"}), monothink_decider_visible=True,
            definition="a description.", family="bearing",
        )
    with pytest.raises(ValueError):
        TagSpec(  # gloss must be descriptive-only (no prescriptive/evaluative words)
            name="prescriptive_tag", kind="failure", owner_layer="monothink",
            editable_by=frozenset({"monothink_pruning_audit"}),
            monothink_decider_visible=True,
            definition="you must fix the step.", family="pruning",
        )


# ── closed enum at scale ──────────────────────────────────────────────────────


def test_taxonomy_is_closed_enum_no_ad_hoc_strings():
    names = list(tax.REGISTRY)
    assert len(names) == len(set(names)), "duplicate canonical name"
    for n in names:
        assert _NAME_RE.match(n), n
    # family partition: every tag in exactly one family; union == all names
    by_family = tax.tags_by_family()
    flat = [n for tags in by_family.values() for n in tags]
    assert sorted(flat) == sorted(names)
    assert len(flat) == len(set(flat)), "a tag appears in two families"


def test_unknown_tag_rejected_by_registry():
    assert tax.is_registered("not_a_real_tag") is False
    assert tax.resolve("not_a_real_tag") is None
    assert tax.is_decider_tag("not_a_real_tag") is False


# ── the migration rail (aliases are ingestion/read-only, never decider-facing) ─


def test_aliases_not_exported_to_decider():
    decider = tax.decider_visible_tags("monothink")
    for alias, canonical in tax.ALIASES.items():
        assert alias not in decider, alias
        assert alias not in ft.FAILURE_TAGS, alias
        assert alias not in tax.REGISTRY, alias  # an alias is not itself canonical
        assert canonical in tax.REGISTRY, canonical  # its target is canonical


def test_resolve_maps_alias_and_canonical():
    # mechanism proof on a LOCAL fixture (production ALIASES is empty in commit 1; this
    # proves the rail that family commits 2-5 will use to migrate legacy names).
    spec = TagSpec(
        name="frame_stale_carried", kind="failure", owner_layer="frames",
        editable_by=frozenset({"frame_selector"}), monothink_decider_visible=False,
        definition="a prior frame was carried into a turn it no longer fit.",
        family="frame",
    )
    reg = {spec.name: spec}
    aliases = {"stale_frame_used": "frame_stale_carried"}
    assert tax.resolve_in("stale_frame_used", reg, aliases) == "frame_stale_carried"
    assert tax.resolve_in("frame_stale_carried", reg, aliases) == "frame_stale_carried"
    assert tax.resolve_in("unknown_xyz", reg, aliases) is None


# ── family is explicit, but a known prefix must match it (lint, not authority) ─


def _frame_spec(name="frame_x", family="frame", legacy=False):
    return TagSpec(
        name=name, kind="failure", owner_layer="frames",
        editable_by=frozenset({"frame_selector"}), monothink_decider_visible=False,
        definition="a frame-layer invariant break.", family=family, legacy=legacy,
    )


def test_known_prefix_must_match_explicit_family():
    # matching prefix/family builds...
    assert _frame_spec(name="frame_stale_carried", family="frame").family == "frame"
    # ...mismatched prefix/family is unbuildable...
    with pytest.raises(ValueError):
        _frame_spec(name="frame_stale_carried", family="branch")
    # ...unless explicitly marked legacy (old/live names that predate the convention).
    assert _frame_spec(name="frame_stale_carried", family="branch", legacy=True).legacy


def test_prefixless_name_never_trips_family_guard():
    # a name with no known prefix carries any family freely (the live slice relies on this).
    spec = TagSpec(
        name="premise_unchecked_clone", kind="failure", owner_layer="monothink",
        editable_by=frozenset({"monothink_pruning_audit"}),
        monothink_decider_visible=True, definition="a clone for the guard test.",
        family="evidence",
    )
    assert spec.family == "evidence"


def test_live_slice_is_marked_legacy():
    # the mirrored live 20 predate the prefix convention and are exempt from the guard
    # (e.g. contract_step_suppressed sits in the conflict family, not the contract_* one,
    # which lands in commit 4).
    for name in LEGACY_MONOTHINK_FAILURE_TAGS_V1:
        assert tax.REGISTRY[name].legacy is True, name


# ── alias invariants (enforced before ALIASES is ever populated) ───────────────


def test_validate_aliases_rejects_alias_equal_to_canonical():
    reg = {"frame_stale_carried": _frame_spec(name="frame_stale_carried")}
    with pytest.raises(ValueError):  # an alias cannot shadow a canonical name
        tax.validate_aliases(reg, {"frame_stale_carried": "frame_stale_carried"})


def test_validate_aliases_rejects_dangling_target():
    reg = {"frame_stale_carried": _frame_spec(name="frame_stale_carried")}
    with pytest.raises(ValueError):  # an alias must target an existing canonical
        tax.validate_aliases(reg, {"stale_frame_used": "no_such_canonical"})


def test_validate_aliases_accepts_clean_alias():
    reg = {"frame_stale_carried": _frame_spec(name="frame_stale_carried")}
    tax.validate_aliases(reg, {"stale_frame_used": "frame_stale_carried"})  # no raise


def test_alias_resolves_to_exactly_one_canonical():
    reg = {"frame_stale_carried": _frame_spec(name="frame_stale_carried")}
    aliases = {"stale_frame_used": "frame_stale_carried"}
    resolved = tax.resolve_in("stale_frame_used", reg, aliases)
    assert resolved == "frame_stale_carried"
    assert isinstance(resolved, str)  # one canonical, never a set/list


def test_production_aliases_validate_and_are_empty_in_framework_commit():
    tax.validate_aliases(tax.REGISTRY, tax.ALIASES)  # real maps satisfy the invariants
    assert tax.ALIASES == {}  # commit 1: rail proven, vocabulary not yet expanded


# ── companion enums are dimensions, not failure tags ──────────────────────────


def test_companion_enums_not_in_failure_tags():
    for dim, values in sch.COMPANION_ENUMS.items():
        assert not (set(values) & set(ft.FAILURE_TAGS)), dim
        assert not (set(values) & tax.all_tag_names()), dim  # nor taxonomy failure names


def test_companion_enums_are_closed_and_nonempty():
    for dim, values in sch.COMPANION_ENUMS.items():
        assert values, dim
        for v in values:
            assert _NAME_RE.match(v), f"{dim}:{v}"
