"""Tests for ``core.monothink`` — model-self-tended scaffold.

Covers the pure helpers (NO_CHANGE detection, diff sizing, journal scan)
and the full ``maybe_evolve_after_rating`` decision tree (flag-off, wrong
tier, idempotence, NO_CHANGE, size cap, diff cap, accept, LLM failure).
A final integration test verifies the ``record_outcome`` hook wires through
when ``kind == "rating"`` on a monothink-tier turn.
"""
from __future__ import annotations

import json

import pytest

from core import monothink as mt
from core import turn_trace as tt


@pytest.fixture(autouse=True)
def _force_monothink_sync(monkeypatch):
    """Production runs the LLM call in a daemon thread (so chat.py's /rating
    handler doesn't freeze on a sync LLM call — observed on the first live
    rating after the join-key fix). Tests assert on the journal entry that
    `maybe_evolve_after_rating` returns; that entry only reflects the final
    decision under the sync path. Force sync here so test assertions stay
    deterministic without relying on thread joins."""
    monkeypatch.setenv("MONOLITH_MONOTHINK_ASYNC", "0")


# ── pure-function helpers ──────────────────────────────────────────────


@pytest.mark.parametrize("variant", [
    "NO_CHANGE",
    "no_change",
    "No_Change",
    "**NO_CHANGE**",
    "`NO_CHANGE`",
    '"NO_CHANGE"',
    "'NO_CHANGE'",
    "NO_CHANGE.",
    "NO_CHANGE:",
    "NO_CHANGE!",
    "NO_CHANGE\nbecause the rating was high",
    "  NO_CHANGE  ",
])
def test_is_no_change_accepts_variants(variant: str) -> None:
    assert mt._is_no_change(variant) is True


@pytest.mark.parametrize("not_match", [
    "",
    "NO CHANGE",          # space instead of underscore
    "no change",
    "change this",
    "NO_CHANGEABLE",       # superstring
    "something\nNO_CHANGE",  # token not on first line
    "Here is the scaffold:\nNO_CHANGE",
])
def test_is_no_change_rejects_non_token(not_match: str) -> None:
    assert mt._is_no_change(not_match) is False


def test_diff_chars_identical_is_zero() -> None:
    assert mt._diff_chars("hello", "hello") == 0


def test_diff_chars_single_char_append() -> None:
    assert mt._diff_chars("abc", "abcd") == 1


def test_diff_chars_full_rewrite_is_max_len() -> None:
    # Disjoint alphabets → no shared subsequence → a full rewrite measures the
    # longer side (whole-file rewrites stay large, well over the cap). Distinct
    # character sets so difflib finds no incidental matches; English sentences
    # share enough subsequence chars that "fully different" isn't truly disjoint.
    a = "x" * 50
    b = "y" * 60
    assert mt._diff_chars(a, b) == len(b)


def test_diff_chars_middle_insertion() -> None:
    a = "before:::after"
    b = "before:NEW:after"
    # Common prefix "before:", common suffix ":after" → middle "::" vs "NEW".
    # NEW (3) wins over :: (2).
    assert mt._diff_chars(a, b) == 3


def test_diff_chars_bracketed_edit_counts_genuine_change_not_span() -> None:
    """Regression (journal entry 27): an edit that changes BOTH ends of a region
    — a header word AND an appended item — must measure the genuine changed
    text, not the whole span between the first and last difference. The old
    single prefix/suffix strip measured a 955-char span for a ~180-char real
    change, mis-routing a line-scale edit into the section path."""
    old = "Two guards:\n- a\n- b\n"
    new = "Three guards:\n- a\n- b\n- c\n"  # 'Two'->'Three' (top) + append '- c' (bottom)
    # Genuine change ≈ 'Two'->'Three' (+5/-3) plus the '- c\n' item (+4); the
    # unchanged middle must NOT be counted. The bracketed span is ~24.
    assert mt._diff_chars(old, new) <= 15


def test_journal_has_turn_returns_false_when_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(mt, "_JOURNAL_PATH", tmp_path / "absent.jsonl")
    assert mt._journal_has_turn("anything") is False


def test_journal_has_turn_finds_existing(tmp_path, monkeypatch) -> None:
    journal = tmp_path / "j.jsonl"
    journal.write_text(
        json.dumps({"turn_id": "T-1", "applied": True}) + "\n"
        + json.dumps({"turn_id": "T-2", "applied": False}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(mt, "_JOURNAL_PATH", journal)
    assert mt._journal_has_turn("T-1") is True
    assert mt._journal_has_turn("T-2") is True
    assert mt._journal_has_turn("T-3") is False


def test_journal_has_turn_ignores_corrupt_lines(tmp_path, monkeypatch) -> None:
    journal = tmp_path / "j.jsonl"
    journal.write_text(
        "this is not json\n"
        + json.dumps({"turn_id": "T-7"}) + "\n"
        + "{broken json\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(mt, "_JOURNAL_PATH", journal)
    assert mt._journal_has_turn("T-7") is True
    assert mt._journal_has_turn("T-other") is False


# ── policy matrix + prompt re-anchor (Phase 3a) ───────────────────────


def test_compose_prompt_includes_before_after_mechanics() -> None:
    """The prompt must teach the BEFORE/AFTER patch mechanism: the patcher
    locates BEFORE as a literal substring of the current scaffold and replaces
    it with AFTER."""
    prompt = mt._compose_prompt("scaffold body", 70, ["restatement_unpruned"])
    assert "BEFORE/AFTER mechanics" in prompt
    assert "byte-for-byte" in prompt
    assert "literal substring" in prompt
    # PROPOSED_SCOPE × actual_scope measurement remains the diagnostic axis.
    assert "scope" in prompt


def test_compose_prompt_biases_toward_small_line_edits() -> None:
    """E's contract (2026-06-25): the prompt strongly biases the decider
    toward ≤_DIFF_CAP-char line edits (which auto-apply), honest section
    otherwise — and it no longer threatens a scope-mismatch REJECT, because
    the code decides from ACTUAL scope and logs the mismatch instead."""
    prompt = mt._compose_prompt("scaffold body", 70, ["restatement_unpruned"])
    low = prompt.lower()
    assert str(mt._DIFF_CAP) in prompt        # the 300-char line target is named
    assert "smallest" in low                  # the strong bias is stated
    # the stale reject-on-mismatch threat must be gone
    assert "rejects if your stated scope" not in low


def test_decide_policy_line_always_applies() -> None:
    # actual=line ⟺ char_diff ≤ _DIFF_CAP by the classifier's definition, so it applies.
    assert mt._decide_policy("line", False) == mt.DECISION_APPLY
    assert mt._decide_policy("line", True) == mt.DECISION_APPLY


def test_decide_policy_section_applies_only_when_concern_repeated() -> None:
    assert mt._decide_policy("section", True) == mt.DECISION_APPLY
    assert mt._decide_policy("section", False) == mt.DECISION_PROPOSE


def test_decide_policy_structural_always_proposes() -> None:
    """Structural never auto-applies; it goes to PROPOSE for manual review."""
    assert mt._decide_policy("structural", True) == mt.DECISION_PROPOSE
    assert mt._decide_policy("structural", False) == mt.DECISION_PROPOSE


def test_decide_policy_none_is_no_change() -> None:
    assert mt._decide_policy("none", False) == mt.DECISION_NO_CHANGE
    assert mt._decide_policy("none", True) == mt.DECISION_NO_CHANGE


def test_decide_policy_decides_from_actual_scope_not_proposed_label() -> None:
    """The contract (E, 2026-06-25): the decision is a pure function of
    (actual_scope, concern_repeated). A wrong PROPOSED_SCOPE label can never
    discard a valid edit — it is logged as a mismatch at the call site, not
    rejected here. The gates are unchanged (section still needs recurrence),
    so this is NOT more permissive — only less wasteful: a line-labeled but
    actually-section edit is governed by the section gate, not thrown away."""
    # mislabeled line→section, first time: PROPOSES (the catch is kept, not discarded)
    assert mt._decide_policy("section", False) == mt.DECISION_PROPOSE
    # mislabeled line→section, recurrent concern: the catch lands
    assert mt._decide_policy("section", True) == mt.DECISION_APPLY
    # _decide_policy takes exactly two args now — proposed_scope is not one of them.
    import inspect
    assert list(inspect.signature(mt._decide_policy).parameters) == [
        "actual_scope", "concern_repeated",
    ]


# ── canary fast-check (Phase 3d) ──────────────────────────────────────


def test_fast_check_passes_well_formed_scaffold() -> None:
    assert mt._fast_check_scaffold(_MULTI_SECTION_SCAFFOLD) is None


def test_fast_check_rejects_oversized() -> None:
    huge = "## A\n" + ("x" * (mt._SIZE_CAP + 100))
    result = mt._fast_check_scaffold(huge)
    assert result is not None
    assert result.startswith("canary_fast_fail:size_cap_exceeded")


def test_fast_check_rejects_half_fenced_code_block() -> None:
    half_fenced = "## A\n```\nunclosed code block\n"
    result = mt._fast_check_scaffold(half_fenced)
    assert result == "canary_fast_fail:half_fenced_code_block"


def test_fast_check_rejects_no_sections() -> None:
    """A scaffold without any `##` headers is structurally broken — the
    model collapsed everything into prose."""
    no_headers = "# Just an H1\nNo section headers anywhere.\nJust prose."
    result = mt._fast_check_scaffold(no_headers)
    assert result == "canary_fast_fail:no_sections"


def test_fast_check_rejects_empty_section_body() -> None:
    """A `##` header with no body under it before the next `##` is broken."""
    empty_body = "## First\n\n## Second\nbody for second\n"
    result = mt._fast_check_scaffold(empty_body)
    assert result is not None
    assert result.startswith("canary_fast_fail:empty_section:")


def test_fast_check_allows_tbd_placeholder_bodies() -> None:
    """`[tbd]` is a legit non-empty body — the current scaffold's
    `## Constraints` uses this pattern. Fast check must not reject it."""
    with_tbd = (
        "## First\n\n[tbd]\n\n## Second\n\nbody\n"
    )
    assert mt._fast_check_scaffold(with_tbd) is None


def test_worker_routes_fast_check_failure_to_propose(
    isolated_paths, monkeypatch,
) -> None:
    """When the policy says APPLY but the fast-check fails, downgrade to
    propose: journal canary_fast_fail, don't write the scaffold.

    Subtle: the half-fenced block must be APPENDED after the last section
    (Conflict Resolution) so the unclosed fence doesn't swallow any
    protected section headers and trigger the protected-section check
    first. The fast-check runs AFTER protected; this test isolates it."""
    _seed_multi_section_scaffold(isolated_paths)
    # BEFORE/AFTER variant: replace the last line and append an unclosed fence
    # so the resulting scaffold has odd fence count. APPLY → fast_check fires.
    response = _apply_response(
        _MULTI_SECTION_SCAFFOLD,
        "Contract wins; conflict annotated.",
        "Contract wins; conflict annotated.\n\n```\nUNCLOSED",
        tag="model_left_unclosed_fence", failure="bad fence",
        target_section="Conflict Resolution", scope="line",
    )
    monkeypatch.setattr(mt, "_call_llm", lambda prompt: response)
    entry = mt.maybe_evolve_after_rating("T-fastfail", 80, ["restatement_unpruned"])
    assert entry["applied"] is False
    assert entry["reject_reason"] == "canary_fast_fail:half_fenced_code_block"
    # Scaffold unchanged
    assert isolated_paths["scaffold"].read_text(encoding="utf-8") == _MULTI_SECTION_SCAFFOLD


# ── repeated-concern detector (Phase 3c) ──────────────────────────────


def _seed_journal_entries(journal_path, entries: list[dict]) -> None:
    """Write a sequence of journal entries to disk (for _concern_repeated tests)."""
    with journal_path.open("a", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def test_concern_repeated_returns_false_when_no_journal(isolated_paths, monkeypatch) -> None:
    """Empty journal → no repetition possible. Defensive base case."""
    monkeypatch.setattr(mt, "_JOURNAL_PATH", isolated_paths["journal"])
    assert isolated_paths["journal"].exists() is False
    assert mt._concern_repeated("any_tag", "T-1") is False


def test_concern_repeated_returns_false_with_no_matching_tag(isolated_paths, monkeypatch) -> None:
    _seed_journal_entries(isolated_paths["journal"], [
        {"turn_id": "T-prev", "primary_failure_tag": "different_tag"},
    ])
    assert mt._concern_repeated("target_tag", "T-current") is False


def test_concern_repeated_returns_true_when_prior_distinct_turn_has_same_tag(
    isolated_paths,
) -> None:
    _seed_journal_entries(isolated_paths["journal"], [
        {"turn_id": "T-old", "primary_failure_tag": "shared_tag"},
    ])
    assert mt._concern_repeated("shared_tag", "T-new") is True


def test_concern_repeated_dedups_same_turn_id(isolated_paths) -> None:
    """Three journal entries with same tag + same turn_id (debug-retry on
    the same prompt) should NOT count as repetition — the distinct-turn_id
    constraint is what prevents path dependency from rating the same
    response multiple times."""
    _seed_journal_entries(isolated_paths["journal"], [
        {"turn_id": "T-retry", "primary_failure_tag": "shared_tag"},
        {"turn_id": "T-retry", "primary_failure_tag": "shared_tag"},
        {"turn_id": "T-retry", "primary_failure_tag": "shared_tag"},
    ])
    assert mt._concern_repeated("shared_tag", "T-current") is True
    # But: from the perspective of T-retry itself, it's not "repeated"
    # since its own turn_id is excluded.
    assert mt._concern_repeated("shared_tag", "T-retry") is False


def test_concern_repeated_skips_current_turn_id(isolated_paths) -> None:
    """The current turn's own prior entries don't count toward repetition."""
    _seed_journal_entries(isolated_paths["journal"], [
        {"turn_id": "T-current", "primary_failure_tag": "loneliness"},
    ])
    assert mt._concern_repeated("loneliness", "T-current") is False


def test_concern_repeated_respects_lookback_window(isolated_paths) -> None:
    """A match that's older than lookback_n distinct turn_ids ago should
    be ignored. Lookback prevents stale matches from triggering applies."""
    # Build 6 distinct prior turns, with the match only on the oldest.
    entries = [{"turn_id": "T-old-match", "primary_failure_tag": "T"}]
    entries += [{"turn_id": f"T-recent-{i}", "primary_failure_tag": "X"} for i in range(6)]
    _seed_journal_entries(isolated_paths["journal"], entries)
    # With lookback_n=5, T-old-match is outside the window.
    assert mt._concern_repeated("T", "T-current", lookback_n=5) is False
    # With lookback_n=100, T-old-match is within the window.
    assert mt._concern_repeated("T", "T-current", lookback_n=100) is True


def test_concern_repeated_ignores_bootstrap_and_rollback_entries(isolated_paths) -> None:
    """Bootstrap/rollback rows don't carry primary_failure_tag; they must be
    skipped in the lookback scan so they don't consume window slots."""
    _seed_journal_entries(isolated_paths["journal"], [
        {"turn_id": "bootstrap-X", "kind": "bootstrap"},
        {"turn_id": "rollback-Y", "kind": "rollback"},
        {"turn_id": "T-real", "primary_failure_tag": "shared"},
    ])
    assert mt._concern_repeated("shared", "T-current") is True


def test_worker_applies_section_when_concern_repeated(
    isolated_paths, monkeypatch,
) -> None:
    """End-to-end: model claims section + classifier agrees + journal has a
    prior matching tag → APPLY. Section-scope auto-apply requires the
    repeated-concern gate to fire."""
    _seed_multi_section_scaffold(isolated_paths)
    # Pre-seed a prior entry with the same tag we're about to use.
    _seed_journal_entries(isolated_paths["journal"], [
        {"turn_id": "T-prior", "primary_failure_tag": "audit_depth", "applied": False},
    ])
    big_after = "Linear traces discharge with " + ("x" * (mt._DIFF_CAP + 100))
    response = _apply_response(
        _MULTI_SECTION_SCAFFOLD,
        "Linear traces discharge with one line.",
        big_after,
        tag="audit_depth", failure="audit needs more depth",
        target_section="Audit", scope="section",
        patch_mode="replace_section",
    )
    monkeypatch.setattr(mt, "_call_llm", lambda prompt: response)
    entry = mt.maybe_evolve_after_rating("T-second", 80, ["restatement_unpruned"])
    assert entry["applied"] is True
    # Scaffold actually written
    written = isolated_paths["scaffold"].read_text(encoding="utf-8")
    assert "x" * 100 in written


# ── rollback substrate (Phase 3e) ─────────────────────────────────────


def test_bootstrap_entry_written_on_first_evolution(isolated_paths, monkeypatch) -> None:
    """Phase 3e: the rollback floor must exist before any apply. On the first
    evolution attempt (any path — accept, reject, or NO_CHANGE), a single
    bootstrap entry captures the current on-disk scaffold so subsequent
    applies can chain parent_scaffold_version back to it."""
    monkeypatch.setattr(mt, "_call_llm", lambda prompt: _schema_response(patch="NO_CHANGE"))

    # Trigger any evolution attempt.
    mt.maybe_evolve_after_rating("T-bootstrap-1", 80, ["restatement_unpruned"])

    entries = _read_journal_entries(isolated_paths["journal"])
    bootstrap_entries = [e for e in entries if e.get("kind") == "bootstrap"]
    assert len(bootstrap_entries) == 1
    boot = bootstrap_entries[0]
    assert boot["applied"] is True
    assert boot["applied_scaffold_full"] == "# MonoThink — seed\n"
    assert boot["parent_scaffold_version"] is None
    assert str(boot["turn_id"]).startswith("bootstrap-")


def test_bootstrap_entry_not_rewritten_when_already_present(
    isolated_paths, monkeypatch,
) -> None:
    """Idempotent: existing bootstrap entries are never duplicated."""
    monkeypatch.setattr(mt, "_call_llm", lambda prompt: _schema_response(patch="NO_CHANGE"))

    # Trigger two evolution attempts on different turn_ids.
    mt.maybe_evolve_after_rating("T-boot-A", 80, ["restatement_unpruned"])
    mt.maybe_evolve_after_rating("T-boot-B", 80, ["restatement_unpruned"])

    entries = _read_journal_entries(isolated_paths["journal"])
    bootstrap_entries = [e for e in entries if e.get("kind") == "bootstrap"]
    assert len(bootstrap_entries) == 1


def test_applied_entry_carries_applied_scaffold_full(
    isolated_paths, monkeypatch,
) -> None:
    """Phase 3e: every applied=True entry stores the new scaffold text in
    applied_scaffold_full so rollback can restore from journal without
    reading the on-disk file."""
    _seed_multi_section_scaffold(isolated_paths)
    response = _apply_response(
        _MULTI_SECTION_SCAFFOLD,
        "Linear traces discharge with one line.",
        "Linear traces discharge with one CLEAN line.",
        tag="line_refine", failure="word choice",
        target_section="Audit", scope="line",
    )
    monkeypatch.setattr(mt, "_call_llm", lambda prompt: response)
    mt.maybe_evolve_after_rating("T-apply-substrate", 80, ["restatement_unpruned"])

    entries = _read_real_journal_entries(isolated_paths["journal"])
    applied = [e for e in entries if e.get("applied") is True]
    assert len(applied) == 1
    a = applied[0]
    assert "applied_scaffold_full" in a
    assert "CLEAN line" in a["applied_scaffold_full"]


def test_applied_entry_links_parent_scaffold_version_to_bootstrap(
    isolated_paths, monkeypatch,
) -> None:
    """First real apply's parent_scaffold_version points to the bootstrap
    entry — the rollback chain starts there."""
    _seed_multi_section_scaffold(isolated_paths)
    response = _apply_response(
        _MULTI_SECTION_SCAFFOLD,
        "Linear traces discharge with one line.",
        "Linear traces discharge with one CLEAN line.",
        tag="first_apply", failure="first edit",
        target_section="Audit", scope="line",
    )
    monkeypatch.setattr(mt, "_call_llm", lambda prompt: response)
    mt.maybe_evolve_after_rating("T-first", 80, ["restatement_unpruned"])

    entries = _read_journal_entries(isolated_paths["journal"])
    bootstrap = [e for e in entries if e.get("kind") == "bootstrap"][0]
    applied = [e for e in entries if e.get("applied") is True and e.get("kind") != "bootstrap"][0]
    assert applied["parent_scaffold_version"] == bootstrap["turn_id"]


def test_rollback_restores_parent_applied_scaffold(
    isolated_paths, monkeypatch,
) -> None:
    """rollback_last_apply restores the on-disk scaffold to the
    parent_scaffold_version's stored text and journals a rollback entry."""
    _seed_multi_section_scaffold(isolated_paths)
    original = isolated_paths["scaffold"].read_text(encoding="utf-8")

    response = _apply_response(
        original,
        "Linear traces discharge with one line.",
        "Linear traces discharge with one CHANGED line.",
        tag="apply_then_revert", failure="bad edit",
        target_section="Audit", scope="line",
    )
    monkeypatch.setattr(mt, "_call_llm", lambda prompt: response)
    mt.maybe_evolve_after_rating("T-revertme", 70, ["restatement_unpruned"])

    # Confirm apply happened
    after_apply = isolated_paths["scaffold"].read_text(encoding="utf-8")
    assert "CHANGED line" in after_apply

    # Now roll back
    rb = mt.rollback_last_apply(reason="canary_failed")
    assert rb is not None
    assert rb["kind"] == "rollback"

    after_revert = isolated_paths["scaffold"].read_text(encoding="utf-8")
    assert "CHANGED line" not in after_revert
    assert after_revert == original.rstrip() or after_revert == original
    # Verify the journal carries a rollback entry referencing the source
    entries = _read_journal_entries(isolated_paths["journal"])
    rollback_entries = [e for e in entries if e.get("kind") == "rollback"]
    assert len(rollback_entries) == 1
    assert rollback_entries[0]["rolled_back_from"] == "T-revertme"


def test_rollback_returns_none_when_nothing_to_revert(
    isolated_paths, monkeypatch,
) -> None:
    """Bootstrap exists but no real apply yet → rollback is a no-op."""
    monkeypatch.setattr(mt, "_call_llm", lambda prompt: _schema_response(patch="NO_CHANGE"))
    mt.maybe_evolve_after_rating("T-noapply", 80, ["restatement_unpruned"])
    assert mt.rollback_last_apply() is None


# ── BEFORE/AFTER patcher (Phase 3.5) ──────────────────────────────────


def test_apply_patch_replaces_unique_before_block() -> None:
    """Happy path: BEFORE appears exactly once → replaced with AFTER."""
    current = "alpha\nbeta\ngamma\n"
    new, err = mt._apply_patch(current, "beta", "BETA")
    assert err is None
    assert new == "alpha\nBETA\ngamma\n"


def test_apply_patch_returns_error_when_before_empty() -> None:
    """Empty BEFORE is a schema bug — caller should use no_change mode."""
    new, err = mt._apply_patch("any text", "", "after text")
    assert new is None
    assert err == "before_block_empty"


def test_apply_patch_returns_error_when_before_not_found() -> None:
    """Model hallucinated BEFORE — text doesn't exist in current scaffold."""
    current = "alpha\nbeta\ngamma\n"
    new, err = mt._apply_patch(current, "DELTA", "EPSILON")
    assert new is None
    assert err == "before_block_mismatch:not_found"


def test_apply_patch_returns_error_when_before_ambiguous() -> None:
    """BEFORE appears more than once — the patch location isn't uniquely
    identified. Model should extend BEFORE with surrounding context."""
    current = "alpha\nbeta\nalpha\n"
    new, err = mt._apply_patch(current, "alpha", "ALPHA")
    assert new is None
    assert err == "before_block_mismatch:ambiguous"


def test_apply_patch_preserves_byte_exactness_outside_before() -> None:
    """The whole point of BEFORE/AFTER: bytes outside BEFORE stay
    IDENTICAL. No 'tightening' of adjacent prose, no whitespace drift."""
    current = (
        "## Section A\n"
        "\n"
        "Line one.\n"
        "Line two — this stays.\n"
        "Line three.\n"
        "\n"
        "## Section B\n"
        "\n"
        "Other content stays.\n"
    )
    new, err = mt._apply_patch(current, "Line one.", "Line ONE.")
    assert err is None
    # Every byte outside "Line one." → "Line ONE." is preserved.
    expected = current.replace("Line one.", "Line ONE.", 1)
    assert new == expected


def test_apply_patch_supports_deletion_via_empty_after() -> None:
    """AFTER may legitimately be empty — that's a deletion of the BEFORE
    block. The parser allows this (only BEFORE empty is rejected)."""
    current = "alpha\nbeta\ngamma\n"
    new, err = mt._apply_patch(current, "beta\n", "")
    assert err is None
    assert new == "alpha\ngamma\n"


def test_apply_patch_supports_multiline_before_block() -> None:
    """BEFORE can span multiple lines. Match remains byte-exact (whitespace
    and line endings included)."""
    current = "head\nmid1\nmid2\ntail\n"
    new, err = mt._apply_patch(current, "mid1\nmid2", "MID")
    assert err is None
    assert new == "head\nMID\ntail\n"


# ── BEFORE/AFTER end-to-end through the worker ────────────────────────


def test_worker_rejects_before_block_not_found(isolated_paths, monkeypatch) -> None:
    """Phase 3.5 end-to-end: the model hallucinated BEFORE content. Phase 3.7
    grants ONE repair attempt; the stub returns the same bad patch, so the
    worker journals the reject with :repair_failed and does not write."""
    _seed_multi_section_scaffold(isolated_paths)
    response = _schema_response(
        tag="hallucinated_before", failure="model wrote text not in scaffold",
        target="Audit", scope="line",
        patch_mode="replace_lines",
        target_lines="42",
        before="This text does not appear anywhere in the scaffold.",
        after="Replacement that will never be applied.",
    )
    monkeypatch.setattr(mt, "_call_llm", lambda prompt: response)
    entry = mt.maybe_evolve_after_rating("T-halluc", 70, ["restatement_unpruned"])
    assert entry["applied"] is False
    assert entry["reject_reason"] == "before_block_mismatch:not_found:repair_failed"
    # Scaffold unchanged
    assert isolated_paths["scaffold"].read_text(encoding="utf-8") == _MULTI_SECTION_SCAFFOLD


def test_worker_rejects_before_block_ambiguous(isolated_paths, monkeypatch) -> None:
    """Phase 3.5: if the model's BEFORE appears more than once in the
    scaffold, the patch location is ambiguous → reject. Uses a custom
    scaffold with duplicate substrings to trigger the path."""
    # Seed a custom scaffold with a string that appears twice on purpose.
    scaffold = (
        "## Section A\n"
        "\n"
        "duplicated phrase here.\n"
        "\n"
        "## Section B\n"
        "\n"
        "duplicated phrase here.\n"
    )
    isolated_paths["scaffold"].write_text(scaffold, encoding="utf-8")
    response = _schema_response(
        tag="ambiguous_before", failure="model picked non-unique BEFORE",
        target="Section A", scope="line",
        patch_mode="replace_lines",
        target_lines="3",
        before="duplicated phrase here.",  # appears twice
        after="DISAMBIGUATED.",
    )
    monkeypatch.setattr(mt, "_call_llm", lambda prompt: response)
    entry = mt.maybe_evolve_after_rating("T-ambig", 70, ["restatement_unpruned"])
    assert entry["applied"] is False
    assert entry["reject_reason"] == "before_block_mismatch:ambiguous:repair_failed"


def test_worker_no_change_mode_short_circuits(isolated_paths, monkeypatch) -> None:
    """Phase 3.5: PATCH_MODE=no_change short-circuits to the no_change
    branch without invoking _apply_patch. BEFORE/AFTER can be empty."""
    _seed_multi_section_scaffold(isolated_paths)
    response = _schema_response(
        tag="declined", failure="no actionable failure",
        target="NONE", scope="none",
        patch_mode="no_change",
        target_lines="NONE", before="", after="",
    )
    monkeypatch.setattr(mt, "_call_llm", lambda prompt: response)
    entry = mt.maybe_evolve_after_rating("T-nochange", 90, ["restatement_unpruned"])
    assert entry["applied"] is False
    assert entry["reject_reason"] == "no_change_requested"


# ── classifier (Phase 3b) ─────────────────────────────────────────────


def test_classify_diff_identical_returns_none() -> None:
    s = "## A\nbody"
    assert mt._classify_diff(s, s) == "none"
    assert mt._classify_diff(s + "\n", s) == "none"  # trailing whitespace tolerant


def test_classify_diff_header_added_returns_structural() -> None:
    old = "## A\nbody\n"
    new = "## A\nbody\n## B\nmore\n"
    assert mt._classify_diff(old, new) == "structural"


def test_classify_diff_header_removed_returns_structural() -> None:
    old = "## A\nbody\n## B\nmore\n"
    new = "## A\nbody\n"
    assert mt._classify_diff(old, new) == "structural"


def test_classify_diff_header_renamed_returns_structural() -> None:
    old = "## A\nbody\n"
    new = "## A renamed\nbody\n"
    assert mt._classify_diff(old, new) == "structural"


def test_classify_diff_multi_section_returns_section() -> None:
    old = "## A\nold a\n## B\nold b\n"
    new = "## A\nnew a\n## B\nnew b\n"
    # Both section bodies changed → span=2 → section
    assert mt._classify_diff(old, new) == "section"


def test_classify_diff_single_section_small_returns_line() -> None:
    old = "## A\nshort body\n"
    new = "## A\nshort body!\n"  # 1-char change
    assert mt._classify_diff(old, new) == "line"


def test_classify_diff_single_section_large_returns_section() -> None:
    """The empirical case from real ratings: model produces a ~3400-char
    diff within a single section. Classifier must call this "section"
    even though the model self-labeled it "line"."""
    old = "## Audit\n" + ("a" * 100) + "\n"
    new = "## Audit\n" + ("a" * 100) + ("b" * (mt._DIFF_CAP + 200)) + "\n"
    assert mt._classify_diff(old, new) == "section"


def test_classify_diff_small_bracketed_edit_in_large_section_returns_line() -> None:
    """Regression (journal entry 27): a tiny edit that brackets a LARGE
    unchanged body — a header word changed at the top and one item appended at
    the bottom — is a line-scale change. The old span measure counted the whole
    section (>cap) and mislabeled it "section", which then needed concern_repeated
    to apply and never did. Genuine change is ~10 chars → must be "line"."""
    body = "x" * (mt._DIFF_CAP + 50)  # unchanged middle, larger than the cap
    old = "## Audit\nTwo guards.\n" + body + "\nend.\n"
    new = "## Audit\nThree guards.\n" + body + "\nend.\n- c\n"
    assert mt._classify_diff(old, new) == "line"


# ── classifier boundary at _DIFF_CAP (the line ⇔ section threshold) ───
#
# E's contract requires the boundary be exact: ≤300 chars in a single section
# is "line"; 301 is "section". These pin 299/300/301 so a future _DIFF_CAP or
# _diff_chars change can't silently move the line/section cutoff.


def test_classify_diff_299_chars_single_section_is_line() -> None:
    old = "## A\nQQ\n"
    new = "## A\nQQ" + ("z" * 299) + "\n"
    assert mt._diff_chars(old, new) == 299
    assert mt._classify_diff(old, new) == "line"


def test_classify_diff_300_chars_single_section_is_line() -> None:
    old = "## A\nQQ\n"
    new = "## A\nQQ" + ("z" * 300) + "\n"
    assert mt._diff_chars(old, new) == 300  # == _DIFF_CAP, still line
    assert mt._classify_diff(old, new) == "line"


def test_classify_diff_301_chars_single_section_is_section() -> None:
    old = "## A\nQQ\n"
    new = "## A\nQQ" + ("z" * 301) + "\n"
    assert mt._diff_chars(old, new) == 301  # one past the cap → section
    assert mt._classify_diff(old, new) == "section"


# ── worker routing through policy (Phase 3a + 3b integration) ─────────


def test_worker_proposes_when_proposed_line_actual_section_logs_mismatch(
    isolated_paths, monkeypatch,
) -> None:
    """Decide-from-actual (E, 2026-06-25): model claims line but BEFORE→AFTER
    measures section. The wrong label does NOT discard the catch — the code
    measures actual=section and, with no prior concern, PROPOSES it. The
    proposed/actual mismatch is LOGGED, never used as a reject reason."""
    _seed_multi_section_scaffold(isolated_paths)
    response = _apply_response(
        _MULTI_SECTION_SCAFFOLD,
        "Linear traces discharge with one line.",
        "Linear traces discharge with " + ("x" * (mt._DIFF_CAP + 100)),
        tag="empirical_misreport", failure="model claimed line but did section",
        target_section="Audit", scope="line",  # mislabel
    )
    monkeypatch.setattr(mt, "_call_llm", lambda prompt: response)
    entry = mt.maybe_evolve_after_rating("T-mismatch", 75, ["restatement_unpruned"])
    assert entry["applied"] is False
    assert entry["reject_reason"] == "proposed_only:section"  # proposed, not discarded
    assert entry["actual_scope"] == "section"
    assert entry["proposed_scope"] == "line"
    assert entry["scope_mismatch"] is True  # logged for telemetry


def test_worker_applies_mislabeled_section_when_concern_repeated(
    isolated_paths, monkeypatch,
) -> None:
    """The payoff: a section-sized catch mislabeled 'line' APPLIES once the
    concern has recurred (the recurrence gate, unchanged). The label is
    advisory; actual scope + recurrence govern. Mismatch still logged."""
    _seed_multi_section_scaffold(isolated_paths)
    response = _apply_response(
        _MULTI_SECTION_SCAFFOLD,
        "Linear traces discharge with one line.",
        "Linear traces discharge with " + ("x" * (mt._DIFF_CAP + 100)),
        tag="restatement_unpruned", failure="big section edit mislabeled line",
        target_section="Audit", scope="line",  # mislabel
    )
    monkeypatch.setattr(mt, "_call_llm", lambda prompt: response)
    monkeypatch.setattr(mt, "_concern_repeated", lambda tag, tid, lookback_n=5: True)
    entry = mt.maybe_evolve_after_rating("T-applies", 40, ["restatement_unpruned"])
    assert entry["applied"] is True  # the catch lands despite the 'line' label
    assert entry["actual_scope"] == "section"
    assert entry["scope_mismatch"] is True


def test_worker_proposes_section_when_concern_not_repeated(
    isolated_paths, monkeypatch,
) -> None:
    """Section-scope edits without a repeated concern route to PROPOSE
    (not APPLY)."""
    _seed_multi_section_scaffold(isolated_paths)
    response = _apply_response(
        _MULTI_SECTION_SCAFFOLD,
        "Linear traces discharge with one line.",
        "Linear traces discharge with " + ("x" * (mt._DIFF_CAP + 100)),
        tag="novel_concern", failure="first-time concern",
        target_section="Audit", scope="section",
        patch_mode="replace_section",
    )
    monkeypatch.setattr(mt, "_call_llm", lambda prompt: response)
    entry = mt.maybe_evolve_after_rating("T-novel", 70, ["restatement_unpruned"])
    assert entry["applied"] is False
    assert entry["reject_reason"] == "proposed_only:section"
    # Scaffold not written
    assert isolated_paths["scaffold"].read_text(encoding="utf-8") == _MULTI_SECTION_SCAFFOLD


def test_worker_applies_line_within_cap(isolated_paths, monkeypatch) -> None:
    """Phase 3.5: model claims line, BEFORE→AFTER is small enough that
    classifier agrees, no protected sections touched → APPLY. The happy
    path enabled by the BEFORE/AFTER mechanism."""
    _seed_multi_section_scaffold(isolated_paths)
    response = _apply_response(
        _MULTI_SECTION_SCAFFOLD,
        "Linear traces discharge with one line.",
        "Linear traces discharge with one CLEAN line.",
        tag="clean_line", failure="word choice refined",
        target_section="Audit", scope="line",
    )
    monkeypatch.setattr(mt, "_call_llm", lambda prompt: response)
    entry = mt.maybe_evolve_after_rating("T-apply", 80, ["restatement_unpruned"])
    assert entry["applied"] is True
    assert entry["reject_reason"] is None
    written = isolated_paths["scaffold"].read_text(encoding="utf-8")
    assert "CLEAN line" in written


def test_worker_proposes_structural_regardless_of_actual_classifier(
    isolated_paths, monkeypatch,
) -> None:
    """Structural changes always route to PROPOSE — never auto-applied.
    Adding a `## New section` at the end via BEFORE/AFTER: BEFORE is the
    last existing chunk + EOF marker, AFTER prepends the new section."""
    _seed_multi_section_scaffold(isolated_paths)
    # Append a new section by patching the last existing line.
    last_chunk = "Contract wins; conflict annotated."  # ends ## Conflict Resolution
    new_last = last_chunk + "\n\n## New section\n\nNew content."
    response = _apply_response(
        _MULTI_SECTION_SCAFFOLD,
        last_chunk,
        new_last,
        tag="new_section", failure="needs a new section",
        target_section="New section", scope="structural",
        patch_mode="replace_section",
    )
    monkeypatch.setattr(mt, "_call_llm", lambda prompt: response)
    entry = mt.maybe_evolve_after_rating("T-struct", 75, ["restatement_unpruned"])
    assert entry["applied"] is False
    assert entry["reject_reason"] == "proposed_only:structural"


# ── section parser + protected core (Phase 2a/2b) ─────────────────────


_MULTI_SECTION_SCAFFOLD = (
    "# MonoThink — test fixture\n"
    "\n"
    "Preamble paragraph that belongs to no section.\n"
    "\n"
    "## Core invariant\n"
    "\n"
    "Single load-bearing rule.\n"
    "\n"
    "## Audit\n"
    "\n"
    "### Threshold\n"
    "\n"
    "Linear traces discharge with one line.\n"
    "\n"
    "### Format\n"
    "\n"
    "Deleted steps tagged.\n"
    "\n"
    "## Constraints\n"
    "\n"
    "[tbd]\n"
    "\n"
    "## Scope boundary\n"
    "\n"
    "MonoThink yields to identity, tool truth, user shape.\n"
    "\n"
    "## Conflict Resolution\n"
    "\n"
    "Contract wins; conflict annotated.\n"
)


def test_parse_sections_returns_expected_h2_set() -> None:
    sections = mt._parse_sections(_MULTI_SECTION_SCAFFOLD)
    assert list(sections.keys()) == [
        "Core invariant",
        "Audit",
        "Constraints",
        "Scope boundary",
        "Conflict Resolution",
    ]


def test_parse_sections_nests_h3_under_parent_h2() -> None:
    """### Threshold and ### Format belong to ## Audit's body, not as
    separate top-level entries."""
    sections = mt._parse_sections(_MULTI_SECTION_SCAFFOLD)
    audit_body = sections["Audit"]
    assert "### Threshold" in audit_body
    assert "### Format" in audit_body
    # And the next h2 has its own body, not the spillover from Audit.
    assert "tbd" in sections["Constraints"]


def test_parse_sections_excludes_h1_preamble() -> None:
    """The `# MonoThink` h1 line and the preamble paragraph are NOT a section
    — section parser only collects ## boundaries."""
    sections = mt._parse_sections(_MULTI_SECTION_SCAFFOLD)
    assert "MonoThink — test fixture" not in sections
    # No preamble bucket key. The "Preamble paragraph" content is discarded
    # from the section map (it lives before any ## boundary).
    for body in sections.values():
        assert "Preamble paragraph" not in body


def test_parse_sections_ignores_hash_inside_code_fence() -> None:
    """A ## inside a ``` code block is content, not a section break."""
    text = (
        "## Real header\n"
        "body before fence\n"
        "```\n"
        "## not a header — inside a fence\n"
        "## also not\n"
        "```\n"
        "body after fence\n"
        "## Another real header\n"
        "second body\n"
    )
    sections = mt._parse_sections(text)
    assert list(sections.keys()) == ["Real header", "Another real header"]
    real_body = sections["Real header"]
    assert "## not a header" in real_body
    assert "body before fence" in real_body
    assert "body after fence" in real_body


def test_parse_sections_returns_empty_dict_when_no_h2() -> None:
    assert mt._parse_sections("# only h1 here\n") == {}
    assert mt._parse_sections("") == {}
    assert mt._parse_sections("just prose, no headers") == {}


def test_parse_sections_preserves_insertion_order() -> None:
    """Python dicts preserve insertion order (3.7+); document the contract
    so callers can rely on it."""
    text = "## Z first\nz body\n## A second\na body\n## M third\nm body\n"
    keys = list(mt._parse_sections(text).keys())
    assert keys == ["Z first", "A second", "M third"]


# ── _diff_touches_protected ──────────────────────────────────────────


def test_diff_touches_protected_returns_empty_when_unchanged() -> None:
    assert mt._diff_touches_protected(_MULTI_SECTION_SCAFFOLD, _MULTI_SECTION_SCAFFOLD) == []


def test_diff_touches_protected_detects_modified_core_invariant() -> None:
    modified = _MULTI_SECTION_SCAFFOLD.replace(
        "Single load-bearing rule.",
        "Single load-bearing rule — REWRITTEN.",
    )
    touched = mt._diff_touches_protected(_MULTI_SECTION_SCAFFOLD, modified)
    assert touched == [("Core invariant", "modified")]


def test_diff_touches_protected_detects_removed_scope_boundary() -> None:
    """Removing the `## Scope boundary` header entirely is the dangerous
    case — the model "consolidates" a protected section out of existence."""
    without = _MULTI_SECTION_SCAFFOLD.replace(
        "## Scope boundary\n\nMonoThink yields to identity, tool truth, user shape.\n\n",
        "",
    )
    touched = mt._diff_touches_protected(_MULTI_SECTION_SCAFFOLD, without)
    assert touched == [("Scope boundary", "removed")]


def test_diff_touches_protected_ignores_unprotected_section_edits() -> None:
    """Mutating `## Audit` (not in _PROTECTED_SECTIONS) returns empty —
    only protected sections are listed."""
    modified = _MULTI_SECTION_SCAFFOLD.replace(
        "Linear traces discharge with one line.",
        "Linear traces discharge differently now.",
    )
    assert mt._diff_touches_protected(_MULTI_SECTION_SCAFFOLD, modified) == []


def test_diff_touches_protected_lists_multiple_when_both_modified() -> None:
    modified = _MULTI_SECTION_SCAFFOLD.replace(
        "Single load-bearing rule.", "different rule",
    ).replace(
        "MonoThink yields to identity, tool truth, user shape.",
        "different yield list",
    )
    touched = mt._diff_touches_protected(_MULTI_SECTION_SCAFFOLD, modified)
    # Order follows _PROTECTED_SECTIONS declaration: Core invariant, then Scope boundary
    assert touched == [
        ("Core invariant", "modified"),
        ("Scope boundary", "modified"),
    ]


# ── _diff_section_span ────────────────────────────────────────────────


def test_diff_section_span_zero_for_identical() -> None:
    assert mt._diff_section_span(_MULTI_SECTION_SCAFFOLD, _MULTI_SECTION_SCAFFOLD) == 0


def test_diff_section_span_one_for_single_section_edit() -> None:
    modified = _MULTI_SECTION_SCAFFOLD.replace(
        "Linear traces discharge with one line.",
        "Linear traces discharge differently.",
    )
    assert mt._diff_section_span(_MULTI_SECTION_SCAFFOLD, modified) == 1


def test_diff_section_span_counts_multiple_changed_sections() -> None:
    modified = _MULTI_SECTION_SCAFFOLD.replace(
        "Single load-bearing rule.", "different rule",
    ).replace(
        "[tbd]", "now populated",
    )
    assert mt._diff_section_span(_MULTI_SECTION_SCAFFOLD, modified) == 2


def test_diff_section_span_counts_section_removal() -> None:
    without = _MULTI_SECTION_SCAFFOLD.replace(
        "## Conflict Resolution\n\nContract wins; conflict annotated.\n",
        "",
    )
    assert mt._diff_section_span(_MULTI_SECTION_SCAFFOLD, without) == 1


# ── Phase 2c/2d: validator wiring (end-to-end) ────────────────────────


def _seed_multi_section_scaffold(isolated_paths) -> None:
    """Overwrite the isolated-paths fixture's single-line scaffold with the
    multi-section test scaffold so section-aware validators have something
    to bite on."""
    isolated_paths["scaffold"].write_text(
        _MULTI_SECTION_SCAFFOLD, encoding="utf-8",
    )


def test_evolve_rejects_protected_section_modification(isolated_paths, monkeypatch) -> None:
    """Phase 2c: a proposed scaffold that modifies `## Core invariant`'s body
    rejects with `protected_section:Core invariant`. Patcher owns enforcement
    — the model can't slip past by editing its own constraints."""
    _seed_multi_section_scaffold(isolated_paths)
    response = _apply_response(
        _MULTI_SECTION_SCAFFOLD,
        "Single load-bearing rule.",
        "Single load-bearing rule plus extra prose.",
        tag="touched_protected", failure="model edited Core invariant",
        target_section="Core invariant", scope="section",
    )
    monkeypatch.setattr(mt, "_call_llm", lambda prompt: response)
    entry = mt.maybe_evolve_after_rating("T-protect-mod", 70, ["restatement_unpruned"])
    assert entry["applied"] is False
    assert entry["reject_reason"] == "protected_section:Core invariant"
    # Scaffold untouched
    assert isolated_paths["scaffold"].read_text(encoding="utf-8") == _MULTI_SECTION_SCAFFOLD


def test_evolve_rejects_protected_section_removal(isolated_paths, monkeypatch) -> None:
    """Phase 2c: header `## Scope boundary` removed entirely — model
    "consolidated" the section out of existence. Reject with
    `protected_section_removed:<name>`."""
    _seed_multi_section_scaffold(isolated_paths)
    # Phase 3.5: removing an entire section is a structural change. We
    # express it as replace_section where BEFORE is the section header +
    # body and AFTER is empty (deletion).
    section_chunk = "## Scope boundary\n\nMonoThink yields to identity, tool truth, user shape.\n\n"
    response = _apply_response(
        _MULTI_SECTION_SCAFFOLD,
        section_chunk,
        "",
        tag="removed_protected", failure="model removed scope boundary",
        target_section="Scope boundary", scope="structural",
        patch_mode="replace_section",
    )
    monkeypatch.setattr(mt, "_call_llm", lambda prompt: response)
    entry = mt.maybe_evolve_after_rating("T-protect-rm", 70, ["restatement_unpruned"])
    assert entry["applied"] is False
    assert entry["reject_reason"] == "protected_section_removed:Scope boundary"


def test_evolve_protected_check_runs_before_size_cap(isolated_paths, monkeypatch) -> None:
    """Order: protected check fires before size cap. The resulting scaffold
    here BOTH touches Core invariant AND blows past _SIZE_CAP — protected
    fires first."""
    _seed_multi_section_scaffold(isolated_paths)
    response = _apply_response(
        _MULTI_SECTION_SCAFFOLD,
        "Single load-bearing rule.",
        "Single load-bearing rule REWRITTEN " + ("x" * mt._SIZE_CAP),
        tag="big_and_protected", failure="oversized + protected",
        target_section="Core invariant", scope="structural",
        patch_mode="replace_section",
    )
    monkeypatch.setattr(mt, "_call_llm", lambda prompt: response)
    entry = mt.maybe_evolve_after_rating("T-order", 60, ["restatement_unpruned"])
    assert entry["reject_reason"] == "protected_section:Core invariant"


def test_evolve_allows_unprotected_section_modification(isolated_paths, monkeypatch) -> None:
    """Sanity: editing `## Audit` (not protected) proceeds past the
    protected-section check. Other checks may still reject, but not this one."""
    _seed_multi_section_scaffold(isolated_paths)
    response = _apply_response(
        _MULTI_SECTION_SCAFFOLD,
        "Linear traces discharge with one line.",
        "Linear traces discharge differently now.",
        tag="edit_audit", failure="refine audit threshold",
        target_section="Audit", scope="section",
    )
    monkeypatch.setattr(mt, "_call_llm", lambda prompt: response)
    entry = mt.maybe_evolve_after_rating("T-allow-audit", 70, ["restatement_unpruned"])
    assert not str(entry.get("reject_reason") or "").startswith("protected_section")


# A multi-section BEFORE block spans an Audit body line through the
# Constraints section body. _apply_patch will substitute the entire range,
# changing both Audit and Constraints — triggering the multi-section span
# detection downstream.
_MULTI_SECTION_BEFORE = (
    "Linear traces discharge with one line.\n"
    "\n"
    "### Format\n"
    "\n"
    "Deleted steps tagged.\n"
    "\n"
    "## Constraints\n"
    "\n"
    "[tbd]"
)
_MULTI_SECTION_AFTER = (
    "Linear traces discharge differently now.\n"
    "\n"
    "### Format\n"
    "\n"
    "Deleted steps tagged.\n"
    "\n"
    "## Constraints\n"
    "\n"
    "now populated with new content"
)


# (SP1: the Phase 2d empty-reason guard was removed. Ratings now require a
#  non-empty failure_tags list to fire evolution — see maybe_evolve_after_rating's
#  triviality gate — so the empty-signal input the guard protected against can no
#  longer occur. The three empty-reason guard tests were deleted with it. Section
#  scope is now governed solely by the _decide_policy / _concern_repeated matrix;
#  protected-section ordering is still covered by the test below and by
#  test_evolve_protected_check_runs_before_size_cap.)


def test_evolve_empty_reason_guard_runs_after_protected_check(
    isolated_paths, monkeypatch,
) -> None:
    """Order: protected check before empty_reason guard. A patch that BOTH
    touches a protected section AND has empty reason + multi-section span
    should report protected_section, not empty_reason_section_span."""
    _seed_multi_section_scaffold(isolated_paths)
    # BEFORE spans Core invariant (protected) body AND the Audit ###
    # Threshold subsection body — touches multiple sections including a
    # protected one.
    bad_before = (
        "Single load-bearing rule.\n"
        "\n"
        "## Audit\n"
        "\n"
        "### Threshold\n"
        "\n"
        "Linear traces discharge with one line."
    )
    bad_after = (
        "different rule\n"
        "\n"
        "## Audit\n"
        "\n"
        "### Threshold\n"
        "\n"
        "Different threshold prose."
    )
    response = _apply_response(
        _MULTI_SECTION_SCAFFOLD,
        bad_before,
        bad_after,
        tag="protected_and_multi", failure="multi and protected",
        target_section="Core invariant", scope="structural",
        patch_mode="replace_section",
    )
    monkeypatch.setattr(mt, "_call_llm", lambda prompt: response)
    entry = mt.maybe_evolve_after_rating("T-double-bad", 25, ["restatement_unpruned"])
    assert entry["reject_reason"] == "protected_section:Core invariant"


# ── parser (Phase 1b) ─────────────────────────────────────────────────


def test_parse_response_well_formed_extracts_all_fields() -> None:
    raw = _schema_response(
        tag="state_action_discipline",
        failure="model conflated outcome with action",
        target="Audit",
        scope="line",
        concerns=["epistemic phrasing too soft", "missing reversibility tracker"],
        patch_mode="replace_lines",
        target_lines="47-49",
        before="OLD LINE",
        after="NEW LINE",
    )
    parsed, err = mt._parse_evolution_response(raw)
    assert err is None
    assert parsed["primary_failure_tag"] == "state_action_discipline"
    assert parsed["primary_failure"] == "model conflated outcome with action"
    assert parsed["target_section"] == "Audit"
    assert parsed["proposed_scope"] == "line"
    assert parsed["deferred_concerns"] == [
        "epistemic phrasing too soft",
        "missing reversibility tracker",
    ]
    assert parsed["patch_mode"] == "replace_lines"
    assert parsed["target_lines"] == "47-49"
    assert parsed["before"] == "OLD LINE"
    assert parsed["after"] == "NEW LINE"


def test_parse_response_missing_primary_failure_tag() -> None:
    raw = (
        "PRIMARY_FAILURE: missing the tag field\n"
        "TARGET_SECTION: NONE\n"
        "PROPOSED_SCOPE: none\n"
        "DEFERRED_CONCERNS:\n"
        "PATCH:\nNO_CHANGE\n"
    )
    parsed, err = mt._parse_evolution_response(raw)
    assert parsed is None
    assert err == "schema_violation:primary_failure_tag"


def test_parse_response_missing_primary_failure() -> None:
    raw = (
        "PRIMARY_FAILURE_TAG: x\n"
        "TARGET_SECTION: NONE\n"
        "PROPOSED_SCOPE: none\n"
        "DEFERRED_CONCERNS:\n"
        "PATCH:\nNO_CHANGE\n"
    )
    parsed, err = mt._parse_evolution_response(raw)
    assert parsed is None
    assert err == "schema_violation:primary_failure"


def test_parse_response_tag_too_long() -> None:
    """Tag cap is 32 chars; the prompt asks for snake_case ≤32. Catches
    models that bake a whole sentence into the tag field."""
    raw = _schema_response(tag="x" * 33)
    parsed, err = mt._parse_evolution_response(raw)
    assert parsed is None
    assert err == "schema_violation:tag_too_long"


def test_parse_response_invalid_scope() -> None:
    raw = _schema_response(scope="huge_restructure")
    parsed, err = mt._parse_evolution_response(raw)
    assert parsed is None
    assert err == "schema_violation:scope_invalid"


def test_parse_response_missing_patch_mode() -> None:
    """Phase 3.5: PATCH_MODE is required. Missing label → schema_violation."""
    raw = (
        "PRIMARY_FAILURE_TAG: x\n"
        "PRIMARY_FAILURE: y\n"
        "TARGET_SECTION: NONE\n"
        "PROPOSED_SCOPE: none\n"
        "DEFERRED_CONCERNS:\n"
        "TARGET_LINES: NONE\n"
        "BEFORE:\n\nAFTER:\n\n"
    )
    parsed, err = mt._parse_evolution_response(raw)
    assert parsed is None
    assert err == "schema_violation:patch_mode"


def test_parse_response_invalid_patch_mode() -> None:
    """Phase 3.5: PATCH_MODE must be one of {replace_lines, replace_section,
    no_change}. Anything else → schema_violation:patch_mode_invalid."""
    raw = _schema_response(patch_mode="rewrite_everything")
    parsed, err = mt._parse_evolution_response(raw)
    assert parsed is None
    assert err == "schema_violation:patch_mode_invalid"


def test_parse_response_missing_before_block() -> None:
    """Phase 3.5: BEFORE label is required (body may be empty for no_change)."""
    raw = (
        "PRIMARY_FAILURE_TAG: x\n"
        "PRIMARY_FAILURE: y\n"
        "TARGET_SECTION: NONE\n"
        "PROPOSED_SCOPE: none\n"
        "DEFERRED_CONCERNS:\n"
        "PATCH_MODE: no_change\n"
        "TARGET_LINES: NONE\n"
        "AFTER:\n"
    )
    parsed, err = mt._parse_evolution_response(raw)
    assert parsed is None
    assert err == "schema_violation:before"


def test_parse_response_empty_before_for_non_no_change_rejects() -> None:
    """Phase 3.5: BEFORE must carry content when PATCH_MODE != no_change.
    Empty BEFORE with replace_lines mode → schema_violation:before."""
    raw = _schema_response(
        patch_mode="replace_lines",
        before="",  # empty
        after="something",
    )
    parsed, err = mt._parse_evolution_response(raw)
    assert parsed is None
    assert err == "schema_violation:before"


def test_parse_response_empty_deferred_concerns_is_valid() -> None:
    """Empty list, not null. Models without deferred concerns leave the
    section empty but the label still appears."""
    raw = _schema_response(concerns=[])
    parsed, err = mt._parse_evolution_response(raw)
    assert err is None
    assert parsed["deferred_concerns"] == []


def test_parse_response_target_none_normalized() -> None:
    """TARGET_SECTION="NONE" maps to Python None for downstream code."""
    raw = _schema_response(target="NONE", scope="none")
    parsed, err = mt._parse_evolution_response(raw)
    assert err is None
    assert parsed["target_section"] is None


def test_parse_response_strips_outer_code_fence() -> None:
    """Model wrapped the entire schema response in ``` fences — parser
    strips them and processes the inner schema normally."""
    inner = _schema_response(tag="model_overfenced", failure="model wrapped output")
    raw = f"```\n{inner}\n```"
    parsed, err = mt._parse_evolution_response(raw)
    assert err is None
    assert parsed["primary_failure_tag"] == "model_overfenced"


def test_parse_response_empty_string_is_schema_violation() -> None:
    parsed, err = mt._parse_evolution_response("")
    assert parsed is None
    assert err == "schema_violation:empty"


def test_parse_response_whitespace_only_is_schema_violation() -> None:
    parsed, err = mt._parse_evolution_response("   \n\n\t  ")
    assert parsed is None
    # The first missing required field is reported (parser short-circuits).
    assert err.startswith("schema_violation:")


def test_journal_carries_structured_fields_on_accept(isolated_paths, monkeypatch) -> None:
    """Phase 1b + 3.5: ACCEPT paths land all structured fields in the
    journal entry, including the new patch_mode field."""
    _seed_multi_section_scaffold(isolated_paths)
    response = _apply_response(
        _MULTI_SECTION_SCAFFOLD,
        "Linear traces discharge with one line.",
        "Linear traces discharge with one TIGHTER line.",
        tag="acceptance_smoke",
        failure="test-only failure",
        target_section="Audit",
        scope="line",
        concerns=["follow-up1", "follow-up2"],
    )
    monkeypatch.setattr(mt, "_call_llm", lambda prompt: response)
    mt.maybe_evolve_after_rating("T-fields-accept", 70, ["restatement_unpruned"])
    entries = _read_real_journal_entries(isolated_paths["journal"])
    assert len(entries) == 1
    e = entries[0]
    assert e["applied"] is True
    assert e["primary_failure_tag"] == "acceptance_smoke"
    assert e["primary_failure"] == "test-only failure"
    assert e["target_section"] == "Audit"
    assert e["proposed_scope"] == "line"
    assert e["deferred_concerns"] == ["follow-up1", "follow-up2"]


# ── think-block injection into the evolution prompt ────────────────────


def test_compose_prompt_omits_think_section_when_absent() -> None:
    prompt = mt._compose_prompt("scaffold body", 70, "thin reasoning")
    assert "Your reasoning trace" not in prompt
    assert "scaffold body" in prompt


def test_compose_prompt_includes_think_section_when_present() -> None:
    trace = "frame: probe-first\naudit: dropped restated step 2\nconclusion: ship"
    prompt = mt._compose_prompt(
        "scaffold body", 60, "missed structural restatement",
        think_block=trace,
    )
    assert "Your reasoning trace" in prompt
    assert trace in prompt
    # Framing explicitly names the scaffold as the lever — not the rating, not
    # the trace. This protects against the model self-grading the trace.
    assert "scaffold is the lever" in prompt


def test_compose_prompt_requires_schema_field_labels() -> None:
    """Phase 1a + 3.5: the prompt must teach the model the schema with the
    new BEFORE/AFTER fields. The old PATCH: field is gone."""
    prompt = mt._compose_prompt("scaffold body", 70, "thin reasoning")
    for label in (
        "PRIMARY_FAILURE_TAG:",
        "PRIMARY_FAILURE:",
        "TARGET_SECTION:",
        "PROPOSED_SCOPE:",
        "DEFERRED_CONCERNS:",
        "PATCH_MODE:",
        "TARGET_LINES:",
        "BEFORE:",
        "AFTER:",
    ):
        assert label in prompt, f"prompt missing schema label {label!r}"


def test_compose_prompt_burned_old_full_scaffold_instruction() -> None:
    """Phase 3.5: the old 'output the entire revised scaffold under PATCH:'
    instruction must be gone — that framing was the load-bearing cause of
    4000-char diffs (the model did exactly what we asked: regenerate the
    whole scaffold). The new contract asks for BEFORE/AFTER blocks instead."""
    prompt = mt._compose_prompt("scaffold body", 70, "thin reasoning")
    # Old Phase 1a instructions should be gone
    assert "your entire reply must be either" not in prompt.lower()
    assert "Reply with EITHER" not in prompt
    # The Phase 1a "entire revised scaffold" framing under PATCH: must be gone
    assert "PATCH:" not in prompt  # the old single-field label
    assert "entire revised scaffold" not in prompt
    # The new contract uses BEFORE/AFTER as the load-bearing surface
    assert "BEFORE:" in prompt
    assert "AFTER:" in prompt
    # Anti-pattern framing from the user's spec must be present
    assert "boring" in prompt
    assert "BYTE-IDENTICAL" in prompt or "byte-for-byte" in prompt


def test_compose_prompt_explains_scope_enum_values() -> None:
    """Phase 1a: the prompt must teach the model what each PROPOSED_SCOPE
    value means. Without this, the model has no shared semantics with the
    scope_mismatch detector that Phase 3b ships."""
    prompt = mt._compose_prompt("scaffold body", 70, "test")
    for scope in ("line", "section", "structural", "none"):
        assert scope in prompt, f"prompt missing scope value {scope!r}"


def test_clip_think_block_returns_none_on_empty() -> None:
    assert mt._clip_think_block(None) is None
    assert mt._clip_think_block("") is None
    assert mt._clip_think_block("   \n\t  ") is None


def test_clip_think_block_passes_short_traces_through() -> None:
    short = "step 1\nstep 2\nstep 3"
    assert mt._clip_think_block(short) == short


def test_clip_think_block_tail_caps_long_traces() -> None:
    long = "x" * (mt._THINK_TAIL_CAP + 500)
    out = mt._clip_think_block(long)
    assert out is not None
    assert out.endswith("x" * mt._THINK_TAIL_CAP)
    assert "trace truncated" in out  # the head-marker is present


# ── async dispatch: don't block the caller (the UI freeze fix) ─────────


def test_async_path_returns_before_llm_call_completes(
    tmp_path, monkeypatch,
) -> None:
    """Production async path must NOT block the caller on the LLM call. We
    prove this by making the worker block until released; the caller should
    return immediately with the scheduled_async sentinel."""
    import threading
    import time

    monkeypatch.setenv("MONOLITH_MONOTHINK_ASYNC", "1")  # override sync autouse
    monkeypatch.setattr(mt, "_JOURNAL_PATH", tmp_path / "j.jsonl")
    monkeypatch.setattr(mt, "_SCAFFOLD_PATH", tmp_path / "scaffold.md")
    monkeypatch.setattr(mt, "_lookup_turn_monothink_active", lambda tid: True)

    worker_started = threading.Event()
    worker_release = threading.Event()
    worker_thread_name = {}

    def slow_worker(tid, rv, rr, tb):
        worker_thread_name["name"] = threading.current_thread().name
        worker_started.set()
        worker_release.wait(timeout=5.0)
        return mt._make_journal_entry(tid, rv, rr, "", "", False, "no_change_requested")

    monkeypatch.setattr(mt, "_run_evolution_blocking", slow_worker)

    caller_thread = threading.current_thread().name
    t0 = time.monotonic()
    result = mt.maybe_evolve_after_rating("T-async", 80, ["restatement_unpruned"], think_block="trace")
    elapsed = time.monotonic() - t0

    # Caller returned in well under a second — it did NOT wait for the worker.
    assert elapsed < 1.0
    assert result is not None
    assert result.get("reject_reason") == "scheduled_async"

    # Worker actually started (the thread was spawned), on a different thread.
    assert worker_started.wait(timeout=2.0)
    assert worker_thread_name["name"] != caller_thread
    assert worker_thread_name["name"].startswith("monothink-evolve-")

    # Let the worker finish so the daemon thread doesn't outlive the test.
    worker_release.set()


def test_async_path_writes_reservation_journal_entry(
    tmp_path, monkeypatch,
) -> None:
    """The reservation entry is what dedups concurrent same-turn ratings.
    Verify it lands in the journal synchronously, before the worker runs."""
    import threading

    monkeypatch.setenv("MONOLITH_MONOTHINK_ASYNC", "1")
    journal = tmp_path / "j.jsonl"
    monkeypatch.setattr(mt, "_JOURNAL_PATH", journal)
    monkeypatch.setattr(mt, "_SCAFFOLD_PATH", tmp_path / "scaffold.md")
    monkeypatch.setattr(mt, "_lookup_turn_monothink_active", lambda tid: True)

    block_worker = threading.Event()
    monkeypatch.setattr(
        mt, "_run_evolution_blocking",
        lambda *a, **kw: block_worker.wait(timeout=5.0) or None,
    )

    mt.maybe_evolve_after_rating("T-reservation", 70, ["restatement_unpruned"])

    # Journal contains the reservation entry immediately, before worker resolves.
    assert journal.exists()
    lines = journal.read_text(encoding="utf-8").splitlines()
    assert any('"reject_reason": "scheduled_async"' in line for line in lines)
    # Second rating on the same turn dedups via _journal_has_turn.
    second = mt.maybe_evolve_after_rating("T-reservation", 60, ["restatement_unpruned"])
    assert second is not None
    assert second.get("reject_reason") == "already_processed"

    block_worker.set()


# ── maybe_evolve_after_rating: decision tree ──────────────────────────


@pytest.fixture
def isolated_paths(tmp_path, monkeypatch):
    """Redirect scaffold + journal to a temp dir; default tier to monothink."""
    scaffold = tmp_path / "monothink.md"
    journal = tmp_path / "monothink.journal.jsonl"
    scaffold.write_text("# MonoThink — seed\n", encoding="utf-8")
    monkeypatch.setattr(mt, "_SCAFFOLD_PATH", scaffold)
    monkeypatch.setattr(mt, "_JOURNAL_PATH", journal)
    monkeypatch.setattr(mt, "_lookup_turn_monothink_active", lambda turn_id: True)
    monkeypatch.setenv("MONOLITH_MONOTHINK_EVOLVE_V1", "1")
    return {"scaffold": scaffold, "journal": journal}


def _read_journal_entries(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_real_journal_entries(path):
    """Read journal entries but exclude bootstrap/rollback rows. Phase 3e
    auto-writes a bootstrap entry on the first monothink evolution attempt,
    which inflates entry counts by 1; tests that assert on the actual
    evolution-attempt count use this helper to filter the substrate noise."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("kind") in ("bootstrap", "rollback"):
            continue
        out.append(e)
    return out


def _schema_response(
    *,
    tag: str = "generic_tag",
    failure: str = "generic failure observation",
    target: str = "NONE",
    scope: str = "none",
    concerns: list[str] | None = None,
    patch_mode: str = "no_change",
    target_lines: str = "NONE",
    before: str = "",
    after: str = "",
    # ── Phase 3.5 migration compatibility ────────────────────────────
    # Legacy `patch=...` is intercepted here and translated. Tests written
    # before the BEFORE/AFTER schema landed used `patch="NO_CHANGE"` (maps
    # to patch_mode=no_change with empty before/after) or `patch=<scaffold
    # text>` (no longer auto-translatable — those tests need to migrate to
    # _apply_response). When patch=<text> is supplied with no before/after,
    # we emit a malformed BEFORE block so the test gets a parser failure
    # instead of a confusing silent pass — that's a tripwire signalling
    # the test needs migration.
    patch: str | None = None,
) -> str:
    """Build a Phase 1a/1b/3.5 schema-compliant evolution response for tests.

    The Phase 3.5 contract uses BEFORE/AFTER instead of a full PATCH body.
    Tests that need a working APPLY should use ``_apply_response(current,
    target_text, replacement_text, ...)`` which builds a response known to
    match the given scaffold. Tests that only need to exercise rejection
    paths can use this helper directly with the patch_mode they want.
    """
    if patch is not None:
        if patch.strip().upper() == "NO_CHANGE":
            patch_mode = "no_change"
            before = ""
            after = ""
        else:
            # Caller passed `patch=<scaffold text>` — legacy path. Translate
            # to before/after with the WHOLE current pre-image as BEFORE,
            # but we don't know the current scaffold here. Best-effort: put
            # the legacy patch text in BEFORE so any byte-exact match will
            # fail and the test gets a clear before_block_mismatch failure
            # instead of a confusing silent pass.
            before = patch
            after = patch
            patch_mode = "replace_section"
    concerns_block = "\n".join(f"- {c}" for c in (concerns or []))
    return (
        f"PRIMARY_FAILURE_TAG: {tag}\n"
        f"PRIMARY_FAILURE: {failure}\n"
        f"TARGET_SECTION: {target}\n"
        f"PROPOSED_SCOPE: {scope}\n"
        f"DEFERRED_CONCERNS:\n"
        f"{concerns_block}\n"
        f"PATCH_MODE: {patch_mode}\n"
        f"TARGET_LINES: {target_lines}\n"
        f"BEFORE:\n"
        f"{before}\n"
        f"AFTER:\n"
        f"{after}\n"
    )


def _apply_response(
    current_scaffold: str,
    target_text: str,
    replacement_text: str,
    *,
    tag: str = "test_apply",
    failure: str = "test apply failure",
    target_section: str = "Audit",
    scope: str = "line",
    concerns: list[str] | None = None,
    target_lines: str = "N/A",
    patch_mode: str = "replace_lines",
) -> str:
    """Build a response that will successfully apply target_text →
    replacement_text against current_scaffold. Verifies target_text appears
    in current_scaffold (fails fast at test-construction time rather than
    producing a confusing before_block_mismatch later).

    Phase 3.5's BEFORE/AFTER schema needs a known-good BEFORE block; this
    helper enforces that the test author actually picked a substring of
    the real scaffold."""
    if target_text not in current_scaffold:
        raise AssertionError(
            f"target_text not in current scaffold; the test can't build a "
            f"working _apply_response. target_text={target_text!r}"
        )
    return _schema_response(
        tag=tag, failure=failure, target=target_section, scope=scope,
        concerns=concerns, patch_mode=patch_mode,
        target_lines=target_lines, before=target_text, after=replacement_text,
    )


def test_flag_off_returns_none(isolated_paths, monkeypatch) -> None:
    monkeypatch.setenv("MONOLITH_MONOTHINK_EVOLVE_V1", "0")
    monkeypatch.setattr(mt, "_call_llm", lambda prompt: "ignored")
    result = mt.maybe_evolve_after_rating("T-flag", 50, ["restatement_unpruned"])
    assert result is None
    assert not isolated_paths["journal"].exists()


def test_non_monothink_tier_returns_none(isolated_paths, monkeypatch) -> None:
    monkeypatch.setattr(mt, "_lookup_turn_monothink_active", lambda turn_id: False)
    monkeypatch.setattr(mt, "_call_llm", lambda prompt: "ignored")
    result = mt.maybe_evolve_after_rating("T-med", 80, ["restatement_unpruned"])
    assert result is None
    assert not isolated_paths["journal"].exists()


def test_already_processed_returns_sentinel(isolated_paths, monkeypatch) -> None:
    # Pre-seed journal with T-dup
    isolated_paths["journal"].write_text(
        json.dumps({"turn_id": "T-dup", "applied": True}) + "\n",
        encoding="utf-8",
    )
    llm_calls = []
    monkeypatch.setattr(mt, "_call_llm", lambda prompt: llm_calls.append(prompt) or "should not run")

    result = mt.maybe_evolve_after_rating("T-dup", 60, ["restatement_unpruned"])
    assert result == {
        "applied": False,
        "reject_reason": "already_processed",
        "turn_id": "T-dup",
    }
    # Critically: NO LLM call. Idempotence is enforced *before* spending tokens.
    assert llm_calls == []


def test_no_change_response_writes_journal_skips_scaffold(isolated_paths, monkeypatch) -> None:
    response = _schema_response(
        tag="no_action_needed",
        failure="rating high; no actionable failure to patch",
        target="NONE",
        scope="none",
        patch="NO_CHANGE",
    )
    monkeypatch.setattr(mt, "_call_llm", lambda prompt: response)
    original_scaffold = isolated_paths["scaffold"].read_text(encoding="utf-8")

    entry = mt.maybe_evolve_after_rating("T-noop", 75, ["restatement_unpruned"])
    assert entry["applied"] is False
    assert entry["reject_reason"] == "no_change_requested"
    # Scaffold unchanged
    assert isolated_paths["scaffold"].read_text(encoding="utf-8") == original_scaffold
    # Journal has one real (non-bootstrap) entry carrying structured fields
    entries = _read_real_journal_entries(isolated_paths["journal"])
    assert len(entries) == 1
    assert entries[0]["reject_reason"] == "no_change_requested"
    assert entries[0]["primary_failure_tag"] == "no_action_needed"
    assert entries[0]["proposed_scope"] == "none"
    assert entries[0]["target_section"] is None


def test_llm_failure_records_failure_entry(isolated_paths, monkeypatch) -> None:
    monkeypatch.setattr(mt, "_call_llm", lambda prompt: None)
    entry = mt.maybe_evolve_after_rating("T-fail", 40, ["restatement_unpruned"])
    assert entry["applied"] is False
    assert entry["reject_reason"] == "llm_call_failed"
    entries = _read_real_journal_entries(isolated_paths["journal"])
    assert len(entries) == 1


def test_empty_response_rejects(isolated_paths, monkeypatch) -> None:
    """Whitespace-only raw response → schema_violation (parser rejects before
    the old empty_response branch can fire). The empty_response branch now
    only triggers when a SCHEMA-VALID response has an empty PATCH body after
    code-fence stripping."""
    monkeypatch.setattr(mt, "_call_llm", lambda prompt: "   \n  \n")
    entry = mt.maybe_evolve_after_rating("T-empty", 50, ["restatement_unpruned"])
    assert entry["applied"] is False
    assert entry["reject_reason"] == "schema_violation:empty"


def test_size_cap_rejects(isolated_paths, monkeypatch) -> None:
    """Phase 3.5: size cap fires when the AFTER block balloons the final
    scaffold past _SIZE_CAP. Uses a multi-section fixture so the parsed
    sections + size check have something coherent to bite on."""
    _seed_multi_section_scaffold(isolated_paths)
    huge_after = "Linear traces discharge differently:\n" + ("x" * mt._SIZE_CAP)
    response = _apply_response(
        _MULTI_SECTION_SCAFFOLD,
        "Linear traces discharge with one line.",
        huge_after,
        tag="oversize", failure="model produced an oversized AFTER block",
        target_section="Audit", scope="section",
        patch_mode="replace_section",
    )
    monkeypatch.setattr(mt, "_call_llm", lambda prompt: response)
    entry = mt.maybe_evolve_after_rating("T-big", 50, ["restatement_unpruned"])
    assert entry["applied"] is False
    assert entry["reject_reason"].startswith("size_cap_exceeded")
    # Scaffold not touched.
    assert isolated_paths["scaffold"].read_text(encoding="utf-8") == _MULTI_SECTION_SCAFFOLD


def test_oversized_line_claim_proposes_not_discards(isolated_paths, monkeypatch) -> None:
    """Decide-from-actual: model claims line, BEFORE→AFTER produces a
    section-sized diff. The catch is PROPOSED from actual scope (mismatch
    logged), never discarded on the label — less wasted evidence."""
    _seed_multi_section_scaffold(isolated_paths)
    huge_after = "Linear traces discharge with " + ("x" * (mt._DIFF_CAP + 50))
    response = _apply_response(
        _MULTI_SECTION_SCAFFOLD,
        "Linear traces discharge with one line.",
        huge_after,
        tag="overdiff", failure="model proposed a too-large AFTER block",
        target_section="Audit", scope="line",  # mislabel
    )
    monkeypatch.setattr(mt, "_call_llm", lambda prompt: response)
    entry = mt.maybe_evolve_after_rating("T-big-diff", 50, ["restatement_unpruned"])
    assert entry["applied"] is False
    assert entry["reject_reason"] == "proposed_only:section"
    assert entry["scope_mismatch"] is True


def test_accept_path_writes_scaffold_and_journals(isolated_paths, monkeypatch) -> None:
    """Accept path runs against a realistic multi-section scaffold via the
    BEFORE/AFTER mechanism. The bare `# MonoThink — seed` fixture has no
    ## sections so the fast-check would reject it as broken structure."""
    _seed_multi_section_scaffold(isolated_paths)
    response = _apply_response(
        _MULTI_SECTION_SCAFFOLD,
        "Linear traces discharge with one line.",
        "Linear traces discharge with one CLEAN line.",
        tag="naming_discipline",
        failure="audit threshold could be sharper",
        target_section="Audit",
        scope="line",
    )
    monkeypatch.setattr(mt, "_call_llm", lambda prompt: response)

    entry = mt.maybe_evolve_after_rating("T-accept", 85, ["restatement_unpruned"])
    assert entry["applied"] is True
    assert entry["reject_reason"] is None
    written = isolated_paths["scaffold"].read_text(encoding="utf-8")
    assert "CLEAN line" in written
    # Journal recorded with structured fields (filter out bootstrap row)
    entries = _read_real_journal_entries(isolated_paths["journal"])
    assert len(entries) == 1
    assert entries[0]["applied"] is True
    assert entries[0]["diff_chars"] > 0
    assert entries[0]["primary_failure_tag"] == "naming_discipline"
    assert entries[0]["proposed_scope"] == "line"
    assert entries[0]["target_section"] == "Audit"
    # Phase 3e: the applied entry carries the full new scaffold and the
    # parent_scaffold_version pointer to the bootstrap row.
    assert "CLEAN line" in entries[0]["applied_scaffold_full"]
    assert str(entries[0]["parent_scaffold_version"]).startswith("bootstrap-")


def test_proposal_equals_current_is_noop(isolated_paths, monkeypatch) -> None:
    current = isolated_paths["scaffold"].read_text(encoding="utf-8")
    response = _schema_response(
        tag="model_echoed",
        failure="model returned the current scaffold verbatim",
        target="NONE",
        scope="line",
        patch=current,
    )
    monkeypatch.setattr(mt, "_call_llm", lambda prompt: response)
    entry = mt.maybe_evolve_after_rating("T-same", 50, ["restatement_unpruned"])
    assert entry["applied"] is False
    assert entry["reject_reason"] == "proposed_equals_current"


def test_code_fence_wrapper_is_stripped(isolated_paths, monkeypatch) -> None:
    """Phase 3.5 obsoletes the original test premise: with BEFORE/AFTER, the
    model doesn't write a fenced full-scaffold block; it writes BEFORE and
    AFTER text directly. The parser still strips a fence wrapping the
    ENTIRE schema response (outer fence) which is what some models do.
    Test repurposed to verify outer-fence stripping survives the schema
    rewrite."""
    _seed_multi_section_scaffold(isolated_paths)
    inner_response = _apply_response(
        _MULTI_SECTION_SCAFFOLD,
        "Linear traces discharge with one line.",
        "Linear traces discharge with one FENCED line.",
        tag="model_fenced_patch",
        failure="model wrapped the whole response in code fences",
        target_section="Audit",
        scope="line",
    )
    # Wrap the whole response in an outer code fence.
    fenced_response = f"```\n{inner_response}\n```"
    monkeypatch.setattr(mt, "_call_llm", lambda prompt: fenced_response)
    entry = mt.maybe_evolve_after_rating("T-fenced", 70, ["restatement_unpruned"])
    assert entry["applied"] is True
    written = isolated_paths["scaffold"].read_text(encoding="utf-8")
    assert "FENCED line" in written


def test_journal_entry_carries_raw_response_full_on_rejection(
    isolated_paths, monkeypatch,
) -> None:
    """Phase 0a + 1b + 3.5: rejection paths persist the full LLM response.
    Model claims line; BEFORE→AFTER measures section → scope_mismatch.
    raw_response_full and structured fields both land on the rejection."""
    _seed_multi_section_scaffold(isolated_paths)
    huge_after = "Linear traces:\n" + ("x" * 1000) + "\nTAIL"
    raw = _apply_response(
        _MULTI_SECTION_SCAFFOLD,
        "Linear traces discharge with one line.",
        huge_after,
        tag="oversized_line_claim",
        failure="model wanted a big restructure but claimed line",
        target_section="Audit",
        scope="line",  # mislabel
    )
    monkeypatch.setattr(mt, "_call_llm", lambda prompt: raw)

    entry = mt.maybe_evolve_after_rating("T-rawfull", 50, ["restatement_unpruned"])

    assert entry["applied"] is False
    assert entry["reject_reason"] == "proposed_only:section"  # proposal, not a discard
    # Full raw (the entire schema-wrapped string) preserved verbatim.
    assert entry["raw_response_full"] == raw
    assert len(entry["raw_response_full"]) > 240
    # Structured fields landed too — parser succeeded before policy fired.
    assert entry["primary_failure_tag"] == "oversized_line_claim"
    assert entry["proposed_scope"] == "line"


def test_journal_entry_raw_preview_equals_full_when_short(
    isolated_paths, monkeypatch,
) -> None:
    """Short rejection responses: preview equals the full text verbatim.
    Whitespace-only output now triggers schema_violation (parser rejects
    before the empty_response branch). Still carries raw_response_full +
    raw_preview as forensic backup."""
    short_whitespace = "   \n\n  \t  "
    monkeypatch.setattr(mt, "_call_llm", lambda prompt: short_whitespace)
    entry = mt.maybe_evolve_after_rating("T-ws", 50, ["restatement_unpruned"])
    assert entry["reject_reason"].startswith("schema_violation:")
    assert entry["raw_response_full"] == short_whitespace
    # Short text: preview is the full text unchanged.
    assert entry["raw_preview"] == short_whitespace


def test_journal_entry_raw_preview_is_ellipsized_tail_when_long(
    isolated_paths, monkeypatch,
) -> None:
    """Long rejection responses: preview is the ellipsized tail; the full
    text is preserved in raw_response_full."""
    _seed_multi_section_scaffold(isolated_paths)
    huge_after = "PREAMBLE\n" + ("y" * 500) + "\nFINAL_TAIL_MARK"
    raw = _apply_response(
        _MULTI_SECTION_SCAFFOLD,
        "Linear traces discharge with one line.",
        huge_after,
        tag="verbose_model",
        failure="model returned an oversized patch",
        target_section="Audit",
        scope="line",  # mislabel
    )
    monkeypatch.setattr(mt, "_call_llm", lambda prompt: raw)
    entry = mt.maybe_evolve_after_rating("T-long", 50, ["restatement_unpruned"])
    assert entry["reject_reason"] == "proposed_only:section"
    assert entry["raw_response_full"] == raw
    assert entry["raw_preview"].startswith("…")
    assert len(entry["raw_preview"]) == 240


def test_lock_releases_on_exception(isolated_paths, monkeypatch) -> None:
    """If the LLM call raises (not returns None), the broad outer except must
    still release ``_evolve_lock`` — otherwise the next rating click hangs."""
    def boom(prompt):
        raise RuntimeError("simulated llm exception")
    monkeypatch.setattr(mt, "_call_llm", boom)

    assert mt._evolve_lock.locked() is False
    entry = mt.maybe_evolve_after_rating("T-boom", 50, ["restatement_unpruned"])
    # Lock released regardless of exception.
    assert mt._evolve_lock.locked() is False
    # Failure entry recorded.
    assert entry is not None
    assert entry["applied"] is False
    assert entry["reject_reason"].startswith("exception:")


# ── integration: record_outcome → hook wire ────────────────────────────


@pytest.fixture
def trace_db(tmp_path, monkeypatch):
    db = tmp_path / "turn_trace.sqlite3"
    tt.set_db_path(db)
    monkeypatch.setenv("MONOLITH_TURN_TRACE_V1", "1")
    yield db
    tt.set_db_path(None)


def test_record_outcome_fires_monothink_hook(trace_db, monkeypatch) -> None:
    """A rating outcome with failure_tags triggers maybe_evolve_after_rating
    with the structured tags (not free text)."""
    captured = {}

    def fake_hook(turn_id, rating_value, failure_tags, think_block=None, replay_input=None, rater_note=None):
        captured["turn_id"] = turn_id
        captured["rating_value"] = rating_value
        captured["failure_tags"] = failure_tags
        captured["think_block"] = think_block
        return None

    # Patch at the import site used by record_outcome (lazy import).
    monkeypatch.setattr("core.monothink.maybe_evolve_after_rating", fake_hook)

    rec = tt.OutcomeTraceRecord(
        turn_id="T-hook",
        recorded_at="2026-05-15T00:00:00Z",
        kind="rating",
        rating_value=88,
        reason="useful pressure on naming",
        metadata={"failure_tags": ["restatement_unpruned"]},
    )
    tt.record_outcome(rec)

    assert captured == {
        "turn_id": "T-hook",
        "rating_value": 88,
        "failure_tags": ["restatement_unpruned"],
        "think_block": None,
    }


def test_record_outcome_forwards_think_block_from_metadata(
    trace_db, monkeypatch,
) -> None:
    """When OutcomeTraceRecord.metadata carries think_block, it flows through
    to the monothink hook so the model can see its own reasoning trace
    alongside the rating signal."""
    captured = {}

    def fake_hook(turn_id, rating_value, failure_tags, think_block=None, replay_input=None, rater_note=None):
        captured["think_block"] = think_block
        return None

    monkeypatch.setattr("core.monothink.maybe_evolve_after_rating", fake_hook)

    trace = "step 1: read the prompt\nstep 2: prune restated step\nstep 3: answer."
    rec = tt.OutcomeTraceRecord(
        turn_id="T-think",
        recorded_at="2026-05-21T00:00:00Z",
        kind="rating",
        rating_value=72,
        reason="missed the structural restatement",
        metadata={"think_block": trace, "failure_tags": ["restatement_unpruned"]},
    )
    tt.record_outcome(rec)

    assert captured["think_block"] == trace


def test_record_outcome_skips_hook_on_passive_actions(trace_db, monkeypatch) -> None:
    """SP1: the evolution hook fires only for kind="rating" with non-empty
    failure_tags. It does NOT fire for copy / regen / delete (passive UI
    signals) — nor for thumbs_up / thumbs_down, which were cut from evolution
    in SP1 (they still record outcomes for telemetry)."""
    called = {"n": 0}

    def fake_hook(*args, **kwargs):
        called["n"] += 1

    monkeypatch.setattr("core.monothink.maybe_evolve_after_rating", fake_hook)

    for kind in ("copy", "regen", "delete"):
        tt.record_outcome(tt.OutcomeTraceRecord(
            turn_id=f"T-{kind}",
            recorded_at="2026-05-15T00:00:00Z",
            kind=kind,
            rating_value=None,
            reason=None,
            metadata=None,
        ))

    assert called["n"] == 0


def test_record_outcome_does_not_fire_hook_on_thumbs_up(
    trace_db, monkeypatch,
) -> None:
    """SP1: thumbs are cut from evolution. thumbs_up records an outcome but does
    NOT trigger maybe_evolve_after_rating — thumbs carry no failure_tags, so no
    directional signal reaches monothink."""
    called = {"n": 0}

    def fake_hook(*args, **kwargs):
        called["n"] += 1

    monkeypatch.setattr("core.monothink.maybe_evolve_after_rating", fake_hook)

    tt.record_outcome(tt.OutcomeTraceRecord(
        turn_id="T-up",
        recorded_at="2026-05-22T00:00:00Z",
        kind="thumbs_up",
        rating_value=None,  # required: thumbs records can't carry rating
        reason=None,
        metadata=None,
    ))

    assert called["n"] == 0


def test_record_outcome_does_not_fire_hook_on_thumbs_down(
    trace_db, monkeypatch,
) -> None:
    """SP1: thumbs_down records an outcome but does NOT trigger evolution."""
    called = {"n": 0}

    def fake_hook(*args, **kwargs):
        called["n"] += 1

    monkeypatch.setattr("core.monothink.maybe_evolve_after_rating", fake_hook)

    tt.record_outcome(tt.OutcomeTraceRecord(
        turn_id="T-down",
        recorded_at="2026-05-22T00:00:00Z",
        kind="thumbs_down",
        rating_value=None,
        reason=None,
        metadata=None,
    ))

    assert called["n"] == 0


# (SP1: test_thumbs_path_carries_think_block_to_hook removed — thumbs no longer
#  reach the evolution hook. think_block forwarding on the rating path is covered
#  by test_record_outcome_forwards_think_block_from_metadata.)


def test_outcome_traces_row_keeps_null_rating_value_on_thumbs(
    trace_db, monkeypatch,
) -> None:
    """Phase 0b semantic invariant: outcome_traces preserves CLEAN signals.
    A thumbs_up row keeps `rating_value=NULL` in the database even though
    the synthesized 85 was passed to monothink. The synthesis is monothink-
    only state; persisting it would conflate "user clicked thumbs" with
    "user assigned a numeric rating" — those are distinct events that
    should remain distinguishable in queries against outcome_traces."""
    monkeypatch.setattr("core.monothink.maybe_evolve_after_rating",
                         lambda *a, **kw: None)

    tt.record_outcome(tt.OutcomeTraceRecord(
        turn_id="T-clean",
        recorded_at="2026-05-22T00:00:00Z",
        kind="thumbs_up",
        rating_value=None,
        reason=None,
        metadata=None,
    ))

    # Read back the actual DB row. rating_value must be NULL, not 85.
    import sqlite3
    db_path = tt._get_db_path()  # honors the test's set_db_path override
    con = sqlite3.connect(str(db_path))
    try:
        row = con.execute(
            "SELECT kind, rating_value FROM outcome_traces WHERE turn_id=?",
            ("T-clean",),
        ).fetchone()
    finally:
        con.close()
    assert row is not None
    assert row[0] == "thumbs_up"
    assert row[1] is None  # NOT 85


# (legacy compatibility: keep the original test name as an alias so any
# external references in docs or other test files keep resolving)
test_record_outcome_skips_hook_on_non_rating = test_record_outcome_skips_hook_on_passive_actions


# ── SP1: failure_tags contract + three-stage adversarial deliberation ──────


def test_evolve_returns_none_when_no_valid_tags(isolated_paths, monkeypatch) -> None:
    """Triviality gate: a rating whose failure_tags normalize to empty (none
    given, or all unknown) does NOT fire evolution — the decider is never
    invoked. This is the 'simple hi' guard, by construction."""
    called = {"n": 0}
    monkeypatch.setattr(mt, "_call_llm", lambda prompt: called.__setitem__("n", called["n"] + 1) or "")
    assert mt.maybe_evolve_after_rating("T-empty-tags", 40, []) is None
    assert mt.maybe_evolve_after_rating("T-bogus-tags", 40, ["not_a_real_tag"]) is None
    assert called["n"] == 0  # decider never reached


def test_compose_prompt_contains_three_deliberation_stages() -> None:
    """D1: the prompt forces steelman-tag / positional-steelman-alternative /
    adjudicate before patching, and feeds the tag's canonical descriptive gloss
    as the directional signal."""
    from core.failure_tags import FAILURE_TAGS
    prompt = mt._compose_prompt("scaffold body", 55, ["missing_branch_pressure"])
    assert "STEELMAN_TAG" in prompt
    assert "STEELMAN_ALTERNATIVE" in prompt
    assert "ADJUDICATION" in prompt
    assert "positional" in prompt.lower()
    assert "argue the alternative" in prompt.lower()
    assert FAILURE_TAGS["missing_branch_pressure"] in prompt  # the gloss is the signal
    # The decision sentinel must precede the field block so the parser can isolate
    # the schema from the deliberation prose.
    assert "=== DECISION ===" in prompt
    assert prompt.index("=== DECISION ===") < prompt.index("PRIMARY_FAILURE_TAG:")


def test_compose_prompt_outputs_only_patch_deliberation_in_reasoning() -> None:
    """Output-only-patch: the 3-stage deliberation must be done IN THINKING, and the
    OUTPUT must be only the patch — no deliberation prose in the output (it's
    redundant with the model's <think> pass, frees output budget for the think, and
    removes a parsing surface). The reasoning INSTRUCTION is kept (the three-stages
    test still holds); only the output-prose REQUIREMENT is removed."""
    prompt = mt._compose_prompt("scaffold body", 70, ["missing_branch_pressure"])
    low = prompt.lower()
    assert "output only the patch" in low
    assert "in your thinking" in low
    # the old 'write all three stages as prose BEFORE the schema' requirement is gone
    assert "as prose before" not in low


def test_compose_prompt_excludes_surface_note_channel() -> None:
    """monothink_visible boundary: _compose_prompt takes failure_tags and has NO
    surface_note parameter — holistic feedback structurally cannot reach the decider."""
    import inspect
    params = set(inspect.signature(mt._compose_prompt).parameters)
    assert "failure_tags" in params
    assert "surface_note" not in params


def test_parse_response_aliases_adjudicated_tag() -> None:
    """D-B: the post-deliberation PRIMARY_FAILURE_TAG is also exposed as
    adjudicated_tag for divergence routing and concern_repeated matching."""
    raw = _schema_response(tag="premise_unchecked", failure="x", patch_mode="no_change")
    parsed, err = mt._parse_evolution_response(raw)
    assert err is None
    assert parsed["adjudicated_tag"] == "premise_unchecked"


def test_surface_note_reaches_hook_as_constrained_rater_note(trace_db, monkeypatch) -> None:
    """The rating hook forwards surface_note as rater_note, where the v2 prompt
    constrains it to locating/rejecting/reserving and forbids apply authority."""
    captured = {}

    def fake_hook(turn_id, rating_value, failure_tags, think_block=None, replay_input=None, rater_note=None):
        captured["failure_tags"] = failure_tags
        captured["think_block"] = think_block
        captured["rater_note"] = rater_note

    monkeypatch.setattr("core.monothink.maybe_evolve_after_rating", fake_hook)

    tt.record_outcome(tt.OutcomeTraceRecord(
        turn_id="T-surface",
        recorded_at="2026-06-03T00:00:00Z",
        kind="rating",
        rating_value=40,
        reason="auto echo",
        metadata={
            "failure_tags": ["assertion_without_argument"],
            "surface_note": "felt rushed and too long overall",
            "think_block": "trace here",
        },
    ))

    assert captured["failure_tags"] == ["assertion_without_argument"]
    assert captured["rater_note"] == "felt rushed and too long overall"
    prompt = mt._compose_prompt(
        "scaffold body",
        40,
        ["assertion_without_argument"],
        think_block="trace",
        rater_note=captured["rater_note"],
    )
    assert "RATER_NOTE_NONCANONICAL" in prompt
    assert "may not satisfy promotion" in prompt


def test_journal_flags_divergent_when_adjudicated_differs(isolated_paths, monkeypatch) -> None:
    """Divergence capture: when the decider adjudicates a different failure than
    the rater flagged, the journal stores both tags and divergent=True."""
    resp = _schema_response(tag="premise_unchecked", failure="reframed", patch_mode="no_change")
    monkeypatch.setattr(mt, "_call_llm", lambda prompt: resp)
    entry = mt.maybe_evolve_after_rating("T-div", 50, ["missing_branch_pressure"])
    assert entry["failure_tags"] == ["missing_branch_pressure"]
    assert entry["adjudicated_tag"] == "premise_unchecked"
    assert entry["divergent"] is True


def test_journal_not_divergent_when_adjudicated_matches(isolated_paths, monkeypatch) -> None:
    """No divergence when the decider's verdict equals the rater's tag."""
    resp = _schema_response(tag="missing_branch_pressure", failure="agreed", patch_mode="no_change")
    monkeypatch.setattr(mt, "_call_llm", lambda prompt: resp)
    entry = mt.maybe_evolve_after_rating("T-agree", 50, ["missing_branch_pressure"])
    assert entry["adjudicated_tag"] == "missing_branch_pressure"
    assert entry["divergent"] is False


def test_parse_ignores_deliberation_prose_before_schema() -> None:
    """The decider writes STEELMAN/ADJUDICATION prose BEFORE the schema, and that
    prose can contain field-looking lines (it argues about scope, names tags). The
    parser must extract the REAL schema block (after the sentinel), not the first
    field-looking line in the deliberation. Without this, every live evolution
    mis-parses while clean-fixture tests stay green — the silent no-op."""
    raw = (
        "STEELMAN_TAG: the strongest case for the rater's tag.\n"
        "PRIMARY_FAILURE_TAG: missing_branch_pressure\n"   # decoy line in prose
        "PROPOSED_SCOPE: section\n"                          # decoy line in prose
        "STEELMAN_ALTERNATIVE: but actually the premise was never checked.\n"
        "ADJUDICATION: I conclude the premise went unchecked.\n"
        "=== PATCH ===\n"
        "PRIMARY_FAILURE_TAG: premise_unchecked\n"
        "PRIMARY_FAILURE: a premise was used unchecked\n"
        "TARGET_SECTION: NONE\n"
        "PROPOSED_SCOPE: none\n"
        "DEFERRED_CONCERNS:\n"
        "PATCH_MODE: no_change\n"
        "TARGET_LINES: NONE\n"
        "BEFORE:\n"
        "\n"
        "AFTER:\n"
        "\n"
    )
    parsed, err = mt._parse_evolution_response(raw)
    assert err is None, err
    assert parsed["primary_failure_tag"] == "premise_unchecked"  # not the decoy
    assert parsed["adjudicated_tag"] == "premise_unchecked"
    assert parsed["proposed_scope"] == "none"  # not the decoy "section"


# -- Phase 3.6: deterministic anchor resolution (whitespace-tolerant tier) --


def test_apply_patch_resolves_before_with_trailing_whitespace_drift() -> None:
    """The decider copies BEFORE from its prompt; trailing spaces / CRLF drift
    must not reject an otherwise-verbatim unique block (observed live as
    before_block_mismatch:not_found on correct-looking blocks)."""
    current = "## A\nalpha line\nbeta line\n\n## B\ngamma\n"
    new, err = mt._apply_patch(current, "alpha line  \nbeta line\r\n", "alpha line\nBETA\n")
    assert err is None
    assert "BETA" in new and "beta line" not in new
    assert "gamma" in new                      # rest of file untouched


def test_apply_patch_resolves_before_with_blank_line_padding() -> None:
    current = "## A\nalpha\n\n## B\nbeta\n"
    new, err = mt._apply_patch(current, "\nalpha\n\n", "alpha2\n")
    assert err is None
    assert "alpha2" in new and "\nalpha\n" not in new


def test_apply_patch_normalized_tier_still_rejects_absent_text() -> None:
    current = "## A\nalpha\n"
    new, err = mt._apply_patch(current, "omega  \n", "x")
    assert new is None and err == "before_block_mismatch:not_found"


def test_apply_patch_normalized_tier_still_rejects_ambiguous() -> None:
    current = "alpha\nx\nalpha\n"
    new, err = mt._apply_patch(current, "alpha  ", "y")
    assert new is None and err == "before_block_mismatch:ambiguous"


def test_apply_patch_exact_match_takes_precedence_over_normalized() -> None:
    # exact unique match applies byte-exactly even when normalization would
    # also match elsewhere
    current = "keep  \nkeep\n"
    new, err = mt._apply_patch(current, "keep  \n", "KEPT\n")
    assert err is None
    assert new == "KEPT\nkeep\n"


def test_compose_prompt_teaches_deletion_and_current_size() -> None:
    """The patch schema has always accepted empty-AFTER deletion (schema line
    ~813) but the prompt never said so — so the model never used it, and at the
    size cap every additive edit dies. The prompt must (a) name deletion as a
    legal move and (b) state the CURRENT size next to the cap so the model can
    see its headroom."""
    prompt = mt._compose_prompt("x" * 7900, 40, ["restatement_unpruned"])
    low = prompt.lower()
    assert "empty after" in low or "after empty" in low or "deletion" in low
    assert "7900" in prompt          # current size shown
    assert str(mt._SIZE_CAP) in prompt


# -- Phase 3.7: bounded repair loop for hallucinated BEFORE anchors --


_REPAIR_SCAFFOLD = (
    "# T\n\npreamble\n\n## Audit\n\nintro\n\n### Hardening\n\n1. rule one\n2. rule two\n\n"
    "### Format\n\nformat body\n\n## Scope boundary\n\nscope body\n"
)


def test_section_excerpt_resolves_h2_section() -> None:
    out = mt._section_excerpt(_REPAIR_SCAFFOLD, "Scope boundary")
    assert "scope body" in out and "rule one" not in out


def test_section_excerpt_resolves_h3_subsection() -> None:
    out = mt._section_excerpt(_REPAIR_SCAFFOLD, "Hardening")
    assert "rule one" in out and "rule two" in out
    assert "format body" not in out          # stops at next header


def test_section_excerpt_falls_back_to_full_scaffold() -> None:
    assert mt._section_excerpt(_REPAIR_SCAFFOLD, "No Such Section") == _REPAIR_SCAFFOLD
    assert mt._section_excerpt(_REPAIR_SCAFFOLD, None) == _REPAIR_SCAFFOLD


def test_repair_before_block_returns_reparsed_patch_and_prompt_carries_excerpt() -> None:
    """One bounded retry: the repair prompt must contain the targeted region's
    CURRENT text verbatim and the model's intended AFTER; the stub returns a
    corrected patch which comes back parsed."""
    parsed = {"target_section": "Hardening", "before": "hallucinated text",
              "after": "2. rule two improved", "patch_mode": "replace_lines",
              "proposed_scope": "line"}
    seen = {}
    def stub_llm(prompt):
        seen["prompt"] = prompt
        return _schema_response(tag="restatement_unpruned", target="Hardening",
                                scope="line", patch_mode="replace_lines",
                                target_lines="1", before="2. rule two",
                                after="2. rule two improved")
    reparsed, raw2 = mt._repair_before_block(_REPAIR_SCAFFOLD, parsed, stub_llm)
    assert reparsed is not None
    assert reparsed["before"] == "2. rule two"
    assert "rule one" in seen["prompt"]              # excerpt present
    assert "2. rule two improved" in seen["prompt"]  # intended AFTER carried
    assert "character-for-character" in seen["prompt"] or "exactly" in seen["prompt"].lower()


def test_repair_before_block_returns_none_on_unparseable_reply() -> None:
    parsed = {"target_section": "Hardening", "before": "x", "after": "y",
              "patch_mode": "replace_lines", "proposed_scope": "line"}
    reparsed, raw2 = mt._repair_before_block(_REPAIR_SCAFFOLD, parsed, lambda p: "garbage")
    assert reparsed is None


def test_repair_before_block_returns_none_when_llm_fails() -> None:
    parsed = {"target_section": "Hardening", "before": "x", "after": "y",
              "patch_mode": "replace_lines", "proposed_scope": "line"}
    reparsed, raw2 = mt._repair_before_block(_REPAIR_SCAFFOLD, parsed, lambda p: None)
    assert reparsed is None


def test_evolution_repairs_hallucinated_before_anchor(isolated_paths, monkeypatch) -> None:
    """Phase 3.7 wiring: first response mis-quotes BEFORE (the dominant live
    reject, 5x in the journal); the loop sends ONE repair prompt carrying the
    targeted region's current text; the corrected patch applies."""
    _seed_multi_section_scaffold(isolated_paths)
    calls = {"n": 0}
    good = _apply_response(
        _MULTI_SECTION_SCAFFOLD,
        "Linear traces discharge with one line.",
        "Linear traces discharge silently.",
        tag="restatement_unpruned", failure="x",
        target_section="Threshold", scope="line",
    )
    bad = _schema_response(
        tag="restatement_unpruned", failure="x", target="Threshold",
        scope="line", patch_mode="replace_lines", target_lines="1",
        before="text that is not in the scaffold at all",
        after="Linear traces discharge silently.",
    )
    def llm(prompt):
        calls["n"] += 1
        return bad if calls["n"] == 1 else good
    monkeypatch.setattr(mt, "_call_llm", llm)
    entry = mt.maybe_evolve_after_rating("T-repair-ok", 40, ["restatement_unpruned"])
    assert calls["n"] == 2
    assert entry["applied"] is True
    assert "discharge silently" in isolated_paths["scaffold"].read_text(encoding="utf-8")


def test_evolution_repair_exhausted_journals_repair_failed(isolated_paths, monkeypatch) -> None:
    """If the single repair attempt also mis-quotes, journal the original
    reject with :repair_failed — no second retry, no loop."""
    _seed_multi_section_scaffold(isolated_paths)
    calls = {"n": 0}
    bad = _schema_response(
        tag="restatement_unpruned", failure="x", target="Threshold",
        scope="line", patch_mode="replace_lines", target_lines="1",
        before="still not in the scaffold", after="whatever",
    )
    def llm(prompt):
        calls["n"] += 1
        return bad
    monkeypatch.setattr(mt, "_call_llm", llm)
    entry = mt.maybe_evolve_after_rating("T-repair-fail", 40, ["restatement_unpruned"])
    assert calls["n"] == 2
    assert entry["applied"] is False
    assert entry["reject_reason"] == "before_block_mismatch:not_found:repair_failed"
    assert isolated_paths["scaffold"].read_text(encoding="utf-8") == _MULTI_SECTION_SCAFFOLD
