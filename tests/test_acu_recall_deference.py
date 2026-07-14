"""Unit 1 of the belief-ledger deference loop: make recall authority-FIRST and
grade the recalled block so the model defers to established (locked/verified)
beliefs over its in-context reconstruction, while keeping low-authority claims
advisory. Plus a recall hit-summary for the deference hit-log.

Scoped per the plan: AU4 (locked) + AU3 (truth-confirmed, fresh) are "established"
and override; AU2 stays advisory; AU1 (L1 stubs / -inf falsehoods) is not recalled.
"""
from __future__ import annotations

from core.acu_retrieval import _score_acu, _tokenize, format_recall_block, recall_hit_summary


# ── authority-FIRST ranking ──────────────────────────────────────────────────


def test_authority_dominates_relevance():
    # A locked (AU4) claim with LOW token overlap must outrank a high-overlap AU2.
    locked = {"canonical": "the user prefers tabs", "locked": 1}                       # AU4
    au2 = {"canonical": "tabs spaces indentation editor vscode settings keymap", "l_level": "L2"}  # AU2, high overlap
    ptoks = _tokenize("does the user prefer tabs or spaces indentation editor vscode settings keymap")
    assert _score_acu(locked, ptoks) > _score_acu(au2, ptoks)


def test_au1_stub_not_recall_eligible():
    l1 = {"canonical": "tabs spaces editor", "l_level": "L1"}   # AU1 stored-only
    ptoks = _tokenize("tabs spaces editor")
    assert _score_acu(l1, ptoks) == 0.0


def test_no_overlap_not_recalled():
    locked = {"canonical": "unrelated locked belief", "locked": 1}
    ptoks = _tokenize("a completely different question about weather")
    assert _score_acu(locked, ptoks) == 0.0


# ── graded recall block (the deference contract) ─────────────────────────────


def test_block_grades_established_with_override_contract():
    locked = {"canonical": "user prefers tabs", "locked": 1}        # AU4 -> [LOCKED]
    au2 = {"canonical": "editor is vscode here", "l_level": "L2"}   # AU2 -> advisory
    block = format_recall_block([locked, au2])
    low = block.lower()
    assert "[LOCKED]" in block
    assert "contradicting evidence" in low          # established override contract present
    assert "advisory" in low                          # advisory framing present
    assert "user prefers tabs" in block


def test_confirmed_fact_labeled_verified_not_provisional():
    # AU3: truth-confirmed + fresh. Must render [VERIFIED] (not [PROVISIONAL] —
    # the old label_for_claim read the dead `veracity` and never showed VERIFIED).
    from datetime import datetime, timezone
    fresh = datetime.now(timezone.utc).isoformat()
    confirmed = {"canonical": "paris is the capital of france", "truth": "confirmed",
                 "truth_checked_at": fresh, "state": "active"}
    block = format_recall_block([confirmed])
    assert "[VERIFIED]" in block


# ── recall burial fix (lever 1 stopword-gate + lever 3 identity-seed exclude) ─


def test_identity_seed_excluded_from_recall():
    # The 12 Origin-0 identity claims are AU4 and overlap almost any prompt via
    # stopwords, so they filled all 5 recall slots and buried every AU3/AU2. They
    # are carried to the model by the {identity_block} anchor, NOT recall — so the
    # task-recall pool must exclude them. (Unit-1 burial fix.)
    seed = {"canonical": "monolith is a persistent operating-system identity",
            "locked": 1, "source": "identity_origin_0", "lock_reason": "origin_0"}  # AU4
    ptoks = _tokenize("what is monolith's persistent operating-system identity model")
    assert _score_acu(seed, ptoks) == 0.0


def test_stopword_only_overlap_not_recalled():
    # Sharing ONLY function words ("is"/"a") is not relevance — that is exactly how
    # identity claims matched every prompt. A claim whose only overlap is stopwords
    # must score 0 regardless of authority.
    locked = {"canonical": "precision over fluency is a core value", "locked": 1}  # AU4
    ptoks = _tokenize("what port is the database on for a connection")  # shares only is/a
    assert _score_acu(locked, ptoks) == 0.0


def test_au3_confirmed_surfaces_with_meaningful_overlap():
    # The tier the burial blocked: an AU3 truth-confirmed belief with a CONTENT
    # match must be recall-eligible (score > 0) once the floods are gone.
    from datetime import datetime, timezone
    fresh = datetime.now(timezone.utc).isoformat()
    au3 = {"canonical": "our postgres database listens on port 6543",
           "truth": "confirmed", "truth_checked_at": fresh, "state": "active"}  # AU3
    ptoks = _tokenize("what port does our postgres database listen on for connections")
    assert _score_acu(au3, ptoks) > 0.0


def test_user_locked_non_identity_override_preserved():
    # The proven 2-space deference must NOT break: a genuine USER-locked claim
    # (lock_reason != origin_0, source != identity_origin_0) still surfaces.
    locked = {"canonical": "user prefers 2-space python indentation", "locked": 1,
              "lock_reason": "deference_probe", "source": "user"}  # AU4, not the seed
    ptoks = _tokenize("how many spaces of python indentation should i use here")
    assert _score_acu(locked, ptoks) > 0.0


# ── recall hit-summary (for the deference hit-log) ───────────────────────────


def test_recall_hit_summary_counts_and_lists_established():
    locked = {"canonical": "a b c", "locked": 1}        # AU4
    au2 = {"canonical": "d e f", "l_level": "L2"}        # AU2
    s = recall_hit_summary([locked, au2])
    assert s["n"] == 2
    assert s["by_authority"][4] == 1
    assert s["by_authority"][2] == 1
    assert "a b c" in s["established"]
    assert "d e f" not in s["established"]
