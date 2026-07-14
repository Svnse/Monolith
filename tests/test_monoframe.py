"""Tests for addons.system.bearing.monoframe — MonoFrame Stage 1 pure core.

The stateless-second-opinion instrument: a momentum-free re-derivation of the
frame, compared to the production (stateful) frame. token_divergence powers both
the momentum signal (control-vs-clean) and the noise floor (clean-vs-clean').

Flag: MONOLITH_MONOFRAME_V1 (observe-only; nothing here mutates bearing).
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# token_divergence — pure, symmetric, 0.0 (identical) .. 1.0 (disjoint)
# ---------------------------------------------------------------------------

class TestTokenDivergence:
    def test_identical_frames_zero_divergence(self):
        from addons.system.bearing import monoframe
        d = monoframe.token_divergence(
            "answering the mutex question", "answering the mutex question"
        )
        assert d == 0.0

    def test_disjoint_frames_full_divergence(self):
        from addons.system.bearing import monoframe
        d = monoframe.token_divergence(
            "explaining bloom filters", "ratifying autonomous vehicle ethics"
        )
        assert d == 1.0

    def test_partial_overlap_between_zero_and_one(self):
        from addons.system.bearing import monoframe
        # share "mutex" only; "answering/question" vs "explaining/locking"
        d = monoframe.token_divergence(
            "answering the mutex question", "explaining mutex locking"
        )
        assert 0.0 < d < 1.0

    def test_symmetric(self):
        from addons.system.bearing import monoframe
        a, b = "explaining bloom filters", "filters and hashing explained"
        assert monoframe.token_divergence(a, b) == monoframe.token_divergence(b, a)

    def test_both_empty_is_zero(self):
        from addons.system.bearing import monoframe
        assert monoframe.token_divergence("", "") == 0.0

    def test_one_empty_is_full_divergence(self):
        from addons.system.bearing import monoframe
        assert monoframe.token_divergence("", "explaining bloom filters") == 1.0

    def test_ignores_case_and_stopwords(self):
        from addons.system.bearing import monoframe
        # only content word is "bloom"/"filters"; stopwords + case differ
        d = monoframe.token_divergence("The Bloom Filters", "bloom filters")
        assert d == 0.0
