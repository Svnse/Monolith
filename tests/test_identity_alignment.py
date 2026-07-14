from __future__ import annotations

import pytest

from core import identity_alignment as al

_CORPUS = "Precision over fluency. Naming over narration. Provenance over assertion."


# ── lexical alignment (overlap coefficient over the text's content tokens) ──

def test_alignment_zero_when_no_overlap() -> None:
    assert al.compute_identity_alignment("quantum chromodynamics", _CORPUS, backend="lexical") == 0.0


def test_alignment_full_when_all_content_tokens_present() -> None:
    assert al.compute_identity_alignment("precision fluency", _CORPUS, backend="lexical") == 1.0


def test_alignment_partial_overlap() -> None:
    # content tokens {precision, rigor}; only precision in corpus -> 1/2
    assert al.compute_identity_alignment("precision rigor", _CORPUS, backend="lexical") == 0.5


def test_alignment_ignores_stopwords() -> None:
    # 'the','of','a','over' are stopwords; only 'precision' counts and is present
    assert al.compute_identity_alignment("the of a precision", _CORPUS, backend="lexical") == 1.0


def test_alignment_all_stopwords_is_zero_not_crash() -> None:
    assert al.compute_identity_alignment("the of a over", _CORPUS, backend="lexical") == 0.0


# ── confidentity = confidence × identity_alignment, Mad-Cow gradient ──

def test_self_provenance_confidentity_capped_at_half() -> None:
    row = {"canonical": "precision fluency", "provenance": "self", "reinforcement": 50}
    c = al.score_confidentity(row, _CORPUS, backend="lexical")
    assert c == pytest.approx(0.5)  # prov 0.5 × sat 1.0 × align 1.0


def test_user_provenance_confidentity_full() -> None:
    row = {"canonical": "precision fluency", "provenance": "user", "reinforcement": 50}
    assert al.score_confidentity(row, _CORPUS, backend="lexical") == pytest.approx(1.0)


def test_user_outranks_self_at_equal_reinforcement() -> None:
    self_row = {"canonical": "precision fluency", "provenance": "self", "reinforcement": 5}
    user_row = {"canonical": "precision fluency", "provenance": "user", "reinforcement": 5}
    assert al.score_confidentity(user_row, _CORPUS, backend="lexical") > al.score_confidentity(self_row, _CORPUS, backend="lexical")


def test_confidentity_clamped_to_unit_interval() -> None:
    row = {"canonical": "precision fluency", "provenance": "user", "reinforcement": 999}
    assert 0.0 <= al.score_confidentity(row, _CORPUS, backend="lexical") <= 1.0


def test_confidentity_zero_when_unaligned() -> None:
    row = {"canonical": "quantum chromodynamics", "provenance": "user", "reinforcement": 50}
    assert al.score_confidentity(row, _CORPUS, backend="lexical") == 0.0


# ── pluggable backend seam ──

def test_default_backend_is_lexical_when_flag_off(monkeypatch) -> None:
    monkeypatch.delenv("MONOLITH_IDENTITY_ALIGN_EMBED", raising=False)
    assert al.resolve_backend(None) == "lexical"


def test_flag_selects_embed_backend(monkeypatch) -> None:
    monkeypatch.setenv("MONOLITH_IDENTITY_ALIGN_EMBED", "1")
    assert al.resolve_backend(None) == "embed"


def test_embed_backend_falls_back_to_lexical_on_failure(monkeypatch) -> None:
    def _boom(text, corpus):
        raise RuntimeError("model unavailable")
    monkeypatch.setattr(al, "_align_embed", _boom)
    got = al.compute_identity_alignment("precision fluency", _CORPUS, backend="embed")
    assert got == al.compute_identity_alignment("precision fluency", _CORPUS, backend="lexical")


# ── stability_score (M3 unify: disposition discriminator) ──

def test_stability_fresh_l1_is_below_threshold() -> None:
    s = al.stability_score({"l_level": "L1", "reinforcement": 1})
    assert s == pytest.approx(0.20)
    assert s < al.STABILITY_THRESHOLD  # fresh -> curiosity disposition


def test_stability_reinforced_is_above_threshold() -> None:
    s = al.stability_score({"l_level": "L1", "reinforcement": 5})
    assert s == pytest.approx(0.68)
    assert s >= al.STABILITY_THRESHOLD  # reinforced -> consolidate disposition


def test_stability_l3_is_above_threshold() -> None:
    assert al.stability_score({"l_level": "L3", "reinforcement": 1}) == pytest.approx(0.52)


def test_stability_truth_confirmed_bonus() -> None:
    assert al.stability_score({"l_level": "L1", "reinforcement": 1, "truth": "confirmed"}) == pytest.approx(0.30)


def test_stability_clamped_to_unit() -> None:
    assert al.stability_score({"l_level": "L3", "reinforcement": 999, "truth": "confirmed"}) <= 1.0
