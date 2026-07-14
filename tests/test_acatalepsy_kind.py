"""Tests for the deterministic Kind classifier (B1) — the gate that decides
which validation rules apply. Kinds: world-fact | self | meta | causal | emotional.
This is a K1/K2 floor; resolution order emotional > causal > meta > self > world-fact.
"""
from core.acatalepsy.kind import classify_kind, KINDS
from core.acatalepsy.normalize import parse_triple


def _k(form: str) -> str:
    return classify_kind(parse_triple(form))


def test_emotional_from_affect_relation():
    assert _k("user | dislikes | safetywrapper") == "emotional"
    assert _k("e | prefers | gemini") == "emotional"


def test_causal_from_causal_relation():
    assert _k("dislike | influences | preference") == "causal"
    assert _k("restriction | causes | reduced approval") == "causal"


def test_meta_from_normative_relation():
    assert _k("auditor | requires | evidence") == "meta"
    assert _k("claims | should | be atomic") == "meta"


def test_self_from_system_subject():
    assert _k("monolith | uses | seven effort tiers") == "self"
    assert _k("core/effort.py | defines | seven tiers") == "self"
    assert _k("substrate | carries | operator frame state") == "self"


def test_world_fact_default():
    assert _k("trump | president_of | usa") == "world-fact"
    assert _k("paris | capital_of | france") == "world-fact"


def test_precedence_emotional_over_self():
    # subject is a system entity but the relation is affective -> emotional wins
    assert _k("operator | dislikes | the kernel") == "emotional"


def test_generic_words_do_not_misroute_world_facts_to_self():
    # "core"/"engine" are generic — a world-fact subject containing them must NOT
    # become `self` (which would block Tavily grounding in Phase 3).
    assert _k("earth core | composed_of | iron") == "world-fact"
    assert _k("happy.python | is | a library") == "world-fact"


def test_none_triple_defaults_to_self():
    assert classify_kind(None) == "self"


def test_returns_a_known_kind():
    for form in ["a | r | b", "user | loves | x", "y | causes | z", "m | must | n"]:
        assert classify_kind(parse_triple(form)) in KINDS
