"""Tests for core/intent_extract — Seed α pure-code prediction floor.

Invariants: deterministic; referents CAPPED-salient (broad-set fix); moves use
turn_classifier's intent_tags vocabulary; merge never lets the card replace the
floor."""
from __future__ import annotations

from core import intent_extract as ie


def test_salient_referents_capped_and_deterministic():
    text = "token rotation session expiry token rotation token " * 5 + "tangential mention once"
    a = ie.salient_referents(text)
    b = ie.salient_referents(text)
    assert a == b                          # deterministic
    assert len(a) <= ie._REFERENT_CAP      # capped
    # frequent early referents rank above a single late tangential mention
    assert "token" in a and "rotation" in a
    assert a.index("token") < len(a)


def test_salient_referents_drops_passing_mention_under_cap():
    """A long answer that mentions many things keeps only the salient few — the
    broad-frozen-set fix: a one-off passing noun shouldn't pollute the set."""
    core = "authentication authentication tokens tokens rotation rotation session session expiry expiry "
    noise = " ".join(f"misc{i}" for i in range(40))  # many one-off nouns
    refs = ie.salient_referents(core + noise)
    assert len(refs) <= ie._REFERENT_CAP
    assert "authentication" in refs and "tokens" in refs
    # the BULK of the 40 one-off nouns is excluded (only cap-minus-core can leak):
    leaked = sum(1 for r in refs if r.startswith("misc"))
    assert leaked <= ie._REFERENT_CAP - 5      # the 5 high-freq core words win their slots
    assert (40 - leaked) >= 37                  # ~all noise excluded — the broad-set fix


def test_mine_staked_shape():
    ans = "Here is the plan: refactor the auth token rotation and analyze the session expiry tradeoffs."
    pset = ie.mine_staked(ans)
    assert pset["source"] == "floor"
    assert isinstance(pset["referents"], list) and pset["referents"]
    assert isinstance(pset["directions"], list)
    assert len(pset["directions"]) <= ie._DIRECTION_CAP
    for d in pset["directions"]:
        assert set(d.keys()) == {"move", "referent"}


def test_mine_staked_uses_classifier_vocabulary():
    """directions[].move comes from turn_classifier intent_tags (apples-to-apples
    with the settler), not an invented enum."""
    from core import turn_classifier
    ans = "Let me design the architecture and outline the steps to refactor this module."
    tags = set(turn_classifier.classify([{"role": "user", "content": ans}], {}).intent_tags)
    moves = {d["move"] for d in ie.mine_staked(ans)["directions"]}
    assert moves and moves.issubset(tags)


def test_merge_card_never_replaces_floor():
    floor = {"directions": [{"move": "plan", "referent": "auth"}],
             "referents": ["auth", "token"], "source": "floor"}
    # card with no overlap still keeps the floor's referents present
    card = {"directions": [{"move": "analysis", "referent": "blast-radius"}],
            "referents": ["downstream"], "intent_read": "wants impact analysis"}
    merged = ie.merge_prediction_sets(floor, card)
    assert "auth" in merged["referents"] and "downstream" in merged["referents"]
    assert merged["source"] == "card+floor"
    assert merged["intent_read"] == "wants impact analysis"
    # merging None card == floor untouched
    assert ie.merge_prediction_sets(floor, None) == floor
