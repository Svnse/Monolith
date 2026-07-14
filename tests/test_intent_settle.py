"""Tests for core/intent_settle — Path B SET-MEMBERSHIP settlement.

The KEYSTONE tests (test_redirect_high_overlap, test_redirect_long_answer) are
the point of attempt 2: they make the v1 failure (a reply-friction scorer blind
to the on-topic redirect) impossible. Do NOT weaken them — fix the settler.
"""
from __future__ import annotations

from core import intent_extract as ie
from core import intent_settle as isettle
from core import friction_store as fs


# ── KEYSTONE 1: redirect fires despite HIGH lexical overlap ────────────


def test_redirect_high_overlap():
    """E stays on-subject (echoes staked words: token/rotation/session) but pulls
    a NEW dominant focus (blast radius / downstream consumers). A reply-friction
    overlap scorer reads this as uptake/unresolved (~0.45) — the v1 failure. The
    membership settler must read REDIRECT because the salient content is mostly
    unstaked, even though staked words ARE present (high overlap)."""
    frozen = {"directions": [{"move": "plan", "referent": "token"}],
              "referents": ["token", "rotation", "session", "expiry", "auth"],
              "source": "floor"}
    reply = ("blast radius downstream consumers — yes the token rotation session "
             "matters, but what's the blast radius on the downstream consumers "
             "and the rollout impact?")
    res = isettle.settle(frozen, reply)
    assert res.friction_type == "redirect", res.channel_json
    assert res.friction_score >= 0.6
    # prove overlap WAS high (staked words present) yet it still settled redirect:
    assert res.channel_json["refs_in"], "test must exercise the high-overlap case"
    assert len(res.channel_json["refs_out"]) > len(res.channel_json["refs_in"])


# ── KEYSTONE 2: redirect fires even for a LONG (broad) answer ──────────


def test_redirect_long_answer():
    """A long answer stakes many referents -> broad set. The capped-salient frozen
    set keeps it tight, so an on-topic reply pulling an UNSTAKED referent still
    settles redirect (proves the broad-set cap works)."""
    long_answer = ("Here is the full plan for session token rotation. " * 6 +
                   "We rotate the token, refresh the session, manage expiry, and "
                   "version the rotation policy across the token store. " * 4)
    frozen = ie.mine_staked(long_answer)
    assert len(frozen["referents"]) <= ie._REFERENT_CAP   # capped despite length
    reply = ("what about the downstream blast radius on the consumers and the "
             "migration rollout if we touch the downstream consumers?")
    res = isettle.settle(frozen, reply)
    assert res.friction_type == "redirect", res.channel_json
    assert res.friction_score >= 0.6


# ── uptake / markers / unresolved ─────────────────────────────────────


def test_uptake_builds_on_staked():
    frozen = {"directions": [{"move": "plan", "referent": "token"}],
              "referents": ["token", "rotation", "session", "windows"], "source": "floor"}
    reply = "great, for the token rotation plan let's design the sliding session windows"
    res = isettle.settle(frozen, reply)
    assert res.friction_type == "uptake", res.channel_json
    assert res.friction_score <= 0.3


def test_markers_override_membership():
    """An explicit correction wins regardless of referent membership."""
    frozen = {"directions": [{"move": "plan", "referent": "token"}],
              "referents": ["token", "rotation"], "source": "floor"}
    res = isettle.settle(frozen, "no, that's wrong — the token rotation isn't the point")
    assert res.friction_type == "correction"
    assert res.friction_score >= 0.9


def test_unresolved_terse_no_confirm():
    """Terse reply -> unresolved, NOT uptake, NOT a confidence raise (asymmetric
    trust: silence is not confirmation, spec §5)."""
    frozen = {"directions": [{"move": "plan", "referent": "token"}],
              "referents": ["token", "rotation"], "source": "floor"}
    res = isettle.settle(frozen, "ok")
    assert res.friction_type == "unresolved"
    assert res.friction_score < 0.6


def test_v1_row_none_set_unresolved():
    """A v1 prediction with no frozen set settles unresolved, never crashes."""
    res = isettle.settle(None, "anything at all about tokens and rotation here")
    assert res.friction_type == "unresolved"


def test_friction_type_in_closed_enum():
    frozen = {"referents": ["token"], "directions": []}
    for reply in ("no wrong", "blast radius downstream new topic entirely here",
                  "ok", "the token rotation is great let's proceed with token"):
        assert isettle.settle(frozen, reply).friction_type in isettle.SETTLE_TYPES


# ── interceptor: flag-gating, peer, once-per-outer-turn ────────────────


def _db(tmp_path):
    return tmp_path / "tt.sqlite3"


def test_interceptor_flag_off_noop(tmp_path, monkeypatch):
    monkeypatch.delenv(fs._FLAG_ENV, raising=False)
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "a"},
            {"role": "user", "content": "no wrong"}]
    out = isettle.friction_settle_interceptor(list(msgs), {})
    assert out == msgs  # unchanged, no crash


def test_interceptor_peer_turn_noop(tmp_path, monkeypatch):
    monkeypatch.setenv(fs._FLAG_ENV, "1")
    monkeypatch.setattr(fs, "_DB_PATH", _db(tmp_path))
    fs.record_prediction("t1", 1, "intent", "c", "f", 0.5, "next_turn",
                         now_iso="2026-06-22T00:00:00",
                         prediction_set_json={"referents": ["token"], "directions": []},
                         db_path=_db(tmp_path))
    msgs = [{"role": "assistant", "content": "a"},
            {"role": "user", "content": "[CHANNEL: connect/Codex] no wrong"}]
    isettle.friction_settle_interceptor(msgs, {"_now_iso": "2026-06-22T01:00:00"})
    # peer turn -> the open prediction is NOT settled
    assert fs.latest_open("intent", db_path=_db(tmp_path)) is not None


def test_interceptor_settles_prior_turn(tmp_path, monkeypatch):
    monkeypatch.setenv(fs._FLAG_ENV, "1")
    db = _db(tmp_path)
    monkeypatch.setattr(fs, "_DB_PATH", db)
    fs.record_prediction("t1", 1, "intent", "wants a plan", "pivots",
                         0.5, "next_turn", now_iso="2026-06-22T00:00:00",
                         prediction_set_json={"referents": ["token", "rotation", "session"],
                                              "directions": [{"move": "plan", "referent": "token"}]},
                         db_path=db)
    msgs = [{"role": "assistant", "content": "answer about token rotation"},
            {"role": "user", "content": "blast radius downstream consumers rollout migration impact"}]
    # current outer turn is AFTER the prediction's created_at -> settle fires
    isettle.friction_settle_interceptor(msgs, {"_now_iso": "2026-06-22T02:00:00", "_turn_id": "t2"})
    settled = fs.recent_settled(db_path=db)
    assert len(settled) == 1
    assert settled[0]["friction_type"] == "redirect"


def test_interceptor_skips_same_outer_turn(tmp_path, monkeypatch):
    """A prediction created in THIS outer turn (tool-loop followup) is NOT settled
    — once-per-outer-turn idempotency keyed on TurnClock _now_iso."""
    monkeypatch.setenv(fs._FLAG_ENV, "1")
    db = _db(tmp_path)
    monkeypatch.setattr(fs, "_DB_PATH", db)
    # prediction created AT the current outer-turn instant (same turn)
    fs.record_prediction("t1", 1, "intent", "c", "f", 0.5, "next_turn",
                         now_iso="2026-06-22T02:00:00",
                         prediction_set_json={"referents": ["token"], "directions": []},
                         db_path=db)
    msgs = [{"role": "assistant", "content": "a"},
            {"role": "user", "content": "blast radius downstream new"}]
    isettle.friction_settle_interceptor(msgs, {"_now_iso": "2026-06-22T02:00:00"})
    # created_at >= now -> skipped, still open
    assert fs.latest_open("intent", db_path=db) is not None
    assert fs.recent_settled(db_path=db) == []
