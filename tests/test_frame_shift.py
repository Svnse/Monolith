"""Tests for the frame_shift consecutive-frame change detector (Phase 1)."""
import json
import importlib

from addons.system.bearing import frame_shift as fs


# ── pure classify() ──────────────────────────────────────────────────────────

def test_identical_frames_hold():
    o = fs.classify("explaining why a gravity misconception is irrelevant",
                    "explaining why a gravity misconception is irrelevant")
    assert o.verdict == "HOLD"
    assert o.sim == 1.0
    assert o.confidence == 1.0


def test_disjoint_frames_shift():
    o = fs.classify("writing the authentication module",
                    "cooking pasta for dinner tonight")
    assert o.verdict == "SHIFT"
    assert o.sim == 0.0
    assert o.confidence == 1.0


def test_partial_overlap_is_ambig():
    # content tokens: {diagnosing, slow, database, query, performance} vs
    #                 {diagnosing, slow, network, latency, problem}
    # intersection 2 / union 8 = 0.25  -> AMBIG band [0.2, 0.6)
    o = fs.classify("diagnosing the slow database query performance",
                    "diagnosing the slow network latency problem")
    assert o.verdict == "AMBIG"
    assert 0.2 <= o.sim < 0.6
    assert o.confidence == 0.0


def test_empty_prev_reads_as_shift():
    # establishing a frame from an empty bearing is a SHIFT (sim 0)
    o = fs.classify("", "recommending a monolith for a 2-person side project")
    assert o.verdict == "SHIFT"
    assert o.sim == 0.0


def test_both_empty_is_hold_safe():
    # no content either side -> jaccard 0 -> SHIFT verdict, but never raises
    o = fs.classify("", "")
    assert o.verdict in ("SHIFT", "HOLD", "AMBIG")
    assert o.prev_tokens == 0 and o.new_tokens == 0


def test_threshold_boundaries_are_inclusive_exclusive():
    # content tokens must be len>=3, so use real words.
    # {alpha,bravo,charlie,delta} vs {alpha,bravo,charlie,echo}:
    # intersection 3 / union 5 = 0.6 exactly -> HOLD (inclusive boundary).
    hold = fs.classify("alpha bravo charlie delta", "alpha bravo charlie echo")
    assert hold.sim == 0.6 and hold.verdict == "HOLD"
    # one shared content word out of eleven -> ~0.09 < 0.2 -> SHIFT
    shift = fs.classify("alpha bravo charlie delta echo foxtrot",
                        "alpha sierra tango uniform victor whiskey")
    assert shift.verdict == "SHIFT"


# ── record() write-gating ────────────────────────────────────────────────────

def test_record_noop_when_flag_off(monkeypatch, tmp_path):
    monkeypatch.delenv("MONOLITH_FRAME_SHIFT_V1", raising=False)
    led = tmp_path / "frame_shift.ledger.jsonl"
    monkeypatch.setattr(fs, "_LEDGER", led)
    out = fs.record("t1", "old frame", "new frame entirely")
    assert out is None
    assert not led.exists()


def test_record_writes_when_flag_on(monkeypatch, tmp_path):
    monkeypatch.setenv("MONOLITH_FRAME_SHIFT_V1", "1")
    led = tmp_path / "frame_shift.ledger.jsonl"
    monkeypatch.setattr(fs, "_LEDGER", led)
    out = fs.record("turn-abc", "writing the auth module",
                    "cooking pasta tonight", turn_n=7,
                    source="frame_heartbeat", session_id="ui:llm_x")
    assert out is not None and out.verdict == "SHIFT"
    rows = [json.loads(l) for l in led.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 1
    r = rows[0]
    # every field E's spec requires is present
    for k in ("turn_id", "turn_n", "previous_frame", "new_frame",
              "lexical_sim", "verdict", "confidence", "source", "session_id"):
        assert k in r, f"missing field {k}"
    assert r["turn_n"] == 7
    assert r["verdict"] == "SHIFT"
    assert r["source"] == "frame_heartbeat"
    assert r["session_id"] == "ui:llm_x"


def test_record_noop_on_empty_new_frame(monkeypatch, tmp_path):
    monkeypatch.setenv("MONOLITH_FRAME_SHIFT_V1", "1")
    led = tmp_path / "frame_shift.ledger.jsonl"
    monkeypatch.setattr(fs, "_LEDGER", led)
    assert fs.record("t1", "some prior frame", "   ") is None
    assert not led.exists()


# ── replay helper ────────────────────────────────────────────────────────────

def test_replay_produces_n_minus_1_verdicts():
    frames = [
        "solving a riddle variant",
        "explaining how to isolate a 502 cause",       # SHIFT
        "explaining how to isolate a 502 cause",       # HOLD
        "recommending a monolith for a side project",  # SHIFT
    ]
    obs = fs.replay_frames(frames)
    assert len(obs) == 3
    assert [o.verdict for o in obs] == ["SHIFT", "HOLD", "SHIFT"]
