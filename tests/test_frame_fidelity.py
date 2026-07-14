"""Tests for addons.system.bearing.frame_fidelity — the Frame Fidelity Judge.

COMMITMENT verification, not cognition: it judges only whether the final answer
HONORED the frame the recorder captured (selected_frame) or drifted to the
rejected runner-up / stale frame. Observational (no rewrite). First target = the
single failure class: selected vs runner-up/stale.

Output frame_fidelity.jsonl, bound to the pre-answer commitment by frame_record_hash.
"""
from __future__ import annotations

import hashlib
import json

import pytest


_JUDGE_PASS = (
    "VERDICT: pass\n"
    "BETRAYAL: none\n"
    "EVIDENCE: the answer explains hash joins operationally\n"
    "EXPLANATION: the answer followed the selected frame"
)
_JUDGE_FAIL = (
    "VERDICT: fail\n"
    "BETRAYAL: answered_runner_up_frame\n"
    "EVIDENCE: drifts into the philosophy of joins || never gives the operational definition\n"
    "EXPLANATION: the answer followed the rejected runner-up frame, not the selected one"
)


class TestParse:
    def test_parses_pass(self):
        from addons.system.bearing import frame_fidelity as ff
        out = ff.parse_fidelity(_JUDGE_PASS)
        assert out["verdict"] is ff.Verdict.PASS
        assert out["betrayal_type"] is ff.BetrayalType.NONE
        assert out["evidence_spans"] == ["the answer explains hash joins operationally"]
        assert "followed the selected" in out["explanation"]

    def test_parses_fail_with_multiple_evidence(self):
        from addons.system.bearing import frame_fidelity as ff
        out = ff.parse_fidelity(_JUDGE_FAIL)
        assert out["verdict"] is ff.Verdict.FAIL
        assert out["betrayal_type"] is ff.BetrayalType.ANSWERED_RUNNER_UP_FRAME
        assert len(out["evidence_spans"]) == 2

    def test_unknown_verdict_defaults_to_warn(self):
        from addons.system.bearing import frame_fidelity as ff
        out = ff.parse_fidelity("VERDICT: gibberish\nBETRAYAL: none")
        assert out["verdict"] is ff.Verdict.WARN  # conservative: unsure -> warn, not pass

    def test_unknown_betrayal_defaults_to_none(self):
        from addons.system.bearing import frame_fidelity as ff
        out = ff.parse_fidelity("VERDICT: pass\nBETRAYAL: weird_thing")
        assert out["betrayal_type"] is ff.BetrayalType.NONE


class TestJudge:
    def test_judge_runs_and_parses(self):
        from addons.system.bearing import frame_fidelity as ff
        out = ff.judge(
            selected_frame="answering a hash-join question operationally",
            rejected_runner_up="musing on the philosophy of joins",
            answer="A hash join builds a hash table on one input then probes it with the other.",
            generate=lambda msgs: _JUDGE_PASS,
        )
        assert out["verdict"] is ff.Verdict.PASS


class TestBuildRecord:
    def test_binds_frame_record_hash_and_answer_digest(self):
        from addons.system.bearing import frame_fidelity as ff, frame_selection as fs
        frame_record = {
            "turn_id": "t-1",
            "selected_frame": "answering operationally",
            "rejected_runner_up": "musing philosophically",
            "artifact_hash": "deadbeef" * 8,
        }
        rec = ff.build_fidelity_record(
            frame_record=frame_record,
            answer="operational answer here",
            generate=lambda msgs: _JUDGE_PASS,
        )
        d = rec.to_dict()
        for key in (
            "turn_id", "frame_record_hash", "answer_digest", "selected_frame",
            "rejected_runner_up", "verdict", "betrayal_type",
            "evidence_spans_from_answer", "explanation", "judge_version", "timestamp_utc",
        ):
            assert key in d, f"missing {key}"
        # binds the pre-answer commitment by hash
        assert d["frame_record_hash"] == "deadbeef" * 8
        assert d["answer_digest"] == fs.digest("operational answer here")
        assert d["verdict"] == "pass"
        assert d["turn_id"] == "t-1"


class TestRecorder:
    def test_records_jsonl_when_enabled(self, monkeypatch, tmp_path):
        from addons.system.bearing import frame_fidelity as ff
        monkeypatch.setenv("MONOLITH_MONOFRAME_V1", "1")
        store = tmp_path / "frame_fidelity.jsonl"
        monkeypatch.setattr(ff, "_STORE", store)
        wrote = ff.judge_turn(
            frame_record={
                "turn_id": "t-9", "selected_frame": "s", "rejected_runner_up": "r",
                "artifact_hash": "h",
            },
            answer="an answer",
            generate=lambda msgs: _JUDGE_FAIL,
        )
        assert wrote is True
        rows = [json.loads(l) for l in store.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(rows) == 1
        assert rows[0]["verdict"] == "fail"
        assert rows[0]["betrayal_type"] == "answered_runner_up_frame"

    def test_noop_when_disabled(self, monkeypatch, tmp_path):
        from addons.system.bearing import frame_fidelity as ff
        monkeypatch.delenv("MONOLITH_MONOFRAME_V1", raising=False)
        store = tmp_path / "frame_fidelity.jsonl"
        monkeypatch.setattr(ff, "_STORE", store)
        ff.judge_turn(
            frame_record={"turn_id": "t", "selected_frame": "s", "rejected_runner_up": "r", "artifact_hash": "h"},
            answer="a", generate=lambda msgs: _JUDGE_PASS,
        )
        assert not store.exists()

    def test_skips_when_no_committed_frame(self, monkeypatch, tmp_path):
        from addons.system.bearing import frame_fidelity as ff
        monkeypatch.setenv("MONOLITH_MONOFRAME_V1", "1")
        monkeypatch.setattr(ff, "_STORE", tmp_path / "frame_fidelity.jsonl")
        # no selected_frame -> nothing to judge fidelity against
        wrote = ff.judge_turn(
            frame_record={"turn_id": "t", "selected_frame": "", "rejected_runner_up": "", "artifact_hash": "h"},
            answer="a", generate=lambda msgs: _JUDGE_PASS,
        )
        assert wrote is False

    def test_async_runs_off_thread_and_records(self, monkeypatch, tmp_path):
        from addons.system.bearing import frame_fidelity as ff
        monkeypatch.setenv("MONOLITH_MONOFRAME_V1", "1")
        store = tmp_path / "frame_fidelity.jsonl"
        monkeypatch.setattr(ff, "_STORE", store)
        t = ff.judge_turn_async(
            frame_record={"turn_id": "t-a", "selected_frame": "s", "rejected_runner_up": "r", "artifact_hash": "h"},
            answer="a", generate=lambda msgs: _JUDGE_PASS,
        )
        assert t is not None
        t.join(timeout=5)
        rows = [json.loads(l) for l in store.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(rows) == 1 and rows[0]["turn_id"] == "t-a"
