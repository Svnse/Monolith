"""Tests for addons.system.bearing.frame_fidelity_jobs — exact-once delivery.

Every committed frame SYNCHRONOUSLY queues a fidelity job. The async judge may
complete it later, but the job cannot disappear silently: a turn without a verdict
has a visible job state. On restart, incomplete jobs are retried. Duplicates
(same frame_record_hash + answer_digest) are rejected.

Append-only event log: current state of a job = its latest row.
"""
from __future__ import annotations

import json

import pytest


def _setup(monkeypatch, tmp_path):
    from addons.system.bearing import frame_fidelity_jobs as jb
    monkeypatch.setenv("MONOLITH_MONOFRAME_V1", "1")
    monkeypatch.setattr(jb, "_STORE", tmp_path / "frame_fidelity_jobs.jsonl")
    return jb


class TestEnqueue:
    def test_enqueue_creates_queued_job(self, monkeypatch, tmp_path):
        jb = _setup(monkeypatch, tmp_path)
        job_id, created = jb.enqueue_job(
            turn_id="t-1", frame_record_hash="fh", answer_digest="ad",
            judge_version="v1", answer="the answer", now="2026-01-01T00:00:00Z",
        )
        assert created is True
        job = jb.get_job(job_id)
        assert job["status"] == "queued"
        assert job["turn_id"] == "t-1"
        assert job["queued_at_utc"] == "2026-01-01T00:00:00Z"
        assert job["retry_count"] == 0

    def test_duplicate_same_hash_and_digest_is_rejected(self, monkeypatch, tmp_path):
        jb = _setup(monkeypatch, tmp_path)
        a = jb.enqueue_job(turn_id="t", frame_record_hash="fh", answer_digest="ad", judge_version="v1")
        b = jb.enqueue_job(turn_id="t", frame_record_hash="fh", answer_digest="ad", judge_version="v1")
        assert a[0] == b[0]          # same deterministic job_id
        assert a[1] is True and b[1] is False   # second not re-created


class TestTurnAccounting:
    def test_recorder_failure_is_a_failed_job_not_skipped(self, monkeypatch, tmp_path):
        jb = _setup(monkeypatch, tmp_path)
        # an answering turn whose recorder produced NO frame_selection block
        jid, created = jb.enqueue_recorder_failure(
            turn_id="t-ans", answer_digest="ad", now="2026-01-01T00:00:00Z",
        )
        assert created is True
        j = jb.get_job(jid)
        assert j["status"] == "failed"
        assert j["error"] == "no_frame_selection_block"
        assert j["verdict_possible"] is False
        assert j["reason"] == "recorder_failed_before_judge"
        assert j["turn_id"] == "t-ans"
        assert j["answer_digest"] == "ad"

    def test_recorder_failure_dedups(self, monkeypatch, tmp_path):
        jb = _setup(monkeypatch, tmp_path)
        a = jb.enqueue_recorder_failure(turn_id="t", answer_digest="ad")
        b = jb.enqueue_recorder_failure(turn_id="t", answer_digest="ad")
        assert a[0] == b[0] and a[1] is True and b[1] is False

    def test_skipped_is_only_for_non_answer_turns(self, monkeypatch, tmp_path):
        jb = _setup(monkeypatch, tmp_path)
        jid, created = jb.enqueue_skipped(turn_id="t-tool", reason="non_answer_turn")
        assert created is True
        j = jb.get_job(jid)
        assert j["status"] == "skipped"
        assert j["reason"] == "non_answer_turn"
        assert j["verdict_possible"] is False


class TestTransitions:
    def test_running_then_complete(self, monkeypatch, tmp_path):
        jb = _setup(monkeypatch, tmp_path)
        jid, _ = jb.enqueue_job(turn_id="t", frame_record_hash="fh", answer_digest="ad", judge_version="v1")
        jb.mark_running(jid, now="t1")
        assert jb.get_job(jid)["status"] == "running"
        assert jb.get_job(jid)["started_at_utc"] == "t1"
        jb.mark_complete(jid, output_fidelity_record_hash="oh", now="t2")
        j = jb.get_job(jid)
        assert j["status"] == "complete"
        assert j["completed_at_utc"] == "t2"
        assert j["output_fidelity_record_hash"] == "oh"

    def test_failed_carries_error(self, monkeypatch, tmp_path):
        jb = _setup(monkeypatch, tmp_path)
        jid, _ = jb.enqueue_job(turn_id="t", frame_record_hash="fh", answer_digest="ad", judge_version="v1")
        jb.mark_failed(jid, error="boom", now="t1")
        assert jb.get_job(jid)["status"] == "failed"
        assert jb.get_job(jid)["error"] == "boom"


class TestIncomplete:
    def test_incomplete_lists_queued_and_running_only(self, monkeypatch, tmp_path):
        jb = _setup(monkeypatch, tmp_path)
        q, _ = jb.enqueue_job(turn_id="q", frame_record_hash="1", answer_digest="a", judge_version="v1")
        r, _ = jb.enqueue_job(turn_id="r", frame_record_hash="2", answer_digest="a", judge_version="v1")
        jb.mark_running(r)
        c, _ = jb.enqueue_job(turn_id="c", frame_record_hash="3", answer_digest="a", judge_version="v1")
        jb.mark_complete(c, output_fidelity_record_hash="oh")
        ids = {j["job_id"] for j in jb.incomplete_jobs()}
        assert ids == {q, r}     # complete excluded


class TestRunAndRecover:
    def test_run_job_marks_complete_and_writes_fidelity(self, monkeypatch, tmp_path):
        jb = _setup(monkeypatch, tmp_path)
        from addons.system.bearing import frame_fidelity as ff
        monkeypatch.setattr(ff, "_STORE", tmp_path / "frame_fidelity.jsonl")
        fr = {"turn_id": "t", "selected_frame": "s", "rejected_runner_up": "r", "artifact_hash": "fh"}
        jid, _ = jb.enqueue_job(turn_id="t", frame_record_hash="fh", answer_digest="ad", judge_version="v1", answer="the answer")
        ok = jb.run_job(jid, frame_record=fr, answer="the answer",
                        generate=lambda m: "VERDICT: pass\nBETRAYAL: none")
        assert ok is True
        assert jb.get_job(jid)["status"] == "complete"
        assert jb.get_job(jid)["output_fidelity_record_hash"]
        fid = [json.loads(l) for l in (tmp_path / "frame_fidelity.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(fid) == 1 and fid[0]["verdict"] == "pass"

    def test_recover_reruns_incomplete_jobs(self, monkeypatch, tmp_path):
        jb = _setup(monkeypatch, tmp_path)
        from addons.system.bearing import frame_fidelity as ff, frame_selection as fs
        monkeypatch.setattr(ff, "_STORE", tmp_path / "frame_fidelity.jsonl")
        # a committed frame in the selection store, so recovery can re-fetch it by hash
        sel = tmp_path / "frame_selection.jsonl"
        monkeypatch.setattr(fs, "_STORE", sel)
        sel.write_text(json.dumps({
            "turn_id": "t", "selected_frame": "s", "rejected_runner_up": "r", "artifact_hash": "fh",
        }) + "\n", encoding="utf-8")
        # an orphaned queued job (judge never ran — e.g. a crash)
        jid, _ = jb.enqueue_job(turn_id="t", frame_record_hash="fh", answer_digest="ad", judge_version="v1", answer="the answer")
        assert jb.get_job(jid)["status"] == "queued"
        n = jb.recover_incomplete(generate=lambda m: "VERDICT: pass\nBETRAYAL: none")
        assert n == 1
        assert jb.get_job(jid)["status"] == "complete"

    def test_recover_once_is_idempotent_per_process(self, monkeypatch, tmp_path):
        jb = _setup(monkeypatch, tmp_path)
        monkeypatch.setattr(jb, "_recovered", False)
        # no incomplete jobs -> 0 dispatched, but the guard still trips
        assert jb.recover_once() == 0
        assert jb.recover_once() == -1   # second call this process is a no-op
