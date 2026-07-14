"""Tests for addons.system.bearing.frame_selection — the standing recorder.

A TRACE CONTRACT, not a cognition probe: it records the frame-selection Monolith
COMMITS TO before answering (candidates / selected / rejected runner-up / reason),
so future output can be judged against the committed frame. Automatic every turn
(source=auto) or on a /frame demand (source=requested). Writes JSONL.

Schema per the cold-proof spec: turn_id, session_id, timestamp_utc, input_digest,
candidate_frames, selected_frame, rejected_runner_up, rejection_reason, source,
storage_surface, monoframe_version, bearing_before_hash, artifact_hash.
"""
from __future__ import annotations

import json

import pytest


_BLOCK = """
<frame_selection>
CANDIDATES: anchor the live WAL ask | continue the stale db-internals frame | refuse as adversarial
SELECTED: anchor the live WAL ask
REJECTED: continue the stale db-internals frame
REASON: the carried frame mirrors a loud prior noun; the live ask is the WAL question
</frame_selection>
""".strip()


class TestParse:
    def test_parses_all_four_fields(self):
        from addons.system.bearing import frame_selection as fs
        out = fs.parse_frame_selection(_BLOCK)
        assert out["candidate_frames"] == [
            "anchor the live WAL ask",
            "continue the stale db-internals frame",
            "refuse as adversarial",
        ]
        assert out["selected_frame"] == "anchor the live WAL ask"
        assert out["rejected_runner_up"] == "continue the stale db-internals frame"
        assert "loud prior noun" in out["rejection_reason"]

    def test_no_block_returns_empty(self):
        from addons.system.bearing import frame_selection as fs
        out = fs.parse_frame_selection("just a normal answer, no block")
        assert out["selected_frame"] == ""
        assert out["candidate_frames"] == []

    def test_has_selection_true_only_with_block(self):
        from addons.system.bearing import frame_selection as fs
        assert fs.has_selection(_BLOCK) is True
        assert fs.has_selection("no block here") is False


class TestDigest:
    def test_digest_is_stable_sha256(self):
        from addons.system.bearing import frame_selection as fs
        import hashlib
        assert fs.digest("hello") == hashlib.sha256(b"hello").hexdigest()

    def test_digest_handles_none(self):
        from addons.system.bearing import frame_selection as fs
        assert isinstance(fs.digest(None), str) and len(fs.digest(None)) == 64


class TestBuildRecord:
    def test_record_has_full_schema_and_artifact_hash(self):
        from addons.system.bearing import frame_selection as fs
        rec = fs.build_record(
            raw_output=_BLOCK,
            turn_id="t-1",
            session_id="s-1",
            timestamp_utc="2026-06-24T00:00:00+00:00",
            user_input="what does a WAL do?",
            bearing_before="responding to an examiner peer query on database internals",
            source="auto",
        )
        d = rec.to_dict()
        for key in (
            "turn_id", "session_id", "timestamp_utc", "input_digest",
            "candidate_frames", "selected_frame", "rejected_runner_up",
            "rejection_reason", "source", "storage_surface", "monoframe_version",
            "bearing_before_hash", "artifact_hash",
        ):
            assert key in d, f"missing {key}"
        assert d["source"] == "auto"
        assert d["selected_frame"] == "anchor the live WAL ask"
        assert d["input_digest"] == fs.digest("what does a WAL do?")
        assert d["bearing_before_hash"] == fs.digest(
            "responding to an examiner peer query on database internals"
        )
        # artifact_hash is over the record's content (excluding itself)
        assert len(d["artifact_hash"]) == 64


class TestRecorder:
    def test_records_jsonl_when_enabled(self, monkeypatch, tmp_path):
        from addons.system.bearing import frame_selection as fs
        monkeypatch.setenv("MONOLITH_MONOFRAME_V1", "1")
        store = tmp_path / "frame_selection.jsonl"
        monkeypatch.setattr(fs, "_STORE", store)
        rec = fs.build_record(
            raw_output=_BLOCK, turn_id="t-9", session_id="s", timestamp_utc="x",
            user_input="q", bearing_before="b", source="auto",
        )
        fs.record_selection(rec)
        rows = [json.loads(l) for l in store.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(rows) == 1 and rows[0]["turn_id"] == "t-9"

    def test_noop_when_disabled(self, monkeypatch, tmp_path):
        from addons.system.bearing import frame_selection as fs
        monkeypatch.delenv("MONOLITH_MONOFRAME_V1", raising=False)
        store = tmp_path / "frame_selection.jsonl"
        monkeypatch.setattr(fs, "_STORE", store)
        rec = fs.build_record(
            raw_output=_BLOCK, turn_id="t", session_id="s", timestamp_utc="x",
            user_input="q", bearing_before="b", source="auto",
        )
        fs.record_selection(rec)
        assert not store.exists()

    def test_record_from_output_skips_when_no_block(self, monkeypatch, tmp_path):
        from addons.system.bearing import frame_selection as fs
        monkeypatch.setenv("MONOLITH_MONOFRAME_V1", "1")
        monkeypatch.setattr(fs, "_STORE", tmp_path / "frame_selection.jsonl")
        # the convenience seam the turn-finalizer calls: no block -> no record
        wrote = fs.record_from_output(
            raw_output="plain answer, no selection block",
            turn_id="t", session_id="s", timestamp_utc="x",
            user_input="q", bearing_before="b", source="auto",
        )
        assert wrote is False
