from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from core import confidence_trajectory


# ── fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def tmp_log(monkeypatch, tmp_path):
    """Redirect the confidence log to a temp file for each test."""
    log_path = tmp_path / "confidence_log.jsonl"
    monkeypatch.setattr(confidence_trajectory, "_LOG_PATH", log_path)
    yield log_path


def _import_scratchpad_executor():
    """Load the dynamic skill executor by file path (matches runtime loader)."""
    spec_path = Path(__file__).parent.parent / "skills" / "scratchpad" / "executor.py"
    spec = importlib.util.spec_from_file_location("scratchpad_exec_test_ct", spec_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def scratchpad(tmp_log, monkeypatch):
    """Scratchpad executor with log and model_id pinned."""
    from core import llm_config
    monkeypatch.setattr(llm_config, "get_current_model_id", lambda: "test-model")
    return _import_scratchpad_executor()


# ── record_confidence: persistence ────────────────────────────────────


def test_record_persists(tmp_log):
    rec = confidence_trajectory.record_confidence(
        value=80,
        claim="the spec is correct",
        premise="user confirmed intent turn 2",
        writer_model_id="model-x",
    )
    assert rec["value"] == 80
    assert rec["claim"] == "the spec is correct"
    assert rec["premise"] == "user confirmed intent turn 2"
    assert rec["writer_model_id"] == "model-x"
    assert "created_at" in rec

    # File must exist and contain exactly one valid JSON line.
    lines = [l for l in tmp_log.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["value"] == 80


# ── record_confidence: value validation ───────────────────────────────


def test_record_rejects_value_out_of_range_high(tmp_log):
    with pytest.raises(ValueError, match="0-100"):
        confidence_trajectory.record_confidence(101, "claim", "premise", "m")
    assert not tmp_log.exists()


def test_record_rejects_value_out_of_range_low(tmp_log):
    with pytest.raises(ValueError, match="0-100"):
        confidence_trajectory.record_confidence(-1, "claim", "premise", "m")
    assert not tmp_log.exists()


def test_record_rejects_value_out_of_range(tmp_log):
    """Boundary test: 0 and 100 are valid, 101 is not."""
    confidence_trajectory.record_confidence(0, "claim", "premise", "m")
    confidence_trajectory.record_confidence(100, "claim", "premise", "m")
    with pytest.raises(ValueError):
        confidence_trajectory.record_confidence(101, "claim", "premise", "m")


def test_record_rejects_bool_value(tmp_log):
    """bool is a subclass of int; must be explicitly rejected."""
    with pytest.raises(ValueError):
        confidence_trajectory.record_confidence(True, "claim", "premise", "m")
    assert not tmp_log.exists()


def test_record_rejects_float_value(tmp_log):
    with pytest.raises(ValueError):
        confidence_trajectory.record_confidence(80.5, "claim", "premise", "m")  # type: ignore[arg-type]
    assert not tmp_log.exists()


# ── record_confidence: claim validation ───────────────────────────────


def test_record_rejects_empty_claim(tmp_log):
    with pytest.raises(ValueError, match="claim"):
        confidence_trajectory.record_confidence(80, "", "premise", "m")
    assert not tmp_log.exists()


def test_record_rejects_whitespace_only_claim(tmp_log):
    with pytest.raises(ValueError, match="claim"):
        confidence_trajectory.record_confidence(80, "   \n  ", "premise", "m")
    assert not tmp_log.exists()


def test_record_rejects_oversize_claim(tmp_log):
    with pytest.raises(ValueError, match="claim"):
        confidence_trajectory.record_confidence(80, "x" * 201, "premise", "m")
    assert not tmp_log.exists()


def test_record_accepts_claim_at_exactly_200(tmp_log):
    rec = confidence_trajectory.record_confidence(80, "x" * 200, "premise", "m")
    assert len(rec["claim"]) == 200


# ── record_confidence: premise validation ─────────────────────────────


def test_record_rejects_empty_premise(tmp_log):
    with pytest.raises(ValueError, match="premise"):
        confidence_trajectory.record_confidence(80, "claim", "", "m")
    assert not tmp_log.exists()


def test_record_rejects_whitespace_only_premise(tmp_log):
    with pytest.raises(ValueError, match="premise"):
        confidence_trajectory.record_confidence(80, "claim", "   ", "m")
    assert not tmp_log.exists()


def test_record_rejects_oversize_premise(tmp_log):
    with pytest.raises(ValueError, match="premise"):
        confidence_trajectory.record_confidence(80, "claim", "x" * 201, "m")
    assert not tmp_log.exists()


def test_record_accepts_premise_at_exactly_200(tmp_log):
    rec = confidence_trajectory.record_confidence(80, "claim", "x" * 200, "m")
    assert len(rec["premise"]) == 200


# ── read_recent ────────────────────────────────────────────────────────


def test_read_recent_newest_first(tmp_log):
    for v in [10, 20, 30, 40, 50]:
        confidence_trajectory.record_confidence(v, f"claim {v}", "premise", "m")
    recent = confidence_trajectory.read_recent(limit=5)
    assert [r["value"] for r in recent] == [50, 40, 30, 20, 10]


def test_read_recent_respects_limit(tmp_log):
    for v in range(10):
        confidence_trajectory.record_confidence(v * 10, "claim", "premise", "m")
    recent = confidence_trajectory.read_recent(limit=3)
    assert len(recent) == 3


def test_read_recent_empty_when_no_records(tmp_log):
    assert confidence_trajectory.read_recent() == []


# ── compute_stats ──────────────────────────────────────────────────────


def test_compute_stats_returns_none_when_empty(tmp_log):
    assert confidence_trajectory.compute_stats() is None


def test_compute_stats_mean_and_recent(tmp_log):
    for v in [60, 70, 80, 90, 100]:
        confidence_trajectory.record_confidence(v, "claim", "premise", "m")
    stats = confidence_trajectory.compute_stats()
    assert stats is not None
    assert stats["n"] == 5
    assert stats["mean"] == 80  # (60+70+80+90+100)/5
    # recent = last 5 in chronological order
    assert stats["recent"] == [60, 70, 80, 90, 100]


def test_compute_stats_window_is_last_20(tmp_log):
    """Window is the last 20 records, not all of them."""
    # Write 25 records; first 5 are value=10, last 20 are value=90.
    for _ in range(5):
        confidence_trajectory.record_confidence(10, "claim", "premise", "m")
    for _ in range(20):
        confidence_trajectory.record_confidence(90, "claim", "premise", "m")
    stats = confidence_trajectory.compute_stats()
    assert stats["n"] == 20
    assert stats["mean"] == 90  # window is last 20, all 90


def test_compute_stats_recent_is_chronological(tmp_log):
    """recent field is oldest-first within the 5-value slice."""
    for v in [10, 20, 30, 40, 50, 60]:
        confidence_trajectory.record_confidence(v, "claim", "premise", "m")
    stats = confidence_trajectory.compute_stats()
    # Last 5 in chronological order: 20, 30, 40, 50, 60
    assert stats["recent"] == [20, 30, 40, 50, 60]


def test_compute_stats_mean_rounds_to_int(tmp_log):
    for v in [70, 71]:
        confidence_trajectory.record_confidence(v, "claim", "premise", "m")
    stats = confidence_trajectory.compute_stats()
    assert isinstance(stats["mean"], int)
    # (70+71)/2 = 70.5; Python's round() does banker's rounding → 70 (rounds to even)
    assert stats["mean"] == 70


# ── 200-record cap ────────────────────────────────────────────────────


def test_log_caps_at_200_records(tmp_log):
    """Write 201 records; file must contain exactly 200 lines, oldest dropped."""
    for i in range(201):
        confidence_trajectory.record_confidence(i % 101, f"claim {i}", "premise", "m")

    lines = [l for l in tmp_log.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 200, f"Expected 200 lines, got {len(lines)}"

    records = [json.loads(l) for l in lines]
    # Oldest (claim "claim 0") must be gone.
    claims = [r["claim"] for r in records]
    assert "claim 0" not in claims
    # Newest (claim "claim 200") must be present.
    assert "claim 200" in claims


def test_log_cap_rewrite_is_atomic(tmp_log, monkeypatch):
    """After a cap rewrite, file has no partial lines."""
    for i in range(205):
        confidence_trajectory.record_confidence(i % 101, f"claim {i}", "premise", "m")
    lines = [l for l in tmp_log.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 200
    # Each line must be valid JSON.
    for line in lines:
        json.loads(line)


# ── render_confidence_block ────────────────────────────────────────────


def test_render_block_none_when_empty(tmp_log):
    assert confidence_trajectory.render_confidence_block() is None


def test_render_block_format(tmp_log):
    for v in [65, 80, 95, 70, 85]:
        confidence_trajectory.record_confidence(v, "claim", "premise", "m")
    block = confidence_trajectory.render_confidence_block()
    assert block is not None
    assert block.startswith("[CONFIDENCE TRAJECTORY]")
    assert "n: 5" in block
    # mean: (65+80+95+70+85)/5 = 395/5 = 79
    assert "mean: 79" in block
    assert "recent: 65, 80, 95, 70, 85" in block


def test_render_block_with_more_than_5_records(tmp_log):
    """recent should be last 5 in chronological order."""
    for v in [10, 20, 30, 40, 50, 60, 70]:
        confidence_trajectory.record_confidence(v, "claim", "premise", "m")
    block = confidence_trajectory.render_confidence_block()
    assert "recent: 30, 40, 50, 60, 70" in block


# ── contribute_section ────────────────────────────────────────────────


def test_contribute_section_returns_none_when_empty(tmp_log):
    result = confidence_trajectory.contribute_section([], {})
    assert result is None


def test_contribute_section_returns_section_when_records_exist(tmp_log):
    confidence_trajectory.record_confidence(75, "claim", "premise", "m")
    result = confidence_trajectory.contribute_section([], {})
    assert result is not None
    assert result.name == "confidence_trajectory"
    assert "[CONFIDENCE TRAJECTORY]" in result.text


def test_contribute_section_respects_flag(tmp_log, monkeypatch):
    monkeypatch.setenv("MONOLITH_CONFIDENCE_TRAJECTORY_V1", "0")
    confidence_trajectory.record_confidence(75, "claim", "premise", "m")
    result = confidence_trajectory.contribute_section([], {})
    assert result is None


def test_contribute_section_flag_on_by_default(tmp_log, monkeypatch):
    # Ensure env var not set.
    monkeypatch.delenv("MONOLITH_CONFIDENCE_TRAJECTORY_V1", raising=False)
    confidence_trajectory.record_confidence(75, "claim", "premise", "m")
    result = confidence_trajectory.contribute_section([], {})
    assert result is not None


# ── scratchpad executor: record_confidence round-trip ─────────────────


def test_scratchpad_op_record_confidence_round_trip(tmp_log, monkeypatch):
    """Full round-trip through executor.run() → persisted → readable."""
    from core import llm_config
    monkeypatch.setattr(llm_config, "get_current_model_id", lambda: "test-model")
    sp = _import_scratchpad_executor()

    out = sp.run(
        {
            "op": "record_confidence",
            "value": 85,
            "claim": "that the spec correctly captures the requirements",
            "premise": "my reading of the user's intent in turn 3",
        },
        None,
    )
    assert "record_confidence" in out
    assert "logged value=85" in out
    assert "that the spec correctly" in out

    # Verify persistence.
    records = confidence_trajectory.read_recent()
    assert records[0]["value"] == 85
    assert records[0]["writer_model_id"] == "test-model"


def test_scratchpad_op_record_confidence_rejects_out_of_range(tmp_log, monkeypatch):
    from core import llm_config
    monkeypatch.setattr(llm_config, "get_current_model_id", lambda: "test-model")
    sp = _import_scratchpad_executor()
    out = sp.run({"op": "record_confidence", "value": 150, "claim": "c", "premise": "p"}, None)
    assert "record_confidence" in out
    assert "0-100" in out or "100" in out
    # Nothing persisted.
    assert confidence_trajectory.read_recent() == []


def test_scratchpad_op_record_confidence_rejects_empty_claim(tmp_log, monkeypatch):
    from core import llm_config
    monkeypatch.setattr(llm_config, "get_current_model_id", lambda: "test-model")
    sp = _import_scratchpad_executor()
    out = sp.run({"op": "record_confidence", "value": 80, "claim": "", "premise": "p"}, None)
    assert "record_confidence" in out
    assert confidence_trajectory.read_recent() == []


def test_scratchpad_op_record_confidence_rejects_empty_premise(tmp_log, monkeypatch):
    from core import llm_config
    monkeypatch.setattr(llm_config, "get_current_model_id", lambda: "test-model")
    sp = _import_scratchpad_executor()
    out = sp.run({"op": "record_confidence", "value": 80, "claim": "claim", "premise": ""}, None)
    assert "record_confidence" in out
    assert confidence_trajectory.read_recent() == []


def test_scratchpad_op_unknown_op_mentions_record_confidence(tmp_log, monkeypatch):
    """Fallback help text must include record_confidence."""
    from core import llm_config
    monkeypatch.setattr(llm_config, "get_current_model_id", lambda: "test-model")
    sp = _import_scratchpad_executor()
    out = sp.run({"op": "nonexistent_op"}, None)
    assert "record_confidence" in out


# ── JSONL record shape ─────────────────────────────────────────────────


def test_record_jsonl_shape(tmp_log):
    """Each persisted JSONL line has the required fields."""
    confidence_trajectory.record_confidence(
        value=72,
        claim="test claim",
        premise="test premise",
        writer_model_id="model-v1",
    )
    lines = [l for l in tmp_log.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert set(rec.keys()) >= {"value", "claim", "premise", "created_at", "writer_model_id"}
    assert rec["value"] == 72
    assert rec["claim"] == "test claim"
    assert rec["premise"] == "test premise"
    assert rec["writer_model_id"] == "model-v1"
    # created_at is ISO-8601
    assert "T" in rec["created_at"]
