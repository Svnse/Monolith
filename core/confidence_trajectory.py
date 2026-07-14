"""Confidence trajectory — records stated confidence numbers from the ANALYSIS loop.

The ANALYSIS synthesis loop (prompts/system.md step 4) says:
"CONFIDENCE — name the load-bearing premise and how confident you are in it."
Monolith produces these numbers each turn but they're transient. This module
persists them and surfaces a compact trajectory so Monolith can see whether
it's been well-calibrated, over-confident, or under-confident over time.

Storage: CONFIG_DIR / "confidence_log.jsonl" — append-only JSONL, one record
per line. Capped at 200 records; overflow triggers atomic rewrite.

Flag: MONOLITH_CONFIDENCE_TRAJECTORY_V1 (default ON). Set =0 to disable.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from core.paths import CONFIG_DIR

_LOG_PATH: Path = CONFIG_DIR / "confidence_log.jsonl"

_RECORD_CAP = 200
_STATS_WINDOW = 20
_RECENT_COUNT = 5
_FIELD_MAX_CHARS = 200

_FLAG_ENV = "MONOLITH_CONFIDENCE_TRAJECTORY_V1"
_BLOCK_TAG = "[CONFIDENCE TRAJECTORY]"

# Self-Check Loop (wire 3): when the loop is on, discount the *displayed*
# calibration mean by the last turn's verifier verdict — an external anchor on
# an otherwise echo-only number (the mean is just the model's own stated
# confidences). Gated by the self-check flag so the whole loop flips together.
# Derived fresh each render; never written back to confidence_log.jsonl.
_SELF_CHECK_FLAG = "MONOLITH_FAULT_TELEMETRY_V1"


# ── helpers ───────────────────────────────────────────────────────────


def _flag_enabled() -> bool:
    raw = str(os.environ.get(_FLAG_ENV, "1")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _self_check_enabled() -> bool:
    # Mirrors core.fault_telemetry's flag (default OFF). The verdict penalty is
    # part of the Self-Check Loop and flips on/off with it.
    raw = str(os.environ.get(_SELF_CHECK_FLAG, "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _verdict_penalty(verdict: str | None) -> float:
    """Multiplier applied to the displayed calibration mean given the last
    verifier verdict. warn -> 0.8, hard_fail -> 0.6, otherwise 1.0."""
    if verdict == "warn":
        return 0.8
    if verdict == "hard_fail":
        return 0.6
    return 1.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_all() -> list[dict]:
    """Read all records from the log file. Returns [] if missing or corrupt."""
    if not _LOG_PATH.exists():
        return []
    records: list[dict] = []
    try:
        with _LOG_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if isinstance(rec, dict):
                        records.append(rec)
                except (ValueError, TypeError):
                    continue
    except OSError:
        return []
    return records


def _append_record(record: dict) -> None:
    """Append one record. If cap exceeded, atomic-rewrite dropping oldest."""
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Append first; then check if we need to trim.
    with _LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")

    # Check cap.
    records = _read_all()
    if len(records) > _RECORD_CAP:
        kept = records[-_RECORD_CAP:]
        tmp = _LOG_PATH.with_name(_LOG_PATH.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")
        os.replace(tmp, _LOG_PATH)


# ── public API ────────────────────────────────────────────────────────


def record_confidence(
    value: int,
    claim: str,
    premise: str,
    writer_model_id: str,
) -> dict:
    """Validate and persist a confidence record. Returns the record dict.

    Validation (raises ValueError on failure):
    1. value must be int (not bool), 0 <= value <= 100.
    2. claim must be non-empty after strip, <= 200 chars.
    3. premise must be non-empty after strip, <= 200 chars.
    """
    # Step 1: value validation — reject bools, non-ints, out-of-range.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"value must be an integer 0-100, got {type(value).__name__}")
    if not (0 <= value <= 100):
        raise ValueError(f"value must be 0-100, got {value}")

    # Step 2: claim validation.
    claim_s = str(claim or "").strip()
    if not claim_s:
        raise ValueError("claim is required and must be non-empty")
    if len(claim_s) > _FIELD_MAX_CHARS:
        raise ValueError(
            f"claim exceeds {_FIELD_MAX_CHARS}-char cap (got {len(claim_s)})"
        )

    # Step 3: premise validation.
    premise_s = str(premise or "").strip()
    if not premise_s:
        raise ValueError("premise is required and must be non-empty")
    if len(premise_s) > _FIELD_MAX_CHARS:
        raise ValueError(
            f"premise exceeds {_FIELD_MAX_CHARS}-char cap (got {len(premise_s)})"
        )

    record: dict = {
        "value": value,
        "claim": claim_s,
        "premise": premise_s,
        "created_at": _now_iso(),
        "writer_model_id": str(writer_model_id or "").strip(),
    }
    _append_record(record)
    return record


def read_recent(limit: int = 20) -> list[dict]:
    """Return up to `limit` most-recent records, newest first."""
    records = _read_all()
    records.reverse()
    return records[:limit]


def compute_stats() -> dict | None:
    """Return stats over the last 20 entries. Returns None if no records.

    Shape: {"n": N, "mean": M, "recent": [v1, v2, ...up to 5]}
    - n: count of records in the window (up to 20).
    - mean: integer mean of those n values.
    - recent: last 5 values in chronological order (oldest first).
    """
    records = _read_all()
    if not records:
        return None

    window = records[-_STATS_WINDOW:]  # chronological order, up to 20
    n = len(window)
    values = [r["value"] for r in window if isinstance(r.get("value"), int) and not isinstance(r.get("value"), bool)]
    if not values:
        return None

    mean = round(sum(values) / len(values))
    recent = values[-_RECENT_COUNT:]  # last 5 in chronological order
    return {"n": n, "mean": mean, "recent": recent}


# ── rendering ─────────────────────────────────────────────────────────


def render_confidence_block() -> str | None:
    """Build the [CONFIDENCE TRAJECTORY] block. Returns None when no records.

    Format:
        [CONFIDENCE TRAJECTORY]
        n: 12 | mean: 78 | recent: 65, 80, 95, 70, 85
    """
    stats = compute_stats()
    if stats is None:
        return None
    recent_str = ", ".join(str(v) for v in stats["recent"])
    mean = stats["mean"]
    note = ""
    if _self_check_enabled():
        from core import turn_trace as _tt
        rec = _tt.get_last_verification_result()
        tier = (rec or {}).get("verdict")
        pen = _verdict_penalty(tier)
        if pen < 1.0:
            mean = round(mean * pen)
            note = f" (×{pen:g} — last turn {tier})"
    return f"{_BLOCK_TAG}\nn: {stats['n']} | mean: {mean}{note} | recent: {recent_str}"


# ── interceptor ───────────────────────────────────────────────────────


def contribute_section(messages: list[dict], config: dict):
    """Section-contributor for the ephemeral_coalescer."""
    from core.ephemeral_coalescer import SectionResult
    if not _flag_enabled():
        return None
    block = render_confidence_block()
    if block is None:
        return None
    return SectionResult(name="confidence_trajectory", text=block)
