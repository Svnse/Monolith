"""Blinded output writer + arm-hash for the Bearing A/B harness.

Per spec §8.1 (locked 2026-05-21): file names use `<fixture_id>_<arm_hash>`,
NOT `arm=on/off`. Raters MUST NOT see the arm decoder until all scoring
is complete; the decoder lives in the same artifacts dir but is gated by
operator process, not by code.

`arm_hash` is deterministic — same (fixture_id, arm, run_id) produces the
same hash, so reruns produce identical filenames and inspecting partial
output across runs works.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ARTIFACTS_ROOT = Path(__file__).resolve().parents[2] / "artifacts" / "bearing_ab"


def arm_hash(fixture_id: str, arm: str, run_id: str) -> str:
    """Deterministic blake2b 12-char hex per (fixture, arm, run_id) triple."""
    digest_input = f"{fixture_id}::{arm}::{run_id}".encode("utf-8")
    return hashlib.blake2b(digest_input, digest_size=6).hexdigest()


def run_dir(run_id: str) -> Path:
    p = ARTIFACTS_ROOT / run_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_arm_output(
    fixture_id: str,
    arm: str,
    run_id: str,
    responses: list[dict[str, Any]],
    fingerprint_summary: dict[str, Any] | None = None,
) -> Path:
    """Write rater-readable response file + meta sidecar; update arm_decoder.

    `responses` is a list of {"turn": N, "user": str, "assistant": str}.
    The rater-facing file does NOT mention `arm` anywhere in the body — only
    the filename hash carries that information, and the hash is opaque
    until the decoder is opened.
    """
    rdir = run_dir(run_id)
    h = arm_hash(fixture_id, arm, run_id)
    body_path = rdir / f"{fixture_id}_{h}.txt"
    meta_path = rdir / f"{fixture_id}_{h}.meta.json"

    lines = [f"# Session: {fixture_id} (hash: {h})", ""]
    for entry in responses:
        lines.append(f"## Turn {entry['turn']}")
        lines.append("")
        lines.append(f"**User:** {entry['user']}")
        lines.append("")
        lines.append(f"**Assistant:** {entry['assistant']}")
        lines.append("")
        lines.append("---")
        lines.append("")
    body_path.write_text("\n".join(lines), encoding="utf-8")

    meta = {
        "fixture_id": fixture_id,
        "arm_hash": h,
        "run_id": run_id,
        "turn_count": len(responses),
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        # fingerprint_summary is OPTIONAL here — the canonical record lives
        # in artifacts/<run_id>/fingerprint.json. This is denormalized for
        # convenience when inspecting a single arm's output.
        "fingerprint_summary": fingerprint_summary or {},
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return body_path


def update_arm_decoder(run_id: str, fixture_id: str, arm: str) -> None:
    """Append (fixture_id, arm) -> arm_hash mapping to the run's decoder.

    Decoder is rater-INACCESSIBLE during scoring. Operator unseals it
    only after both raters have submitted.
    """
    rdir = run_dir(run_id)
    decoder_path = rdir / "arm_decoder.json"
    decoder: dict[str, dict[str, str]] = {}
    if decoder_path.exists():
        try:
            decoder = json.loads(decoder_path.read_text(encoding="utf-8"))
        except Exception:
            decoder = {}
    h = arm_hash(fixture_id, arm, run_id)
    decoder.setdefault(fixture_id, {})[arm] = h
    decoder_path.write_text(json.dumps(decoder, indent=2), encoding="utf-8")
