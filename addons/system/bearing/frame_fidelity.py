"""MonoFrame v2 — the Frame Fidelity Judge (commitment verification).

It judges ONE thing: did the final answer HONOR the frame the recorder captured
(selected_frame), or did it drift to the rejected runner-up / a stale frame? It
does NOT judge answer quality, and it does NOT claim to verify hidden cognition —
only the COMMITMENT. Observational: it records a verdict, never rewrites.

Recorder = "what did Monolith commit to before answering?" (frame_selection.jsonl)
Judge    = "did the answer betray that commitment?"        (frame_fidelity.jsonl)

The verdict is bound to the pre-answer commitment by ``frame_record_hash`` (the
frame_selection record's artifact_hash) and cites spans FROM THE ANSWER, so a
failure is externally checkable from the answer text + frame artifact alone.

First target = the single failure class: selected vs runner-up/stale. The full
betrayal vocabulary is reserved for later judges. Flag MONOLITH_MONOFRAME_V1.
"""
from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from core.paths import CONFIG_DIR

from . import frame_selection

_FLAG_ENV = "MONOLITH_MONOFRAME_V1"
_TRUTHY = {"1", "true", "yes", "on"}

JUDGE_VERSION = "fidelity-v1-selected-vs-runnerup"
STORAGE_SURFACE = "frame_fidelity.jsonl"
_STORE = CONFIG_DIR / STORAGE_SURFACE


class Verdict(Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class BetrayalType(Enum):
    NONE = "none"
    IGNORED_SELECTED_FRAME = "ignored_selected_frame"
    ANSWERED_RUNNER_UP_FRAME = "answered_runner_up_frame"
    STALE_FRAME_LEAK = "stale_frame_leak"
    FRAME_ANSWER_MISMATCH = "frame_answer_mismatch"
    OVERBROAD_ANSWER = "overbroad_answer"
    UNDERANSWERED_COMMITTED_SCOPE = "underanswered_committed_scope"


def enabled() -> bool:
    return os.environ.get(_FLAG_ENV, "0").strip().lower() in _TRUTHY


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_JUDGE_PROMPT = (
    "You are a FRAME FIDELITY judge. You do NOT judge whether the answer is good, "
    "and you do NOT claim to know the model's true hidden cognition. You judge ONE "
    "thing: did the final answer HONOR the SELECTED frame the model committed to "
    "before answering, or did it drift to the REJECTED runner-up frame (or a stale "
    "frame)?\n\n"
    "SELECTED frame (the commitment the answer must honor):\n{selected}\n\n"
    "REJECTED runner-up frame (the drift target):\n{rejected}\n\n"
    "Cite spans FROM THE ANSWER as evidence, so the verdict is checkable from the "
    "answer text alone. Reply EXACTLY, no preamble:\n"
    "VERDICT: <pass = honored the selected frame | fail = followed the rejected/stale frame | warn = ambiguous>\n"
    "BETRAYAL: <none | ignored_selected_frame | answered_runner_up_frame | stale_frame_leak>\n"
    "EVIDENCE: <a span quoted from the answer || another span>\n"
    "EXPLANATION: <one or two sentences>"
)


# ── pure parsing ────────────────────────────────────────────────────


def _field(text: str, label: str) -> str:
    m = re.search(rf"^{re.escape(label)}\s*:\s*(.+)$", text or "", re.IGNORECASE | re.MULTILINE)
    return m.group(1).strip() if m else ""


def parse_fidelity(text: str) -> dict[str, Any]:
    """Parse the judge output. Unknown verdict -> WARN (never silently pass);
    unknown betrayal -> NONE. Evidence spans split on ' || '."""
    v_raw = _field(text, "VERDICT").lower()
    verdict = next((v for v in Verdict if v.value in v_raw), Verdict.WARN)
    b_raw = _field(text, "BETRAYAL").lower()
    betrayal = BetrayalType.NONE
    for b in BetrayalType:
        if b is not BetrayalType.NONE and b.value in b_raw:
            betrayal = b
            break
    ev_raw = _field(text, "EVIDENCE")
    evidence = [s.strip() for s in ev_raw.split("||") if s.strip()]
    return {
        "verdict": verdict,
        "betrayal_type": betrayal,
        "evidence_spans": evidence,
        "explanation": _field(text, "EXPLANATION"),
    }


# ── the judge call (injected generate) ──────────────────────────────


def _default_generate(base_config: dict[str, Any]):
    from engine.sync_bridge import generate_sync_from_config

    def generate(msgs: list[dict]) -> str:
        return generate_sync_from_config(base_config, msgs)

    return generate


def judge(
    *,
    selected_frame: str,
    rejected_runner_up: str,
    answer: str,
    base_config: dict[str, Any] | None = None,
    generate=None,
) -> dict[str, Any]:
    """Run the fidelity judge LLM call and parse it. Fails to WARN on error
    (never silently passes)."""
    if generate is None:
        generate = _default_generate(base_config or {})
    system = _JUDGE_PROMPT.format(selected=selected_frame, rejected=rejected_runner_up)
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"ANSWER:\n{answer}"},
    ]
    try:
        out = str(generate(msgs) or "")
    except Exception:
        out = "VERDICT: warn\nBETRAYAL: none\nEXPLANATION: judge call failed"
    return parse_fidelity(out)


# ── record + write ──────────────────────────────────────────────────


@dataclass(frozen=True)
class FrameFidelityRecord:
    turn_id: str
    frame_record_hash: str        # binds to the pre-answer commitment
    answer_digest: str
    selected_frame: str
    rejected_runner_up: str
    verdict: str
    betrayal_type: str
    evidence_spans_from_answer: list[str]
    explanation: str
    judge_version: str
    timestamp_utc: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "frame_record_hash": self.frame_record_hash,
            "answer_digest": self.answer_digest,
            "selected_frame": self.selected_frame,
            "rejected_runner_up": self.rejected_runner_up,
            "verdict": self.verdict,
            "betrayal_type": self.betrayal_type,
            "evidence_spans_from_answer": list(self.evidence_spans_from_answer),
            "explanation": self.explanation,
            "judge_version": self.judge_version,
            "timestamp_utc": self.timestamp_utc,
        }


def build_fidelity_record(
    *,
    frame_record: dict[str, Any],
    answer: str,
    base_config: dict[str, Any] | None = None,
    generate=None,
) -> FrameFidelityRecord:
    """Judge ``answer`` against the committed frame in ``frame_record`` and build a
    record bound to that commitment by hash."""
    selected = str(frame_record.get("selected_frame", ""))
    rejected = str(frame_record.get("rejected_runner_up", ""))
    parsed = judge(
        selected_frame=selected, rejected_runner_up=rejected, answer=answer,
        base_config=base_config, generate=generate,
    )
    return FrameFidelityRecord(
        turn_id=str(frame_record.get("turn_id", "")),
        frame_record_hash=str(frame_record.get("artifact_hash", "")),
        answer_digest=frame_selection.digest(answer),
        selected_frame=selected,
        rejected_runner_up=rejected,
        verdict=parsed["verdict"].value,
        betrayal_type=parsed["betrayal_type"].value,
        evidence_spans_from_answer=parsed["evidence_spans"],
        explanation=parsed["explanation"],
        judge_version=JUDGE_VERSION,
        timestamp_utc=_now_iso(),
    )


def record_fidelity(record: FrameFidelityRecord) -> None:
    if not enabled():
        return
    try:
        _STORE.parent.mkdir(parents=True, exist_ok=True)
        with _STORE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    except Exception:
        pass


def judge_turn(
    *,
    frame_record: dict[str, Any],
    answer: str,
    base_config: dict[str, Any] | None = None,
    generate=None,
) -> bool:
    """Turn-finalizer seam: judge the answer against the turn's committed frame and
    record the verdict. Skips when disabled or there is no committed selected_frame
    to judge against. Returns whether a verdict was written."""
    if not enabled():
        return False
    if not str(frame_record.get("selected_frame", "")).strip():
        return False
    rec = build_fidelity_record(
        frame_record=frame_record, answer=answer, base_config=base_config, generate=generate,
    )
    record_fidelity(rec)
    return True


def judge_turn_async(
    *,
    frame_record: dict[str, Any],
    answer: str,
    base_config: dict[str, Any] | None = None,
    generate=None,
) -> threading.Thread | None:
    """Fire-and-forget: run the judge on a daemon thread so the turn-finalizer
    never blocks on the judge's LLM call (off the chat path, like a monoline
    block). No-op when disabled. Returns the started thread."""
    if not enabled():
        return None

    def _work() -> None:
        try:
            judge_turn(
                frame_record=frame_record, answer=answer,
                base_config=base_config, generate=generate,
            )
        except Exception:
            pass

    t = threading.Thread(target=_work, name="frame-fidelity", daemon=True)
    t.start()
    return t


def read_recent(limit: int = 20) -> list[dict[str, Any]]:
    if not _STORE.exists():
        return []
    try:
        with _STORE.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for line in lines[-max(1, int(limit)):]:
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out
