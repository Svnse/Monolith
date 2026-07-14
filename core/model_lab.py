from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.paths import ARTIFACTS_DIR


MODEL_LAB_DIR = ARTIFACTS_DIR / "model_lab"


@dataclass(frozen=True)
class ModelCandidate:
    candidate_id: str
    model_identity: dict[str, Any]
    output: str
    blind_label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelLabRun:
    run_id: str
    prompt: str
    created_at: str
    candidates: tuple[ModelCandidate, ...]
    fixture_id: str | None = None
    outcome_turn_id: str | None = None
    votes: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def to_dict(self, *, reveal_identity: bool = True) -> dict[str, Any]:
        candidates = []
        for candidate in self.candidates:
            data = candidate.to_dict()
            if not reveal_identity:
                data["model_identity"] = {"hidden": True}
            candidates.append(data)
        return {
            "run_id": self.run_id,
            "prompt": self.prompt,
            "created_at": self.created_at,
            "fixture_id": self.fixture_id,
            "outcome_turn_id": self.outcome_turn_id,
            "candidates": candidates,
            "votes": list(self.votes),
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_path(run_id: str, *, root: Path | None = None) -> Path:
    base = root or MODEL_LAB_DIR
    return base / f"{run_id}.json"


def _labels(count: int) -> list[str]:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if count <= len(alphabet):
        return list(alphabet[:count])
    return [f"M{i + 1}" for i in range(count)]


def create_compare_run(
    prompt: str,
    candidates: list[dict[str, Any]],
    *,
    fixture_id: str | None = None,
    root: Path | None = None,
) -> ModelLabRun:
    prompt_s = str(prompt or "").strip()
    if not prompt_s:
        raise ValueError("model lab run requires a prompt")
    if len(candidates) < 2:
        raise ValueError("model lab compare requires at least two candidates")
    labels = _labels(len(candidates))
    run_candidates: list[ModelCandidate] = []
    for idx, item in enumerate(candidates):
        if not isinstance(item, dict):
            raise ValueError("candidate must be a dict")
        output = str(item.get("output", "") or "")
        identity = item.get("model_identity") if isinstance(item.get("model_identity"), dict) else {}
        run_candidates.append(
            ModelCandidate(
                candidate_id=str(item.get("candidate_id") or uuid.uuid4().hex),
                model_identity=dict(identity),
                output=output,
                blind_label=labels[idx],
            )
        )
    run = ModelLabRun(
        run_id=uuid.uuid4().hex,
        prompt=prompt_s,
        created_at=_now_iso(),
        fixture_id=fixture_id,
        candidates=tuple(run_candidates),
    )
    save_run(run, root=root)
    return run


def save_run(run: ModelLabRun, *, root: Path | None = None) -> Path:
    path = _run_path(run.run_id, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_run(run_id: str, *, root: Path | None = None) -> ModelLabRun | None:
    path = _run_path(run_id, root=root)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return ModelLabRun(
        run_id=str(data["run_id"]),
        prompt=str(data["prompt"]),
        created_at=str(data["created_at"]),
        fixture_id=data.get("fixture_id"),
        outcome_turn_id=data.get("outcome_turn_id"),
        candidates=tuple(ModelCandidate(**item) for item in data.get("candidates", [])),
        votes=tuple(data.get("votes", [])),
    )


def record_vote(
    run_id: str,
    *,
    winning_label: str,
    reason: str = "",
    rating_value: int | None = None,
    turn_id: str | None = None,
    root: Path | None = None,
    train_monothink: bool = False,
) -> ModelLabRun:
    run = load_run(run_id, root=root)
    if run is None:
        raise ValueError(f"model lab run not found: {run_id}")
    label = str(winning_label or "").strip().upper()
    labels = {candidate.blind_label for candidate in run.candidates}
    if label not in labels:
        raise ValueError(f"unknown blind label: {winning_label!r}")
    vote = {
        "winning_label": label,
        "reason": str(reason or ""),
        "rating_value": rating_value,
        "recorded_at": _now_iso(),
        "train_monothink": bool(train_monothink),
    }
    updated = ModelLabRun(
        run_id=run.run_id,
        prompt=run.prompt,
        created_at=run.created_at,
        fixture_id=run.fixture_id,
        outcome_turn_id=turn_id or run.outcome_turn_id,
        candidates=run.candidates,
        votes=tuple(list(run.votes) + [vote]),
    )
    save_run(updated, root=root)
    if turn_id and rating_value is not None:
        from core import turn_trace

        winner = next(candidate for candidate in updated.candidates if candidate.blind_label == label)
        metadata = {
            "model_lab_run_id": run_id,
            "model_identity": winner.model_identity,
            "blind_label": label,
            "training_enabled": bool(train_monothink),
            "rating_value": max(0, min(100, int(rating_value))),
        }
        kind = "rating" if train_monothink else "copy"
        turn_trace.record_outcome(
            turn_trace.OutcomeTraceRecord(
                turn_id=str(turn_id),
                recorded_at=_now_iso(),
                kind=kind,
                rating_value=metadata["rating_value"] if train_monothink else None,
                reason=reason,
                metadata=metadata,
            )
        )
    return updated
