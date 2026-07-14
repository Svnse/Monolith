"""
core/vision_artifact_bridge.py - bridges async vision engine output to chat.

Subscribes to VisionProcess.sig_image. For gen_ids registered as
"skill-origin" by execute_generate_image(), it:

  1. Auto-saves the arriving PIL image as PNG + sidecar JSON to
     artifacts/vision/YYYY/MM/DD/{artifact_id}_b{batch_index}.png
  2. Appends a one-line entry to artifacts/vision/index.jsonl
  3. Emits sig_artifact_ready(artifact_id, batch_index, png_path, sidecar)
     for the chat session to render in the matching tool-result bubble.

Skill-origin filtering uses a pending-set: the executor calls
register_pending(gen_id, call_id, config) before submitting the generate
op to the engine. Arrivals whose gen_id is NOT in the pending-set are
ignored (those are user-clicked VISION-tab generations, which the SD
module already handles via its own connection to the same signal).

The bridge is intentionally a thin QObject. It owns no UI, no addon
state. The chat session connects to its signals; the executor calls
register_pending / drop_pending. Nothing else couples to it.
"""
from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import QObject, Signal

from core.paths import MONOLITH_ROOT


VISION_ARTIFACT_ROOT = MONOLITH_ROOT / "artifacts" / "vision"
INDEX_PATH = VISION_ARTIFACT_ROOT / "index.jsonl"


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _artifact_day_dir(now: datetime | None = None) -> Path:
    now = now or datetime.now()
    return VISION_ARTIFACT_ROOT / f"{now.year:04d}" / f"{now.month:02d}" / f"{now.day:02d}"


def _mint_artifact_id() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"vision_{ts}_{uuid.uuid4().hex[:8]}"


@dataclass
class PendingArtifact:
    artifact_id: str
    call_id: str
    config: dict
    received_paths: list[str] = field(default_factory=list)


class VisionArtifactBridge(QObject):
    """Translates VisionProcess.sig_image into chat-ready artifact events."""

    # (artifact_id, batch_index, png_path, sidecar)
    sig_artifact_ready = Signal(str, int, str, dict)
    # (artifact_id, current_step, total_steps)
    sig_artifact_progress = Signal(str, int, int)
    # (artifact_id, message) - emitted when a pending generation fails
    sig_artifact_failed = Signal(str, str)

    def __init__(self, vision_engine: Any = None, parent: Any = None) -> None:
        super().__init__(parent)
        self._lock = threading.Lock()
        self._pending: dict[int, PendingArtifact] = {}
        self._engine = vision_engine
        if vision_engine is not None:
            if hasattr(vision_engine, "sig_image"):
                vision_engine.sig_image.connect(self._on_image)
            if hasattr(vision_engine, "sig_progress"):
                vision_engine.sig_progress.connect(self._on_progress)

    # ── public API ────────────────────────────────────────────────────────

    def register_pending(self, gen_id: int, call_id: str, config: dict) -> str:
        """Mark gen_id as skill-origin. Returns the artifact_id to surface
        in the tool-result text so the chat session can correlate."""
        artifact_id = _mint_artifact_id()
        with self._lock:
            self._pending[int(gen_id)] = PendingArtifact(
                artifact_id=artifact_id,
                call_id=str(call_id or ""),
                config=dict(config or {}),
            )
        return artifact_id

    def drop_pending(self, gen_id: int, reason: str = "") -> Optional[str]:
        """Clean up a pending entry (engine error, stop, or completion). Returns
        the artifact_id if there was one, so callers can fan out the failure."""
        with self._lock:
            row = self._pending.pop(int(gen_id), None)
        if row is None:
            return None
        if reason:
            self.sig_artifact_failed.emit(row.artifact_id, str(reason))
        return row.artifact_id

    def is_pending(self, gen_id: int) -> bool:
        with self._lock:
            return int(gen_id) in self._pending

    # ── slots ────────────────────────────────────────────────────────────

    def _on_image(self, image: object, batch_index: int, gen_id: int) -> None:
        with self._lock:
            row = self._pending.get(int(gen_id))
        if row is None:
            return
        png_path, sidecar = self._save(image, batch_index, row)
        if png_path is None:
            return
        with self._lock:
            row.received_paths.append(str(png_path))
        self.sig_artifact_ready.emit(
            row.artifact_id,
            int(batch_index),
            str(png_path),
            sidecar,
        )

    def _on_progress(self, current: int, total: int, gen_id: int) -> None:
        with self._lock:
            row = self._pending.get(int(gen_id))
        if row is None:
            return
        self.sig_artifact_progress.emit(row.artifact_id, int(current), int(total))

    # ── persistence ──────────────────────────────────────────────────────

    def _save(self, image: object, batch_index: int, row: PendingArtifact) -> tuple[Optional[Path], dict]:
        if image is None:
            return None, {}
        day_dir = _artifact_day_dir()
        try:
            day_dir.mkdir(parents=True, exist_ok=True)
            INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            return None, {}

        png_path = day_dir / f"{row.artifact_id}_b{batch_index}.png"
        meta_path = day_dir / f"{row.artifact_id}_b{batch_index}.json"

        try:
            image.save(png_path)
        except Exception:
            return None, {}

        cfg = row.config
        sidecar = {
            "artifact_id": row.artifact_id,
            "call_id": row.call_id,
            "modality": "image",
            "kind": "generated",
            "producer": "vision_skill",
            "created_at": _now_iso(),
            "batch_index": int(batch_index),
            "path": str(png_path),
            "metadata_path": str(meta_path),
            "prompt": str(cfg.get("prompt") or ""),
            "negative_prompt": str(cfg.get("negative_prompt") or ""),
            "scheduler": str(cfg.get("scheduler") or "dpm++"),
            "seed_requested": cfg.get("seed"),
            "width": int(cfg.get("width", 0) or 0),
            "height": int(cfg.get("height", 0) or 0),
            "steps": int(cfg.get("steps", 0) or 0),
            "guidance_scale": float(cfg.get("guidance_scale", 0.0) or 0.0),
            "batch_size": int(cfg.get("batch_size", 1) or 1),
            "model_path": str(cfg.get("model_path") or ""),
        }
        try:
            meta_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
            with INDEX_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(sidecar) + "\n")
        except Exception:
            pass
        return png_path, sidecar
