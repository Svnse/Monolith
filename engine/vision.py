"""
engine/vision.py  —  VisionProcess
Subprocess-isolated image generation engine.

Public interface is backward-compatible with the old VisionEngine so
SDModule and existing bridge wiring require no changes.

New signals vs old VisionEngine:
  sig_image    now (object, int)  — PIL image + batch_index
  sig_progress (int, int)         — (current_step, total_steps)
  sig_resource (dict)             — {"vram_used_mb": int, "vram_free_mb": int}

Supported models (auto-detected):
  SD 1.5 / 2.x   (.safetensors / .ckpt / HuggingFace dir)
  SDXL
  Flux.1  (diffusers >= 0.30)

Generation payload keys (passed inside "config" sub-dict or top-level):
  prompt, negative_prompt, width, height, steps, guidance_scale,
  seed (-1 = random), scheduler ("euler"|"dpm++"|"ddim"|"lcm"),
  lora_path, lora_scale, batch_size
"""
from __future__ import annotations

from PySide6.QtCore import QTimer, Signal

from core.state import SystemStatus
from engine.engine_process import EngineProcess


class VisionProcess(EngineProcess):
    """Subprocess-isolated diffusers inference engine."""

    # ── signals ────────────────────────────────────────────────────────────
    sig_image    = Signal(object, int)   # (PIL.Image, batch_index)
    sig_finished = Signal()
    sig_progress = Signal(int, int)      # (current_step, total_steps)
    sig_resource = Signal(dict)          # vram stats after load / generate

    def __init__(self) -> None:
        super().__init__()
        self._model_path: str = ""

    # ── worker entry point (runs in child process) ─────────────────────────
    @staticmethod
    def _worker_fn(to_worker, from_worker) -> None:
        from engine._workers import vision_worker
        vision_worker.main(to_worker, from_worker)

    # ── EnginePort overrides ───────────────────────────────────────────────

    def set_model_path(self, payload: dict) -> None:
        self._model_path = str(payload.get("path") or "")
        QTimer.singleShot(0, lambda: self.sig_status.emit(SystemStatus.READY))

    def load_model(self) -> None:
        if not self._model_path:
            self.sig_trace.emit("VISION: ERROR: No model selected.")
            self.sig_status.emit(SystemStatus.ERROR)
            return
        if not self._ensure_proc():
            return
        self.sig_status.emit(SystemStatus.LOADING)
        self.sig_trace.emit(f"VISION: loading {self._model_path}")
        self._send("load", model_path=self._model_path, config={})

    def unload_model(self) -> None:
        if self._proc and self._proc.is_alive():
            self.sig_status.emit(SystemStatus.UNLOADING)
            self._send("unload")
        else:
            QTimer.singleShot(0, lambda: self.sig_status.emit(SystemStatus.READY))

    def generate(self, payload: dict) -> None:
        if not (self._proc and self._proc.is_alive()):
            self.sig_trace.emit("VISION: ERROR: Model offline.")
            self.sig_status.emit(SystemStatus.READY)
            return

        # Accept both flat payload and nested "config" sub-dict
        cfg    = payload.get("config", payload)
        prompt = str(cfg.get("prompt") or payload.get("prompt") or "")

        self._gen_id += 1
        self._active_gen_id = self._gen_id

        full_cfg = {
            "prompt":          prompt,
            "negative_prompt": str(cfg.get("negative_prompt") or ""),
            "width":           int(cfg.get("width",  512)),
            "height":          int(cfg.get("height", 512)),
            "steps":           int(cfg.get("steps",  25)),
            "guidance_scale":  float(cfg.get("guidance_scale", 7.5)),
            "seed":            cfg.get("seed"),
            "scheduler":       str(cfg.get("scheduler", "dpm++")),
            "lora_path":       cfg.get("lora_path"),
            "lora_scale":      float(cfg.get("lora_scale", 0.8)),
            "batch_size":      int(cfg.get("batch_size", 1)),
        }
        # Normalise seed: -1 or None → random
        if isinstance(full_cfg["seed"], int) and full_cfg["seed"] < 0:
            full_cfg["seed"] = None

        self.sig_status.emit(SystemStatus.RUNNING)
        self._send("generate", gen_id=self._gen_id, config=full_cfg)

    def stop_generation(self) -> None:
        self._active_gen_id = 0
        if self._proc and self._proc.is_alive():
            self._send("stop")

    # ── event dispatch ─────────────────────────────────────────────────────

    def _dispatch_event(self, event: dict) -> None:
        # Let the base class handle status / trace / error / sig_event
        super()._dispatch_event(event)

        kind = str(event.get("event") or "")

        if kind == "result":
            if int(event.get("gen_id") or 0) == self._active_gen_id:
                img = event.get("image")
                idx = int(event.get("batch_index") or 0)
                if img is not None:
                    self.sig_image.emit(img, idx)

        elif kind == "progress":
            if int(event.get("gen_id") or 0) == self._active_gen_id:
                self.sig_progress.emit(
                    int(event.get("step")  or 0),
                    int(event.get("total") or 0),
                )

        elif kind == "resource":
            self.sig_resource.emit({
                "vram_used_mb": int(event.get("vram_used_mb") or 0),
                "vram_free_mb": int(event.get("vram_free_mb") or 0),
            })

        elif kind in ("status", "stopped", "unloaded"):
            # Fire sig_finished on any completion transition so callers that
            # used the old VisionEngine.sig_finished still work
            s = str(event.get("status") or kind).lower()
            if s in ("ready", "unloaded", "stopped"):
                self.sig_finished.emit()

        elif kind == "trace":
            # Base EngineProcess already emitted this trace and raw sig_event.
            # Do not re-emit here or the UI gets duplicate lines.
            return
