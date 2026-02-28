"""
engine/_workers/vision_worker.py
Runs inside the vision subprocess.  All heavy imports (torch, diffusers)
are deferred until actually needed — they never load in the main process.

Supported backends (auto-detected from model metadata):
  sd15  — StableDiffusionPipeline
  sdxl  — StableDiffusionXLPipeline
  flux  — FluxPipeline  (diffusers >= 0.30)

Generation config keys (all optional, sensible defaults):
  prompt, negative_prompt, width, height, steps, guidance_scale,
  seed, scheduler ("euler"|"dpm++"|"ddim"|"lcm"),
  lora_path, lora_scale, batch_size
"""
from __future__ import annotations

import traceback
from pathlib import Path


# ── helpers ───────────────────────────────────────────────────────────────

def _emit(q, event: str, **kw) -> None:
    try:
        q.put_nowait({"event": event, **kw})
    except Exception:
        pass


def _vram_snapshot() -> tuple[int, int]:
    """(vram_used_mb, vram_free_mb) — (0, 0) if CUDA unavailable."""
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            return (total - free) // (1024 * 1024), free // (1024 * 1024)
    except Exception:
        pass
    return 0, 0


def _detect_backend(model_path: str) -> str:
    """Heuristic: peek at model_index.json or filename to choose backend."""
    import json as _json
    p = Path(model_path)
    if p.is_dir():
        idx = p / "model_index.json"
        if idx.exists():
            try:
                data = _json.loads(idx.read_text())
                cls = str(data.get("_class_name", "")).lower()
                if "xl" in cls:
                    return "sdxl"
                if "flux" in cls:
                    return "flux"
            except Exception:
                pass
        return "sd15"
    name = p.name.lower()
    if "xl" in name:
        return "sdxl"
    if "flux" in name:
        return "flux"
    return "sd15"


def _load_pipeline(model_path: str, backend: str, dtype):
    p = Path(model_path)
    is_file = p.is_file()
    no_safety = dict(safety_checker=None, requires_safety_checker=False)

    if backend == "sdxl":
        from diffusers import StableDiffusionXLPipeline as Cls
    elif backend == "flux":
        from diffusers import FluxPipeline as Cls
        no_safety = {}      # Flux has no safety_checker arg
    else:
        from diffusers import StableDiffusionPipeline as Cls

    kwargs = dict(torch_dtype=dtype, **no_safety)
    if is_file:
        return Cls.from_single_file(model_path, **kwargs)
    return Cls.from_pretrained(model_path, **kwargs)


def _get_scheduler(name: str):
    from diffusers import (
        DPMSolverMultistepScheduler,
        EulerDiscreteScheduler,
        DDIMScheduler,
        LCMScheduler,
    )
    return {
        "euler": EulerDiscreteScheduler,
        "dpm++": DPMSolverMultistepScheduler,
        "ddim":  DDIMScheduler,
        "lcm":   LCMScheduler,
    }.get(str(name or "dpm++").lower(), DPMSolverMultistepScheduler)


def _pipeline_device_type(pipe) -> str:
    """
    Diffusers pipelines are not nn.Module instances, so they do not expose
    `.parameters()`. Resolve device from common pipeline components instead.
    """
    # Newer diffusers pipelines often expose `.device`
    dev = getattr(pipe, "device", None)
    if dev is not None:
        return str(getattr(dev, "type", dev))

    # Fall back to common module components
    for attr in ("unet", "transformer", "text_encoder", "vae"):
        module = getattr(pipe, attr, None)
        if module is None:
            continue
        try:
            return next(module.parameters()).device.type
        except Exception:
            continue

    return "cpu"


# ── main loop ─────────────────────────────────────────────────────────────

def main(to_worker, from_worker) -> None:
    """Entry point — called by VisionProcess in the child process."""
    pipe    = None
    backend = "sd15"
    _interrupt = [False]

    while True:
        # Block until we get a message (avoids busy-spin)
        try:
            msg = to_worker.get(timeout=2.0)
        except Exception:
            continue

        op = str(msg.get("op") or "")

        # ── shutdown ───────────────────────────────────────────────────────
        if op == "shutdown":
            _emit(from_worker, "status", status="unloaded")
            break

        # ── unload ─────────────────────────────────────────────────────────
        elif op == "unload":
            if pipe is not None:
                del pipe
                pipe = None
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            _emit(from_worker, "status", status="unloaded")

        # ── load ───────────────────────────────────────────────────────────
        elif op == "load":
            model_path = str(msg.get("model_path") or "")
            if not model_path:
                _emit(from_worker, "error", message="load: model_path is empty")
                continue

            # Unload any existing pipeline first
            if pipe is not None:
                del pipe
                pipe = None
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

            _emit(from_worker, "status", status="loading")
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                dtype  = torch.float16 if device == "cuda" else torch.float32

                backend = _detect_backend(model_path)
                _emit(from_worker, "trace",
                      message=f"backend={backend} device={device} path={model_path}")

                pipe = _load_pipeline(model_path, backend, dtype)
                pipe = pipe.to(device)

                used_mb, free_mb = _vram_snapshot()
                _emit(from_worker, "resource",
                      vram_used_mb=used_mb, vram_free_mb=free_mb)
                _emit(from_worker, "status", status="ready")
                _emit(from_worker, "trace",
                      message=f"pipeline ready — vram_used={used_mb}MB free={free_mb}MB")

            except Exception as exc:
                tb = traceback.format_exc()[-600:]
                _emit(from_worker, "error", message=f"load failed: {exc}\n{tb}")

        # ── generate ───────────────────────────────────────────────────────
        elif op == "generate":
            if pipe is None:
                _emit(from_worker, "error", message="generate: no pipeline loaded")
                continue

            gen_id = int(msg.get("gen_id") or 0)
            cfg    = msg.get("config") or {}

            prompt          = str(cfg.get("prompt")          or "")
            negative_prompt = str(cfg.get("negative_prompt") or "")
            width           = int(cfg.get("width",           512))
            height          = int(cfg.get("height",          512))
            steps           = int(cfg.get("steps",           25))
            guidance        = float(cfg.get("guidance_scale", 7.5))
            seed_val        = cfg.get("seed")
            scheduler_name  = str(cfg.get("scheduler",      "dpm++"))
            lora_path       = cfg.get("lora_path")
            lora_scale      = float(cfg.get("lora_scale",   0.8))
            batch_size      = max(1, int(cfg.get("batch_size", 1)))

            _interrupt[0] = False

            try:
                import torch

                # Swap scheduler (no model reload needed)
                try:
                    sched_cls = _get_scheduler(scheduler_name)
                    pipe.scheduler = sched_cls.from_config(pipe.scheduler.config)
                except Exception as sched_exc:
                    _emit(from_worker, "trace",
                          message=f"scheduler swap failed ({sched_exc}), keeping default")

                # LoRA
                lora_loaded = False
                if lora_path:
                    try:
                        pipe.load_lora_weights(lora_path)
                        pipe.set_adapters(["default_0"], [lora_scale])
                        lora_loaded = True
                        _emit(from_worker, "trace",
                              message=f"LoRA loaded: {lora_path} scale={lora_scale}")
                    except Exception as lora_exc:
                        _emit(from_worker, "trace",
                              message=f"LoRA load failed: {lora_exc}")

                # Seed
                device = _pipeline_device_type(pipe)
                generator = None
                if isinstance(seed_val, int) and seed_val >= 0:
                    generator = torch.Generator(device=device).manual_seed(seed_val)

                # Step callback — emits progress + respects stop requests
                total_ref = [steps]

                def _step_cb(pipe_inst, step_idx, _timestep, cb_kwargs):
                    if _interrupt[0]:
                        pipe_inst._interrupt = True
                    _emit(from_worker, "progress",
                          gen_id=gen_id,
                          step=step_idx + 1,
                          total=total_ref[0])
                    return cb_kwargs

                _emit(from_worker, "status", status="running")
                _emit(from_worker, "trace",
                      message=f"gen_id={gen_id} steps={steps} size={width}x{height}")

                # Build kwargs per backend
                gen_kwargs: dict = dict(
                    num_inference_steps=steps,
                    generator=generator,
                    callback_on_step_end=_step_cb,
                )
                if backend != "flux":
                    gen_kwargs["guidance_scale"]         = guidance
                    gen_kwargs["width"]                  = width
                    gen_kwargs["height"]                 = height
                    gen_kwargs["num_images_per_prompt"]  = batch_size
                    if negative_prompt:
                        gen_kwargs["negative_prompt"] = negative_prompt

                result = pipe(prompt, **gen_kwargs)

                # Unload LoRA
                if lora_loaded:
                    try:
                        pipe.unload_lora_weights()
                    except Exception:
                        pass

                if _interrupt[0]:
                    _emit(from_worker, "stopped")
                    _emit(from_worker, "status", status="ready")
                    continue

                for idx, img in enumerate(result.images):
                    _emit(from_worker, "result",
                          gen_id=gen_id, image=img, batch_index=idx)

                used_mb, free_mb = _vram_snapshot()
                _emit(from_worker, "resource",
                      vram_used_mb=used_mb, vram_free_mb=free_mb)
                _emit(from_worker, "status", status="ready")
                _emit(from_worker, "trace",
                      message=f"gen_id={gen_id} complete — vram_used={used_mb}MB")

            except Exception as exc:
                tb = traceback.format_exc()[-600:]
                _emit(from_worker, "error",
                      message=f"generate failed: {exc}\n{tb}")

        # ── stop ───────────────────────────────────────────────────────────
        elif op == "stop":
            _interrupt[0] = True
