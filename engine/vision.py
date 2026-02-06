from __future__ import annotations

from PySide6.QtCore import QObject, QThread, Signal, QTimer

from core.state import AppState, SystemStatus


class PipelineLoader(QThread):
    trace = Signal(str)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = model_path

    def run(self) -> None:
        try:
            try:
                import torch
                from diffusers import StableDiffusionPipeline
            except ImportError as exc:
                raise RuntimeError(
                    "diffusers is not installed. pip install diffusers"
                ) from exc

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32

            if self.isInterruptionRequested():
                return

            self.trace.emit(f"loading pipeline: {self.model_path}")
            if self.model_path.endswith((".safetensors", ".ckpt")):
                pipe = StableDiffusionPipeline.from_single_file(
                    self.model_path,
                    torch_dtype=dtype,
                    safety_checker=None,
                    requires_safety_checker=False,
                )
            else:
                pipe = StableDiffusionPipeline.from_pretrained(
                    self.model_path,
                    torch_dtype=dtype,
                    safety_checker=None,
                    requires_safety_checker=False,
                )

            pipe = pipe.to(device)
            self.finished.emit(pipe)
        except Exception as exc:
            self.error.emit(str(exc))


class GenerationWorker(QThread):
    image = Signal(object)
    trace = Signal(str)
    done = Signal(bool, str)

    def __init__(
        self,
        pipe,
        prompt: str,
        steps: int,
        guidance: float,
        seed: int | None,
    ):
        super().__init__()
        self.pipe = pipe
        self.prompt = prompt
        self.steps = steps
        self.guidance = guidance
        self.seed = seed

    def run(self) -> None:
        completed = False
        err_msg = ""
        try:
            import torch

            if self.isInterruptionRequested():
                return

            device = "cuda" if torch.cuda.is_available() else "cpu"
            generator = None
            if self.seed is not None:
                generator = torch.Generator(device=device).manual_seed(self.seed)

            def _callback(step: int, timestep: int, latents) -> None:
                if self.isInterruptionRequested():
                    raise RuntimeError("Generation interrupted")

            self.trace.emit("generation started")
            result = self.pipe(
                self.prompt,
                num_inference_steps=self.steps,
                guidance_scale=self.guidance,
                generator=generator,
                callback=_callback,
                callback_steps=1,
            )
            if self.isInterruptionRequested():
                return
            self.image.emit(result.images[0])
            self.trace.emit("generation complete")
            completed = True
        except Exception as exc:
            err_msg = str(exc)
        finally:
            self.done.emit(completed, err_msg)


class VisionEngine(QObject):
    sig_token = Signal(str)
    sig_trace = Signal(str)
    sig_status = Signal(SystemStatus)
    sig_usage = Signal(int)
    sig_finished = Signal()
    sig_image = Signal(object)

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.pipe = None
        self.model_path: str | None = None
        self._loaded_path: str | None = None
        self.loader: PipelineLoader | None = None
        self.worker: GenerationWorker | None = None
        self._load_cancel_requested = False
        self._shutdown_requested = False

    def set_model_path(self, path: str) -> None:
        self.model_path = path
        QTimer.singleShot(0, lambda: self.sig_status.emit(SystemStatus.READY))

    def load_model(self) -> None:
        if self.loader and self.loader.isRunning():
            self.sig_trace.emit("VISION: load already in progress.")
            QTimer.singleShot(0, lambda: self.sig_status.emit(SystemStatus.READY))
            return

        if not self.model_path:
            self.sig_trace.emit("VISION: ERROR: No model selected.")
            self.sig_status.emit(SystemStatus.ERROR)
            return

        if self.pipe and self._loaded_path == self.model_path:
            self.sig_trace.emit("VISION: pipeline already loaded.")
            QTimer.singleShot(0, lambda: self.sig_status.emit(SystemStatus.READY))
            return

        if self.pipe and self._loaded_path != self.model_path:
            self.unload_model()

        self.sig_status.emit(SystemStatus.LOADING)
        self.sig_trace.emit("VISION: loading pipeline")
        self._load_cancel_requested = False
        self.loader = PipelineLoader(self.model_path)
        self.loader.trace.connect(self._emit_trace)
        self.loader.error.connect(self._on_load_error)
        self.loader.finished.connect(self._on_load_success)
        self.loader.finished.connect(self._cleanup_loader)
        self.loader.error.connect(self._cleanup_loader)
        self.loader.start()

    def _emit_trace(self, message: str) -> None:
        self.sig_trace.emit(f"VISION: {message}")

    def _on_load_success(self, pipe) -> None:
        if self._shutdown_requested:
            del pipe
            self.sig_status.emit(SystemStatus.READY)
            return

        if self._load_cancel_requested:
            del pipe
            self.pipe = None
            self._loaded_path = None
            self.sig_status.emit(SystemStatus.READY)
            self.sig_trace.emit("VISION: load cancelled")
            self.loader = None
            return

        self.pipe = pipe
        self._loaded_path = self.model_path
        self.sig_trace.emit("VISION: pipeline ready")
        self.sig_status.emit(SystemStatus.READY)
        self.loader = None

    def _on_load_error(self, err_msg: str) -> None:
        self.sig_trace.emit(f"VISION: ERROR: {err_msg}")
        self.sig_status.emit(SystemStatus.ERROR)
        self.loader = None

    def _cleanup_loader(self, *args, **kwargs) -> None:
        self.loader = None

    def unload_model(self) -> None:
        if self.loader and self.loader.isRunning():
            self._load_cancel_requested = True
            self.sig_trace.emit(
                "VISION: unload requested during load; will cancel after init completes"
            )
            return

        if self.worker and self.worker.isRunning():
            self.sig_trace.emit("VISION: ERROR: Cannot unload while generating.")
            return

        self.sig_status.emit(SystemStatus.UNLOADING)
        if self.pipe:
            del self.pipe
            self.pipe = None
            self._loaded_path = None

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        QTimer.singleShot(0, lambda: self.sig_status.emit(SystemStatus.READY))

    def generate(self, payload: dict) -> None:
        if not self.pipe:
            self.sig_trace.emit("VISION: ERROR: Model offline.")
            self.sig_status.emit(SystemStatus.READY)
            return

        if self.worker and self.worker.isRunning():
            self.sig_trace.emit("VISION: ERROR: Busy. Wait for completion.")
            return

        config = payload.get("config", payload)
        prompt = config.get("prompt", payload.get("prompt", ""))

        steps = int(config.get("steps", 25))
        guidance_scale = float(config.get("guidance_scale", 7.5))
        seed = config.get("seed")
        if isinstance(seed, int) and seed < 0:
            seed = None

        self.sig_status.emit(SystemStatus.RUNNING)
        self.worker = GenerationWorker(
            self.pipe,
            prompt,
            steps,
            guidance_scale,
            seed,
        )
        self.worker.image.connect(self.sig_image)
        self.worker.trace.connect(self._emit_trace)
        self.worker.done.connect(self._on_gen_finish)
        self.worker.start()

    def stop_generation(self) -> None:
        if self.loader and self.loader.isRunning():
            self._load_cancel_requested = True
            self.sig_trace.emit(
                "VISION: load cancel requested; will stop after initialization completes"
            )
            return

        if self.worker and self.worker.isRunning():
            self.worker.requestInterruption()

    def _on_gen_finish(self, completed: bool, err_msg: str) -> None:
        if completed:
            self.sig_finished.emit()
            self.sig_status.emit(SystemStatus.READY)
        elif err_msg == "Generation interrupted":
            self.sig_trace.emit("VISION: generation interrupted")
            self.sig_status.emit(SystemStatus.READY)
        else:
            self.sig_trace.emit(f"VISION: ERROR: {err_msg}")
            self.sig_status.emit(SystemStatus.ERROR)
        self.worker = None

    def shutdown(self) -> None:
        self._shutdown_requested = True
        self.stop_generation()

        if self.worker:
            self.worker.requestInterruption()
            self.worker.wait(1500)
            self.worker = None

        if self.loader and self.loader.isRunning():
            self._load_cancel_requested = True
            self.loader.wait(150)
