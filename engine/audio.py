"""
engine/audio.py  —  AudioProcess + MicRecorder
Subprocess-isolated audio engine (STT / TTS / MusicGen) + main-process
microphone capture.

AudioProcess signals
────────────────────
  sig_transcription  (str)              — STT result text
  sig_audio_ready    (bytes, int, int)  — (wav_bytes, sample_rate, channels)
  sig_music_result   (object, int)      — (numpy audio array, sample_rate)
  sig_progress       (str)              — human-readable progress message
  sig_resource       (dict)             — vram stats
  sig_finished       (str)             — mode that completed ("stt"|"tts"|"music")
  sig_loaded         (str, str)         — (mode, backend) on successful load

Usage
─────
  engine = AudioProcess()

  # Speech-to-text
  engine.load_stt()                          # model_size="base" by default
  engine.transcribe(audio_path="clip.wav")
  # → sig_transcription("hello world")

  # Text-to-speech
  engine.load_tts()                          # kokoro preferred
  engine.speak("Hello, I'm Monolith.")
  # → sig_audio_ready(wav_bytes, 24000, 1)

  # Music generation  (drop-in replacement for old AudioGenModule)
  engine.load_music(model_path="facebook/musicgen-small")
  engine.generate_music({"prompt": "lo-fi chill beat", "duration": 8.0})
  # → sig_music_result(np.array, 32000)

  # Microphone capture (stays in main process — no GPU)
  mic = MicRecorder()
  mic.sig_pcm_chunk.connect(lambda pcm, sr: engine.transcribe(pcm_bytes=pcm, sample_rate=sr))
  mic.start_recording()
  ...
  mic.stop_recording()
"""
from __future__ import annotations

from PySide6.QtCore import QObject, QThread, QTimer, Signal

from core.state import SystemStatus
from engine.engine_process import EngineProcess


# ── AudioProcess ──────────────────────────────────────────────────────────

class AudioProcess(EngineProcess):
    """Subprocess-isolated audio engine (STT + TTS + MusicGen)."""

    sig_transcription = Signal(str)              # STT text result
    sig_audio_ready   = Signal(bytes, int, int)  # (wav_bytes, sample_rate, channels)
    sig_music_result  = Signal(object, int)      # (np audio array, sample_rate)
    sig_progress      = Signal(str)              # progress text
    sig_resource      = Signal(dict)
    sig_finished      = Signal(str)              # mode that finished
    sig_loaded        = Signal(str, str)          # (mode, backend_name)

    def __init__(self) -> None:
        super().__init__()
        self._stt_config:   dict = {}
        self._tts_config:   dict = {}
        self._music_config: dict = {}

    # ── worker entry point ─────────────────────────────────────────────────
    @staticmethod
    def _worker_fn(to_worker, from_worker) -> None:
        from engine._workers import audio_worker
        audio_worker.main(to_worker, from_worker)

    # ── load ops ───────────────────────────────────────────────────────────

    def load_stt(self, model_size: str = "base", model_path: str = "") -> None:
        """Load Whisper STT model.  model_size: tiny|base|small|medium|large-v3"""
        if not self._ensure_proc():
            return
        self.sig_status.emit(SystemStatus.LOADING)
        self._send("load_stt", model_size=model_size, model_path=model_path)

    def load_tts(self, model_path: str = "", backend: str = "auto") -> None:
        """Load TTS model.  backend: "kokoro" | "piper" | "auto" """
        if not self._ensure_proc():
            return
        self.sig_status.emit(SystemStatus.LOADING)
        self._send("load_tts", model_path=model_path, backend=backend)

    def load_music(self, model_path: str = "") -> None:
        """Load MusicGen model.  model_path: local dir or HuggingFace id."""
        if not self._ensure_proc():
            return
        self.sig_status.emit(SystemStatus.LOADING)
        self._send("load_music", model_path=model_path)

    # ── EnginePort compat: generate routes based on payload["mode"] ────────

    def set_model_path(self, payload: dict) -> None:
        """Route to the appropriate sub-loader via payload["mode"]."""
        mode = str(payload.get("mode") or "music").lower()
        path = str(payload.get("path") or "")
        if mode == "stt":
            self._stt_config = {"model_path": path,
                                "model_size": payload.get("model_size", "base")}
        elif mode == "tts":
            self._tts_config = {"model_path": path,
                                "backend":    payload.get("backend", "auto")}
        else:
            self._music_config = {"model_path": path}
        QTimer.singleShot(0, lambda: self.sig_status.emit(SystemStatus.READY))

    def load_model(self) -> None:
        """Load using the most recent set_model_path config (defaults to music)."""
        if self._stt_config:
            self.load_stt(**self._stt_config)
        elif self._tts_config:
            self.load_tts(**self._tts_config)
        else:
            self.load_music(**self._music_config)

    def unload_model(self) -> None:
        if self._proc and self._proc.is_alive():
            self.sig_status.emit(SystemStatus.UNLOADING)
            self._send("unload_all")
        else:
            QTimer.singleShot(0, lambda: self.sig_status.emit(SystemStatus.READY))

    def generate(self, payload: dict) -> None:
        """
        Route based on payload["mode"]:
          "transcribe" → STT
          "speak"      → TTS
          "music" (default) → MusicGen
        """
        mode = str(payload.get("mode") or "music").lower()
        if mode == "transcribe":
            self.transcribe(
                audio_path=payload.get("audio_path"),
                pcm_bytes=payload.get("pcm_bytes"),
                sample_rate=int(payload.get("sample_rate") or 16000),
            )
        elif mode == "speak":
            self.speak(
                text=str(payload.get("text") or ""),
                voice=str(payload.get("voice") or "af_heart"),
                speed=float(payload.get("speed") or 1.0),
            )
        else:
            self.generate_music(payload)

    # ── task methods ───────────────────────────────────────────────────────

    def transcribe(self, audio_path: str | None = None,
                   pcm_bytes: bytes | None = None,
                   sample_rate: int = 16000) -> None:
        """Transcribe audio file or raw PCM bytes to text."""
        if not (self._proc and self._proc.is_alive()):
            self.sig_trace.emit("AUDIO: STT model not loaded")
            return
        self._gen_id += 1
        self._active_gen_id = self._gen_id
        self.sig_status.emit(SystemStatus.RUNNING)
        kwargs: dict = {"gen_id": self._gen_id, "sample_rate": sample_rate}
        if audio_path:
            kwargs["audio_path"] = audio_path
        if pcm_bytes is not None:
            kwargs["pcm_bytes"] = pcm_bytes
        self._send("transcribe", **kwargs)

    def speak(self, text: str, voice: str = "af_heart", speed: float = 1.0) -> None:
        """Synthesise speech from text."""
        if not (self._proc and self._proc.is_alive()):
            self.sig_trace.emit("AUDIO: TTS model not loaded")
            return
        self._gen_id += 1
        self._active_gen_id = self._gen_id
        self.sig_status.emit(SystemStatus.RUNNING)
        self._send("speak", gen_id=self._gen_id,
                   text=text, voice=voice, speed=speed)

    def generate_music(self, payload: dict) -> None:
        """Generate music from a text prompt (MusicGen)."""
        if not (self._proc and self._proc.is_alive()):
            self.sig_trace.emit("AUDIO: Music model not loaded")
            return
        cfg = payload.get("config", payload)
        self._gen_id += 1
        self._active_gen_id = self._gen_id
        self.sig_status.emit(SystemStatus.RUNNING)
        self._send("generate_music", gen_id=self._gen_id, config=cfg)

    # ── event dispatch ─────────────────────────────────────────────────────

    def _dispatch_event(self, event: dict) -> None:
        super()._dispatch_event(event)

        kind   = str(event.get("event") or "")
        gen_id = int(event.get("gen_id") or 0)
        is_cur = (gen_id == self._active_gen_id) or (gen_id == 0)

        if kind == "transcription" and is_cur:
            text = str(event.get("text") or "")
            self.sig_transcription.emit(text)
            self.sig_finished.emit("stt")

        elif kind == "audio" and is_cur:
            wav   = event.get("wav_bytes") or b""
            sr    = int(event.get("sample_rate") or 24000)
            ch    = int(event.get("channels")    or 1)
            self.sig_audio_ready.emit(wav, sr, ch)
            self.sig_finished.emit("tts")

        elif kind == "music_result" and is_cur:
            audio = event.get("audio")
            sr    = int(event.get("sample_rate") or 32000)
            self.sig_music_result.emit(audio, sr)
            self.sig_finished.emit("music")

        elif kind == "loaded":
            mode    = str(event.get("mode")    or "")
            backend = str(event.get("backend") or "")
            self.sig_loaded.emit(mode, backend)
            self.sig_trace.emit(f"AUDIO: {mode} loaded via {backend}")

        elif kind == "progress_text" and is_cur:
            self.sig_progress.emit(str(event.get("message") or ""))

        elif kind == "resource":
            self.sig_resource.emit({
                "vram_used_mb": int(event.get("vram_used_mb") or 0),
                "vram_free_mb": int(event.get("vram_free_mb") or 0),
            })

        elif kind == "trace":
            msg = str(event.get("message") or "")
            self.sig_trace.emit(f"AUDIO: {msg}" if not msg.startswith("AUDIO:") else msg)


# ── MicRecorder ───────────────────────────────────────────────────────────

class _MicThread(QThread):
    """Records PCM from the default microphone until stop() is called."""
    sig_chunk = Signal(bytes, int)   # (pcm_int16_bytes, sample_rate)
    sig_error = Signal(str)

    _SAMPLE_RATE   = 16000
    _CHUNK_FRAMES  = 1600    # 100 ms per chunk
    _CHANNELS      = 1

    def __init__(self) -> None:
        super().__init__()
        self._running = False

    def run(self) -> None:
        try:
            import sounddevice as sd
            import numpy as np
        except ImportError:
            self.sig_error.emit(
                "sounddevice not installed: pip install sounddevice"
            )
            return

        self._running = True
        try:
            with sd.InputStream(
                samplerate=self._SAMPLE_RATE,
                channels=self._CHANNELS,
                dtype="int16",
                blocksize=self._CHUNK_FRAMES,
            ) as stream:
                while self._running:
                    data, _ = stream.read(self._CHUNK_FRAMES)
                    self.sig_chunk.emit(data.tobytes(), self._SAMPLE_RATE)
        except Exception as exc:
            self.sig_error.emit(str(exc))

    def stop(self) -> None:
        self._running = False


class MicRecorder(QObject):
    """
    Push-to-talk / continuous microphone capture.
    Stays in the main process — no GPU needed.

    Signals:
      sig_pcm_chunk   (bytes, int)  — raw int16 PCM chunk + sample rate
      sig_recording   (bool)        — recording state changed
      sig_error       (str)
    """
    sig_pcm_chunk = Signal(bytes, int)
    sig_recording = Signal(bool)
    sig_error     = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._thread: _MicThread | None = None

    @property
    def is_recording(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def start_recording(self) -> None:
        if self.is_recording:
            return
        self._thread = _MicThread()
        self._thread.sig_chunk.connect(self.sig_pcm_chunk)
        self._thread.sig_error.connect(self.sig_error)
        self._thread.finished.connect(lambda: self.sig_recording.emit(False))
        self._thread.start()
        self.sig_recording.emit(True)

    def stop_recording(self) -> None:
        if self._thread and self._thread.isRunning():
            self._thread.stop()
            self._thread.wait(1000)
        self._thread = None

    def shutdown(self) -> None:
        self.stop_recording()
