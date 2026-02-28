"""
engine/_workers/audio_worker.py
Runs inside the audio subprocess.  Three modes share one process:

  STT  — speech-to-text via faster-whisper (preferred) / openai-whisper
  TTS  — text-to-speech via kokoro (preferred) / piper-tts
  MUSIC — music generation via audiocraft MusicGen

Ops received from AudioProcess:
  {"op": "load_stt",    "model_size": str, "model_path": str}
  {"op": "load_tts",    "model_path": str, "backend": "kokoro"|"piper"}
  {"op": "load_music",  "model_path": str}
  {"op": "transcribe",  "gen_id": int, "audio_path": str}
  {"op": "transcribe",  "gen_id": int, "pcm_bytes": bytes, "sample_rate": int}
  {"op": "speak",       "gen_id": int, "text": str, "voice": str, "speed": float}
  {"op": "generate_music", "gen_id": int, "config": dict}
  {"op": "stop"}
  {"op": "unload_all"}
  {"op": "shutdown"}
"""
from __future__ import annotations

import io
import traceback


# ── helpers ───────────────────────────────────────────────────────────────

def _emit(q, event: str, **kw) -> None:
    try:
        q.put_nowait({"event": event, **kw})
    except Exception:
        pass


def _vram_snapshot() -> tuple[int, int]:
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            return (total - free) // (1024 * 1024), free // (1024 * 1024)
    except Exception:
        pass
    return 0, 0


# ── STT ───────────────────────────────────────────────────────────────────

def _load_stt(model_size: str, model_path: str, q):
    """Returns (model, backend_name) or raises."""
    # Try faster-whisper first (CTranslate2, much faster)
    try:
        from faster_whisper import WhisperModel
        path_or_size = model_path or model_size or "base"
        _emit(q, "trace", message=f"STT: loading faster-whisper ({path_or_size})")
        model = WhisperModel(path_or_size, device="auto", compute_type="auto")
        _emit(q, "trace", message="STT: faster-whisper ready")
        return model, "faster_whisper"
    except ImportError:
        pass

    # Fall back to openai-whisper
    try:
        import whisper
        size = model_size or "base"
        _emit(q, "trace", message=f"STT: loading openai-whisper ({size})")
        model = whisper.load_model(size)
        _emit(q, "trace", message="STT: openai-whisper ready")
        return model, "openai_whisper"
    except ImportError:
        raise RuntimeError(
            "No STT backend found. "
            "Install faster-whisper:  pip install faster-whisper\n"
            "  or openai-whisper:     pip install openai-whisper"
        )


def _transcribe(model, backend: str, audio_path: str | None,
                pcm_bytes: bytes | None, sample_rate: int, q) -> str:
    if backend == "faster_whisper":
        import numpy as np
        if audio_path:
            segments, info = model.transcribe(audio_path)
        else:
            # PCM bytes → float32 numpy
            arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            segments, info = model.transcribe(arr)
        text = " ".join(s.text for s in segments).strip()
        _emit(q, "trace", message=f"STT: lang={info.language} prob={info.language_probability:.2f}")
        return text
    else:
        import whisper, numpy as np
        if audio_path:
            result = model.transcribe(audio_path)
        else:
            arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            result = model.transcribe(arr)
        return result["text"].strip()


# ── TTS ───────────────────────────────────────────────────────────────────

def _load_tts(model_path: str, backend_pref: str, q):
    """Returns (model, backend_name) or raises."""
    if backend_pref in ("kokoro", "auto", ""):
        try:
            from kokoro import KPipeline
            _emit(q, "trace", message="TTS: loading kokoro")
            model = KPipeline(lang_code="a")   # "a" = American English
            _emit(q, "trace", message="TTS: kokoro ready")
            return model, "kokoro"
        except ImportError:
            if backend_pref == "kokoro":
                raise RuntimeError("kokoro not installed: pip install kokoro>=0.9")

    if backend_pref in ("piper", "auto", ""):
        try:
            from piper.voice import PiperVoice
            if not model_path:
                raise RuntimeError("piper requires a model_path (.onnx file)")
            _emit(q, "trace", message=f"TTS: loading piper from {model_path}")
            model = PiperVoice.load(model_path)
            _emit(q, "trace", message="TTS: piper ready")
            return model, "piper"
        except ImportError:
            pass

    raise RuntimeError(
        "No TTS backend found. "
        "Install kokoro:  pip install kokoro>=0.9\n"
        "  or piper-tts:  pip install piper-tts"
    )


def _speak(model, backend: str, text: str, voice: str, speed: float) -> bytes:
    """Returns raw WAV bytes."""
    if backend == "kokoro":
        import soundfile as sf
        import numpy as np
        buf = io.BytesIO()
        # kokoro KPipeline yields (graphemes, phonemes, audio_array)
        audio_chunks = []
        for _, _, audio in model(text, voice=voice or "af_heart", speed=speed):
            audio_chunks.append(audio)
        if not audio_chunks:
            return b""
        arr = np.concatenate(audio_chunks)
        sf.write(buf, arr, samplerate=24000, format="WAV")
        return buf.getvalue()

    elif backend == "piper":
        buf = io.BytesIO()
        with model.wave_writer(buf) as wav_writer:
            model.synthesize(text, wav_writer)
        return buf.getvalue()

    return b""


# ── MusicGen ──────────────────────────────────────────────────────────────

def _load_music(model_path: str, q):
    try:
        from audiocraft.models import MusicGen
        path_or_id = model_path or "facebook/musicgen-small"
        _emit(q, "trace", message=f"MUSIC: loading {path_or_id}")
        model = MusicGen.get_pretrained(path_or_id)
        _emit(q, "trace", message="MUSIC: model ready")
        return model
    except ImportError:
        raise RuntimeError(
            "audiocraft not installed: pip install audiocraft"
        )


def _generate_music(model, cfg: dict, q, gen_id: int) -> tuple[object, int]:
    """Returns (audio_array, sample_rate)."""
    prompt   = str(cfg.get("prompt") or "")
    duration = float(cfg.get("duration", 5.0))
    sr       = int(cfg.get("sample_rate", 32000))
    model.set_generation_params(duration=duration)
    _emit(q, "progress_text", gen_id=gen_id, message="generating music...")
    wav = model.generate([prompt])
    audio = wav[0].cpu().numpy()
    return audio, sr


# ── main loop ─────────────────────────────────────────────────────────────

def main(to_worker, from_worker) -> None:
    """Entry point — called by AudioProcess in the child process."""
    stt_model    = None
    stt_backend  = ""
    tts_model    = None
    tts_backend  = ""
    music_model  = None
    _interrupt   = [False]

    while True:
        try:
            msg = to_worker.get(timeout=2.0)
        except Exception:
            continue

        op = str(msg.get("op") or "")

        # ── shutdown ───────────────────────────────────────────────────────
        if op == "shutdown":
            _emit(from_worker, "status", status="unloaded")
            break

        # ── unload all ────────────────────────────────────────────────────
        elif op == "unload_all":
            stt_model = tts_model = music_model = None
            stt_backend = tts_backend = ""
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            _emit(from_worker, "status", status="unloaded")

        # ── load STT ──────────────────────────────────────────────────────
        elif op == "load_stt":
            _emit(from_worker, "status", status="loading")
            try:
                stt_model, stt_backend = _load_stt(
                    msg.get("model_size") or "base",
                    msg.get("model_path") or "",
                    from_worker,
                )
                _emit(from_worker, "status", status="ready")
                _emit(from_worker, "loaded", mode="stt", backend=stt_backend)
            except Exception as exc:
                _emit(from_worker, "error",
                      message=f"STT load failed: {exc}\n{traceback.format_exc()[-400:]}")

        # ── load TTS ──────────────────────────────────────────────────────
        elif op == "load_tts":
            _emit(from_worker, "status", status="loading")
            try:
                tts_model, tts_backend = _load_tts(
                    msg.get("model_path") or "",
                    msg.get("backend")    or "auto",
                    from_worker,
                )
                _emit(from_worker, "status", status="ready")
                _emit(from_worker, "loaded", mode="tts", backend=tts_backend)
            except Exception as exc:
                _emit(from_worker, "error",
                      message=f"TTS load failed: {exc}\n{traceback.format_exc()[-400:]}")

        # ── load MusicGen ─────────────────────────────────────────────────
        elif op == "load_music":
            _emit(from_worker, "status", status="loading")
            try:
                music_model = _load_music(msg.get("model_path") or "", from_worker)
                _emit(from_worker, "status", status="ready")
                _emit(from_worker, "loaded", mode="music", backend="audiocraft")
            except Exception as exc:
                _emit(from_worker, "error",
                      message=f"Music load failed: {exc}\n{traceback.format_exc()[-400:]}")

        # ── transcribe ────────────────────────────────────────────────────
        elif op == "transcribe":
            if stt_model is None:
                _emit(from_worker, "error", message="transcribe: STT model not loaded")
                continue
            gen_id     = int(msg.get("gen_id") or 0)
            audio_path = msg.get("audio_path")
            pcm_bytes  = msg.get("pcm_bytes")
            sample_rate = int(msg.get("sample_rate") or 16000)
            _interrupt[0] = False
            _emit(from_worker, "status", status="running")
            try:
                text = _transcribe(stt_model, stt_backend,
                                   audio_path, pcm_bytes, sample_rate, from_worker)
                _emit(from_worker, "transcription",
                      gen_id=gen_id, text=text)
                _emit(from_worker, "status", status="ready")
            except Exception as exc:
                _emit(from_worker, "error",
                      message=f"transcribe failed: {exc}\n{traceback.format_exc()[-400:]}")

        # ── speak ─────────────────────────────────────────────────────────
        elif op == "speak":
            if tts_model is None:
                _emit(from_worker, "error", message="speak: TTS model not loaded")
                continue
            gen_id = int(msg.get("gen_id") or 0)
            text   = str(msg.get("text") or "")
            voice  = str(msg.get("voice") or "af_heart")
            speed  = float(msg.get("speed") or 1.0)
            _emit(from_worker, "status", status="running")
            try:
                wav_bytes = _speak(tts_model, tts_backend, text, voice, speed)
                _emit(from_worker, "audio",
                      gen_id=gen_id,
                      wav_bytes=wav_bytes,
                      sample_rate=24000,
                      channels=1)
                _emit(from_worker, "status", status="ready")
            except Exception as exc:
                _emit(from_worker, "error",
                      message=f"speak failed: {exc}\n{traceback.format_exc()[-400:]}")

        # ── generate music ────────────────────────────────────────────────
        elif op == "generate_music":
            if music_model is None:
                _emit(from_worker, "error", message="generate_music: music model not loaded")
                continue
            gen_id = int(msg.get("gen_id") or 0)
            cfg    = msg.get("config") or {}
            _emit(from_worker, "status", status="running")
            try:
                audio_arr, sr = _generate_music(music_model, cfg, from_worker, gen_id)
                _emit(from_worker, "music_result",
                      gen_id=gen_id, audio=audio_arr, sample_rate=sr)
                _emit(from_worker, "status", status="ready")
            except Exception as exc:
                _emit(from_worker, "error",
                      message=f"music gen failed: {exc}\n{traceback.format_exc()[-400:]}")

        # ── stop ──────────────────────────────────────────────────────────
        elif op == "stop":
            _interrupt[0] = True
