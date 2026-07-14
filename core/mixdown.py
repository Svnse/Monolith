from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torchaudio

DEFAULT_SR = 44100


def _beats_to_samples(beats: float, bpm: float, sr: int) -> int:
    seconds = (float(beats) / max(1.0, float(bpm))) * 60.0
    return int(round(seconds * sr))


def _load_mono(path: str, target_sr: int) -> torch.Tensor:
    """Load an audio file as a 1-D mono float tensor at target_sr."""
    waveform, sr = torchaudio.load(str(path))  # [channels, frames]
    if waveform.dim() == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    elif waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform.reshape(-1)


def render_arrangement(
    project: dict[str, Any],
    clips_by_id: dict[str, dict[str, Any]],
    *,
    target_sr: int | None = None,
    tail_seconds: float = 0.25,
) -> tuple[torch.Tensor, int]:
    bpm = float(project.get("bpm") or 120.0)
    sr = int(target_sr or DEFAULT_SR)
    rendered: list[tuple[int, torch.Tensor]] = []
    end_sample = 0

    for track in project.get("tracks") or []:
        if track.get("muted"):
            continue
        track_gain = float(track.get("volume") or 1.0)
        for placement in track.get("placements") or []:
            if placement.get("muted"):
                continue
            clip = clips_by_id.get(str(placement.get("clip_id") or ""))
            if not clip:
                continue
            path = str(clip.get("path") or "")
            if not path or not Path(path).exists():
                continue
            samples = _load_mono(path, sr)
            samples = _fit_length(samples, placement, bpm, sr)
            gain = track_gain * float(placement.get("gain") or 1.0)
            samples = samples * gain
            start = _beats_to_samples(placement.get("start_beat") or 0.0, bpm, sr)
            rendered.append((start, samples))
            end_sample = max(end_sample, start + samples.shape[0])

    total = end_sample + int(round(tail_seconds * sr))
    master = torch.zeros(max(1, total))
    for start, samples in rendered:
        master[start : start + samples.shape[0]] += samples

    peak = float(master.abs().max()) if master.numel() else 0.0
    if peak > 1.0:
        master = master / peak
    return master, sr


def _fit_length(samples: torch.Tensor, placement: dict[str, Any], bpm: float, sr: int) -> torch.Tensor:
    length_beats = placement.get("length_beats")
    if length_beats is None:
        return samples                                     # natural length (one-shot)
    target_len = _beats_to_samples(length_beats, bpm, sr)
    n = samples.shape[0]
    if target_len <= 0 or n == 0:
        return torch.zeros(max(0, target_len))
    if n >= target_len:
        return samples[:target_len]                        # truncate
    reps = (target_len + n - 1) // n
    return samples.repeat(reps)[:target_len]               # loop-tile


def write_wav(buffer: torch.Tensor, sr: int, path: str | Path) -> str:
    out = buffer.unsqueeze(0) if buffer.dim() == 1 else buffer
    torchaudio.save(str(path), out, int(sr))
    return str(path)


def render_to_wav(
    project: dict[str, Any],
    clips_by_id: dict[str, dict[str, Any]],
    path: str | Path,
    *,
    target_sr: int | None = None,
) -> str:
    buffer, sr = render_arrangement(project, clips_by_id, target_sr=target_sr)
    return write_wav(buffer, sr, path)
