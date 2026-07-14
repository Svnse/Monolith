from pathlib import Path

import torch
import torchaudio

from core.mixdown import render_arrangement, render_to_wav, write_wav, DEFAULT_SR


def _write_clip(tmp_path: Path, name: str, value: float, seconds: float, sr: int) -> str:
    n = int(seconds * sr)
    buf = torch.full((1, n), float(value))
    path = tmp_path / f"{name}.wav"
    torchaudio.save(str(path), buf, sr)
    return str(path)


def _project(bpm, tracks):
    return {"bpm": bpm, "tracks": tracks}


def test_placement_offset_and_silence(tmp_path):
    sr = 8000
    clip = _write_clip(tmp_path, "a", 0.5, 1.0, sr)  # 1s of 0.5
    project = _project(60.0, [{  # 60 bpm -> 1 beat == 1 second == sr samples
        "volume": 1.0, "muted": False,
        "placements": [{"clip_id": "c1", "start_beat": 2.0, "length_beats": None,
                        "gain": 1.0, "muted": False}],
    }])
    buf, out_sr = render_arrangement(project, {"c1": {"path": clip}}, target_sr=sr)
    assert out_sr == sr
    assert float(buf[: 2 * sr].abs().max()) < 1e-3          # silent before beat 2
    assert float(buf[2 * sr : 3 * sr].abs().max()) > 0.4    # energy during the clip
    assert float(buf[3 * sr + 100 :].abs().max()) < 1e-3    # silent after (within tail)


def test_gain_scales_amplitude(tmp_path):
    sr = 8000
    clip = _write_clip(tmp_path, "a", 0.8, 1.0, sr)
    project = _project(60.0, [{
        "volume": 0.5, "muted": False,
        "placements": [{"clip_id": "c1", "start_beat": 0.0, "length_beats": None,
                        "gain": 0.5, "muted": False}],
    }])
    buf, _ = render_arrangement(project, {"c1": {"path": clip}}, target_sr=sr)
    # 0.8 * track 0.5 * placement 0.5 = 0.2
    assert abs(float(buf[: sr].abs().max()) - 0.2) < 0.02


def test_normalize_on_overlap(tmp_path):
    sr = 8000
    clip = _write_clip(tmp_path, "a", 0.8, 1.0, sr)
    project = _project(60.0, [{
        "volume": 1.0, "muted": False,
        "placements": [
            {"clip_id": "c1", "start_beat": 0.0, "length_beats": None, "gain": 1.0, "muted": False},
            {"clip_id": "c1", "start_beat": 0.0, "length_beats": None, "gain": 1.0, "muted": False},
        ],
    }])
    buf, _ = render_arrangement(project, {"c1": {"path": clip}}, target_sr=sr)
    assert float(buf.abs().max()) <= 1.0 + 1e-4             # 0.8+0.8=1.6 -> normalized
    assert float(buf.abs().max()) > 0.99                    # peak pushed to ~1.0


def test_missing_clip_is_skipped(tmp_path):
    sr = 8000
    project = _project(60.0, [{
        "volume": 1.0, "muted": False,
        "placements": [{"clip_id": "nope", "start_beat": 0.0, "length_beats": None,
                        "gain": 1.0, "muted": False}],
    }])
    buf, _ = render_arrangement(project, {}, target_sr=sr)
    assert float(buf.abs().max()) < 1e-6


def test_loop_fill_tiles_short_clip(tmp_path):
    sr = 8000
    clip = _write_clip(tmp_path, "a", 0.5, 1.0, sr)        # 1s clip
    project = _project(60.0, [{                            # length 2 beats == 2s
        "volume": 1.0, "muted": False,
        "placements": [{"clip_id": "c1", "start_beat": 0.0, "length_beats": 2.0,
                        "gain": 1.0, "muted": False}],
    }])
    buf, _ = render_arrangement(project, {"c1": {"path": clip}}, target_sr=sr)
    assert float(buf[: 2 * sr].abs().min()) > 0.4          # energy across the full 2s (tiled)
    assert float(buf[2 * sr + 100 :].abs().max()) < 1e-3   # silent after


def test_truncate_long_clip(tmp_path):
    sr = 8000
    clip = _write_clip(tmp_path, "a", 0.5, 2.0, sr)        # 2s clip
    project = _project(60.0, [{                            # length 1 beat == 1s
        "volume": 1.0, "muted": False,
        "placements": [{"clip_id": "c1", "start_beat": 0.0, "length_beats": 1.0,
                        "gain": 1.0, "muted": False}],
    }])
    buf, _ = render_arrangement(project, {"c1": {"path": clip}}, target_sr=sr)
    assert float(buf[: sr].abs().min()) > 0.4
    assert float(buf[sr + 100 :].abs().max()) < 1e-3


def test_track_mute_silences(tmp_path):
    sr = 8000
    clip = _write_clip(tmp_path, "a", 0.5, 1.0, sr)
    project = _project(60.0, [{
        "volume": 1.0, "muted": True,
        "placements": [{"clip_id": "c1", "start_beat": 0.0, "length_beats": None,
                        "gain": 1.0, "muted": False}],
    }])
    buf, _ = render_arrangement(project, {"c1": {"path": clip}}, target_sr=sr)
    assert float(buf.abs().max()) < 1e-6


def test_placement_mute_silences(tmp_path):
    sr = 8000
    clip = _write_clip(tmp_path, "a", 0.5, 1.0, sr)
    project = _project(60.0, [{
        "volume": 1.0, "muted": False,
        "placements": [{"clip_id": "c1", "start_beat": 0.0, "length_beats": None,
                        "gain": 1.0, "muted": True}],
    }])
    buf, _ = render_arrangement(project, {"c1": {"path": clip}}, target_sr=sr)
    assert float(buf.abs().max()) < 1e-6


def test_resample_preserves_time(tmp_path):
    src_sr = 8000
    clip = _write_clip(tmp_path, "a", 0.5, 1.0, src_sr)    # 1s at 8k
    project = _project(60.0, [{
        "volume": 1.0, "muted": False,
        "placements": [{"clip_id": "c1", "start_beat": 0.0, "length_beats": None,
                        "gain": 1.0, "muted": False}],
    }])
    buf, out_sr = render_arrangement(project, {"c1": {"path": clip}}, target_sr=16000)
    assert out_sr == 16000
    # 1 second of audio must still be ~1 second at the new rate
    assert float(buf[: 16000].abs().max()) > 0.4
    assert float(buf[16000 + 200 :].abs().max()) < 1e-3


def test_write_wav_roundtrip_mono(tmp_path):
    sr = 8000
    buf = torch.full((sr,), 0.4)                            # 1-D mono buffer
    out = write_wav(buf, sr, tmp_path / "w.wav")
    back, back_sr = torchaudio.load(out)
    assert back_sr == sr
    assert back.dim() == 2 and back.shape[0] == 1           # written as mono [1, frames]
    assert back.shape[1] == sr
    assert abs(float(back.abs().max()) - 0.4) < 0.02        # value survives the round-trip


def test_render_to_wav_produces_playable_file(tmp_path):
    sr = 8000
    clip = _write_clip(tmp_path, "a", 0.5, 1.0, sr)
    project = _project(60.0, [{
        "volume": 1.0, "muted": False,
        "placements": [{"clip_id": "c1", "start_beat": 1.0, "length_beats": None,
                        "gain": 1.0, "muted": False}],
    }])
    out = render_to_wav(project, {"c1": {"path": clip}}, tmp_path / "mix.wav", target_sr=sr)
    back, back_sr = torchaudio.load(out)
    assert back_sr == sr
    assert back.shape[0] == 1                               # mono file
    assert float(back[0, : sr].abs().max()) < 1e-2         # silent during beat 0..1
    assert float(back[0, sr : 2 * sr].abs().max()) > 0.4   # energy where the clip sits
