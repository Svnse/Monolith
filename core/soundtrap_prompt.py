from __future__ import annotations

INSTRUMENTS = ["drums", "hats", "bass", "keys", "guitar", "lead", "pad", "fx"]

_BASE = {
    "drums": "solo acoustic drum kit groove, no other instruments",
    "hats": "solo hi-hat percussion pattern, no other instruments",
    "bass": "solo bass guitar riff, no drums, no other instruments",
    "keys": "solo electric piano chords, no other instruments",
    "guitar": "solo electric guitar riff, no other instruments",
    "lead": "solo synth lead melody, no other instruments",
    "pad": "solo warm synth pad chords, no other instruments",
    "fx": "atmospheric sound effect texture, no other instruments",
}


def build_loop_request(
    instrument: str,
    *,
    bpm: float,
    key: str = "",
    vibe: str = "",
    bars: int = 2,
) -> dict:
    name = str(instrument or "").strip().lower()
    base = _BASE.get(name, f"solo {name or 'instrument'}, no other instruments")
    parts = [base]
    if vibe.strip():
        parts.append(vibe.strip())
    parts.append(f"{int(round(float(bpm)))} bpm")
    if key.strip():
        parts.append(f"key of {key.strip()}")
    parts.append("seamless loop")
    seconds = (int(bars) * 4.0 / max(1.0, float(bpm))) * 60.0
    duration = max(1.0, min(30.0, seconds))
    return {"prompt": ", ".join(parts), "duration": round(duration, 2), "instrument": name}
