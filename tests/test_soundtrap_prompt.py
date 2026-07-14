from core.soundtrap_prompt import build_loop_request, INSTRUMENTS


def test_prompt_contains_instrument_bpm_key():
    req = build_loop_request("bass", bpm=120.0, key="A minor")
    assert "bass" in req["prompt"].lower()
    assert "120 bpm" in req["prompt"]
    assert "key of A minor" in req["prompt"]
    assert req["instrument"] == "bass"


def test_duration_two_bars_at_120_is_4s():
    req = build_loop_request("drums", bpm=120.0)
    assert abs(req["duration"] - 4.0) < 1e-6


def test_duration_clamped_to_musicgen_range():
    assert build_loop_request("pad", bpm=10.0)["duration"] <= 30.0   # very slow -> clamp 30
    assert build_loop_request("lead", bpm=600.0)["duration"] >= 1.0  # very fast -> clamp 1


def test_unknown_instrument_falls_back_to_solo_phrase():
    req = build_loop_request("kalimba", bpm=100.0)
    assert "kalimba" in req["prompt"].lower()
    assert "no other instruments" in req["prompt"].lower()


def test_vibe_is_included_when_given():
    req = build_loop_request("keys", bpm=90.0, vibe="dark lo-fi")
    assert "dark lo-fi" in req["prompt"]


def test_palette_is_nonempty_and_lowercase():
    assert INSTRUMENTS
    assert all(name == name.lower() for name in INSTRUMENTS)
