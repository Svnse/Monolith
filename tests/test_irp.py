from __future__ import annotations

from core import irp


def test_veracity_ranges_map_to_labels() -> None:
    assert irp.map_veracity_to_label(100) == "VERIFIED"
    assert irp.map_veracity_to_label(5) == "ACCEPTED"
    assert irp.map_veracity_to_label(1) == "PROVISIONAL"
    assert irp.map_veracity_to_label(0) == "CONTESTED"
    assert irp.map_veracity_to_label(-20) == "REJECTED"


def test_locked_claim_wins_over_numeric_veracity() -> None:
    row = {
        "canonical": "Origin 0 names Monolith's refusal posture",
        "veracity": -100,
        "source": "identity_origin_0",
    }
    assert irp.label_for_claim(row) == "LOCKED"


def test_label_text_applies_only_to_claim_and_observer_scopes() -> None:
    assert irp.label_text("claim text", scope="claim", label="ACCEPTED") == "[ACCEPTED] claim text"
    assert irp.label_text("observer text", scope="observer", label="PROVISIONAL") == "[PROVISIONAL] observer text"
    assert irp.label_text("normal answer", scope="chat", label="REJECTED") == "normal answer"


def test_rejected_state_overrides_veracity_for_claim_rows() -> None:
    row = {"canonical": "bad claim", "veracity": 100, "state": "rejected"}
    assert irp.label_for_claim(row) == "REJECTED"
