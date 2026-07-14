from __future__ import annotations


def test_observer_surfaces_emergence_signal_readonly(monkeypatch, tmp_path) -> None:
    from core import identity_milestones as m
    monkeypatch.setattr(m, "STORE_PATH", tmp_path / "identity_milestones.json")
    m.set_latest_emergence_signal({
        "message": "3 self-derived claim(s) at or above 0.3 confidentity; 12 new ACU(s).",
        "candidate_count": 3,
    })

    from addons.system.observer import runtime
    snap = runtime.build_observer_snapshot(turn_id="t-emergence")
    block = snap.get("block", "")

    assert "identity emergence" in block.lower()
    assert "identity_review" in block  # points the model at the bidden skill


def test_observer_silent_when_no_emergence_signal(monkeypatch, tmp_path) -> None:
    from core import identity_milestones as m
    monkeypatch.setattr(m, "STORE_PATH", tmp_path / "identity_milestones.json")
    # No signal set → ledger empty.
    from addons.system.observer import runtime
    snap = runtime.build_observer_snapshot(turn_id="t-quiet")
    assert "identity emergence" not in snap.get("block", "").lower()
