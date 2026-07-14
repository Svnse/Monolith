from __future__ import annotations

from pathlib import Path

from core import model_lab, turn_trace


def test_create_compare_run_uses_blind_labels(tmp_path: Path) -> None:
    run = model_lab.create_compare_run(
        "answer this",
        [
            {"model_identity": {"id": "m1"}, "output": "one"},
            {"model_identity": {"id": "m2"}, "output": "two"},
        ],
        root=tmp_path,
    )

    assert [candidate.blind_label for candidate in run.candidates] == ["A", "B"]
    assert (tmp_path / f"{run.run_id}.json").exists()
    assert run.to_dict(reveal_identity=False)["candidates"][0]["model_identity"] == {"hidden": True}


def test_record_vote_defaults_to_non_training_outcome(tmp_path: Path) -> None:
    turn_trace.set_db_path(tmp_path / "turn_trace.sqlite3")
    try:
        artifact_root = tmp_path / "artifacts"
        run = model_lab.create_compare_run(
            "prompt",
            [
                {"model_identity": {"id": "m1"}, "output": "one"},
                {"model_identity": {"id": "m2"}, "output": "two"},
            ],
            root=artifact_root,
        )

        updated = model_lab.record_vote(
            run.run_id,
            winning_label="B",
            rating_value=88,
            reason="clearer",
            turn_id="turn-model-lab",
            root=artifact_root,
        )
        outcomes = turn_trace.list_outcomes_for_turn("turn-model-lab")

        assert updated.votes[-1]["winning_label"] == "B"
        assert outcomes[0].kind == "copy"
        assert outcomes[0].metadata["training_enabled"] is False
        assert outcomes[0].metadata["rating_value"] == 88
    finally:
        turn_trace.set_db_path(None)
