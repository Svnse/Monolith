from __future__ import annotations

from pathlib import Path

from core import investigation_runs


def test_investigation_run_writes_json_and_markdown(tmp_path: Path) -> None:
    run = investigation_runs.create_investigation("map the repo", root=tmp_path)

    run = investigation_runs.add_source_ref(
        run.run_id,
        kind="file",
        ref="core/plans.py",
        title="plans",
        note="planner store",
        root=tmp_path,
    )
    run = investigation_runs.update_synthesis(
        run.run_id,
        "The planner is propose-only.",
        linked_plan_uids=["plan-1"],
        linked_acu_ids=[7],
        root=tmp_path,
    )
    loaded = investigation_runs.load_investigation(run.run_id, root=tmp_path)

    assert loaded is not None
    assert loaded.source_refs[0].ref == "core/plans.py"
    assert loaded.linked_plan_uids == ("plan-1",)
    markdown_files = list(tmp_path.glob(f"{run.run_id}-*.md"))
    assert markdown_files
    assert "The planner is propose-only." in markdown_files[0].read_text(encoding="utf-8")
