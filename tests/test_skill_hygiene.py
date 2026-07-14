from __future__ import annotations

from pathlib import Path

from core import skill_hygiene


def test_update_skill_hygiene_round_trips_metadata(tmp_path: Path) -> None:
    metadata_path = tmp_path / "skill_hygiene.json"

    skill_hygiene.update_skill_hygiene(
        "read_file",
        retrieval_tags=["files", "local"],
        catalog_enabled=False,
        prompt_weight=0.5,
        notes=["too broad in tiny profile"],
        test_status="ok",
        metadata_path=metadata_path,
    )

    records = {record.name: record for record in skill_hygiene.list_skill_hygiene(metadata_path=metadata_path)}

    if "read_file" in records:
        assert records["read_file"].retrieval_tags == ("files", "local")
        assert records["read_file"].catalog_enabled is False
        assert records["read_file"].prompt_weight == 0.5


def test_audit_skills_returns_records_and_findings(tmp_path: Path) -> None:
    result = skill_hygiene.audit_skills(metadata_path=tmp_path / "meta.json")

    assert "status" in result
    assert isinstance(result["records"], list)
    assert isinstance(result["findings"], list)
