from __future__ import annotations

import zipfile
from pathlib import Path

import yaml

from core import substrate_snapshot


def test_snapshot_export_redacts_config_secrets(monkeypatch, tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    skills_dir = tmp_path / "skills"
    log_dir = tmp_path / "logs"
    artifact_dir = tmp_path / "artifacts"
    config_dir.mkdir()
    (skills_dir / "demo").mkdir(parents=True)
    log_dir.mkdir()
    (config_dir / "config.yaml").write_text(
        yaml.safe_dump({"llm": {"api_key": "secret", "api_base": "http://x"}}),
        encoding="utf-8",
    )
    (skills_dir / "demo" / "SKILL.md").write_text("---\nname: demo\ndescription: Demo\n---\n", encoding="utf-8")
    monkeypatch.setattr(substrate_snapshot, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(substrate_snapshot, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr(substrate_snapshot, "LOG_DIR", log_dir)
    monkeypatch.setattr(substrate_snapshot, "SNAPSHOT_DIR", artifact_dir)

    result = substrate_snapshot.export_snapshot(output_path=artifact_dir / "snap.zip")

    assert result["dry_run"] is False
    with zipfile.ZipFile(result["output_path"], "r") as zf:
        config_text = zf.read("config/config.yaml").decode("utf-8")
    assert "secret" not in config_text
    assert "<redacted>" in config_text


def test_restore_snapshot_dry_run_reports_targets(monkeypatch, tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    skills_dir = tmp_path / "skills"
    monkeypatch.setattr(substrate_snapshot, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(substrate_snapshot, "SKILLS_DIR", skills_dir)
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("config/config.yaml", "version: 1\n")
        zf.writestr("skills/demo/SKILL.md", "demo")

    result = substrate_snapshot.restore_snapshot_dry_run(archive)

    assert result["ok"] is True
    assert {change["kind"] for change in result["changes"]} == {"config", "skill"}
