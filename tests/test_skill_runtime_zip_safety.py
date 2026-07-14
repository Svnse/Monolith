from __future__ import annotations

import zipfile
from pathlib import Path

import core.skill_runtime as skill_runtime


def _ctx(tmp_path: Path) -> skill_runtime.ToolExecutionContext:
    return skill_runtime.ToolExecutionContext(archive_dir=tmp_path)


def test_unzip_blocks_zip_slip_entry(tmp_path: Path) -> None:
    archive = tmp_path / "slip.zip"
    out_dir = tmp_path / "out"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("../escape.txt", "nope")

    result = skill_runtime.execute_unzip_file(
        {"archive": str(archive), "output_dir": str(out_dir)},
        _ctx(tmp_path),
    )

    assert "blocked" in result
    assert "zip-slip" in result
    assert not (tmp_path / "escape.txt").exists()


def test_unzip_blocks_total_uncompressed_size_limit(monkeypatch, tmp_path: Path) -> None:
    archive = tmp_path / "too_large.zip"
    out_dir = tmp_path / "out"
    monkeypatch.setattr(skill_runtime, "_MAX_ZIP_TOTAL_UNCOMPRESSED", 100)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("big.txt", "x" * 200)

    result = skill_runtime.execute_unzip_file(
        {"archive": str(archive), "output_dir": str(out_dir)},
        _ctx(tmp_path),
    )

    assert "blocked" in result
    assert "expands to" in result


def test_unzip_blocks_suspicious_compression_ratio(monkeypatch, tmp_path: Path) -> None:
    archive = tmp_path / "ratio.zip"
    out_dir = tmp_path / "out"
    monkeypatch.setattr(skill_runtime, "_MAX_ZIP_COMPRESSION_RATIO", 2)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("high_ratio.txt", "A" * 10_000)

    result = skill_runtime.execute_unzip_file(
        {"archive": str(archive), "output_dir": str(out_dir)},
        _ctx(tmp_path),
    )

    assert "blocked" in result
    assert "compression ratio" in result
