from __future__ import annotations

from pathlib import Path

import core.skill_runtime as skill_runtime


def _ctx(tmp_path: Path) -> skill_runtime.ToolExecutionContext:
    return skill_runtime.ToolExecutionContext(archive_dir=tmp_path)


def test_calculate_allows_small_expressions(tmp_path: Path) -> None:
    result = skill_runtime.execute_calculate({"expr": "2 ** 8"}, _ctx(tmp_path))
    assert result == "[calculate: 2 ** 8 = 256]"


def test_calculate_blocks_excessive_power_growth(tmp_path: Path) -> None:
    result = skill_runtime.execute_calculate({"expr": "2 ** 100000"}, _ctx(tmp_path))
    assert "error" in result
    assert "too large" in result


def test_calculate_blocks_overly_complex_expression(tmp_path: Path) -> None:
    expr = "+".join(["1"] * 400)
    result = skill_runtime.execute_calculate({"expr": expr}, _ctx(tmp_path))
    assert "expression too complex" in result
