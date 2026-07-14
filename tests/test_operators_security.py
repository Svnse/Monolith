from __future__ import annotations

from core.operators import OperatorManager


def test_save_operator_strips_system_prompt_recursively(tmp_path) -> None:
    manager = OperatorManager()
    manager._operators_dir = tmp_path / "operators"

    payload = {
        "config": {"temperature": 0.7, "system_prompt": "secret"},
        "modules": [
            {"id": "chat", "config": {"system_prompt": "hidden", "max_tokens": 100}},
            {"id": "nested", "config": {"inner": {"system_prompt": "also hidden"}}},
        ],
        "extra": [{"system_prompt": "nope"}],
    }
    path, _drift = manager.save_operator("Atlas", payload)
    loaded = manager.load_operator("Atlas")

    assert path.exists()
    assert "system_prompt" not in str(loaded)
    assert loaded["config"]["temperature"] == 0.7
    assert loaded["modules"][0]["config"]["max_tokens"] == 100
