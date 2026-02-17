from engine.meritocracy import CompetenceHorizon, calculate_score


def test_calculate_score_zero_calls_no_errors():
    score = calculate_score(
        {"steps_used": 5, "budget": 10, "tool_calls": [], "error_flags": [], "completed": True}
    )
    assert score == 87.5


def test_calculate_score_critical_error_is_zero_correctness():
    score = calculate_score(
        {
            "steps_used": 1,
            "budget": 10,
            "tool_calls": [{"name": "read_file", "arguments": {"path": "a"}}],
            "error_flags": ["CRITICAL_ERROR"],
            "completed": True,
        }
    )
    assert score == 0.0


def test_calculate_score_incomplete_trace_is_zero():
    score = calculate_score({"steps_used": 1, "budget": 10, "tool_calls": [], "error_flags": []})
    assert score == 0.0


def test_promotion_and_immediate_reset(tmp_path):
    horizon = CompetenceHorizon(db_path=tmp_path / "merit.sqlite3")
    user_id = "u1"

    for _ in range(3):
        horizon.record_score(user_id, 81)

    config = horizon.resolve_config(user_id)
    assert config["max_steps"] == 25


def test_immediate_demote_to_c_on_critical_error(tmp_path):
    horizon = CompetenceHorizon(db_path=tmp_path / "merit.sqlite3")
    user_id = "u2"
    for _ in range(3):
        horizon.record_score(user_id, 81)

    score = horizon.update_after_trace(
        user_id,
        {
            "steps_used": 10,
            "budget": 25,
            "tool_calls": [{"name": "search", "arguments": {"q": "x"}}],
            "error_flags": ["CRITICAL_ERROR"],
            "completed": True,
        },
    )

    assert score == 0.0
    config = horizon.resolve_config(user_id)
    assert config["max_steps"] == 12
