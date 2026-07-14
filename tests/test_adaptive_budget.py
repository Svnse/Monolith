from __future__ import annotations

from core.adaptive_budget import (
    AdaptiveBudgetConfig,
    BudgetTier,
    adaptive_budget_interceptor,
    compute_complexity_score,
    score_to_tier,
    _adjust_tier_for_model,
    _TIERS,
)


def _msgs(*pairs: tuple[str, str]) -> list[dict]:
    """Build message list from (role, content) pairs."""
    return [{"role": r, "content": c} for r, c in pairs]


def test_simple_greeting_scores_low():
    msgs = _msgs(("system", "You are Monolith."), ("user", "hi"))
    cfg = AdaptiveBudgetConfig()
    score, feats = compute_complexity_score(msgs, {}, cfg)
    assert score < 25, f"Simple greeting should score < 25, got {score}"


def test_complex_multipart_scores_high():
    msgs = _msgs(
        ("system", "You are Monolith."),
        ("user", (
            "I need you to do several things:\n"
            "1. Find all Python files in C:\\Users\\test\\project containing 'TODO'\n"
            "2. Read each file and extract the TODO comments\n"
            "3. If any file has more than 5 TODOs, flag it\n"
            "4. Then write a summary to C:\\Users\\test\\report.md\n"
            "5. Finally, run the tests to make sure nothing broke\n"
            "Also, if the grep fails, try a different pattern."
        )),
    )
    cfg = AdaptiveBudgetConfig()
    score, feats = compute_complexity_score(msgs, {}, cfg)
    assert score > 50, f"Complex multipart should score > 50, got {score}"
    assert feats["multipart"] > 0
    assert feats["technical"] > 0
    assert feats["conditionals"] > 0


def test_error_context_boosts_score():
    msgs = _msgs(
        ("system", "You are Monolith."),
        ("user", "read the file"),
        ("assistant", "Error: file not found. Traceback follows..."),
        ("user", "try again"),
    )
    cfg = AdaptiveBudgetConfig()
    score_with_errors, feats = compute_complexity_score(msgs, {}, cfg)

    msgs_clean = _msgs(
        ("system", "You are Monolith."),
        ("user", "read the file"),
        ("assistant", "Here is the file content."),
        ("user", "try again"),
    )
    score_clean, _ = compute_complexity_score(msgs_clean, {}, cfg)

    assert score_with_errors > score_clean, (
        f"Error context should boost score: {score_with_errors} vs {score_clean}"
    )


def test_tier_boundaries():
    cfg = AdaptiveBudgetConfig()
    assert score_to_tier(0, cfg).name == "MINIMAL"
    assert score_to_tier(25, cfg).name == "MINIMAL"
    assert score_to_tier(26, cfg).name == "STANDARD"
    assert score_to_tier(50, cfg).name == "STANDARD"
    assert score_to_tier(51, cfg).name == "DETAILED"
    assert score_to_tier(75, cfg).name == "DETAILED"
    assert score_to_tier(76, cfg).name == "EXHAUSTIVE"
    assert score_to_tier(100, cfg).name == "EXHAUSTIVE"


def test_interceptor_injects_guidance():
    msgs = _msgs(
        ("system", "You are Monolith."),
        ("user", "Explain quantum computing in detail with examples"),
    )
    config = {"max_tokens": 8192}
    result = adaptive_budget_interceptor(msgs, config)

    assert result is not None
    assert len(result) == 3  # system + injected tagged guidance + user
    guidance_msg = result[-2]
    assert guidance_msg["role"] == "user"
    assert guidance_msg.get("source") == "adaptive_budget"
    assert "[BUDGET GUIDANCE]" in guidance_msg["content"]
    assert result[-1]["role"] == "user"
    assert result[-1]["content"] == "Explain quantum computing in detail with examples"


def test_interceptor_caps_max_tokens():
    msgs = _msgs(("system", "You are Monolith."), ("user", "hi"))
    config = {"max_tokens": 8192}
    result = adaptive_budget_interceptor(msgs, config)

    assert result is not None
    # Simple greeting → MINIMAL tier → 512 cap
    assert config["max_tokens"] <= 512


def test_interceptor_disabled_returns_none():
    from core.adaptive_budget import _cached_config
    import core.adaptive_budget as ab

    # Force disabled config
    old = ab._cached_config
    ab._cached_config = AdaptiveBudgetConfig(enabled=False)
    try:
        msgs = _msgs(("system", "You are Monolith."), ("user", "hello"))
        result = adaptive_budget_interceptor(msgs, {"max_tokens": 2048})
        assert result is None
    finally:
        ab._cached_config = old


def test_no_double_injection():
    msgs = _msgs(
        ("system", "You are Monolith."),
        ("user", "[BUDGET GUIDANCE] Respond concisely."),
        ("user", "what time is it"),
    )
    config = {"max_tokens": 2048}
    result = adaptive_budget_interceptor(msgs, config)
    assert result is None, "Should not double-inject budget guidance"


def test_budget_guidance_does_not_mutate_user_message():
    msgs = _msgs(
        ("system", "You are Monolith."),
        ("user", "original user content"),
    )
    config = {"max_tokens": 8192}
    result = adaptive_budget_interceptor(msgs, config)

    assert result is not None
    assert result[-1]["role"] == "user"
    assert result[-1]["content"] == "original user content"


def test_model_aware_thinking_boost():
    tier = _TIERS[1]  # STANDARD, 1024
    preset = {"capabilities": {"supports_thinking": True}, "context_window": 32768}
    adjusted = _adjust_tier_for_model(tier, preset)
    assert adjusted.max_tokens == 2048, "Thinking model should get 2x tokens"


def test_model_aware_small_context_clamp():
    tier = _TIERS[2]  # DETAILED, 2048
    preset = {"capabilities": {}, "context_window": 4096}
    adjusted = _adjust_tier_for_model(tier, preset)
    assert adjusted.max_tokens == 1024, "Small context model should clamp to 1024"


def test_max_tokens_never_exceeds_user_setting():
    msgs = _msgs(
        ("system", "You are Monolith."),
        ("user", (
            "1. Do this\n2. Do that\n3. Also this\n"
            "If it fails, try something else. Check C:\\project\\src\\main.py"
        )),
    )
    user_max = 256
    config = {"max_tokens": user_max}
    adaptive_budget_interceptor(msgs, config)
    assert config["max_tokens"] <= user_max
