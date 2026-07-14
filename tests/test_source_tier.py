from core.source_tier import SourceTier, classify_source_tiers, tier_label


def test_tool_only_turn_is_tool_not_generation():
    # no <think>, but a tool ran — absence of trace != generation
    r = classify_source_tiers(
        raw='<tool_call>{"name":"grep"}</tool_call> done', public="done", tools_used=()
    )
    assert r.answer_tier is SourceTier.TOOL
    assert r.had_tool is True
    assert r.had_trace is False
    assert r.region_tiers["answer"] == "tool"
    assert r.region_tiers.get("tool") == "tool"


def test_tools_used_param_signals_tool_even_without_tag():
    r = classify_source_tiers(
        raw="here is the result", public="here is the result", tools_used=("grep",)
    )
    assert r.answer_tier is SourceTier.TOOL
    assert r.had_tool is True


def test_trace_without_tool_is_generation_floor():
    r = classify_source_tiers(
        raw="<think>reasoning</think> my answer", public="my answer", tools_used=()
    )
    assert r.answer_tier is SourceTier.GENERATION
    assert r.had_trace is True
    assert r.region_tiers["answer"] == "generation"
    # V0: an untested trace stays at the floor (FAITHFUL_TRACE is Stage 2 only)
    assert r.region_tiers.get("trace") == "generation"


def test_bare_answer_no_tool_no_trace_is_generation():
    r = classify_source_tiers(
        raw="just an assertion", public="just an assertion", tools_used=()
    )
    assert r.answer_tier is SourceTier.GENERATION
    assert r.had_tool is False
    assert r.had_trace is False
    assert "trace" not in r.region_tiers and "tool" not in r.region_tiers


def test_tool_plus_trace_rolls_up_to_tool():
    r = classify_source_tiers(
        raw="<think>plan</think><tool_call>{}</tool_call> answer",
        public="answer",
        tools_used=("grep",),
    )
    assert r.answer_tier is SourceTier.TOOL
    assert r.region_tiers["answer"] == "tool"
    assert r.region_tiers.get("trace") == "generation"  # trace still untested in V0


def test_tier_label_roundtrip():
    assert tier_label(SourceTier.TOOL) == "tool"
    assert tier_label(SourceTier.FAITHFUL_TRACE) == "faithful-trace"
    assert tier_label(SourceTier.GENERATION) == "generation"


def test_classify_is_pure_no_flag_dependence(monkeypatch):
    # classify_source_tiers has NO flag gate; the flag gates side effects only.
    monkeypatch.delenv("MONOLITH_SOURCE_TIER_V1", raising=False)
    r = classify_source_tiers(raw="<tool_call>x</tool_call>", public="", tools_used=())
    assert r.answer_tier is SourceTier.TOOL
