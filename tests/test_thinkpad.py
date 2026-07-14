"""Thinkpad — the reasoning-branch generator (Seed A, advisory verdict).

Key contracts under test:
  * cite-parse runs on the split-out ANSWER, not the raw branch output — a handle
    merely MENTIONED in <think> deliberation must not count as a cite (the selection
    gate caught this false-positive live).
  * ALL branches are surfaced (it's a generator, not an auto-picker).
  * the grounded verdict ranks branches ADVISORILY — for display, not to commit.
"""
from core import thinkpad, grounded_verdict


def _gen(outputs):
    seq = list(outputs)
    return lambda messages: seq.pop(0)


def test_cite_parse_runs_on_answer_not_deliberation():
    raw = "<think>I'll weigh [R2] against [R1] here</think>The answer rests here. [cite: R1]"
    res = thinkpad.run_thinkpad([{"role": "user", "content": "q"}], {}, n=1,
                                generate=_gen([raw]), resolve=lambda h: {"R1": 4, "R2": 1}.get(h))
    b = res.branches[0]
    assert b.answer == "The answer rests here. [cite: R1]"   # reasoning stripped
    assert b.think == "I'll weigh [R2] against [R1] here"     # reasoning kept (the MonoTrace lane)
    assert b.cites == ("R1",)                                 # R2 (a deliberation mention) excluded


def test_all_branches_surfaced_not_just_a_winner():
    raws = ["<think>x</think>A [cite: R1]", "<think>y</think>B [cite: R2]", "<think>z</think>C, reasoned directly"]
    res = thinkpad.run_thinkpad([{"role": "user", "content": "q"}], {}, n=3,
                                generate=_gen(raws), resolve=lambda h: {"R1": 2, "R2": 4}.get(h))
    assert {b.id for b in res.branches} == {"c0", "c1", "c2"}   # all surfaced, none dropped


def test_advisory_ranking_returned_but_branches_not_committed():
    raws = ["<think>x</think>A [cite: R1]", "<think>y</think>B [cite: R2]"]
    res = thinkpad.run_thinkpad([{"role": "user", "content": "q"}], {}, n=2,
                                generate=_gen(raws), resolve=lambda h: {"R1": 2, "R2": 4}.get(h))
    assert res.advisory[0].candidate.id == "c1"     # R2 (auth 4) ranks above R1 (auth 2)
    assert res.advisory[0].winning_cite == "R2"
    assert len(res.branches) == 2                   # advisory ranks; it does NOT prune/commit


def test_tokenless_branch_surfaced_and_ranked_ungrounded():
    raws = ["<think>x</think>A grounded [cite: R1]", "<think>y</think>B, I reason it directly"]
    res = thinkpad.run_thinkpad([{"role": "user", "content": "q"}], {}, n=2,
                                generate=_gen(raws), resolve=lambda h: {"R1": 4}.get(h))
    by_id = {b.id: b for b in res.branches}
    assert by_id["c1"].cites == ()                  # tokenless, but still a surfaced branch
    assert res.advisory[0].candidate.id == "c0"     # grounded one ranks first (advisory)
    assert res.advisory[-1].grounded is False       # tokenless ranked last


def test_failed_generation_is_skipped():
    def gen(messages):
        if not getattr(gen, "hit", False):
            gen.hit = True
            raise RuntimeError("backend hiccup")
        return "<think>x</think>B [cite: R1]"
    res = thinkpad.run_thinkpad([{"role": "user", "content": "q"}], {}, n=2,
                                generate=gen, resolve=lambda h: {"R1": 4}.get(h))
    assert len(res.branches) == 1                   # the surviving branch
    assert res.branches[0].cites == ("R1",)


# ── branch-frame builder ─────────────────────────────────────────────

def test_branch_frame_injects_scaffold_lane_and_think_convention():
    frame = thinkpad.build_branch_frame(
        [{"role": "user", "content": "answer this"}],
        recall_lane="From recalled memory:\n- [R1] [LOCKED] x | is | y",
        scaffold="## Grounding cite\ncite handles.",
    )
    assert frame[0]["role"] == "system"
    s = frame[0]["content"]
    assert "Grounding cite" in s                       # cite-slot present
    assert "[R1] [LOCKED] x | is | y" in s             # render-once recall lane present
    assert "</think>" in s                             # the answer/reasoning convention present
    assert frame[-1] == {"role": "user", "content": "answer this"}   # conversation preserved


def test_branch_frame_replaces_main_system_keeps_conversation():
    frame = thinkpad.build_branch_frame(
        [{"role": "system", "content": "OLD MAIN SYSTEM"},
         {"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
         {"role": "user", "content": "c"}],
        scaffold="SCAF",
    )
    assert "OLD MAIN SYSTEM" not in frame[0]["content"]   # branch gets the branch frame, not main system.md
    assert frame[0]["content"].startswith("SCAF")
    assert [m["role"] for m in frame[1:]] == ["user", "assistant", "user"]


def test_branch_frame_convention_toggle():
    on = thinkpad.build_branch_frame([{"role": "user", "content": "q"}], scaffold="S")
    off = thinkpad.build_branch_frame([{"role": "user", "content": "q"}], scaffold="S", think_convention=False)
    assert "</think>" in on[0]["content"] and "</think>" not in off[0]["content"]


# ── live orchestration (run_subagent fan-out) ────────────────────────

def test_run_thinkpad_live_fans_out_subagents_links_traces_and_renders_once(monkeypatch):
    from core import runtime_state_projection as rsp, subagent, recall_handles
    render_calls = {"n": 0}
    def fake_lane(msgs):
        render_calls["n"] += 1                                  # must be called ONCE for the whole fan-out
        recall_handles.reset(); recall_handles.register("R1", {"locked": 1})
        return "From recalled memory:\n- [R1] [LOCKED] x | is | y"
    monkeypatch.setattr(rsp, "render_recall_lane", fake_lane)

    outs = iter([
        subagent.SubagentResult(True, "<think>weigh [R1]</think>A rests here [cite: R1]", "f", "trace0", 3),
        subagent.SubagentResult(True, "<think>weigh</think>B also here [cite: R1]", "f", "trace1", 3),
    ])
    monkeypatch.setattr(subagent, "run_subagent", lambda *a, **k: next(outs))

    res, traces = thinkpad.run_thinkpad_live(
        [{"role": "user", "content": "q"}], {}, n=2, parent_turn_id="outer", scaffold="SCAF")

    assert render_calls["n"] == 1                               # render-once contract
    assert len(res.branches) == 2
    assert res.branches[0].cites == ("R1",)                     # [R1] in <think> excluded; answer cite kept
    assert res.branches[0].trace_id == "trace0"                 # branch linked to its MonoTrace
    assert traces == ("trace0", "trace1")
    assert res.advisory[0].winning_cite == "R1" and res.advisory[0].authority == 4


def test_run_thinkpad_live_drops_refused_branch(monkeypatch):
    from core import runtime_state_projection as rsp, subagent, recall_handles
    monkeypatch.setattr(rsp, "render_recall_lane",
                        lambda m: (recall_handles.reset() or recall_handles.register("R1", {"locked": 1}) or "lane"))
    outs = iter([
        subagent.SubagentResult(False, "", "f", "tX", 3, halt_reason="busy"),     # refused
        subagent.SubagentResult(True, "answer [cite: R1]", "f", "trace1", 3),
    ])
    monkeypatch.setattr(subagent, "run_subagent", lambda *a, **k: next(outs))
    res, traces = thinkpad.run_thinkpad_live(
        [{"role": "user", "content": "q"}], {}, n=2, parent_turn_id="outer", scaffold="S")
    assert len(res.branches) == 1                               # refused branch dropped
    assert traces == ("trace1",)


def test_run_thinkpad_live_falls_back_to_native_thinking_for_branch_think(monkeypatch):
    """When the model emits no <think> tags but the engine separated native
    thinking, the branch's think lane carries that native trace (the rating
    decider's input) instead of silently dropping it."""
    from core import runtime_state_projection as rsp, subagent, recall_handles
    monkeypatch.setattr(rsp, "render_recall_lane",
                        lambda m: (recall_handles.reset() or "lane"))
    outs = iter([
        subagent.SubagentResult(True, "bare answer no tags", "f", "trace0", 3,
                                thinking="native trace"),
        subagent.SubagentResult(True, "<think>tagged</think>answer", "f", "trace1", 3,
                                thinking="ignored - tags win"),
    ])
    monkeypatch.setattr(subagent, "run_subagent", lambda *a, **k: next(outs))
    res, _ = thinkpad.run_thinkpad_live(
        [{"role": "user", "content": "q"}], {}, n=2, parent_turn_id="outer", scaffold="S")
    assert res.branches[0].think == "native trace"     # fallback engaged
    assert res.branches[0].answer == "bare answer no tags"
    assert res.branches[1].think == "tagged"           # explicit tags still win


def test_run_thinkpad_live_gives_branches_reasoning_token_headroom(monkeypatch):
    """A branch is a single reasoning-heavy inference: run_subagent's default
    1024-token budget gets consumed by the think phase and the answer never
    arrives (observed live: every branch folded status=error, raw empty,
    ~4.5k chars of thinking). The fan-out must pass an explicit llm_config
    with real headroom."""
    from core import runtime_state_projection as rsp, subagent, recall_handles
    monkeypatch.setattr(rsp, "render_recall_lane",
                        lambda m: (recall_handles.reset() or "lane"))
    seen = {}
    def fake_run(m, cfg, **kw):
        seen.update(kw)
        return subagent.SubagentResult(True, "a", "f", "t0", 3)
    monkeypatch.setattr(subagent, "run_subagent", fake_run)
    thinkpad.run_thinkpad_live(
        [{"role": "user", "content": "q"}], {}, n=1, parent_turn_id="outer", scaffold="S")
    assert seen.get("llm_config", {}).get("max_tokens", 0) >= 3072
