from __future__ import annotations

import core.subagent as sub


def test_subagent_engine_raw_returns_text_not_fence(monkeypatch):
    # run_subagent is the heavy path; stub it to a known SubagentResult.
    fake = sub.SubagentResult(
        ok=True, text="RAW BODY",
        fenced="[SUBAGENT_RESULT ok=true]\nRAW BODY\n[/SUBAGENT_RESULT]",
        child_turn_id="x", level=3, tools_run=0)
    monkeypatch.setattr(sub, "run_subagent", lambda *a, **k: fake)
    out = sub.subagent_engine_raw([{"role": "user", "content": "hi"}],
                                  {"_subagent": {"level": 3}})
    assert out == "RAW BODY"
    assert "[SUBAGENT_RESULT" not in out
