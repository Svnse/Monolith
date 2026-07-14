"""Workshop 'no output' fix (2026-06-14): the monolith-provider LLM block must hand
run_subagent a REAL token budget (not the 1024-token leaf default) and regenerate on an
empty result before failing the block. Root cause: DeepSeek-class thinking models charge
reasoning AGAINST max_tokens, so a 1024 cap truncates before the final answer
(finish_reason=length, empty content) -> the block silently returns "" and the flow dies.
See .workshop_empty_output_rootcause.md."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import core.subagent as sub
from engine import monoline_bridge as br
from tests._monoline_requirement import requires_monoline


pytestmark = requires_monoline


def _result(text="", *, ok=None, halt_reason=None, thinking=""):
    if ok is None:
        ok = bool(text)
    if halt_reason is None:
        halt_reason = None if ok else "error"
    return sub.SubagentResult(ok=ok, text=text, fenced="[r]", child_turn_id="c",
                              level=3, tools_run=0, halt_reason=halt_reason, thinking=thinking)


def _engine(monkeypatch, *, config_max=8192):
    cfg = {"api_base": "https://m", "api_model": "deepseek", "api_key": "k", "temp": 0.4}
    if config_max is not None:
        cfg["max_tokens"] = config_max
    monkeypatch.setattr(br, "load_config", lambda: cfg)
    return br.make_engine_func(parent_turn_id="ROOT", spawn_budget=None,
                               should_cancel=lambda: False, is_busy=lambda: False,
                               busy_retries=2, busy_backoff=0.0)


def test_monolith_block_passes_real_budget_not_1024(monkeypatch):
    # Must hand run_subagent a real max_tokens (>= the runtime's 8192), NOT rely on the
    # 1024-token leaf default that truncates DeepSeek's reasoning before the answer.
    seen = {}

    def fake_run(messages, config, **kw):
        seen.update(kw)
        return _result("OK")
    monkeypatch.setattr(sub, "run_subagent", fake_run)
    fn = _engine(monkeypatch, config_max=8192)
    out = fn([{"role": "user", "content": "hi"}], {"provider": "monolith", "max_tokens": 2048})
    assert out == "OK"
    # floored to the runtime budget (8192), NOT the block's stale 2048 default, NOT 1024
    assert seen.get("llm_config", {}).get("max_tokens", 0) >= 8192


def test_monolith_block_honors_larger_block_budget(monkeypatch):
    seen = {}

    def fake_run(messages, config, **kw):
        seen.update(kw)
        return _result("OK")
    monkeypatch.setattr(sub, "run_subagent", fake_run)
    fn = _engine(monkeypatch, config_max=8192)
    fn([{"role": "user", "content": "hi"}], {"provider": "monolith", "max_tokens": 20000})
    assert seen.get("llm_config", {}).get("max_tokens") == 20000  # honors the block's larger budget


def test_monolith_block_does_not_inherit_runtime_max_tokens_ceiling(monkeypatch):
    # Regression (2026-06-22): load_config() carries the GLOBAL runtime max_tokens, which for a
    # long-context cloud profile is the CONTEXT window (e.g. 1,000,000) -- FAR above a provider's
    # OUTPUT cap (DeepSeek rejects >393216 with HTTP 400 "Invalid max_tokens value"). The per-block
    # budget must NOT inherit that runtime ceiling as a floor: a block with a small authored budget
    # must floor to _MONOLINE_LLM_MIN_TOKENS (8192), and EVERY escalating regen attempt must stay
    # clamped to the cloud output cap (_MONOLINE_LLM_MAX_TOKENS). Without the clamp the Axioms block
    # of the Axiomatic Synthesis Forge sent 1M/2M/3M and failed on every attempt.
    budgets = []

    def always_empty(messages, config, **kw):
        budgets.append(kw.get("llm_config", {}).get("max_tokens", 0))
        return _result("", thinking="reasoning, no answer")  # force all 3 regens
    monkeypatch.setattr(sub, "run_subagent", always_empty)
    fn = _engine(monkeypatch, config_max=1_000_000)  # runtime ceiling = context-window-sized
    with pytest.raises(Exception):
        fn([{"role": "user", "content": "hi"}], {"provider": "monolith", "max_tokens": 1600})
    assert budgets, "expected at least one inference attempt"
    # NEVER hand the provider the 1,000,000 runtime ceiling (or its 2x/3x escalation):
    assert max(budgets) <= br._MONOLINE_LLM_MAX_TOKENS
    # block's small 1600 floored to the thinking-headroom floor, NOT raised to the runtime ceiling:
    assert budgets[0] == br._MONOLINE_LLM_MIN_TOKENS


def test_monolith_block_regens_on_empty_then_succeeds(monkeypatch):
    budgets = []
    seq = [_result("", thinking="reasoning, no answer"), _result("REAL ANSWER")]

    def fake_run(messages, config, **kw):
        budgets.append(kw.get("llm_config", {}).get("max_tokens", 0))
        return seq[len(budgets) - 1]
    monkeypatch.setattr(sub, "run_subagent", fake_run)
    fn = _engine(monkeypatch)
    out = fn([{"role": "user", "content": "hi"}], {"provider": "monolith"})
    assert out == "REAL ANSWER"
    assert len(budgets) == 2          # regenerated once after the empty result
    assert budgets[1] > budgets[0]    # escalated the budget on regen


def test_monolith_block_fails_after_three_empty_regens(monkeypatch):
    calls = {"n": 0}

    def always_empty(messages, config, **kw):
        calls["n"] += 1
        return _result("", thinking="We need to produce... [reasoning, never an answer]")
    monkeypatch.setattr(sub, "run_subagent", always_empty)
    fn = _engine(monkeypatch)
    with pytest.raises(Exception) as ei:
        fn([{"role": "user", "content": "hi"}], {"provider": "monolith", "label": "Unifier"})
    assert calls["n"] == 3                          # bounded: 3 attempts, then fail the block
    assert "Unifier" in str(ei.value)               # error names the block
    assert "no final answer" in str(ei.value).lower()


def test_monolith_block_does_not_regen_on_cancel(monkeypatch):
    # Guard: a user-cancel must NOT be treated as an empty result to regenerate.
    calls = {"n": 0}

    def cancelled(messages, config, **kw):
        calls["n"] += 1
        return _result("", halt_reason="cancelled")
    monkeypatch.setattr(sub, "run_subagent", cancelled)
    fn = _engine(monkeypatch)
    out = fn([{"role": "user", "content": "hi"}], {"provider": "monolith"})
    assert out == ""        # returns empty; the caller handles cancellation
    assert calls["n"] == 1  # no regen on cancel


def test_monolith_block_returns_immediately_on_content(monkeypatch):
    # Guard: a good first answer never triggers a needless regen.
    calls = {"n": 0}

    def ok_once(messages, config, **kw):
        calls["n"] += 1
        return _result("DONE")
    monkeypatch.setattr(sub, "run_subagent", ok_once)
    fn = _engine(monkeypatch)
    out = fn([{"role": "user", "content": "hi"}], {"provider": "monolith"})
    assert out == "DONE"
    assert calls["n"] == 1


def _three_block_chain_blueprint() -> dict:
    # input -> first(monolith) -> second(local) -> output. No "schema_version" -> the lenient
    # build_preset_from_blueprint path (see test_monoline_bridge).
    return {
        "name": "MonolithEmptyChain",
        "blocks": [
            {"id": "input", "kind": "port",
             "config": {"direction": "in", "label": "request", "source": "user_input"}},
            {"id": "first", "kind": "llm", "config": {"provider": "monolith", "label": "First"}},
            {"id": "second", "kind": "llm", "config": {"provider": "local", "label": "Second"}},
            {"id": "output", "kind": "port",
             "config": {"direction": "out", "label": "response", "source": "subgraph"}},
        ],
        "connections": [["input.value", "first.prompt"],
                        ["first.response", "second.prompt"],
                        ["second.response", "output.value"]],
    }


def test_empty_block_halts_run_and_downstream_never_activates(tmp_path, monkeypatch):
    # End-to-end (real Monoline runtime): an always-empty monolith block raises after all regens.
    # Monoline marks it block_status=error and writes NO output -> the downstream 'second' block's
    # wired input never arrives -> it is never ready -> NEVER activates. summarize_run_failure (the
    # channel the chat dispatch uses) reports the failure. This is the chosen 'fail the block +
    # nothing flows downstream' semantics. (ActivationResult.error stays None by design -- the
    # activation runtime surfaces block errors via block_status/step_log, not result.error.)
    import core.turn_trace as tt
    from core.workflow_registry import Workflow

    calls = {"n": 0}

    def always_empty(messages, config, **kw):
        calls["n"] += 1
        return _result("", thinking="reasoning, never an answer")
    monkeypatch.setattr(sub, "run_subagent", always_empty)
    monkeypatch.setattr(br, "load_config",
                        lambda: {"api_base": "x", "api_model": "y", "max_tokens": 8192, "temp": 0.4})
    # the downstream 'local' block would emit this sentinel IF it ever activated:
    m = br.load_monoline()
    monkeypatch.setattr(m["engine"], "engine_call", lambda msgs, cfg: "DOWNSTREAM_RAN")

    p = Path(tmp_path) / "chain.monoline"
    p.write_text(json.dumps(_three_block_chain_blueprint()), encoding="utf-8")
    wf = Workflow(id="ch", name="Chain", description="", kind="monoline", source_path=p)

    tt.set_db_path(Path(tmp_path) / "tt.sqlite3")
    try:
        run = br.run_monoline_world(
            wf, user_input="hi", parent_turn_id="",
            spawn_budget=None, should_cancel=lambda: False, is_busy=lambda: False,
            on_step=None, should_stop=None)
    finally:
        tt.set_db_path(None)

    failure = br.summarize_run_failure(run)
    assert failure and "no final answer" in failure.lower()      # the run is REPORTED failed...
    assert "call_llm" in failure.lower()                         # ...at the LLM block that went empty
    assert (run.result.output or "") == ""                       # no final output produced
    all_step_outputs = " ".join(str(getattr(sr, "outputs", "")) for sr in run.result.step_log)
    assert "DOWNSTREAM_RAN" not in all_step_outputs              # downstream block NEVER activated
    assert calls["n"] == 3                                       # 3 regens, then fail the block


def test_cancel_during_block_halts_run_cleanly_without_hard_failure(tmp_path, monkeypatch):
    # Enforce-stop UX: when a stop arrives mid-generation, run_subagent reports halt_reason=
    # "cancelled" and the monolith block returns "" WITHOUT raising. Unlike the empty-after-regens
    # case above (a hard failure), a user stop must NOT surface as a workflow failure -> the chat
    # dispatch then routes to sig_pipeline_done (clean) rather than a red "Pipeline run failed".
    import core.turn_trace as tt
    from core.workflow_registry import Workflow

    state = {"stopped": False}

    def cancel_on_first(messages, config, **kw):
        state["stopped"] = True                       # a stop is observed during the first block
        return _result("", halt_reason="cancelled")
    monkeypatch.setattr(sub, "run_subagent", cancel_on_first)
    monkeypatch.setattr(br, "load_config",
                        lambda: {"api_base": "x", "api_model": "y", "max_tokens": 8192, "temp": 0.4})
    m = br.load_monoline()
    monkeypatch.setattr(m["engine"], "engine_call", lambda msgs, cfg: "DOWNSTREAM_RAN")

    p = Path(tmp_path) / "chain.monoline"
    p.write_text(json.dumps(_three_block_chain_blueprint()), encoding="utf-8")
    wf = Workflow(id="ch", name="Chain", description="", kind="monoline", source_path=p)

    tt.set_db_path(Path(tmp_path) / "tt.sqlite3")
    try:
        run = br.run_monoline_world(
            wf, user_input="hi", parent_turn_id="",
            spawn_budget=None, should_cancel=lambda: state["stopped"], is_busy=lambda: False,
            on_step=None, should_stop=lambda: state["stopped"])
    finally:
        tt.set_db_path(None)

    failure = br.summarize_run_failure(run)
    # No "no final answer"-style HARD failure for an intentional stop:
    assert not (failure and "no final answer" in str(failure).lower())
    # The run halted: downstream never activated.
    all_step_outputs = " ".join(str(getattr(sr, "outputs", "")) for sr in run.result.step_log)
    assert "DOWNSTREAM_RAN" not in all_step_outputs


def test_stop_emits_stopped_events_not_error(tmp_path, monkeypatch):
    # On STOP the run-event stream must carry a CLEAN stopped state: the interrupted block emits
    # BlockFinished(status="stopped") and the run emits RunFinished(stopped=True, error="") — so
    # the RunView renders "stopped", not a red error.
    import core.turn_trace as tt
    from core.workflow_registry import Workflow
    from core.run_model import BlockFinished, RunFinished

    state = {"stopped": False}

    def cancel_on_first(messages, config, **kw):
        state["stopped"] = True
        return _result("", halt_reason="cancelled")
    monkeypatch.setattr(sub, "run_subagent", cancel_on_first)
    monkeypatch.setattr(br, "load_config",
                        lambda: {"api_base": "x", "api_model": "y", "max_tokens": 8192, "temp": 0.4})
    m = br.load_monoline()
    monkeypatch.setattr(m["engine"], "engine_call", lambda msgs, cfg: "X")

    p = Path(tmp_path) / "chain.monoline"
    p.write_text(json.dumps(_three_block_chain_blueprint()), encoding="utf-8")
    wf = Workflow(id="ch", name="Chain", description="", kind="monoline", source_path=p)

    events: list = []
    tt.set_db_path(Path(tmp_path) / "tt.sqlite3")
    try:
        br.run_monoline_world(
            wf, user_input="hi", parent_turn_id="",
            spawn_budget=None, should_cancel=lambda: state["stopped"], is_busy=lambda: False,
            on_step=None, should_stop=lambda: state["stopped"], on_event=events.append)
    finally:
        tt.set_db_path(None)

    finished = [e for e in events if isinstance(e, RunFinished)]
    assert finished and finished[-1].stopped is True      # run reports a clean STOP...
    assert finished[-1].error == ""                       # ...with no error text
    first = [e for e in events if isinstance(e, BlockFinished) and e.block_id == "first"]
    assert first and first[-1].status == "stopped"        # the interrupted block reads "stopped"
