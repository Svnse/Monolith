"""Second Opinion card: a Monoline flow where a different-family model answers BLIND and
DISAGREEMENT is the error signal (operationalizes Tier-3 of docs/research/INDEPENDENT_ERROR_SIGNAL.md).

The contract is tested deterministically with a scripted engine_func (no live model, no egress):
- A and B agree   -> the verdict says "corroborated"
- A and B disagree -> the verdict flags "disagree"/"uncertain"
- the adjudicator's prompt received BOTH answers (the fan-in wiring is correct)
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SEED = Path(__file__).resolve().parents[1] / "assets" / "workshop_seeds" / "second-opinion.monoline"


@pytest.fixture
def headless():
    """Load Monoline via the bridge; skip cleanly if the plugin dir is absent (INV-#0)."""
    from engine import monoline_bridge as br
    try:
        m = br.load_monoline()
    except RuntimeError as e:
        pytest.skip(f"Monoline plugin not present: {e}")
    return m["headless"]


def _scripted_engine(scenario: dict, captured: dict):
    """Route by provider + prompt CONTENT — the signals the real engine actually carries.
    (We must NOT route on system_prompt: Monoline ignores an llm block's system field, so the
    instructions live in the wired prompt. The adjudicator's prompt is the only one carrying
    both answer slots + the contract.)"""
    def _engine(messages, config):
        provider = str(config.get("provider", "") or "").lower()
        user = ""
        for mm in reversed(messages):
            if str(mm.get("role", "")).lower() == "user":
                user = str(mm.get("content", "") or "")
                break
        ul = user.lower()
        if "answer a" in ul and "answer b" in ul:           # the adjudicator's prompt
            captured["adj_provider"] = provider
            captured["adjudicate_prompt"] = user
            if scenario["a"] == scenario["b"]:
                return f"CORROBORATED - two independent models agree: {scenario['a']}"
            return (f"UNCERTAIN - two independent models DISAGREE. "
                    f"A: {scenario['a']} / B: {scenario['b']}")
        if provider == "api":                               # the independent second model
            captured["b_provider"] = provider
            return scenario["b"]
        captured["a_provider"] = provider                   # the bound model
        return scenario["a"]
    return _engine


def _run(headless, scenario, captured):
    return headless.run_workflow(
        str(SEED), user_input="What is the capital of France?",
        session_messages=[],
        engine_func=_scripted_engine(scenario, captured),
        tool_func=lambda *a, **k: "",
        activation_mode="on_activate",
        on_step=None, should_stop=lambda: False,
        max_steps=60)


def test_seed_lists_in_registry() -> None:
    from core.workflow_registry import WorkflowRegistry
    reg = WorkflowRegistry(workflows_dir=SEED.parent)
    wf = reg.get("second-opinion")
    assert wf is not None and wf.kind == "monoline"


def test_seed_pins_providers_in_json() -> None:
    import json
    d = json.loads(SEED.read_text(encoding="utf-8"))
    prov = {b["id"]: b.get("config", {}).get("provider")
            for b in d["blocks"] if b.get("kind") == "llm"}
    # A and the adjudicator are the bound model (gated L3 atom); only B egresses.
    assert prov == {"answer_a": "monolith", "answer_b": "api", "adjudicate": "monolith"}


def test_corroborated_when_models_agree(headless) -> None:
    captured: dict = {}
    run = _run(headless, {"a": "Paris", "b": "Paris"}, captured)
    out = str(run.result.output).lower()
    assert "corroborat" in out
    adj_prompt = str(captured.get("adjudicate_prompt", "")).lower()
    # fan-in wiring: the adjudicator saw BOTH answers
    assert "paris" in adj_prompt
    # the CONTRACT reached the adjudicator's PROMPT (not system_prompt, which Monoline ignores)
    assert "pick a winner" in adj_prompt
    # the runtime preserved the saved providers through load (A = gated atom, B = egress)
    assert captured.get("a_provider") == "monolith"
    assert captured.get("b_provider") == "api"
    assert captured.get("adj_provider") == "monolith"


def test_flagged_uncertain_when_models_disagree(headless) -> None:
    captured: dict = {}
    run = _run(headless, {"a": "Paris", "b": "Lyon"}, captured)
    out = str(run.result.output).lower()
    assert ("disagree" in out) or ("uncertain" in out)
    adj = str(captured.get("adjudicate_prompt", "")).lower()
    assert "paris" in adj and "lyon" in adj  # blind B's answer reached the adjudicator
