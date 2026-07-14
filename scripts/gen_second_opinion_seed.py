"""Generate assets/workshop_seeds/second-opinion.monoline (the Second Opinion card).

Builds the blueprint via Monoline's own build_preset_from_blueprint (schema-4 correct by
construction), then FORCES the per-block config (providers + system prompts + api placeholders)
because Monoline's native llm provider enum does not list "monolith" — that provider is supplied
by Monolith's bridge at runtime (engine/monoline_bridge.make_engine_func), so we pin it in the
saved config rather than trust the builder to preserve an enum value it doesn't recognize.

Run from the Monolith repo root:  python scripts/gen_second_opinion_seed.py
Kept (not throwaway) so the seed is reproducible/editable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so engine/core import

from engine.monoline_bridge import load_monoline

_A_SYS = ("You are the primary model. Answer the user's question directly and "
          "concisely. Do not hedge.")
_B_SYS = ("You are an independent second model, a DIFFERENT family from the primary. "
          "Answer the user's question directly and concisely, on your own. Do not hedge.")
_ADJ_SYS = (
    "You compare two independently-produced answers (A and B) to the same question, "
    "produced by two DIFFERENT models. Your ONLY job is to detect AGREEMENT — not to "
    "decide which is correct.\n"
    "- If A and B make the same substantive claims: reply 'CORROBORATED — two independent "
    "models agree:' then give the agreed answer. Agreement is NOT proof of truth (shared "
    "mistakes survive); never say 'verified'.\n"
    "- If A and B make different or conflicting claims: reply '⚠ UNCERTAIN — two "
    "independent models DISAGREE.' then show briefly where they diverge (A says ... / B "
    "says ...). Do NOT pick a winner; flag it low-confidence and recommend checking.\n"
    "Be terse. Output only the verdict.")
# The adjudicator's contract MUST live in its PROMPT, not system_prompt: Monoline ignores an
# llm block's system field (a block's only instruction is its wired prompt) AND provider="monolith"
# runs the atom, which uses Monolith's own system assembly. So fold the contract into this text block.
_VERDICT_CONTENT = (
    _ADJ_SYS + "\n\n"
    "Question:\n{{question}}\n\n"
    "Answer A (one model):\n{{answer_a}}\n\n"
    "Answer B (a DIFFERENT model, answered independently):\n{{answer_b}}\n\n"
    "Now produce ONLY the verdict.")
_DESC = (
    "Cross-model disagreement as an error signal. A different-family model answers the same "
    "question BLIND; an adjudicator reports whether the two agree. CORROBORATED means two "
    "independent models agree — NOT that the answer is verified (shared mistakes survive). "
    "DISAGREEMENT is the signal: treat it as uncertain and check. Block B must be a DIFFERENT "
    "model family than your bound model — that is the whole point: set it to a second local "
    "GGUF (no egress), or to the 'api' provider for a cloud model (EGRESS — the question "
    "leaves the machine; needs per-flow egress opt-in). ~2-3x compute.")


def main() -> int:
    m = load_monoline()
    bp = {
        "name": "Second Opinion",
        "blocks": [
            {"id": "question_in", "kind": "port",
             "config": {"direction": "in", "label": "question", "source": "user_input"}},
            {"id": "answer_a", "kind": "llm", "label": "Answer A (bound model)",
             "config": {"provider": "monolith", "system_prompt": _A_SYS,
                        "temperature": 0.7, "max_tokens": 2048}},
            {"id": "answer_b", "kind": "llm", "label": "Answer B (independent, different family)",
             "config": {"provider": "api", "api_url": "", "api_token": "", "model": "",
                        "system_prompt": _B_SYS, "temperature": 0.7, "max_tokens": 2048}},
            {"id": "verdict_prompt", "kind": "text", "config": {"content": _VERDICT_CONTENT}},
            {"id": "adjudicate", "kind": "llm", "label": "Adjudicator",
             "config": {"provider": "monolith", "temperature": 0.2, "max_tokens": 1024}},
            {"id": "output", "kind": "port",
             "config": {"direction": "out", "label": "verdict", "source": "subgraph"}},
        ],
        "connections": [
            ["question_in.value", "answer_a.prompt"],
            ["question_in.value", "answer_b.prompt"],          # B is BLIND: only the question
            ["question_in.value", "verdict_prompt.question"],
            ["answer_a.response", "verdict_prompt.answer_a"],
            ["answer_b.response", "verdict_prompt.answer_b"],
            ["verdict_prompt.text", "adjudicate.prompt"],
            ["adjudicate.response", "output.value"],
        ],
    }
    preset = m["blueprint"].build_preset_from_blueprint(bp, strict=False)
    preset.id = "second-opinion"
    preset.name = "Second Opinion"
    preset.description = _DESC
    d = preset.to_dict()

    # FORCE the critical per-block config (providers + prompts + api placeholders), since the
    # builder may coerce/drop the unknown "monolith" provider and trim unrecognized keys.
    forced = {
        "answer_a": {"provider": "monolith", "system_prompt": _A_SYS},
        "answer_b": {"provider": "api", "api_url": "", "api_token": "", "model": "",
                     "system_prompt": _B_SYS},
        "adjudicate": {"provider": "monolith"},  # contract is in verdict_prompt, NOT system
        "verdict_prompt": {"content": _VERDICT_CONTENT},
    }
    for b in d.get("blocks", []):
        if b.get("id") in forced:
            b.setdefault("config", {}).update(forced[b["id"]])

    out = Path("assets/workshop_seeds/second-opinion.monoline")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
    providers = {b["id"]: b.get("config", {}).get("provider")
                 for b in d.get("blocks", []) if b.get("kind") == "llm"}
    print("wrote", out, "id=", d.get("id"), "schema_version=", d.get("schema_version"),
          "blocks=", len(d.get("blocks", [])), "providers=", providers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
