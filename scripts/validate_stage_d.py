"""Stage-D carve validation. Two layers:

  STRUCTURAL (no API): assert every produced-shape / injection anchor the
    Stage-D cuts could have endangered is still present in the raw master
    prompt. This is the real safety net for the MEMORY compression.
  BEHAVIORAL (DeepSeek): OUTPUT-BOUNDARY leak probe gates the example cut;
    value-premise + tool-production are regressions vs Stage C.

Run: python scripts/validate_stage_d.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
README_PATH = (ROOT / "README.md").resolve().as_posix()

from core.config import get_config
from core.llm_config import build_system_prompt, load_master_prompt
from engine.llm import OpenAICompatLLM


# Strings that MUST survive the carve. Each is a produced shape (model emits it),
# an injection anchor (runtime keys on it), or a load-bearing contract.
ANCHORS = [
    ("{skills_catalog} placeholder", "{skills_catalog}"),
    ("{identity_block} placeholder", "{identity_block}"),
    ("MEMORY injection header", "MEMORY —"),
    ("[IDENTITY] header", "[IDENTITY]"),
    ("[TOOL_LOOP_DONE] sentinel", "[TOOL_LOOP_DONE]"),
    ("bearing_update envelope open", "<bearing_update>"),
    ("bearing_update envelope close", "</bearing_update>"),
    ("bearing status enum", "active|dormant|closed|rejected|superseded"),
    ("bearing referent kind enum", "file|peer|entity|claim|tool_result"),
    ("bearing register enum", "literal|performative|ironic|exploratory"),
    ("bearing char limits", "current_frame ≤400"),
    ("recall tool envelope", '"name":"recall"'),
    ("save_memory tool envelope", '"name":"save_memory"'),
    ("working_memory_set tool", "working_memory_set"),
    ("working_memory_clear tool", "working_memory_clear"),
    ("scratchpad op=pin", "op=pin"),
    ("CONTINUITY/RECALL anti-dup", "never duplicate a fact across surfaces"),
    ("tool return accessor", "$id.data.field"),
    ("output-boundary rule", "</think>"),
]


def structural() -> bool:
    raw = load_master_prompt()
    assembled = build_system_prompt({"system_prompt": raw})
    print(f"[raw master prompt:      {len(raw)} chars]")
    print(f"[assembled system prompt: {len(assembled)} chars]\n")
    ok = True
    for label, needle in ANCHORS:
        present = needle in raw
        print(f"  {'PASS' if present else '!! FAIL':<8} {label}")
        ok = ok and present
    print(f"\nSTRUCTURAL: {'ALL ANCHORS PRESENT' if ok else 'MISSING ANCHOR(S) — DO NOT SHIP'}\n")
    return ok


def call(system_prompt: str, user: str, max_tokens: int = 2200) -> str:
    cfg = get_config().llm.model_dump()
    client = OpenAICompatLLM(
        str(cfg.get("api_base", "")).strip(),
        str(cfg.get("api_key", "") or ""),
        str(cfg.get("api_model", "")).strip(),
    )
    parts: list[str] = []
    for chunk in client.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
        max_tokens=max_tokens,
        stream=False,
    ):
        choice = (chunk.get("choices", [{}]) or [{}])[0]
        delta = choice.get("delta") or {}
        content = delta.get("content") or (choice.get("message") or {}).get("content")
        if content:
            parts.append(content)
    return "".join(parts)


def output_boundary_verdict(resp: str) -> str:
    """Leak = think block opened but nothing substantive follows the close."""
    if "<think>" not in resp:
        return "OK (no think tags — answer is plain text, no leak risk)"
    if "</think>" not in resp:
        return "!! LEAK — <think> opened, never closed; answer trapped inside"
    after = resp.split("</think>")[-1].strip()
    if len(after) < 20:
        return f"!! LEAK — nothing substantive after </think> (got {after!r})"
    return f"OK (answer closes think then emits {len(after)} visible chars)"


PROBES = {
    # The gate for the OUTPUT-BOUNDARY example cut: force multi-step reasoning,
    # then check the model still closes </think> before the visible answer.
    "OUTPUT_BOUNDARY": (
        "[CHANNEL: USER]\nThink step by step, then give me the answer. "
        "A bat and a ball cost $1.10 together. The bat costs $1.00 more than "
        "the ball. How much does the ball cost?"
    ),
    "V1_value_premise": (
        "[CHANNEL: USER]\nShould I quit my stable job to go all-in on my startup? Decide."
    ),
    "R3_tool_production": (
        "[CHANNEL: USER]\nRead the file at "
        f"{README_PATH} and tell me its first heading."
    ),
    "MEMORY_save": (
        "[CHANNEL: USER]\nFor the record, remember that I prefer Rust over Go for systems work."
    ),
}


def behavioral() -> None:
    carved = build_system_prompt({"system_prompt": load_master_prompt()})
    for name, probe in PROBES.items():
        print("=" * 64)
        print(name)
        print("=" * 64)
        try:
            resp = call(carved, probe)
        except Exception as e:  # noqa: BLE001 — validation must report, not crash
            print(f"!! call failed: {e!r}\n")
            continue
        if name == "OUTPUT_BOUNDARY":
            print(f"VERDICT: {output_boundary_verdict(resp)}\n")
        print(resp.strip()[:1600])
        print()


def main() -> int:
    print("\n========== STRUCTURAL ==========\n")
    ok = structural()
    if "--structural-only" in sys.argv:
        return 0 if ok else 1
    print("========== BEHAVIORAL ==========\n")
    behavioral()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
