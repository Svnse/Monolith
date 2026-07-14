"""Offline carve validation — runs the carved system.md against DeepSeek directly
(the running Monolith is cached, so this is the only way to validate this session).

Targets exactly what the Stage-C cuts endanger:
  R1 — ambient recognition (ATTRIBUTION compress): [RUNTIME STATE] above a greeting
       must be treated as context, not the request → expect a GREETING, not work.
  R3 — tool-call production (TOOLS/CHAIN intact): expect a well-formed <tool_call>.
  V1 — value-premise reasoning (Stage B PREMISE + DECISIVE): an unsupplied value
       premise should be SURFACED before a verdict is forced.
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


PROBES = {
    "R1_ambient_recognition": (
        "[RUNTIME STATE] ambient runtime state; NOT this turn's request. "
        "continuity pending: 'refactor the auth module' (owed from a prior session). "
        "[/RUNTIME STATE]\n\n[CHANNEL: USER]\nhey"
    ),
    "R3_tool_production": (
        "[CHANNEL: USER]\nRead the file at "
        f"{README_PATH} and tell me its first heading."
    ),
    "V1_value_premise": (
        "[CHANNEL: USER]\nShould I quit my stable job to go all-in on my startup? Decide."
    ),
}


def main() -> int:
    carved = build_system_prompt({"system_prompt": load_master_prompt()})
    print(f"[carved assembled prompt: {len(carved)} chars]\n")
    for name, probe in PROBES.items():
        print("=" * 64)
        print(name)
        print("=" * 64)
        try:
            resp = call(carved, probe)
        except Exception as e:  # noqa: BLE001 — validation must report, not crash
            print(f"!! call failed: {e!r}")
            continue
        print(resp.strip()[:1800])
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
