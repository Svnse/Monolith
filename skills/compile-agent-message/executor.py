"""
compile_agent_message — Intent-Preserving Agent Message Compiler.

Preserves raw_input verbatim and extracts structured metadata using an internal
LLM call. If the LLM backend is unavailable, extraction fields are left null
and a fallback note instructs the caller to run llm_call manually.
"""
from __future__ import annotations

import json
import re
from typing import Any

# ── prompt template for extraction ──────────────────────────────────────────
EXTRACT_PROMPT = """\
You are an Intent-Preserving Message Compiler. Given a raw human message,
extract the following fields accurately without altering the original words.

Fields to extract:
- intent: what the sender wants to accomplish (one sentence)
- task_type: category (query, command, analysis, creative, etc.)
- target: system, file, component, or agent aimed at (or "none")
- constraints: limitations, must-haves, don't-do's (list or "none")
- ambiguity_risks: potential misinterpretations, typos that could cause errors
- missing_info: what's not given that might be needed (or "none")
- execution_packet: clean call to action for the receiver (one sentence)
- verification_checks: how to confirm the task succeeded
- confidence: 0-100 your confidence in the extraction

Return ONLY a JSON object with these fields and no other text.
Do NOT include the raw_input field.

Raw message:
{raw_input}
"""

SYSTEM_PROMPT = "You are a precision extraction engine. Cold, factual, no commentary."


def _extract(raw_input: str, ctx: Any) -> dict[str, Any]:
    """Call the internal LLM to extract fields. Returns a dict of fields or
    a dict with a single _fallback_note key if extraction is not possible."""
    prompt = EXTRACT_PROMPT.format(raw_input=raw_input)
    try:
        # ctx.llm_call may be provided by the runtime; attempt common patterns.
        if hasattr(ctx, "llm_call") and callable(ctx.llm_call):
            response = ctx.llm_call(
                prompt=prompt, system=SYSTEM_PROMPT, max_tokens=1024
            )
        elif hasattr(ctx, "execute_tool"):
            # Try using the llm_call tool via the runtime's tool dispatcher.
            raw = ctx.execute_tool("llm_call", {
                "prompt": prompt,
                "system": SYSTEM_PROMPT,
                "max_tokens": 1024,
            })
            if isinstance(raw, dict) and "data" in raw:
                response = raw["data"].get("response", "")
            else:
                response = str(raw)
        else:
            return {
                "_fallback_note": (
                    "LLM backend not available in executor context. "
                    "Run llm_call manually with EXTRACT_PROMPT from "
                    "executor.py and merge the result with raw_input."
                )
            }

        # Strip markdown fences if present
        clean = re.sub(r"```(?:json)?\s*", "", str(response)).strip()
        data = json.loads(clean)
        expected = {
            "intent", "task_type", "target", "constraints",
            "ambiguity_risks", "missing_info", "execution_packet",
            "verification_checks", "confidence",
        }
        return {k: data.get(k, "") for k in expected}

    except Exception as e:
        return {"_extraction_error": f"LLM extraction failed: {e}"}


def run(cmd: dict, ctx: Any) -> str:
    """Main entry point for compile_agent_message tool."""
    raw_input = cmd.get("raw_input") or ""
    if not raw_input.strip():
        return json.dumps({"error": "raw_input is required and must be non-empty"})

    context = cmd.get("context") or None

    extraction = _extract(raw_input, ctx)

    result: dict[str, Any] = {"raw_input": raw_input}
    if context:
        result["context"] = context
    result.update(extraction)

    return json.dumps(result, indent=2, ensure_ascii=False)
