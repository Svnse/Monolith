"""Preflight planning call for the loop runtime."""

from __future__ import annotations

import json
from typing import Any, Callable

from engine.loop.contracts import PreflightResult

InferFn = Callable[[list[dict[str, str]]], str]

_PREFLIGHT_SYSTEM = (
    "You are a planning and comprehension engine. You do not execute. "
    "You do not write code. You analyze a user request against an environment "
    "and produce a structured reasoning artifact that an execution agent will follow. "
    "Respond with ONLY a JSON object. No markdown fences."
)


def _build_preflight_user_message(
    user_prompt: str,
    env_block: str,
    *,
    tools: list[Any] | None = None,
    target_lines_per_step: int | None = None,
) -> str:
    tools_block = _format_tools_block(tools)
    msg = f"""ENVIRONMENT:
{env_block}

USER REQUEST:
{user_prompt}

TOOLS AVAILABLE:
{tools_block}

Analyze this request against the environment. Produce a JSON object with these exact keys:

{{
  "variant_literal": "exact restatement of what user asked",
  "variant_constraint": "restate emphasizing what CANNOT be done given the environment",
  "variant_intent": "what user wants to experience/have when this is done",
  "variant_failure": "this task fails if...",
  "variant_minimal": "smallest version that satisfies the request",

  "intent_type": "coding | modification | question | conversation | research | debugging",
  "response_mode": "reply | act | reply_then_act",
  "complexity_class": "atomic | short_multi | deep_multi",
  "primary_constraint": "single hardest constraint from the environment or request",
  "implicit_assumptions": ["things the user did not say but expects"],

  "plan_granularity": 1-5,
  "execution_weight": 1-5,

  "vision_primary": "concrete description of the fully-done state",
  "vision_minimum": "concrete description of minimum viable done state",

  "approach": "given environment constraints, the right technical approach",
  "risks": ["what could go wrong during execution"],
  "sequencing": "what depends on what, what must happen first",
  "verification_strategy": "how the executor will know each step worked",

  "action_steps": [
    {{"what": "description", "tool_hint": "tool_name", "depends_on": [], "success_signal": "expected result"}}
  ],

  "todo": [
    {{"directive": "imperative one-liner", "tool_hint": "tool_name", "blocking": true/false}}
  ],

  "invariants": ["hard constraints that must never be violated"]
}}

Scale detail to complexity. If simple (plan_granularity 1-2), keep todo minimal.
If complex (4-5), be thorough.
"""
    if int(target_lines_per_step or 0) > 0:
        target = int(target_lines_per_step)
        msg += (
            "\n"
            "CONSTRAINT - OUTPUT CAPACITY:\n"
            f"The execution model can produce at most ~{target} lines of code per action.\n"
            "Each todo item that involves writing or modifying code MUST be scoped to fit within\n"
            "this limit. If a logical unit of work exceeds "
            f"{target} lines, decompose\n"
            "it into multiple sequential todo items. Prefer many small focused items over few large ones.\n"
            'Do NOT create todo items like "Implement all game logic" - break them into discrete,\n'
            'independently writable units (e.g. "Add Ball class with move/bounce methods",\n'
            '"Add score tracking and reset", "Wire keyboard input to paddle movement").\n'
        )
    return msg


def run_preflight(
    user_prompt: str,
    env_block: str,
    infer_fn: InferFn,
    *,
    tools: list[Any] | None = None,
    target_lines_per_step: int | None = None,
) -> PreflightResult | None:
    """Execute a single preflight LLM call. Returns None on parse/infer failure."""
    messages = [
        {"role": "system", "content": _PREFLIGHT_SYSTEM},
        {
            "role": "user",
            "content": _build_preflight_user_message(
                str(user_prompt or ""),
                str(env_block or ""),
                tools=tools,
                target_lines_per_step=target_lines_per_step,
            ),
        },
    ]
    try:
        raw = infer_fn(messages)
    except Exception:
        return None
    return _parse_preflight(raw)


def _parse_preflight(raw: str) -> PreflightResult | None:
    text = str(raw or "").strip()
    if not text:
        return None

    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None

    if not isinstance(data, dict):
        return None

    def s(key: str, default: str = "") -> str:
        return str(data.get(key, default) or default).strip()

    def sl(key: str) -> list[str]:
        val = data.get(key, [])
        if isinstance(val, list):
            return [str(v).strip() for v in val if str(v).strip()]
        return []

    def i(key: str, default: int = 3) -> int:
        try:
            return max(1, min(5, int(data.get(key, default))))
        except Exception:
            return default

    def dl(key: str) -> list[dict[str, Any]]:
        val = data.get(key, [])
        if isinstance(val, list):
            return [v for v in val if isinstance(v, dict)]
        return []

    try:
        return PreflightResult(
            variant_literal=s("variant_literal"),
            variant_constraint=s("variant_constraint"),
            variant_intent=s("variant_intent"),
            variant_failure=s("variant_failure"),
            variant_minimal=s("variant_minimal"),
            intent_type=s("intent_type", "coding"),
            response_mode=s("response_mode", "act"),
            complexity_class=s("complexity_class", "short_multi"),
            primary_constraint=s("primary_constraint"),
            implicit_assumptions=sl("implicit_assumptions"),
            plan_granularity=i("plan_granularity", 3),
            execution_weight=i("execution_weight", 3),
            vision_primary=s("vision_primary"),
            vision_minimum=s("vision_minimum"),
            approach=s("approach"),
            risks=sl("risks"),
            sequencing=s("sequencing"),
            verification_strategy=s("verification_strategy"),
            action_steps=dl("action_steps"),
            todo=dl("todo"),
            invariants=sl("invariants"),
        )
    except Exception:
        return None


def _format_tools_block(tools: list[Any] | None) -> str:
    if not tools:
        return "(none provided)"
    lines: list[str] = []
    for item in tools:
        name = ""
        scope = ""
        if isinstance(item, dict):
            name = str(item.get("name") or "").strip()
            scope = str(item.get("scope") or "").strip()
        else:
            name = str(getattr(item, "name", "") or "").strip()
            scope = str(getattr(item, "scope", "") or "").strip()
        if not name:
            continue
        label = f"{name} [{scope}]" if scope else name
        lines.append(f"- {label}")
    return "\n".join(lines) if lines else "(none provided)"
