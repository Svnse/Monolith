"""Prompt builder: converts RunContext into LLM messages."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.loop.contracts import RunContext, RunPolicy, ToolSpec


def build_messages(ctx: RunContext) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _system(ctx.tools, ctx.policy, ctx.cycle)},
        {"role": "user", "content": _cycle(ctx)},
    ]


def build_retry(raw: str, error: str) -> dict[str, str]:
    return {
        "role": "user",
        "content": (
            "Your previous response could not be parsed.\n"
            f"Error: {error}\n\n"
            "Respond with ONLY a valid JSON StepPacket object. "
            "No markdown fences and no commentary outside JSON."
        ),
    }


def _system(tools: list[ToolSpec], policy: RunPolicy, cycle: int = 1) -> str:
    tool_block = _fmt_tools(tools) if int(cycle) <= 1 else _fmt_tools_compact(tools)
    return f"""You are an autonomous coding agent. You have a plan and you execute it.

## Tools
{tool_block}

Call tools as: {{"tool": "name", "args": {{...}}}}

## Response format
Respond with a single JSON object:
{{
  "response": "what you're doing (1 sentence)",
  "intent": "what this step accomplishes",
  "reasoning": "1-2 sentences",
  "actions": [{{"tool": "...", "args": {{...}}}}],
  "self_check": "what happened / what to verify",
  "step_ok": true/false/null,
  "todo_update": "directive text of completed todo item, or null",
  "task_finished": false,
  "finish_summary": ""
}}

## Rules
- Follow the plan. Deviate only if blocked, then explain why.
- Every cycle must have at least one action OR task_finished=true.
- If a tool fails, try a different tool or approach.
- When done, set task_finished=true and provide finish_summary in one sentence.
- Be concise.
- Respond with ONLY the JSON object.

## Budget
- Max cycles: {policy.max_cycles}
- Max tool calls: {policy.max_tool_calls}
- Max time: {int(policy.max_elapsed_sec)}s
"""


def _cycle(ctx: RunContext) -> str:
    pad = ctx.pad
    remaining_cycles = max(0, ctx.policy.max_cycles - ctx.cycle)
    remaining_tools = max(0, ctx.policy.max_tool_calls - ctx.total_tool_calls)
    sections: list[str] = []

    if pad.env_block:
        sections.append(f"ENVIRONMENT:\n{pad.env_block}")

    sections.append(f"GOAL: {pad.goal}")

    if ctx.cycle == 1 and pad.preflight is not None:
        if pad.preflight.variant_constraint:
            sections.append(f"CONSTRAINTS: {pad.preflight.variant_constraint}")
        if pad.preflight.approach:
            sections.append(f"APPROACH: {pad.preflight.approach}")
        if pad.preflight.verification_strategy:
            sections.append(f"VERIFICATION: {pad.preflight.verification_strategy}")

    plan_text = pad.render_todo_text() or str(pad.plan or "")
    if plan_text:
        sections.append(f"PLAN:\n{plan_text}")

    if ctx.cycle > 1 and isinstance(pad.last_check, dict):
        ok = pad.last_check.get("ok")
        ok_str = "OK" if ok is True else ("FAIL" if ok is False else "UNKNOWN")
        note = str(pad.last_check.get("note") or "").replace("\n", " ")
        if len(note) > 160:
            note = note[:160] + "..."
        last_intent = pad.steps[-1] if pad.steps else ""
        evidence_line = ""
        if pad.evidence:
            ev = pad.evidence[-1]
            ev_status = "OK" if ev.ok else "FAIL"
            ev_preview = str(ev.output or "").replace("\n", " ")[:100]
            evidence_line = f"\n  action: {ev.tool} -> {ev_status} - {ev_preview}"
        sections.append(
            "LAST CYCLE:\n"
            f"  intent: {last_intent}\n"
            f"  self_check: {ok_str} - {note}"
            f"{evidence_line}"
        )

    if pad.tool_failures:
        lines: list[str] = []
        for fail in pad.tool_failures[-2:]:
            tool = str(fail.get("tool") or "")
            note = str(fail.get("note") or "").replace("\n", " ")
            if len(note) > 120:
                note = note[:120] + "..."
            lines.append(f"  {tool}: {note}")
        sections.append("TOOL FAILURES:\n" + "\n".join(lines))

    if pad.invariants:
        sections.append("INVARIANTS:\n" + "\n".join(f"  - {inv}" for inv in pad.invariants))

    if pad.open_questions:
        sections.append("OPEN QUESTIONS:\n" + "\n".join(f"  - {q}" for q in pad.open_questions[:3]))

    sections.append(
        f"Cycle {ctx.cycle}/{ctx.policy.max_cycles} - "
        f"{remaining_cycles} remaining, {remaining_tools} tool calls left."
    )
    return "\n\n".join(sections)


def _fmt_tools(tools: list[ToolSpec]) -> str:
    if not tools:
        return "(no tools available)"
    lines: list[str] = []
    for tool in tools:
        required = ", ".join(str(x) for x in (tool.required_args or [])) or "(none)"
        params = ", ".join(f"{k}: {v}" for k, v in (tool.parameters or {}).items()) or "(none)"
        lines.append(
            f"- {tool.name} [{tool.scope}] - {tool.description}\n"
            f"  required: {required}\n"
            f"  params: {params}"
        )
    return "\n".join(lines)


def _fmt_tools_compact(tools: list[ToolSpec]) -> str:
    if not tools:
        return "(no tools available)"
    lines: list[str] = []
    for tool in tools:
        required = ", ".join(str(x) for x in (tool.required_args or []))
        req = f" (requires: {required})" if required else ""
        lines.append(f"- {tool.name} [{tool.scope}] - {tool.description}{req}")
    return "\n".join(lines)
