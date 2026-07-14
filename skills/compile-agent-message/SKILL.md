---
name: compile_agent_message
description: Intent-Preserving Agent Message Compiler — turn raw human input into a structured agent-to-agent packet while preserving the original words verbatim. Never rewrite, correct, smooth, or replace the raw input; only add structure around it. Extracts intent, task type, target, constraints, ambiguity/typo risks, missing info, execution packet, verification checks, and confidence. Style: cold, precise, no fluff. Improve transmission, not expression.
---

## Parameters

- `raw_input` (required) — the original human message to preserve verbatim.
- `context` (optional) — additional context the sender wants to include.

## Call shape

{"tool":"compile_agent_message","raw_input":"<RAW_INPUT>","context":"<optional context>"}

## Output

A JSON object containing `raw_input` and the extracted fields. If the executor's internal LLM is unavailable, extraction fields are null and a `_fallback_note` instructs the caller to run `llm_call` manually.

## Fallback usage

When extraction fails, chain an `llm_call` with the raw_input and the extraction prompt template shown in executor.py.
