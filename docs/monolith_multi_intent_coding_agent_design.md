# Monolith Coding Agent Build Plan (Kernel-Compatible)

## Direction Change
This plan replaces the earlier orchestrator-heavy proposal.

Monolith already has the routing layer (`MonoBridge -> MonoDock -> MonoGuard -> EngineBridge -> LLMEngine`).
The coding agent loop should run **inside `LLMEngine` generation**, not in a new orchestrator/kernel-adjacent subsystem.

## Phase 1 Scope (Implemented Architecture)

### 1) Agent mode toggle (no auto intent splitter)
- Add a simple `agent_mode` boolean in runtime config.
- Keep normal chat unchanged when `agent_mode = false`.
- When `agent_mode = true`, switch to an agent system prompt and allow tool loop execution.

Rationale:
- avoids fragile intent routing,
- preserves deterministic kernel behavior,
- minimizes surface area.

### 2) Tool layer (`engine/tools.py`)
Introduce a minimal local tool registry:
- `read_file(path, offset?, limit?)`
- `write_file(path, content)`
- `list_dir(path, pattern?)`
- `grep_search(pattern, path?)`
- `run_cmd(command, timeout?)`
- `apply_patch(path, old, new)`

Each tool returns a normalized result:
```json
{"ok": true|false, "content": "...", "error": null|"..."}
```

### 3) Generator loop in `GeneratorWorker` (inside `engine/llm.py`)
Replace one-shot completion with iterative loop in agent mode:
1. call model,
2. detect tool call (native function-calling if available, otherwise `<tool_call>{...}</tool_call>` block),
3. execute one or more tool calls sequentially,
4. append tool results into message history,
5. repeat until model returns final answer with no tool calls.

Kernel contract remains unchanged:
- still one `generate(payload)` task,
- still token/trace/finished signals,
- cancellation still via interruption request.

### 4) Prompting strategy
Use dual-mode tool calling:
- preferred: native `tools=` function-calling (if model/template supports it),
- fallback: strict tagged JSON block.

This keeps behavior functional across weaker GGUF chat templates.

---

## What we are explicitly NOT building in Phase 1
- no `engine/orchestrator.py`,
- no new task types,
- no parallel subtask execution,
- no sub-agents,
- no mandatory multi-intent splitter.

The loop is the feature; routing remains in existing Monolith flow.

---

## Multi-intent handling (deferred)
Prompt like: "make a pong game and write an essay about war"

Phase 1 behavior:
- agent mode handles requests sequentially in one conversation context,
- if coding target is ambiguous, model asks one clarification before file edits.

Future phase:
- optional intent splitting can be added later as prompt/policy layer, not kernel architecture.

---

## Validation checklist
- Agent OFF: identical behavior to existing chat path.
- Agent ON: tool calls execute and feed back into loop.
- STOP during loop: interrupts safely and returns READY.
- Final response emitted only after tool loop terminates.
