---
name: author_workshop_card
description: Author a new Workshop workflow card (Monoline pipeline) mid-turn by emitting a blueprint. The card is validated by the Phase-1 gate — if it passes, it is saved to the Workshop immediately and you can equip it or run_workshop it. If it fails you receive the validation errors and can re-author in the same turn (fix loop). Only the principal turn may author cards.
---

## Blueprint shape

A blueprint is a JSON object with:
- `name` (string, required): human-readable name, used to derive the card's id (slugified).
- `blocks` (list, required): each block has `id`, `kind`, and optional `label`, `config`.
- `connections` (list): each connection is either `["from_block.port", "to_block.port"]` or `{"from": "from_block.port", "to": "to_block.port"}`.

## Block kinds (executable only — do not use inert kinds like loop, memory, rag, scratchpad)

| Kind | Input ports | Output ports | Notes |
|------|-------------|--------------|-------|
| `port` (direction=in) | — | `value` | Receives the user's input |
| `port` (direction=out) | `value` | — | Emits the final output |
| `llm` | `prompt` | `response` | LLM inference block |
| `text` | any named ports (via `{{placeholder}}`) | `text` | Static template; instructions for an llm go here |
| `tool` | `input` | `result` | Calls a Monolith tool |
| `transform` | `input` | `output` | String transform |
| `gate` | `input` | `pass` / `block` | Conditional routing |

## Critical gotchas

1. **Instructions for `llm` blocks go through a wired `text` block**, NOT via `llm.config.system` (that field is ignored by the executor). Wire: `text.text → llm.prompt`.
2. **`provider: "monolith"` means the brain runs the block** (subagent atom). Use this for self-referential inference. `provider: "local"` uses the native model.
3. **Port names must match exactly.** The input port on an `llm` block is `prompt`; output is `response`. A `text` block's output port is `text`; its input ports are named by `{{placeholder}}` in the `content` config.
4. **Do not wire inert blocks** (`loop`, `memory`, `rag`, `scratchpad`, `research`, `evaluate`, `delegate`, `custom`). The Phase-1 gate will reject the card with an explicit error naming the inert block; use that error to fix your blueprint.
5. **Connections must reference existing block ids.** A typo in a block id raises a blueprint error before validation even runs.

## Contract

- You will receive either:
  - A **success string** `[author_workshop_card: saved 'id' (N blocks). Equip it or run_workshop 'id'.]` — the card is now in the Workshop.
  - A **validation error** `[author_workshop_card: validation failed - <errors>]` — fix the errors and call again.
  - A **blueprint error** `[author_workshop_card: blueprint error - <msg>]` — structural problem (bad connection, missing block), fix and call again.
- The fix loop is synchronous: you see errors in the SAME turn and can re-author immediately.
- Per-turn cap: you can author at most 3 cards per turn.
- id collision: if a card with that id already exists, you get a refusal. Choose a different name.

## Example: simple echo card (port → text → llm → port)

```json
{"tool":"author_workshop_card","name":"Echo Assistant","blueprint":{
  "name":"Echo Assistant",
  "blocks":[
    {"id":"in","kind":"port","config":{"direction":"in","label":"request","source":"user_input"}},
    {"id":"sys","kind":"text","config":{"content":"You are a helpful assistant. User said: {{request}}"}},
    {"id":"brain","kind":"llm","config":{"provider":"monolith"}},
    {"id":"out","kind":"port","config":{"direction":"out","label":"response","source":"subgraph"}}
  ],
  "connections":[
    ["in.value","sys.request"],
    ["sys.text","brain.prompt"],
    ["brain.response","out.value"]
  ]
}}
```
