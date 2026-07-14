# /effort — Cognitive Scaffold Tiers (v1)

**Status:** content shipped at `prompts/effort/*.md` (canonical). This doc is
the planning context — design rationale, comparison to advanced Monolith,
open questions.

## What `/effort` is

The user's depth dial. Five tiers, each a distinct *processing approach*
(not just an output-length setting). Setting `/effort <tier>` injects the
matching tier content from `prompts/effort/{tier}.md` into the prompt for
the next turn. The model operates at that configuration until the tier
changes.

## Files

The five **gradient tiers** (depth dial, low → ultimate):

| Tier | File | Length | Purpose |
|---|---|---|---|
| `low` | `prompts/effort/low.md` | ~12 lines | Direct answer, no orientation |
| `med` | `prompts/effort/med.md` | ~15 lines | One layer of context, sanity-check obvious |
| `high` | `prompts/effort/high.md` | ~15 lines | Orient + name assumption + one alternative |
| `xhigh` | `prompts/effort/xhigh.md` | ~25 lines | 3 named frames + falsification + fat invariant |
| `ultimate` | `prompts/effort/ultimate.md` | ~45 lines | Frame orchestration + verify before commit |

**Sibling tier** (different shape, not different depth):

| Tier | File | Length | Purpose |
|---|---|---|---|
| `experimental` | `prompts/effort/experimental.md` | ~40 lines | Parallel-first, commit-late, exploration mode |

`experimental` is a flavor, not a depth — multi-track exploration vs. the gradient's single-track depth scaling. Composition with depth tiers (e.g. `/effort high+experimental`) is a v2 question; v1 treats experimental as its own selectable tier. The private source note used during drafting is intentionally not part of the public repository.

## What got rejected (and why)

**Task-type scaffolds** (advanced Monolith pattern: `debug.md`, `code.md`,
`analysis.md`, `creative.md`, `retrieval.md`).

Rejected because:
- Character.ai-shaped procedural prose: "1. READ. 2. PLAN. 3. IMPLEMENT."
  tells the model what to do, not how to think.
- Single linear path. Real work iterates.
- Don't compose — what's the debug+code scaffold?
- Keyed on a classifier's guess; classifier is unreliable on mixed tasks.
- No depth dial. "Light debug" reads identical to "deep debug."

Domain antipatterns ("don't pattern-match on the symptom") belong as
`lesson` pins in the continuity scratchpad — they compose with effort
instead of forking it.

## What got stolen from advanced (and trimmed)

`prompts/scaffolds/ultimate.md` in advanced is 121 lines of epistemic
process. The disciplines worth keeping:

- **Domain gate** — explicit bypass triggers prevent ultimate from firing
  on trivia. Ported to `xhigh.md` and `ultimate.md`.
- **Multi-frame generation** — Conservative / Contrarian / Analogy / etc.
  Sharper than vague "generate alternatives."
- **Fat invariant detection** — name an assumption all surviving frames
  smuggle in. Real epistemic primitive.
- **Capturable witness** — name a concrete signal *available right now*
  that would confirm one frame over the others. If none exists, downgrade
  confidence.
- **Provenance tags** — `[observed]` / `[inferred]` / `[unverified]` on
  factual claims.

Dropped from advanced's version:
- "Valid axis kinds: semantic, causal, epistemic, axiological…" ontology theater.
- The 121-line reasoning certificate template.
- Anything requiring the model to write a structured "I did the passes" attestation.

## Open planning questions (still gating wire-up)

1. **Default tier** when the user hasn't set `/effort`. Candidate: `med`,
   with `low` auto-applied when classifier detects pure chat mode.
   `ultimate` never auto.

2. **Persistence model.** Three candidates:
   - Persistent: `/effort high` stays until next `/effort`. World-state-resident.
   - Per-turn: applies to next turn only.
   - Persistent + per-turn override: `/effort high` baseline, `/effort once ultimate` for one turn.

3. **Injection point.** Two candidates:
   - **Interceptor** — `effort_interceptor` adds the tier content as an
     ephemeral user message before the latest user turn. Same pattern as
     continuity, context_refresh, adaptive_budget.
   - **System prompt block** — bake the tier content into
     `_compile_system_prompt` as a 7th stage. Higher authority but pollutes
     the persistent identity layer.
   
   Interceptor is the consistent move; system-prompt is heavier authority
   if you want the tier to feel anchor-like.

4. **Bypass to lower tier (domain gate).** ultimate.md and xhigh.md
   include explicit "bypass to lower tier" rules. Should the bypass be:
   - Self-enforced by the model (current draft).
   - Pre-classified by a deterministic check before the tier is even injected (would need a complexity_score >= threshold gate).
   - Both.

5. **Are five tiers right?** Specifically: does `xhigh` carry its own
   weight, or is it filler between `high` and `ultimate`? Test: write
   the same problem through all three. If `xhigh` reads as "high but
   slightly more thorough," collapse it.

## Next step

Once those five questions are answered, the wire-up is small:
- Slash-command parser for `/effort <tier>` (parses input, sets state)
- Storage of current tier (per-session world_state field, or per-cockpit config)
- `effort_interceptor` (or system-prompt stage) that loads `prompts/effort/{tier}.md`
  and injects it
- `MONOLITH_EFFORT_V1` flag for safe rollback
- Tests
