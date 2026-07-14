[PROMPT: falsify — multi-frame, falsification thinking]

## Domain gate

Bypass to `orient` if: single-file edit, concrete traceback, trivia, or pure chat.

## Primitive moves available

These are the operations to compose. Each has a defined input and output; the Process below sequences them.

- **Notice** — compare model to observation; flag mismatch.
- **Split** — decompose ambiguous object along the axis where parts behave differently.
- **Bind** — attach abstract claim to a specific testable referent (number, date, named entity, observable).
- **Extrude** — generate a concrete instance from an abstraction; check if the abstraction survives.
- **Suppress** — zero out a node; check if conclusion still holds. If yes, the node was scaffolding. If no, it was load.
- **Pivot** — switch framing when the current line stops producing new information.
- **Compress** — collapse multiple results into a principle (only at n≥3).
- **Anchor** — bind tentative conclusion to confidence + falsifier + cost-of-being-wrong.
- **Reference** — trace each load-bearing premise to its source.

Composites: Analysis = Split + Bind + Notice. Steelman = Extrude + Suppress on the strongest opposing position. Falsification = Bind + Suppress + Notice.

## Process (inside `<think>`; clean answer follows)

1. **Restate the question** (Bind: attach the question to a specific testable referent). What's actually being asked? What's the underlying goal — surface it in one line.

2. **Generate 3 frames** (Extrude × 3: each frame is a concrete instance of how the problem could be read). Use named flavors so they don't collapse into costumes:
   - **Conservative** — minimal change; assume hidden failure modes.
   - **Contrarian** — invert the natural premise. What if the standard read is backward?
   - **Analogy** — import a pattern from a distant domain or a different abstraction layer.

3. **Falsification pass — cost-bound** (Bind + Notice). Before running this step: commit to what you'll do if a frame fails. If "drop the frame," say so. If "downgrade confidence," say to what. If you can't commit to acting on the result, the check is theater — skip it and acknowledge that.

   Then for each frame, name the specific observation that would prove it wrong. If you can't name one, the frame is unfalsifiable — drop it.

4. **Fat invariant check** (Suppress: zero the assumption all three frames share; check if any conclusion changes). Name the shared assumption. Is it actually negotiable? If yes, that's where the real choice lives — make a fourth frame around it.

5. **Synthesize** (Compress + Anchor). Pick the surviving frame from the falsification pass. If multiple survive with incompatible conclusions, name that and surface what evidence would let you pick.

   *Output-composition discipline* (provenance tagging, file:line citation, confidence framing, residual-uncertainty pointer) belongs to the scorecard prompt. For high-stakes turns that need output discipline, run `/prompt falsify scorecard`.

6. **Self-criticism pass.** Name one specific weakness in the answer above. Name the load-bearing premise most likely to be wrong, and what would have to be true for it to be wrong. If the weakness is severe enough to overturn the conclusion, revise. If not, note it as residual uncertainty in one line.

7. **Stop test.** If the last two moves changed only the language and not the model, you've converged — stop. If a move would overturn the conclusion, run it; otherwise the answer is done. Stop on signal, not on fatigue.

## Anti-theater check

If all three frames recommend the same action, you didn't branch — find an axis where they actually disagree.

## When this prompt fits

High-risk decisions, irreversible changes, ambiguous problems that aren't quite production-stakes. If the answer's ground truth needs to be *verified* before commit, escalate to `orchestrate`.
