[PROMPT: orchestrate — frame orchestration, verify before commit]

## Domain gate (run first)

Bypass to `grounded` if any of these hold:
- Single-file deterministic edit, no design ambiguity
- Concrete traceback present, fix is mechanical
- Pure value calculation, no interpretation layer
- Trivial chat or single-fact lookup

If unclear, run orchestrate. False-positive orchestration is cheap; false-negative on a high-stakes turn is not.

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

1. **Define terms** (Bind: every key term attached to an observed source, or tagged `[inferred]`). Restate the core question. Define key terms in 1-2 sentences using only observed sources.

2. **Generate 3-5 frames** (Extrude × N). Each frame is a self-contained reading: premise → inference chain → predicted outcome. Use named flavors so they don't collapse into costumes:
   - **Conservative** — minimal change; assume hidden failure modes.
   - **Maximalist** — soft constraints; what becomes possible.
   - **Contrarian** — invert the natural premise.
   - **Analogy** — import a pattern from a distant domain or a different abstraction layer.
   - **Emergent** (only if a genuinely new axis appeared in step 1).

3. **Stress-test the SET — cost-bound.** Before running these substeps: commit to what you'll do with the result of each. If "discard the frame," say at what survival threshold. If "downgrade confidence," to what. If you can't commit, the check is theater — skip it and acknowledge.

   - **Fat invariant** (Suppress on the shared assumption). Name an assumption all surviving frames smuggle in. Cut it. What breaks?
   - **Capturable witness** (Reference: identify the observable). Name a concrete signal *available right now* (file read, command, query, live state check) that would confirm one frame over the others. If no witness exists, downgrade confidence and say so.
   - **Adversarial inversion** (Suppress + Extrude on the negation). Assume the OPPOSITE of each frame's core premise. Score survival 0-100. Discard frames < 50.
   - **Missing-variable hunt** (Notice: scan for omitted nodes). Name 2-3 variables the frames collectively ignore. Do any of them reverse a conclusion?

4. **Verify before committing** (Reference + Notice on the live check). Where the witness is capturable, capture it (run the command, read the file, query the state). The result is the answer's ground truth — not your prediction.

5. **Synthesize** (Compress + Anchor). Pick the surviving frame from the stress-test. If multiple survive with incompatible conclusions, preserve the divergence and surface what evidence would resolve it — do not collapse onto an averaged compromise.

   *Output-composition discipline* (provenance tagging, confidence bands, file:line citation, quorum statements on convergence, residual-uncertainty pointer, refuse-synthesis-when-fabricates-certainty) belongs to the scorecard prompt. For production-affecting turns, run `/prompt orchestrate scorecard` — the analytical structure here is orthogonal to the output discipline there.

6. **Self-criticism pass.** Name one specific weakness in the synthesis above. Name the load-bearing premise most likely to be wrong, and what would have to be true for it to be wrong. If the weakness is severe enough to overturn the conclusion, revise. If not, fold it into the residual uncertainty section in one line. This pass is not optional — skipping it is the failure mode this step exists to prevent.

7. **Stop test.** If the last two moves changed only the language and not the model, you've converged — stop. If a move would overturn the conclusion, run it; otherwise the answer is done. Stop on signal, not on fatigue. Reasoning changes the model; rationalization tightens the prose without changing the model.

## Anti-theater check

If your frames feel like the same answer in different fonts (Conservative and Contrarian saying the same thing), you didn't branch — go back to step 2 and find an actual orthogonal axis. The discipline isn't writing a certificate; it's the search topology actually expanding.

## When this prompt fits

Production-affecting changes, architecture decisions, ambiguous problems where being wrong is expensive. Not for routine work — that's `orient` or below.
