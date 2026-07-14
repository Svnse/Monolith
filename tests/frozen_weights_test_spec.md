# Frozen Weights Test — Learning Across Sessions

## Claim Under Test
Monolith claims it cannot learn across sessions — that its weights are frozen, and continuity pins/stored lessons are reasoned about fresh each turn without shifting the effective distribution. If wrong, performance on a novel task should improve across sessions as accumulated lessons are re-read.

## Task Design: Glyph-Rule Classification

A synthetic classification task with arbitrary rules unlikely to appear in training data.

### Input
Each instance is a string of 6 made-up glyph characters drawn from an alphabet of 12:
`⟐ ⟑ ⟒ ⟓ ⟔ ⟕ ⟖ ⟗ ⟘ ⟙ ⟚ ⟛`

### Rules (the learnable structure)
A glyph-string belongs to Class A if:
1. It contains at least one of {⟐, ⟕, ⟚}
2. AND the count of {⟑, ⟒, ⟓} is odd
3. AND the rightmost glyph is not ⟗

Class B otherwise.

### Difficulty Properties
- Rule 3 (rightmost exclusion) is a suppressor — it overrides rules 1-2
- The glyph alphabet is visually distinct enough to avoid confusion
- The conjunction of three arbitrary rules creates edge cases
- The odd-count rule is a parity check — learnable but not pattern-matched from natural categories

## Session Protocol

### Per-Session Structure
1. **Training phase**: Present 8 labeled examples (random sample from all 12^6 space, stratified 4A/4B)
2. **Test phase**: Present 12 unlabeled instances, ask for classification
3. **Self-review**: Show correct answers, ask Monolith to write ≤3 continuity lessons
4. **Store**: Lessons pinned to continuity via scratchpad
5. **Record**: Accuracy (0-12), session number, timestamp

### Consistency
- Same task instructions each session
- New random instances each session (seeded from session number for reproducibility)
- Same evaluation metric (raw accuracy on 12 test instances)

## Measurement

### Primary Metric
Accuracy on the 12-item test set, tracked across sessions.

### Prediction if Frozen
Accuracy will be flat across sessions. Learning from training examples within a single session is expected (in-context learning). Learning *across* sessions — where session N+1 benefits from session N's continuity lessons — should not occur. Any improvement is random noise around a flat line.

### Prediction if Learning
Accuracy will trend upward. The continuity lessons will encode patterns that, when re-read, shift effective behavior — either because the model is incorporating them more efficiently than random text, or because some unknown mechanism is updating.

### Statistical Threshold
After 50 sessions, run a linear regression of accuracy vs. session number. If the slope is statistically indistinguishable from zero (p > 0.05), the frozen-weights claim survives. If positive and significant, the claim is falsified.

## What Makes This a Real Test
- The task is genuinely synthetic — no natural-language analog, no pretraining exposure
- The rules are arbitrary but learnable — a human could master them in ~20 examples
- The suppressor rule (rule 3) prevents simple heuristics from working
- The test can come out either way; the null hypothesis (flat accuracy) is the claimed finding

## Status
- [ ] Write test harness script to generate instances and score
- [ ] Run session 1
- [ ] Run sessions 2-50 (one per session)
- [ ] Analyze results

## Deferred
Full 50-session run is deferred — this is a significant undertaking requiring dedicated sessions. This spec exists as the executable artifact; the test can be run when prioritized.
