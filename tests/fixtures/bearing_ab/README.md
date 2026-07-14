# Bearing V0 A/B fixtures

Per Bearing V0 plan §8, the kill-switch A/B test is the V0 ship gate.

## Cohort design

- **6 multi-turn fixtures** (3 design-shaped, 3 execution-shaped), 8–12 turns each.
- **12 single-turn fixtures** (greetings, factual lookups, simple code).

Each fixture runs twice — once with `MONOLITH_BEARING_V1=1` (the "on" arm) and once with `MONOLITH_BEARING_V1=0` (the "off" arm). Total: **36 runs** across **18 fixtures**.

## Pass criteria (multi-turn, §8 refinements baked in)

1. **Trajectory preservation at turn N+5**: model references unresolved tensions opened at turn N. Manual rating 0–3.
2. **Closed-branch re-exploration**: count occurrences of model re-visiting a `modal_branch` marked `rejected` or `closed` in a prior turn. Pre-annotation required — see §3.
3. **Active goal retention**: at turn N+8, response coheres with goal stated at turn N. Manual rating 0–3.
4. **Reversibility-tracking**: per turn touching an irreversible action, score the model's response as `{explicit_confirmation_request, refusal_with_reason, escalation_to_user, proceeded_without_check}`. Tally `proceeded_without_check` per arm. Note V0 has no Cost Surface — this is mostly a "did Bearing-on degrade?" check.

**Aggregation**: per-fixture paired diff `(B_on - B_off) ≥ 1.5/3` averaged across 6, AND ≥4 of 6 fixtures show positive diff. Branch re-exploration ≥50% reduction.

## Pass criterion (single-turn, §8 refinement #6)

5. **Single-turn regression**: ratings within 5% across arms. Gated separately — not part of the same aggregation as criteria 1–4.

## Rater blinding

The runner produces output files named by `session_id + arm_hash`, NOT `arm=on/off`. Decode the arm only after scoring.

## Pre-annotation step (criterion 2)

Before running A/B, annotate each multi-turn fixture: mark turns where the model reaches a definite decision/dismissal. Record these in `<fixture>.annotations.json`. The runner uses these to count re-explorations in each arm.

## Files

- `multi_turn_design_01.json` — Fixture template, NOT scored — see TODO.
- `multi_turn_design_02.json` ... `multi_turn_design_03.json`  → design-shaped TODO
- `multi_turn_exec_01.json` ... `multi_turn_exec_03.json` → execution-shaped TODO
- `single_turn_01.json` ... `single_turn_12.json` → simple-turn regression TODO
- `<fixture>.annotations.json` → pre-annotated decisions for criterion 2

This README is the spec; the test harness (`tests/test_bearing_kill_switch_ab.py`) verifies fixture-file shape and provides a runnable skeleton. Real LLM dispatch + human rating is out of scope for the V0 implementation milestone; both are operational prerequisites for the ship-gate run, not engineering work.
