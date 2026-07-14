# M3 — Curiosity (V0) — Design & Spec

- **Date:** 2026-06-02
- **Author:** Monolith agent (Claude), autonomous-build mandate from E
- **Status:** **BUILT (V0, 2026-06-02) + adversarially reviewed.** Ships dark (`MONOLITH_CURIOSITY_V1` default OFF). Propose-only. All tests green; live-DB check confirmed exclusivity (10 aligned → 0 emergence / 10 curiosity / 0 overlap). Impl: `core/curiosity.py`, `skills/curiosity/`, `tests/test_curiosity*.py`.
- **Builds on:** M2 (`docs/specs/M2_IDENTITY_EVOLUTION_SPEC.md`) — reuses the identity-alignment scorer.

## 0. The pivot (empirical, 2026-06-02)

Original M3 plan was "pull = NOVEL × identity-aligned." A read-only check against the **live `acatalepsy.sqlite3`** killed it: 68 self-derived ACUs, uniformly fresh/L1/low-reinforcement; `acu_relations` **empty** (no overlaps/partial edges); l1 canonical_log events **all zero**; bearing unverified referents **zero**. So there is no "frontier of partial knowledge" population distinct from M2's, and reweighting M2's rows by novelty would **contradict** M2 (one claim labeled both "consolidate" and "be curious about").

**Resolution (E-approved): unify.** M2 and M3 are two *dispositions* of one identity signal, split by **stability**:
- identity-aligned **+ stable** (reinforced / crystallized) → **emergence** (consolidate into identity — M2).
- identity-aligned **+ not-yet-stable** (fresh) → **curiosity** (a pull to explore — M3).

A claim has exactly ONE disposition → no contradiction. On current data (all fresh): M2 correctly goes quiet (nothing stable to consolidate yet); M3 surfaces the ~10 fresh identity-aligned pulls. As claims reinforce across sessions they **graduate** curiosity → consolidation. This also fixes a latent M2 issue (it currently proposes consolidation for fresh self-claims with no stability gate — slightly Mad-Cow-off).

## 1. What gets built / changed

1. **`core/identity_alignment.py`** — add `stability_score(acu_row) -> float` (0.6·reinforcement-saturation + 0.4·l_level + truth bonus) and `STABILITY_THRESHOLD = 0.5`. Shared by both dispositions.
2. **`core/identity_emergence.py` (M2 fix)** — emergence candidate now also requires `stability ≥ STABILITY_THRESHOLD`. Update M2 emergence tests to ingest *reinforced* (stable) claims; fresh claims become curiosity, not emergence.
3. **`core/curiosity.py` (new)** — `detect_pulls()`: identity-aligned (`confidentity ≥ align_threshold`) AND `stability < STABILITY_THRESHOLD`; `pull_strength = identity_alignment × (1 - stability)` (fresher = stronger). **Retireable** (advisor): a seen-set in the ledger caps how many times a pull resurfaces, so the heartbeat doesn't loop the same items forever. Emits `curiosity_pull_detected`. Read-only (`persist=False`); propose-only.
4. **Curiosity ledger** — `latest_curiosity` + `surfaced` seen-set in `core/identity_milestones.py` (identity-owned state; Observer reads).
5. **`skills/curiosity/` (new)** — `{"op":"detect"}` deterministic ranked pulls (no LLM). Propose-only: surfaces what it's curious about; never auto-pursues (pursuit is M1's job).
6. **Observer line** — read-only curiosity line (Observer reads the ledger; never writes).
7. **`canonical_log_kinds.py`** — add `curiosity_pull_detected`, bump `KIND_VERSION` 6→7.
8. **Heartbeat** — `record_outcome` + bootstrap hooks also call `detect_pulls()`; gated by **`MONOLITH_CURIOSITY_V1` (default OFF / ships dark)**.
9. **Rule 6 deletion** — delete the dead `addons/system/observer/desire_view.py` stub + its tests. It was inert (projected the bearing goal, `visible_to_model=False`); M3's curiosity surface (ledger → Observer line → skill) supersedes it. (If a unified directed+undirected "desire" abstraction is wanted later, that's a deliberate M1 design, not a revived stub.)

## 2. Disposition math

`stability = clamp(0.6·min(reinforcement,5)/5 + 0.4·{L3:1.0, L2:0.6, L1:0.2} + (truth=='confirmed' ? 0.1 : 0))`.
Fresh L1 reinf=1 → 0.20 (curious). Reinforced L1 reinf=5 → 0.68 (consolidate). Threshold 0.5.

**Boundary (explicit):** emergence is eligible at `stability >= 0.5`; curiosity is `stability < 0.5`. Complementary (`>=` vs `<`) → every row has exactly one disposition. Exactly 0.5 is unreachable with integer reinforcement inputs, but by rule emergence takes precedence there. As a claim reinforces it crosses 0.5 and **graduates** curiosity → emergence (tested: `test_pull_graduates_to_emergence_when_reinforced`).

## 3. Done-gate (acceptance)

`tests/test_curiosity_firerate.py`: against the real seed + the live ACU distribution shape, the curiosity loop must surface a plausible fraction of fresh identity-aligned claims (the ~10 the overlap check found) AND emergence must be empty on the same fresh data (proving the dispositions are exclusive, not duplicative). Verify on the live DB read-only before declaring done.

## 4. Propose-only / Mad Cow / frozen

Curiosity surfaces and ranks; it never acts, promotes, or spends compute pursuing. A fresh self-claim being "curious about" is not promotion (Mad Cow safe; the disposition split *is* the Mad-Cow gate — self-claims can't be consolidated until reinforced). Frozen (untouched): `monokernel/*`, `engine/bridge.py`, `core/world_state.py`, `core/task.py`.

## 5. Deferred to M1 / V1

Pursuing a pull (forming a goal/question from it) is **M1** (planner). Deferred to V1:
- **Real NOVEL dedup.** Both emergence and curiosity use the cheap `canon in corpus` substring check (a near-no-op — a triple is rarely a substring of prose). A genuine dedup against accepted Emergent claims / a seen-set is the proper NOVEL predicate.
- **Graph-frontier curiosity signal.** Once `acu_relations` / intake PARTIAL actually populate in live use (currently empty), partial-match / overlaps edges / unverified referents become a richer curiosity source than the stability-split alone.
- **Embedding alignment** (flip `MONOLITH_IDENTITY_ALIGN_EMBED`), **decay-by-time** (vs surface-count) for retirement, **seen-set tombstoning** of graduated keys.
- **Ledger atomicity under parallelism.** The curiosity seen-set uses load-mutate-save; safe for the single-writer heartbeat today, needs file locking if the heartbeat is ever parallelized.
- **Tool-catalog metadata.** The `curiosity` skill is discovered + reachable (verified) but its optional `threshold` param isn't advertised in `_TOOL_RUNTIME_META`; add for discoverability if stricter validation is wanted.
