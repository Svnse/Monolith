# M2 — Identity Evolution (V0) — Design & Spec

- **Date:** 2026-06-02
- **Author:** Monolith agent (Claude), under autonomous-build authorization from E
- **Status:** **V0 BUILT + adversarially reviewed (2026-06-02).** All tests green; ships dark (auto-run flag default OFF). Approval gate waived by E; this doc + `canonical_log` are E's audit trail. See §12 for what shipped.
- **Mode:** propose-only (nothing spends compute unbidden; detection is cheap/deterministic, drafting is invoked)

---

## 0. Decisions locked

| # | Decision | Who | Rationale |
|---|----------|-----|-----------|
| 1 | Autonomy boundary = **propose-only** (always) | E | Rides existing rails (`proposals.py`, `decisions`/`canonical_log`, `PolicyDecision`); nothing acts unbidden. |
| 2 | Design order = **M2 (identity) → M3 (curiosity) → M1 (planner)** | E | Identity is the referent: M2 defines M3's target; what it's curious about shapes M1's goals. |
| 3 | Identity unit = **two-layer** (prose operative face + ACU governor underneath) | E | Human-readable & E-correctable; honors emergence/mass-budget; powers M3 (alignment = confidentity vs the cluster). |
| 4 | Where change lands = **frozen Origin-0 + growing Emergent region** | delegated → agent | Origin-0 stays the byte-stable diffable gravity well; authority transfers to self-derived patterns without erasing the seed. Protection becomes trivial (Origin-0 byte-locked). origin-N→N+1 snapshotting layers on as V1. |

---

## 1. Concept

Identity = **Origin-0** (frozen, hand-authored seed; the gravity well) **+ Emergent** (self-derived, grows only by E-approved amendments).

The accumulated ACU distribution is the **evidence**. A cheap deterministic **emergence detector** notices when high-confidentity self-derived claims have accumulated that the operative identity does not yet reflect, and **surfaces a signal** (advisory). When **bidden** (an `identity_review` skill the model or E invokes), a single grounded LLM call **drafts an amendment** to the Emergent region and files it to `proposals.json`. E reviews/applies through the existing manual gate. A **confidentity-mass budget** governs how much the Emergent region may change per milestone. Every step emits a `canonical_log` event.

```
self-derived ACUs accrue (intake)  ── confidentity scored (M0 alignment) ──┐
                                                                            ▼
   bootstrap/idle ── detect_emergence() [deterministic, no LLM] ── divergence ≥ threshold?
                                                                            │ yes
                                              Observer line + canonical_log: identity_emergence_detected
                                                                            │
                                    (bidden) /identity_review ── ONE LLM draft, grounded in evidence ACUs
                                                                            │
                              validate: Origin-0 untouched + mass budget + caps
                                                                            ▼
                          proposals.propose_amendment(target="identity.md", section="Emergent", …)
                                                                            │
                                              E reads queue → applies to Emergent region (manual; V1 = gated auto-apply)
                                                                            ▼
                                              canonical_log: identity_amendment_applied → milestone ledger updated
```

---

## 2. Existing substrate reused (do not rebuild)

| Need | Reuse | File |
|------|-------|------|
| Propose-only review surface for identity edits | `propose_amendment()` / `list_proposals()` — `identity.md` already an allowed target, ≤2000-char caps, human gate | `core/proposals.py` |
| Audit floor / processing trace | `canonical_log.append` (+ `append_on`) | `core/acatalepsy/canonical_log.py` |
| Decision vocabulary (accept/reject/edit/defer, `user_e` authority) | pattern mirror; events already typed | `core/acatalepsy/decisions.py` |
| Identity load/save/seed | `load_identity`, `save_identity`, `IDENTITY_PATH`, `_DEFAULT_IDENTITY` | `core/identity.py` |
| Identity → locked ACUs | `extract_origin0_claims`, `ensure_origin0_acus_loaded` (confidentity=1.0, lock_reason='origin_0') | `core/identity_acus.py` |
| ACU storage / confidentity column / retrieve | `ACUStore.ingest`, `ingest_locked`, `retrieve`, `confidentity` column | `core/acu_store.py` |
| Operative `[IDENTITY]` injection (renders file verbatim) | `build_system_prompt` | `core/llm_config.py` |
| Advisory turn-boundary surface | Observer `build_observer_snapshot` already reads locked identity + bearing + recent log | `addons/system/observer/runtime.py` |
| Bounded-autonomy template (caps, model-proposes/code-decides, journal, rollback, protected sections) | MonoThink — pattern, not code reuse | `core/monothink.py` |

**Frozen — do NOT touch:** `monokernel/*`, `engine/bridge.py`, `core/world_state.py`, `core/task.py` (Kernel Contract v2). Observer is **advisory / no mutation authority** — M2 must not make Observer write proposals; Observer only *surfaces* the detector's read-only signal.

---

## 3. The missing verbs (what M2 builds), mapped to nouns

1. **M0 — Identity-alignment scoring (the load-bearing prerequisite).** Self-derived ACUs currently have no `confidentity`. Build a scorer `confidentity = confidence × identity_alignment`:
   - `confidence` (V0): behavioural — normalized reinforcement / provenance (user>self), already in the row.
   - `identity_alignment` (V0): **deterministic lexical/entity overlap** between the ACU canonical form and the current identity corpus (Origin-0 + Emergent claims). *To be confirmed by code review:* if an embedding utility already exists in-repo, use cosine similarity instead; otherwise lexical overlap is the V0 default with embeddings as a V1 swap behind one function.
   - Write the computed `confidentity` onto self-derived ACU rows (new write path; locked Origin-0 rows untouched).
   - **Make it load-bearing without regressing recall:** add a dedicated confidentity-aware selection (`retrieve_identity_candidates()`), do **not** change the global `retrieve()` default ordering (other systems depend on reinforcement+recency). Any change to default ordering is V1 and flag-gated.

2. **Emergent region in `identity.md`.** Structural convention: everything above the marker is frozen Origin-0; an appended, mutable Emergent region holds approved self-derived claims.
   ```
   …Origin-0 seed (frozen)…

   <!-- EMERGENT:BEGIN — self-derived; Origin-0 above is frozen & diffable -->
   ## Emergent
   - <approved emergent claim> (milestone N, confidentity X.XX)
   <!-- EMERGENT:END -->
   ```
   New helper `core/identity_regions.py`: `split_regions(text) -> (origin0, emergent)`, `render_identity()`, `apply_emergent_amendment(before, after)` that **refuses any byte change above `EMERGENT:BEGIN`** (code-enforced protection, MonoThink-style). `extract_origin0_claims` becomes region-aware (stops at `EMERGENT:BEGIN`) so Emergent prose is never ingested as Origin-0-locked. Because `build_system_prompt` already renders the file verbatim, no prompt-assembly change is needed once amendments live in the file.

3. **Emergence detector (deterministic, no LLM).** `core/identity_emergence.py: detect_emergence() -> EmergenceReport`. Reads self-derived L2+ ACUs with confidentity ≥ threshold not yet reflected in the identity corpus; computes a divergence/mass signal vs a watermark (ACU count at last check). Emits `canonical_log: identity_emergence_detected` when over threshold. Surfaced as an advisory Observer line (read-only). Gated so it only fires when ≥ N new qualifying ACUs accrued since last check.

4. **`identity_review` skill (bidden drafting).** Takes the `EmergenceReport`; ONE grounded LLM call drafts an Emergent amendment (`current_text`/`proposed_text` for the Emergent region, with rationale citing evidence ACU ids). Validates against Origin-0 protection + mass budget + `proposals.py` caps. Files via `propose_amendment(target="identity.md", section="Emergent", …)`. Emits `canonical_log: identity_amendment_proposed`. Never applies (propose-only).

5. **Confidentity-mass budget + milestone ledger.** `core/identity_milestones.py` (JSON at `CONFIG_DIR/identity_milestones.json`, atomic write; canonical_log-derived view as source of truth): tracks milestone N, change-budget spent, the accumulation watermark, and an Origin-0 hash for diffability. V0 budget = conservative caps (one pending emergent amendment at a time; ≤ proposals' 2000-char cap; a confidentity-mass cap per milestone window). Graduated schedule (tighter→looser) = V1.

---

## 4. Layer accounting (Rule 6 — additions pair with deletions)

- M2 is mostly additive (it builds missing verbs), but it **subsumes** two existing ad-hoc paths:
  - The **manual-only identity use of `propose_amendment`** (model spontaneously proposing identity edits) is subsumed by the evidence-grounded emergence loop. `propose_amendment` is *kept* (it's the reused surface) but its identity use becomes detector-driven, not ad-hoc.
  - `core/identity.py:append_identity()` (ad-hoc line appends to identity.md) is **superseded** by the Emergent-region amendment path and flagged deprecated (kept for back-compat; no new callers).
- Net new cognitive layer = "Emergent identity"; the layer it pairs against = the now-deprecated ad-hoc append/propose path for identity prose.

## 5. Flags (follow `MONOLITH_*_V1` convention)

- `MONOLITH_IDENTITY_EMERGENCE_V1` — gates the auto-run detector (the `record_outcome` + bootstrap heartbeats). **Ships DARK (default OFF)**, matching `acu_retrieval`'s "ships dark for first observation" precedent: the heartbeat is a no-op until E enables it, so nothing spends compute / writes the DB unbidden. The `identity_review` skill bypasses the flag via `force=True`, so manual detect/draft always works. (Superseded the earlier "default ON" draft — see §10/§11; defaulting ON would fire DB-writing detection on every rated turn before first observation, violating the propose-only / nothing-unbidden mode.)

## 6. canonical_log event kinds added (bump `KIND_VERSION`)

`identity_emergence_detected`, `identity_amendment_proposed`, `identity_amendment_applied` (V1 apply path), `identity_milestone_snapshot`. *Mechanism to confirm in `core/acatalepsy/canonical_log_kinds.py`.*

## 7. Test plan (TDD — write tests first)

- `tests/test_identity_regions.py` — region split; Origin-0 byte-stability enforcement (apply refuses to touch Origin-0); render; first-amendment creates region.
- `tests/test_identity_alignment.py` — confidentity = confidence × alignment; deterministic; Origin-0 rows untouched.
- `tests/test_identity_emergence.py` — watermark/threshold gating; divergence math; canonical_log event emitted; idempotence (no re-fire without new accrual).
- `tests/test_identity_review_skill.py` — drafting validates Origin-0 protection + mass budget + caps; files to proposals.json; no-op handling; never applies.
- `tests/test_identity_milestones.py` — budget accounting; watermark; Origin-0 hash.
- Extend `tests/test_identity_acus.py` — region-aware Origin-0 extraction stops at EMERGENT:BEGIN.

## 8. V0 scope vs deferred

- **IN (V0):** M0 alignment scorer (deterministic) + confidentity-aware identity selection; Emergent region + protection; deterministic detector + Observer surfacing + canonical_log; `identity_review` bidden drafting → proposals.json; milestone/watermark ledger + simple mass budget; tests.
- **OUT (V1+):** gated accept→auto-apply to Emergent region; embedding-based alignment; graduated mass-budget schedule; origin-N→N+1 full snapshotting/versioning; authority-transfer weighting of Emergent over Origin-0 in the prompt; changing global `retrieve()` ordering.

## 9. Open risks (to resolve in review/build)

- **Alignment signal availability** — does an embedding util exist in-repo? If not, lexical V0 (confirm quality is enough to be non-noise).
- **Where self-derived confidentity is written** — `intake.ingest_l1` doesn't set confidentity; need a safe post-intake scoring write that doesn't fight the one-writer rule or locked rows.
- **canonical_log kind registration** — exact extension mechanism + version bump.
- **Detector trigger site** — bootstrap vs idle vs explicit; must not add per-turn compute (propose-only) and must not make Observer a writer.
- **Mad Cow interaction** — self-derived identity claims are `self` provenance; ensure the emergence loop doesn't let self-provenance claims self-promote into identity without the human gate (the propose-only gate already enforces this; document it).

---

## 10. RECONCILIATION — post buildability-verification + advisor (2026-06-02)

A 6-probe adversarial verification + an advisor pass refined the plan. Final locked decisions:

**Verified holds (no change):** verbatim prompt render (do NOT touch `build_system_prompt`); one-writer rule via `authorized_write` + `GUARDED_TABLES`; `canonical_log` kind registration = add to `KNOWN_KINDS` + bump `KIND_VERSION` 5→6 in `core/acatalepsy/canonical_log_kinds.py`; Mad Cow satisfied by propose-only + `can_crystallize` + locked rows (document, no new enforcement).

**Corrected — region awareness (was a latent bug):** `extract_origin0_claims` is NOT region-aware and would lock Emergent prose as Origin-0. Build `core/identity_regions.py` (`split_regions`, `apply_emergent_amendment`), make extraction stop at `EMERGENT:BEGIN`, and **code-enforce Origin-0 protection inside `proposals.propose_amendment` for `target="identity.md"`** (reject if `current_text`/`section` lands in the Origin-0 region). Enforcement lives in code, not the mutable substrate (MonoThink doctrine).

**Decision A — identity-alignment scorer (default chosen EMPIRICALLY, not by ideology):**
- Build a pluggable seam `compute_identity_alignment(text, corpus, backend)` with **both** a deterministic lexical backend (normalized overlap coefficient over the *triple's* tokens + stopword filtering — tuned for short-triple-vs-prose, not raw token count) **and** an embedding backend (`transformers`+`torch` already in `requirements.txt`; lazy thread-safe singleton; pin model+version for reproducibility).
- **Default is decided by the fire-rate test (§11), not asserted.** Rationale for *preferring* lexical if it fires adequately: dependency/operational minimalism for a propose-only V0 + the human gate makes false positives cheap (a bad proposal costs a "reject", not identity corruption). NOTE: replay-determinism is NOT a valid rationale here — confidentity is computed read-time and not replayed (Decision B), so it never participates in replay.
- **The dominant risk is SILENCE (false negatives), not noise.** If short triples rarely overlap the prose corpus, nothing crosses threshold and the loop is inert while looking shipped. The fire-rate test is the guard.

**Decision B — confidentity write path (third option):**
- Compute confidentity = confidence × identity_alignment at READ time in the identity path (source of truth), **and** have the **identity machinery's own detector pass persist** the computed value onto self-derived rows via its own `authorized_write("identity_scoring:…")` UPDATE (mirroring the `crystallize.py` post-write pattern), skipping locked rows. **Intake's one-writer path is NOT modified.** This keeps intake pristine AND populates the `confidentity` column so E can see the weight in `get_by_id` / `monobase_dev` triage (E asked specifically whether ACUs have confidentity — it must be visible, not just internal).

**Decision C — trigger heartbeats (two):**
- (1) Feedback heartbeat: a deterministic side-hook in `core/turn_trace.record_outcome` next to the monothink hook (fires on `rating`/`thumbs_*`, outside the DB lock, try/except isolated, NO LLM). (2) Bootstrap heartbeat: a cheap watermark-throttled check at app start (`bootstrap.py`, after `ensure_origin0_acus_loaded`). Both update `core/identity_milestones.py` (watermark + `latest_emergence_signal`). Observer READS the ledger and surfaces a read-only `[OBSERVER]` line — Observer never writes. Spec note: the loop only "breathes" on these events, so apparent inactivity between them is expected.

## 11. DONE-GATE (acceptance) — empirical fire-rate test

`tests/test_identity_emergence_firerate.py`: feed a realistic set of self-derived ACU triples + the **actual Origin-0 corpus** through the scorer + detector and assert a plausible non-trivial fraction crosses the emergence threshold under the chosen backend. **M2 is not "done" because unit tests pass — it is done when this test shows the loop actually proposes.** If lexical fires near-zero, flip the embedding backend default (the seam makes this a one-line change) and re-run. This test, not the unit suite, certifies the feature does what E asked.

**Real-data calibration (2026-06-02, post-build).** Ran the detector read-only (`persist=False`) against the live `acatalepsy.sqlite3` (68 accumulated self-derived ACUs): 56/68 score nonzero (lexical is NOT sparse-dead), but at the original 0.30 threshold only **1** crossed — real triples are multi-token, so scores cluster at 0.20–0.30. Decision: keep the lexical backend (real signal, dependency-light) and **lower the default threshold 0.30 → 0.20**, surfacing ~10 genuinely identity-relevant claims (continuity, self-modeling, verification, "failure that doesn't hide"). Embedding-default stays the one-line escape hatch (`MONOLITH_IDENTITY_ALIGN_EMBED=1`) if the lexical distribution ever regresses. Also verified live: `identity_review` is discovered by `skill_registry.list_tools()` and its `{op,threshold,min_new}` args pass `tool_validation.validate_tool_arguments` — the model can actually invoke it.

---

## 12. SHIPPED (V0) — 2026-06-02

Built test-first (TDD), then adversarially reviewed (4 dimensions × verify pass);
3 confirmed findings fixed. ~45 new tests; full suite collects 1462 with zero errors.

**New modules**
- `core/identity_regions.py` — `split_regions`, `locate_snippet`, `targets_origin0`, `apply_emergent_amendment` (V1 apply helper, tested, intentionally unwired in V0).
- `core/identity_alignment.py` — `compute_identity_alignment` (pluggable: lexical default, embed opt-in via `MONOLITH_IDENTITY_ALIGN_EMBED`), `score_confidentity` = provenance_weight × alignment (self caps 0.5 = Mad-Cow ceiling).
- `core/identity_milestones.py` — watermark / milestone / `latest_emergence_signal` / origin0_hash ledger (atomic JSON).
- `core/identity_emergence.py` — deterministic `detect_emergence` (no LLM); persists confidentity onto self-derived rows via its OWN `authorized_write("identity_scoring:emergence")` (intake untouched; locked rows skipped); emits `identity_emergence_detected`.
- `skills/identity_review/` — `detect` (no LLM) + `draft` (one grounded LLM call → Emergent amendment → proposals queue; emits `identity_amendment_proposed`).

**Edits**
- `core/identity_acus.py` — `extract_origin0_claims` region-aware (stops at `EMERGENT:BEGIN`).
- `core/proposals.py` — Origin-0 guard at the shared chokepoint for `target="identity.md"` (covers the scratchpad op + every caller).
- `core/acatalepsy/canonical_log_kinds.py` — 4 kinds, `KIND_VERSION` 5→6.
- `core/turn_trace.py` (record_outcome) + `bootstrap.py` — emergence heartbeats; ship dark.
- `addons/system/observer/runtime.py` — read-only emergence line (Observer reads the ledger; never writes).
- `skills/scratchpad/SKILL.md` — example moved off Origin-0 (now system.md / Emergent), points to `identity_review`.

**Layer accounting (Rule 6).** M2 adds the "Emergent identity" layer and *subsumes* freeform `identity.md` amendment: Origin-0 sections are now frozen at the proposals chokepoint, so the scratchpad `propose_amendment` path can no longer target Origin-0 (only Emergent or system.md). `core/identity.py:append_identity()` is superseded by the Emergent amendment path (kept for back-compat, no new callers).

**Deferred to V1 (unbuilt, by design):** gated accept→auto-apply to the Emergent region (wire `apply_emergent_amendment`); embedding-default alignment (flip the flag if the fire-rate test ever regresses on real ACUs); graduated confidentity-mass budget schedule; origin-N→N+1 snapshotting (emits `identity_milestone_snapshot`); authority-transfer weighting of Emergent vs Origin-0 in the prompt; persisting confidentity into the global `retrieve()` ordering.

**How E uses it now.** Invoke the `identity_review` skill (`{"op":"detect"}` to see emerged claims, `{"op":"draft"}` to queue an Emergent amendment). To turn on the background heartbeat: set `MONOLITH_IDENTITY_EMERGENCE_V1=1`. Proposals land in `proposals.json` for manual review/apply (nothing auto-applies).
