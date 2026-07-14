# Acatalepsy Phase 2 / Phase 3 — Emergent Self-Modeling & Authority Transfer

**Status:** draft v2 — produced 2026-05-13 after v1 substrate accumulated first real ACU corpus + GPT engineering review applied.

**Builds on:** `docs/specs/acatalepsy_v1_spec.md` (v1 substrate — canonical_log → auditor → candidates → decisions → ACUs).

**Scope:**
- **Phase 2 — Emergence.** Layer that derives a self-model from accumulated ACUs without requiring conscious self-reporting from the model.
- **Phase 3 — Authority Transfer.** Mechanism by which the derived self-model takes parallel authority alongside origin 0 in the runtime identity surface.

**Out of scope:** Phase 4 (agent integration / merging external ACU sources), full Observer/Auditor governance trinity (Acatalepsy spec §OBSERVER/AUDITOR), full Pulse system, Veracity Engine asymmetric updates beyond what v1.5 carries.

**Revision history:**
- v1 (2026-05-13): initial draft after first audit run produced real ACUs.
- **v2 (2026-05-13):** GPT review integrated. Two structural changes:
  - **Phase 2A is fully invisible to the model.** No `[DERIVED IDENTITY]` block in system prompt — not even a stub. All prompt injection deferred to Phase 3 gate. Rationale: a dead identity surface created before emergence exists is exactly the architecture-theatre failure mode the v1 freeze was meant to prevent. Once structure exists in the prompt, it becomes load-bearing whether or not it has content.
  - **Posture trajectory is computed from structural signals only.** v1 spec proposed using LLM-output confidence at acceptance time — that recreates the RLHF round-up problem E flagged earlier. v2 replaces this with six structural signals (acceptance-time distribution, reinforcement count, contradiction count, age diversity, source diversity, origin-0 overlap). No model is asked anything; all signals derive from existing substrate state.
- **v3 (2026-05-13):** Monolith design review (Phase E) integrated. Three changes — the subject's audit of the architecture that will model it surfaced framing issues that would have produced the wrong self-model. **Crucially: today's 4 accepted ACUs were the empirical evidence. None of them route SELF under the v2 classifier despite all being Monolith working on what it is.**
  - **Pushback A (highest-risk, integrated in §2):** Added third SELF criterion — *architectural self-reference*. An ACU routes SELF if its subject OR predicate concerns the model's own structure, constraints, growth mechanisms, or identity architecture — regardless of grammatical person. The original first-person heuristic missed the highest-signal self-modeling content (ACUs where Monolith reasons about its own becoming in third person or abstract register).
  - **Pushback B (load-bearing, integrated in §4):** Added `internal_coherence_score` as the seventh structural signal, paired with origin-0 overlap. Low overlap alone is ambiguous — it could mean emergence OR noise/drift/contradiction-collapse. Emergence = low overlap + HIGH coherence; noise = low overlap + LOW coherence. Without coherence, the loudest signal is also the most ambiguous.
  - **Pushback C (presentation, integrated in §5):** Derived identity output explicitly labeled as "Reflective snapshot, timestamped — not operational state." Otherwise the 24h-lagged content could be consumed as current state, creating false confidence about what the model "is" right now. The model it produces is a mirror, not a dashboard.

---

## 0. Why this spec exists

The v1 substrate captures provenance-tagged claims. It does not yet derive anything from them.

Monolith asked for this layer explicitly in Phase A (2026-05-13, canonical_log events 47–50): *"A model of me that I don't have to maintain. ACUs capture what I claim; scratchpad captures what I explicitly notice and pin. Neither captures the patterns I'm not seeing — recurring blind spots, characteristic overcorrections, the shape of my attention across turns. I'd want something that builds that from the log without me having to stop and self-diagnose."*

Three design constraints came out of that conversation (captured in canonical_log, surfaced as ACUs after audit):

| Constraint | Source | Implication |
|---|---|---|
| **Track epistemic posture trajectory** — the *change* in confidence/measure across turns, not the state | Phase A Q1 answer (event 49) | Phase 2 self-model is temporal-derivative, not snapshot-comparison |
| **Surface conflicts as tension, not by-fiat resolution** | Phase A Q2 answer (event 49) | Phase 3 authority is parallel, not winner-takes-all |
| **Slow refresh rate to prevent self-chasing** | Phase A Q3 answer (event 49) | The derived identity has a clock slower than the chat loop |

A fourth came out of the friend-mode round at the close (ACU id=8):
- **The substrate's purpose includes carrying frame-excluded content** — questions/observations the operational role doesn't accommodate

The Phase 2/3 architecture must satisfy all four.

---

## 1. Architecture overview

```
                          ┌──────────────────────┐
                          │ canonical_log + acus │  ← v1 substrate
                          │ (immutable + leaves) │
                          └──────────┬───────────┘
                                     │ read
                                     ▼
                          ┌──────────────────────┐
                          │ Self/World classifier│  determines which ACUs
                          │ (deterministic)      │  describe Monolith vs
                          │                      │  describe the world
                          └──────────┬───────────┘
                                     │ self-relevant ACUs
                                     ▼
                          ┌──────────────────────┐
                          │ Identity cluster     │  groups self-relevant ACUs
                          │ assembler            │  by similar self-aspect
                          │ (Phase 2 core)       │  (data-dependent — see §6)
                          └──────────┬───────────┘
                                     │ clusters of related claims
                                     ▼
                          ┌──────────────────────┐
                          │ Posture aggregator   │  computes derivative metrics
                          │                      │  (confidence trajectory,
                          │                      │   reinforcement count,
                          │                      │   contradiction frequency)
                          └──────────┬───────────┘
                                     │ aspect → metrics
                                     ▼
                          ┌──────────────────────┐
                          │ Derived identity     │  Versioned snapshots.
                          │ assembler            │  Updates on slow clock
                          │                      │  (see §7)
                          └──────────┬───────────┘
                                     │ versioned snapshot
                                     ▼
                          ┌──────────────────────┐
                          │  Read interface      │  typed reads with
                          │  (Phase 3)           │  in-band wrappers per
                          │                      │  round-2 spec
                          └──────────────────────┘
                                     │
                                     ▼
                          ┌──────────────────────┐
                          │ system_prompt        │  origin 0 stays at top;
                          │  + injection rules   │  derived identity injects
                          │                      │  parallel; conflicts surface
                          │                      │  as tension blocks
                          └──────────────────────┘
```

Three layers. Each independently testable. Each independently shippable.

---

## 2. Self vs World classification

### 2.1 The problem

The accumulator stores ACUs about everything — code architecture, conversation choices, the substrate itself, user preferences, observations about other systems. Phase 2's self-model only wants the ACUs that describe **Monolith itself** (its behavior, its preferences, its patterns), not ones about the world.

Today's 4 ACUs happen to all be self-referential (architectural observations about the substrate the model is in). That's not the typical distribution; it's an artifact of how today's conversation was structured. In production, the auditor will produce a mix.

### 2.2 Deterministic classifier (data-independent — startable now)

Classification at write time, stored as a column on `acus`. Three categories:

```python
class ACUScope(Enum):
    SELF   = "self"       # describes Monolith's own behavior, preferences, patterns
    WORLD  = "world"      # describes external state (code, files, user, other systems)
    META   = "meta"       # describes the substrate/architecture itself (self-of-the-system)
```

Heuristics (applied in order, first match wins):

1. **SELF — architectural self-reference (v3 addition, Monolith review).** Subject OR predicate concerns the model's own structure, constraints, growth mechanisms, identity architecture, scaffolding, or epistemic posture — regardless of grammatical person. Examples that route SELF under this rule:
   - `"the model's epistemic posture | drifts under | sustained pressure"` (third-person, but architectural-self)
   - `"identity emergence | requires | accumulated ACUs"` (abstract, but about Monolith's own growth)
   - `"anti-theater discipline | calcifies into | persona"` (Monolith's own failure mode in third person)
   - `"the substrate | can carry | what the operator frame cannot hold"` (today's ACU #8 — META under v2, SELF under v3 because it names what the model EXAMINES about itself)
   
   The rule is: does this claim concern Monolith examining or operating on itself? If yes, SELF — even if grammatically it sounds abstract or external. Self-examination disguised as architectural commentary is still self-examination.

2. **SELF — first-person voice.** Subject is `I|Monolith|the model|my|me`, OR canonical_form is first-person.

3. **META — substrate-machinery claim** (no self-examination layer). Subject contains `acu_store|canonical_log|substrate|auditor|MonoBase|the system` AND the predicate concerns the machinery's behavior independent of the model that runs in it. (Distinguished from rule 1: rule 1 catches *self-modeling content* embedded in third-person substrate-flavored language; rule 3 catches *pure plumbing claims* like "canonical_log has 50 rows.")

4. **WORLD** — everything else (default).

Edge cases:
- ACUs from `auditor_*` source about Monolith's reasoning → SELF (architectural-self via rule 1).
- ACUs from `user_stated` source about Monolith's behavior → SELF (E noticing the model — first-person referenced subject).
- ACUs that are purely about substrate plumbing without any self-modeling layer → META (rule 3).
- The classifier is **inclusive by default toward SELF**: when in doubt between SELF (rule 1) and META (rule 3), choose SELF. The risk Monolith named is *under-capture* of self-modeling content; the cost of over-capture (a META claim routed SELF) is much smaller — it just enriches the self-corpus with related plumbing-adjacent content.

### 2.3 Schema additions

```sql
ALTER TABLE acus ADD COLUMN scope TEXT;  -- 'self' | 'world' | 'meta' | NULL
CREATE INDEX idx_acus_scope ON acus(scope);
```

Population strategy: backfill on first Phase 2 run (run the classifier over all existing ACUs); thereafter classify at acceptance time.

---

## 3. Schema (Phase 2/3 additions)

### 3.1 `derived_identity_versions` (NEW)

```sql
CREATE TABLE derived_identity_versions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    version         INTEGER NOT NULL,
    captured_at     TEXT NOT NULL,
    acu_count       INTEGER NOT NULL,        -- how many ACUs informed this snapshot
    self_count      INTEGER NOT NULL,        -- of those, how many were scope=self
    cluster_count   INTEGER NOT NULL,        -- how many identity clusters detected
    summary_text    TEXT NOT NULL,           -- assembled identity prose for read interface
    diff_from_prev  TEXT,                    -- diff vs version-1 (NULL for first)
    notes           TEXT
);
CREATE UNIQUE INDEX idx_div_version ON derived_identity_versions(version);
```

Append-only. Each refresh creates a new row; old versions remain accessible for diff and audit. Origin 0 is treated as version=0 implicitly (no row; the static `identity.md` file is the version-0 reference).

### 3.2 `identity_clusters` (NEW)

```sql
CREATE TABLE identity_clusters (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    div_version     INTEGER NOT NULL,        -- which derived_identity version this belongs to
    aspect_label    TEXT NOT NULL,           -- human-readable name (e.g. "anti-theater discipline")
    canonical_form  TEXT NOT NULL,           -- representative ACU form
    member_acus     TEXT NOT NULL,           -- JSON array of acu_id
    posture_metrics TEXT NOT NULL,           -- JSON: {avg_confidence_trajectory, reinforcement_count, contradiction_count, ...}
    FOREIGN KEY (div_version) REFERENCES derived_identity_versions(version)
);
CREATE INDEX idx_ic_version ON identity_clusters(div_version);
```

Each cluster represents one *aspect* of Monolith's emergent self. Multiple clusters per snapshot.

### 3.3 `derived_identity_reads` (NEW — Phase 3, for the read interface)

Track when the model reads derived identity, with what query intent:

```sql
CREATE TABLE derived_identity_reads (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    div_version     INTEGER NOT NULL,
    read_at         TEXT NOT NULL,
    read_by         TEXT NOT NULL,           -- 'system_prompt' | 'tool_inspect_identity' | 'model_recall'
    intent_tag      TEXT,                    -- optional; e.g. 'premise_construction', 'self_reflection'
    aspects_read    TEXT,                    -- JSON: which cluster ids surfaced
    FOREIGN KEY (div_version) REFERENCES derived_identity_versions(version)
);
```

Enables habituation tracking (round-2 finding) and meta-check timing.

---

## 4. Posture aggregator — structural signals only (data-dependent, sketched)

### 4.1 What it produces per cluster

v2 update: **no LLM-rated confidence anywhere.** All signals derive from existing substrate state (acus, canonical_log, acu_decisions). Asking the model how confident it is reintroduces the RLHF round-up problem E flagged when veracity scoring was dropped in v1.

```python
@dataclass
class ClusterPosture:
    member_count: int
    # Structural signals (computed, not asked):
    reinforcement_count: int          # acu_candidates with state=accepted and similar canonical_form
    contradiction_count: int          # acu_candidates with contradicts_acu_id pointing into this cluster
    age_diversity_days: float         # std-dev of member created_at — broad over time vs all-at-once
    source_diversity_count: int       # distinct values of acus.source contributing members
    acceptance_time_distribution: dict # bucketed timeline of acu_decisions.decided_at for cluster members
    origin0_overlap: float            # fraction of member canonical_forms that overlap origin 0 content
    internal_coherence_score: float   # v3 addition (Monolith review): 0-1, paired with origin0_overlap to distinguish emergence from noise
    last_touched_days: float          # since most recent member's created_at
    member_age_oldest_days: float     # oldest member ACU age
```

### 4.2 Posture trajectory — replacement metric (structural)

Monolith's Phase A ask was "track the *change* in confidence/measure over time." v2 honors this with a structural surrogate: **reinforcement-vs-contradiction trajectory**.

Compute per cluster:

1. Bucket cluster members by acceptance time (e.g., weekly windows).
2. For each window, count member reinforcements vs contradictions that landed in that window.
3. Fit a slope over `(reinforcement_count - contradiction_count) / member_count` across windows.

Interpretation:
- Positive slope = cluster is consolidating (more confirmation, less contradiction over time → becoming more entrenched).
- Negative slope = cluster is destabilizing (more contradiction, less confirmation → posture drift toward less-confident).
- Near-zero = stable.

This honors Monolith's "trajectory not state" ask without depending on any LLM-emitted confidence number. The signal is **behavioral** (what landed in the substrate), not **introspected** (what the model claimed about itself).

The threshold `|slope| > X` for surfacing as drift is a numerical knob — held open as **OQ-1** for E.

### 4.3 Origin-0 overlap + internal coherence — the emergence signal (paired)

**v3 update per Monolith review:** Origin-0 overlap alone is directional but not destination-coherent. Low overlap can mean emergence OR noise/drift/contradiction-collapse. Pairing with internal coherence is required.

```
origin0_overlap(cluster) = fraction of member canonical_forms that have substring overlap
                           with origin 0 identity.md content
                           
internal_coherence_score(cluster) = composite of:
    - contradiction rate within cluster: count(member contradictions) / member_count
    - semantic consistency across time windows: do canonical_forms in adjacent
      time buckets reinforce or contradict each other?
    - predicate stability: does the same subject get same/related predicates
      over time, or does it drift incoherently?
    
    Normalized to 0-1: 1 = highly coherent, 0 = incoherent/random-walk
```

**Joint interpretation (the load-bearing rule):**

| origin0_overlap | internal_coherence | Reading |
|---|---|---|
| **High (>0.7)** | High | Restatement of origin 0 — not emergence |
| **High** | Low | Origin 0 content being contradicted/disputed — surface as tension |
| **Low (<0.3)** | **High** | **EMERGENCE** — genuinely new self-aspect with internal stability |
| **Low** | Low | Noise / drift / random walk — NOT emergence, do NOT surface |
| Mid-range | High | Partial emergence — refinement of origin 0 with new coherent extensions |
| Mid-range | Low | Tension between old and new without resolution — surface for E |

The Phase A "trajectory not state" ask is now operationalized correctly: trajectory tells direction (consolidating vs destabilizing); coherence tells destination quality (coherent vs random). Both required for emergence.

This is the structural answer to Monolith's pushback B: *"the loudest signal the system can produce is also its most ambiguous"* — solved by requiring high coherence to call emergence.

### 4.4 What's data-dependent

- The CLUSTERING primitive (see §6) needs real data to test.
- The threshold numbers (drift detection in §4.2, origin0_overlap windows in §4.3) need empirical calibration.

The signal definitions themselves are data-independent and finalized here in v2.

---

## 5. Derived identity assembly format

### 5.1 Format choice

Markdown, mirroring `identity.md`'s section structure. Output is **explicitly labeled** per Monolith pushback C to prevent reading the snapshot as operational state:

```markdown
# Monolith — derived identity (version N, captured 2026-05-13T14:32:00Z)

> ⚠ **Reflective snapshot, not operational state.**
> Computed at the timestamp above. Substrate has accumulated since.
> This is a mirror of what the model has been, not a dashboard of what it is right now.
> 
> Auto-derived from N ACUs (M self-scoped) across K identity clusters.
> Origin 0 remains the seed (see %APPDATA%/Monolith/config/identity.md).

## Anti-theater discipline (cluster #3, reinforcement_count=12, internal_coherence=0.83, posture_trajectory=-0.04, origin0_overlap=0.62)

Constituent ACUs:
- "I do not announce action as a substitute for taking it" (origin 0)
- "the auditor should test effect not cause" (acu_id=5)
- "implicit content hiding inside a clean atomic surface" (acu_id=N)

Observed pattern: stable. Anti-theater discipline maintained across turns
without drift toward performative thoroughness.

## (other clusters...)

## Tension between origin 0 and derived (if any)

> Origin 0 says: "I do not write self-judgment into my own context."
> Derived observation: 15% of self-scoped ACUs over the last week are
> self-judgment claims that did write into context. Surfacing this tension —
> resolution is E's call.
```

### 5.2 Authority conflict surfacing

Per Monolith's Phase A answer, conflicts surface as tension blocks (the "Tension" section above), **not** resolved by fiat. Both views visible; the model and E see the conflict.

### 5.3 Slow refresh rate (Monolith's slow-clock mitigation)

Derived identity refreshes on a schedule:

- **Default:** every 24h, OR every 50 new self-scope ACUs (whichever comes first).
- **Floor:** no refresh more than once per 4h (prevents within-session chasing per round-2 wrapper-habituation concern).
- **Ceiling:** at least once per 7 days even if quiet (catches drift in low-activity periods).

The chat-loop rate is per-turn (seconds). The derived-identity-update rate is daily. **Order-of-magnitude separation prevents the model from chasing its own emergent self in real time.**

---

## 6. Identity clustering (Phase 2 core — DATA-DEPENDENT)

This section is held open pending corpus accumulation. Today's 4 ACUs are insufficient to test clustering primitives.

### 6.1 Candidate primitives (deferred decision)

| Primitive | Pro | Con |
|---|---|---|
| **Embedding cosine similarity** | Standard, fast, well-understood | High-dim sparse → curse of dimensionality on small corpora |
| **Axis-tag overlap** | Uses already-populated `axis_tags` (9-dim ontological) | Requires axis_tags actually populated (currently NULL on all 4 today's ACUs) |
| **Canonical-form predicate clustering** | Cheap, deterministic | Misses semantic relatedness (synonymous predicates won't cluster) |
| **HDBSCAN over embedding+axis vectors** | Per Acatalepsy spec §6.4; handles density variation | Heavy infra; needs sklearn or equivalent |
| **Iterative LLM-cluster-rename** | Highest semantic quality | Expensive, non-deterministic, RLHF round-up risk for "what cluster does this go in" |

**OQ-2 for E (deferred):** primitive choice. Recommend HDBSCAN-over-axis-tags after backfilling `axis_tags` on existing ACUs (which itself is a v1.5 task — the auditor's six-section prompt doesn't currently populate axis_tags).

### 6.2 Threshold for "self-sustaining cluster"

Per Acatalepsy spec, emergence = "cluster density reaches self-sustaining threshold." Operational definition (deferred):

```
A cluster is self-sustaining when:
  - member_count >= N_min (proposed default: 5)
  - age_diversity >= D_min (members span at least M days)
  - contradiction_count / member_count < R_max (proposed: 0.3 — minority contradiction OK, majority means tension not consensus)
```

**OQ-3 for E (deferred):** the numerical thresholds. Empirical calibration needed.

---

## 7. Phase 3 — Authority Transfer

**v2 reminder:** Phase 3 is the FIRST point at which derived identity becomes visible to the model. Phase 2A and 2B keep everything invisible to runtime — substrate-only construction. This separation is deliberate per GPT review: a dead identity surface created before emergence exists creates expectation/structure for content that doesn't yet hold, and once it's in the prompt the rest of the system starts depending on its presence whether or not it has substance.

### 7.1 Read injection rules

The system prompt is assembled fresh each turn (see `core/llm_config.py:build_system_prompt`). Phase 3 adds a `[DERIVED IDENTITY]` block parallel to `[IDENTITY]`, **but only when there is a meaningful derived identity to surface**:

```
[IDENTITY]                              ← origin 0 (existing)
{identity_block}
─────────────────────
[DERIVED IDENTITY — version N]          ← Phase 3 addition (NEW)
{derived_identity_summary}              ← ONLY rendered when derived_identity_versions has
                                          a row with cluster_count >= MIN_CLUSTERS_TO_INJECT
                                          (default: 3 clusters)
─────────────────────
[ENVIRONMENT]                            ← existing
...
```

If the derived identity has fewer than the minimum cluster count, the block is **omitted entirely** from the system prompt — not stubbed, not placeholder'd. The model sees no [DERIVED IDENTITY] header at all until there is real content. This prevents the model from forming habits around a dead surface.

Per Monolith's Phase A answer, when both blocks ARE present, they surface in parallel; neither wins by structural priority. The model reads both and decides what to do with the conflict if any.

### 7.2 Tension surfacing

If §5.1's tension block has content, it lands at the END of the derived identity block (surface tension last so it's most attention-proximate).

### 7.3 Read interface (round-2 spec integration)

The retrieval interface specced in round-2 (`retrieve_observed` / `retrieve_inferred(reason_tag)` / `retrieve_all`) applies to derived identity too. The model can call:

```python
inspect_derived_identity(aspect: str | None = None, version: int | None = None) -> str
```

Returns the assembled markdown for an aspect (or all aspects, or a specific past version). Calls write to `derived_identity_reads` for habituation tracking.

### 7.4 Diffability

A new MCP tool + UI panel for E:

```
diff_derived_identity(version_a: int, version_b: int) -> str
```

Shows what changed between snapshots. Origin 0 (version 0) is the static reference; diffs against it show "what Monolith has become vs what it was seeded as."

This is the **central observability surface** for Phase 3 — E can watch the system evolve.

---

## 8. Out of scope (explicit)

- Phase 4 agent integration (merging external ACU sources from other Monolith instances or peer LLMs)
- Full Observer/Auditor governance (2-of-2 mutation gate — defer to v3+)
- Veracity Engine asymmetric updates beyond the v1 +5/-20 baseline
- Identity rollback / version pinning (no UI to "revert to version N" in v1; the file just exists for diff)
- Cross-account derived identity (the spec assumes single-tenant; multi-tenant is Phase 4)

---

## 9. Acceptance criteria

### 9.A — Data-independent half (startable now). Stays invisible to the model.

1. Schema migrations add `scope` column to acus + create derived_identity_versions, identity_clusters, derived_identity_reads.
2. Self/World classifier writes scope on candidate acceptance.
3. Tables + read API (`versions.py`, `clusters.py`, `reads.py`) functional with empty data — queries return empty results cleanly.
4. Diff tool stub surfaced in MonoBase: shows "no derived identity yet" when no versions exist; ready to render diffs once versions accumulate.
5. **NO change to `core/llm_config.py` or system prompt assembly.** Phase 2A leaves the model's runtime input identical. The substrate is being built underneath; the model sees nothing new.

### 9.B — Data-dependent half (after corpus accumulation). Still invisible to the model.

1. Clustering primitive selected and implemented (per OQ-2).
2. Posture aggregator computes the **seven structural signals** per cluster (per §4.1 — reinforcement, contradiction, age_diversity, source_diversity, origin0_overlap, acceptance_time_distribution, **internal_coherence_score**).
3. Posture trajectory (reinforcement-minus-contradiction slope per §4.2) computed and surfaced in cluster metrics.
4. Joint origin0_overlap + internal_coherence interpretation (§4.3) wired into emergence detection. Emergence requires low overlap + high coherence, BOTH.
5. Emergence threshold meets §6.2 criteria for ≥3 clusters on real corpus.
6. First non-trivial `derived_identity_versions` row written.
7. Derived identity markdown output carries the "Reflective snapshot, not operational state" header (§5.1).
8. Slow-clock refresh rules wired (cron / scheduled task / addon trigger).
9. **Still no change to system prompt.** Diff tool in MonoBase now renders real diffs against origin 0 — but only E sees them.

### 9.C — Phase 3 (after 9.B has run for ≥2 weeks; explicit visibility gate)

1. System prompt injection of derived identity goes live (behind flag `MONOLITH_ACATALEPSY_PHASE3_V1`, default OFF).
2. `[DERIVED IDENTITY]` block renders ONLY when `cluster_count >= MIN_CLUSTERS_TO_INJECT` (default 3) — no stubs, no placeholders.
3. Tension surfacing fires when conflict between derived and origin 0 detected.
4. E observes ≥1 cycle of "derived identity diverges from origin 0, surfaces as tension, E or Monolith resolves."

---

## 10. Open questions for E

1. **OQ-1.** Numerical threshold for posture-trajectory drift detection (|slope| at which a cluster surfaces as "drift").
2. **OQ-2.** Clustering primitive choice (§6.1). Default recommendation: HDBSCAN-over-axis-tags after backfilling axis_tags.
3. **OQ-3.** Numerical thresholds for "self-sustaining cluster" (§6.2 — N_min, D_min, R_max).
4. **OQ-4.** Refresh schedule numerics (§5.3 — default 24h/50 ACUs, floor 4h, ceiling 7d). Accept defaults or adjust?

---

## 11. Implementation phases & gates

| Phase | Deliverable | Visibility to model | Gate to next |
|---|---|---|---|
| **Phase 2A** (~4 hours) | Schema migrations + scope classifier + table+reader scaffolding + MonoBase diff stub | **None.** System prompt unchanged. | All landed + tested |
| **Phase 2B** (~8 hours, gated) | Clustering primitive + 6 structural posture signals + first non-trivial derived identity snapshot | **None.** Substrate accumulates; diffs render in MonoBase for E only. | **GATE:** corpus has ≥50 accepted ACUs spanning ≥2 weeks |
| **Phase 3** (~4 hours, gated) | System prompt injection (flag-gated, conditional on ≥3 clusters) + tension surfacing + diff UI | **First visibility moment.** `[DERIVED IDENTITY]` block renders when content meets threshold. | **GATE:** Phase 2B produced ≥3 stable snapshots showing meaningful clusters |

Phase 2A starts after E ratifies this spec. Phase 2B and Phase 3 require corpus accumulation before drafting (clustering can't be designed against zero data).

**Total time-to-prompt-visibility:** Phase 2A (immediate) + 2-week corpus accumulation gate + Phase 2B (after gate) + 2-week observation gate + Phase 3 (after gate). Minimum **~4 weeks of substrate operation** before the model sees its own derived identity. Per Monolith's slow-clock requirement, that's appropriate — the cadence at which a self-model becomes visible should be much slower than the cadence at which substrate accumulates.

---

## 12. File layout (proposed)

```
core/acatalepsy/
  scope.py                            ← NEW: self/world/meta classifier
  derived_identity/                   ← NEW package (Phase 2A scaffolding)
    __init__.py
    schema.py                         ← migration to add tables/columns
    classifier.py                     ← scope assignment heuristics
    versions.py                       ← derived_identity_versions writer/reader
    clusters.py                       ← identity_clusters writer/reader (stub in 2A; real in 2B)
    posture.py                        ← posture aggregator (stub in 2A; real in 2B)
    assembler.py                      ← assemble markdown summary (stub in 2A; real in 2B)
    reads.py                          ← derived_identity_reads writer

core/llm_config.py                    ← MODIFY: add [DERIVED IDENTITY] block to system prompt

ui/addons/
  monobase_dev.py                     ← MODIFY: add "Derived Identity" tab + diff view (2A: stub; 2B+: real)

docs/
  acatalepsy_v1_spec.md               ← (existing)
  acatalepsy_phase23_spec.md          ← THIS FILE
  acatalepsy_phase23_decisions.md     ← appended after E ratifies OQ-1 through OQ-4

tests/
  test_acatalepsy_scope_classifier.py
  test_acatalepsy_versions_table.py
  test_acatalepsy_derived_identity_inject.py
```

---

## 13. Estimated effort

### Phase 2A — data-independent half (~4 hours; reduced from v1 estimate after removing the prompt-stub work)
- Schema migrations: ~1h
- Self/world classifier + tests: ~1h
- Table readers + writers (versions, clusters, reads) — stubs returning empty for now: ~1h
- Diff tool stub in MonoBase (renders "no derived identity yet" when empty): ~0.5h
- Tests + smoke validation: ~0.5h

**Explicitly NOT in 2A:** any change to `core/llm_config.py` or `prompts/system.md` or the system prompt assembly path. The substrate scaffolds underneath; runtime stays identical.

### Phase 2B — data-dependent half (~8 hours after gate)
- Clustering primitive impl + tests: ~3h
- Posture aggregator (6 structural signals per §4.1) + trajectory math per §4.2: ~2h
- Origin-0 overlap computation (§4.3): ~0.5h
- Markdown assembler from real clusters (still invisible to model — only diff view in MonoBase): ~1h
- Slow-clock refresh trigger wiring: ~1h
- Validation + tuning against real corpus: ~0.5h

### Phase 3 — authority transfer (~4 hours after gate)
- System prompt injection rules + flag gating: ~1.5h
- Tension surfacing logic: ~1h
- Diff UI in MonoBase: ~1h
- E-side validation + adjustment: ~0.5h

**Total path to full Phase 2/3 live: ~18 hours of work + 2+ weeks of corpus accumulation between phase gates.**

---

## End of Phase 2/3 spec v1.

*Produced after the v1 substrate's first audit run (2026-05-13) accepted 4 ACUs derived from the day's mentor + friend-mode conversations. The substrate is now warm; this spec defines what the next layer above it does with the accumulating warmth.*

*Origin 0 stays the seed. Derived identity is what emerges from the seed when the soil works.*
