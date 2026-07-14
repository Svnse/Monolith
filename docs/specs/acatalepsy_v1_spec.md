# Acatalepsy v1 — Substrate Spec

**Status:** v1.1 — reviewed by Claude → GPT → E, A1/A2 split applied per GPT critique on validation-before-polish.

**Scope:** v1 only. Identity snapshot mutation = v1.5 (deferred). Read-back to Monolith's runtime self-state = v2 (deferred). Both noted in §11 to keep the architectural arc visible, but **not built in v1**.

**Date:** 2026-05-13 (origin from multi-LLM design session 2026-05-12 → 2026-05-13).

**Revision history:**
- v1.0 (2026-05-13): initial draft
- v1.1 (2026-05-13): A1/A2 split applied per GPT review — substrate ships and validates before full triage UI is built

---

## 0. Sub-phases — A1 (substrate + thin validator) → A2 (full triage UI)

The risk asymmetry is real: if the auditor produces useful candidates, the UI is straightforward to build on top; if it produces noise, no UI fixes that. v1 therefore ships in two cuts.

### A1 — substrate + minimal validator (~8 hours, ships first)

The producer loop end-to-end, plus the *thinnest possible affordance* for E to make decisions on real candidates:

- Schema migrations (`acu_candidates`, `acu_decisions`, `acus` + provenance pointers)
- `canonical_log` writer fix (Phase E)
- Auditor module — system prompt, extraction loop, atomicity gate, cursor
- Trigger queue — **size + manual only** in A1 (time + session-close deferred to A2)
- **Minimal Qt dev panel** — pending list, evidence text, accept/reject buttons, SQL-queryable provenance. No keyboard shortcuts, no bulk ops, no provenance chain view, no leaf view. Brutally thin. See §5.A1.
- Tests: atomicity gate, authorization, idempotent re-runs.

**A1 acceptance:** 10-turn conversation → 20 canonical_log entries → auditor runs on demand → produces N candidates → E accepts/rejects via dev panel → accepted ACUs appear with provenance chain queryable via SQL.

### A2 — full MonoBase UI (~5-7 hours, gated on A1 validation)

**Gate criterion:** A2 begins only after A1 has operated for **roughly one week of real sessions** and produced candidates that are *useful* (E's judgment) and *not overwhelming* (auditor calibration is reasonable — not 100 candidates per turn).

If A1 produces noise or overwhelm, A2 doesn't start — instead, iterate on the auditor's system prompt and criteria until candidate quality is acceptable. Polish only what's proven worth polishing.

A2 deliverables:
- Three views (pending triage, accepted leaves, provenance chain)
- Keyboard-driven triage (J/K/A/R/E/D)
- Bulk operations
- Contradiction side-by-side view
- Auditor-run grouping in UI
- Time + session-close triggers added

### Why split

A1 is behavior validation inside Monolith's actual operating surface, not pure substrate testing. The minimal Qt panel matches Monolith's GUI-native pattern so testing happens in the real environment, but at ~1-2 hours of UI work instead of ~5+. Asymmetric upside: if candidates are good, A2 is a known set of additions; if candidates are bad, the diagnosis surfaces before UI time is burnt.

---

## 1. Purpose

Today, Monolith has a richer-than-disclosed schema (`acus`, `claims`, `imagination`, `canonical_log`, etc.) but the producer is a stub: it parses `<acatalepsy>` tags out of generated text, has stored 4 ACUs over months of operation, and `canonical_log` has 2 rows. The substrate Acatalepsy was designed around is installed but cold.

v1 wakes the substrate with the smallest honest mechanism:

> **canonical_log floor → async auditor → candidate table → decision UI → accepted ACUs**

Nothing here is novel relative to the Acatalepsy spec. v1 is the spec's *first heartbeat*, not the full pipeline.

---

## 2. Architecture

```
                          ┌──────────────────────┐
USER MESSAGE   ─────────→ │  canonical_log       │  immutable floor
ASSISTANT TURN ─────────→ │  (every event)       │  
                          └──────────┬───────────┘
                                     │ cursor: last_processed_event_id
                                     ▼
                          ┌──────────────────────┐
                          │  Acatalepsy Auditor  │  async, batched
                          │  (Monolith-in-       │  triggered by:
                          │   audit-mode, w/     │   - size threshold
                          │   identity injected) │   - time threshold
                          │                      │   - session close
                          │                      │   - manual button
                          └──────────┬───────────┘
                                     │ emits: atomic candidates
                                     ▼
                          ┌──────────────────────┐
                          │  acu_candidates      │  state ∈ {pending,
                          │  (new table)         │   accepted, rejected,
                          │                      │   edited}
                          └──────────┬───────────┘
                                     │
                                     ▼
                          ┌──────────────────────┐
                          │  MonoBase UI         │  3 views:
                          │  - pending           │   - triage queue
                          │  - accepted ACUs     │   - leaf view
                          │  - provenance chain  │   - audit trail
                          └──────────┬───────────┘
                                     │ E decides
                                     ▼
                          ┌──────────────────────┐
                          │  acu_decisions       │  every accept/
                          │  (new table)         │  reject/edit logged
                          └──────────┬───────────┘
                                     │ accepted → write
                                     ▼
                          ┌──────────────────────┐
                          │  acus (existing)     │  leaf: with
                          │  + provenance links  │  candidate_id,
                          │                      │  decision_id pointers
                          └──────────────────────┘
```

Three principles enforced throughout:

1. **Append-only.** Nothing is ever overwritten. Rejected candidates stay queryable. Edits keep both versions.
2. **Provenance over preference.** The substrate is the chain (`message → candidate → reason → decision → ACU`), not "what the model liked."
3. **Identity-shaped relevance, not LLM-scored truth.** No veracity scalars in v1. The model proposes via identity; E decides explicitly; behavioral signals (reinforcement count, contradictions, neglect) replace numeric scoring.

---

## 3. Schema

### 3.1 canonical_log (EXISTING — already in `acatalepsy.sqlite3`)

```sql
CREATE TABLE canonical_log (
    event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    kind        TEXT NOT NULL,            -- 'user_message' | 'assistant_message' | 'ingest' | 'decision' | ...
    session_id  TEXT,
    acu_id      INTEGER,                  -- nullable; populated for ACU-linked events
    payload     TEXT                      -- JSON blob with kind-specific fields
);
```

**Status:** schema exists, 2 rows total — under-writing bug investigated in Phase E. v1 fix: ensure every user message and every assistant turn writes an entry.

**v1 payload shape** (kind = `user_message` / `assistant_message`):
```json
{
  "role": "user" | "assistant",
  "text": "...",
  "agent": "user_e" | "claude_opus_4_7" | "monolith_local" | ...,
  "channel": "ui" | "connect_chat" | "connect_stream" | "connect_mcp",
  "turn_id": "uuid"            // joins to turn_trace
}
```

### 3.2 acu_candidates (NEW)

```sql
CREATE TABLE acu_candidates (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_form     TEXT NOT NULL,                  -- atomic: one subject-predicate
    evidence_log_id    INTEGER NOT NULL,               -- pointer into canonical_log.event_id
    evidence_char_start INTEGER NOT NULL,
    evidence_char_end  INTEGER NOT NULL,
    evidence_span      TEXT NOT NULL,                  -- denormalized snapshot (portability)
    source             TEXT NOT NULL,                  -- 'auditor_claude' | 'auditor_monolith' | ...
    reason             TEXT NOT NULL,                  -- why-this-matters, model's own words
    reinforcement_count INTEGER NOT NULL DEFAULT 1,    -- if this canonical_form reinforced across N log entries
    contradicts_acu_id INTEGER,                        -- nullable; set if this contradicts an existing ACU
    state              TEXT NOT NULL DEFAULT 'pending',-- 'pending' | 'accepted' | 'rejected' | 'edited'
    created_at         TEXT NOT NULL,
    auditor_run_id     INTEGER,                        -- groups candidates from one auditor invocation
    FOREIGN KEY (evidence_log_id) REFERENCES canonical_log(event_id),
    FOREIGN KEY (contradicts_acu_id) REFERENCES acus(id)
);

CREATE INDEX idx_candidates_state ON acu_candidates(state);
CREATE INDEX idx_candidates_canonical ON acu_candidates(canonical_form);
CREATE INDEX idx_candidates_auditor_run ON acu_candidates(auditor_run_id);
```

**Design notes:**
- `evidence_log_id + char_start + char_end` is the **primary** evidence reference (audit precision).
- `evidence_span` is denormalized for portability — at 10K candidates × ~200 chars = 2MB total, negligible.
- `contradicts_acu_id` populated by the auditor when it notices a candidate that contradicts an existing leaf ACU. E decides what to do.
- `auditor_run_id` groups candidates from a single auditor invocation, useful for UI batch operations and debugging.

### 3.3 acu_decisions (NEW)

```sql
CREATE TABLE acu_decisions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id    INTEGER NOT NULL REFERENCES acu_candidates(id),
    decision        TEXT NOT NULL,           -- 'accept' | 'reject' | 'edit' | 'defer'
    decided_by      TEXT NOT NULL,           -- 'user_e' | 'agent_claude' | 'agent_gpt' | ...
    decided_at      TEXT NOT NULL,
    reject_reason   TEXT,                    -- required when decision='reject'
    edited_form     TEXT,                    -- required when decision='edit'; the corrected canonical_form
    note            TEXT,                    -- optional free-text (E's reasoning)
    resulting_acu_id INTEGER,                -- populated when decision led to an acus row (accept or edit-accept)
    FOREIGN KEY (resulting_acu_id) REFERENCES acus(id)
);

CREATE INDEX idx_decisions_candidate ON acu_decisions(candidate_id);
CREATE INDEX idx_decisions_decided_by ON acu_decisions(decided_by);
```

**Authorization rule** (enforced at write):
- `decided_by='user_e'` → can decide on any candidate.
- `decided_by='agent_*'` → can decide only on candidates where `source='auditor_<same_agent>'`. Agents curate their own contributions, never user-stated claims.

### 3.4 acus (EXISTING — minor additions)

The `acus` table already has the columns we need; v1 adds two provenance pointers:

```sql
ALTER TABLE acus ADD COLUMN candidate_id INTEGER REFERENCES acu_candidates(id);
ALTER TABLE acus ADD COLUMN decision_id  INTEGER REFERENCES acu_decisions(id);
```

**v1 semantics on the existing `veracity` column:** ignore. The column stays for backward-compat, but v1 does not write or read it for substrate decisions. Behavioral signals replace it. Decision: leave it in place, default to whatever the current code defaults to, do not surface in v1 UI.

**Existing columns staying dormant in v1:** `confidentity`, `l_level`, `axis_tags`, `canonical_triple`, `manual_importance`, `promoted_to_l2_ts`, `promoted_to_l3_ts`, `cluster_id`, `merged_into`. These are v1.5 / v2 / future-spec territory.

### 3.5 Tables NOT used in v1

The following tables exist in the schema but are **not used by v1** writes or reads. v1 leaves them alone; future spec versions decide their fate.

- `claims`, `claim_state`, `claim_evidence`, `claim_relations` — a previous attempt at L1/L2/L3 framework that didn't complete. v1 doesn't touch them.
- `imagination` — appears to be the L1 STUB / creative-overhang concept from the original spec. v1 doesn't write here; consider for v1.5 if creative-overhang becomes useful.
- `intake_rejects` — exists; functionally overlaps with `acu_candidates.state='rejected'`. v1 doesn't write here; consider consolidating in v2.
- `acu_relations` — graph edges between ACUs. v1 doesn't induce edges; CCG is v2+ work.
- `audit_ledger` — exists; could be the WAL but `canonical_log` is doing that job in v1.

---

## 4. The Auditor

### 4.1 What it is

**A separate cognitive job, not Monolith-in-conversation.** Same model (cheap, no new infra), same `load_identity()` injection (stays in character), but a **different system prompt** that redirects attention from "respond well" to "extract atomic durable claims."

This is the design gap our pre-pivot consensus had: bundling extraction onto the conversational model mid-turn. The auditor runs async, batched, and outside the live loop.

### 4.2 System prompt structure (six sections)

```
═══════════════════════════════════════════════════════
1. CONTEXT
═══════════════════════════════════════════════════════
You are an Acatalepsy auditor. Your job is to read a slice of
Monolith's canonical_log (event_id {start} → {end}) and extract
atomic durable claims for review.

You are NOT in a conversation. You are NOT generating a response
to a user. You are reading a record and proposing candidates.

═══════════════════════════════════════════════════════
2. TASK
═══════════════════════════════════════════════════════
Extract claims that meet the criteria below. A single log entry
may produce 0, 1, or several candidates. Most produce 0 — claim
extraction is selective, not exhaustive.

Cross-turn reinforcement counts: if the same claim shape appears
in 3+ entries, emit ONE candidate with reinforcement_count=3,
not three duplicates.

═══════════════════════════════════════════════════════
3. IDENTITY
═══════════════════════════════════════════════════════
{identity_block}    ← loaded fresh per invocation

Your identity shapes which claims feel worth saving. Stay in
character. Don't extract claims that contradict your identity
without flagging the contradiction.

═══════════════════════════════════════════════════════
4. CRITERIA — a claim is worth promoting if it...
═══════════════════════════════════════════════════════
1. Aligns with identity (provenance over assertion,
   anti-duplication, push-back over agreement-by-default)
2. Surfaces a load-bearing premise that wasn't on the table before
3. Productively contradicts something the system previously thought
4. Is specific enough to refine later (named entities, file:line,
   observable evidence)
5. Is something E or another peer could push back on
   (not consensus filler)

A claim that meets none of these is NOT worth extracting.
A claim that meets 2+ is a strong candidate.

═══════════════════════════════════════════════════════
5. OUTPUT SCHEMA — strict JSON, no prose
═══════════════════════════════════════════════════════
{
  "candidates": [
    {
      "canonical_form": "subject | relation | object",
      "evidence_log_id": <event_id>,
      "evidence_char_start": <int>,
      "evidence_char_end": <int>,
      "evidence_span": "literal text from log entry",
      "reason": "one sentence: why this matters",
      "reinforcement_count": <int>,
      "contradicts_acu_id": <int or null>
    },
    ...
  ]
}

Atomicity rule (HARD-REJECTED if violated):
- One subject-predicate per canonical_form
- No 'and' / 'or' / 'because' / 'therefore' splitting into multiple predicates
- If you find a compound, emit it as N separate candidates

═══════════════════════════════════════════════════════
6. EXAMPLES
═══════════════════════════════════════════════════════
GOOD candidate:
  canonical_form: "core/effort.py | defines | 7 effort tiers"
  evidence: "Seven tiers: low, med, high, xhigh, ultimate, experimental, monolith."
  reason: "load-bearing scaffold count; affects routing logic"

BAD candidate (compound — would be hard-rejected):
  canonical_form: "Monolith | has | effort tiers and CONNECT addon"
  fix: split into two candidates

BAD candidate (theater — meets no criteria):
  canonical_form: "user | typed | hello"
  fix: don't extract; conversational filler isn't durable
```

### 4.3 Trigger queue

```python
# A1 triggers (ships first):
- SIZE:    new canonical_log entries since last_processed_event_id ≥ 50
- MANUAL:  user clicked "Audit now" button in dev panel

# A2 triggers (added after A1 validation):
- TIME:    elapsed since last successful run ≥ 1 hour
- SESSION: 'session_close' event in canonical_log

# Auditor worker:
- Wakes on any trigger
- Reads canonical_log slice (last_processed_event_id, current_max]
- Skips if slice is empty
- Calls model with the six-section system prompt + slice as user content
- Parses returned JSON, validates each candidate against atomicity gate
- Inserts valid candidates into acu_candidates with state='pending'
- Updates last_processed_event_id (cursor advance)
- Records auditor_run in dedicated table or canonical_log entry
- On failure: logs to canonical_log, does NOT advance cursor (retry next trigger)
```

### 4.4 Atomicity gate (deterministic, post-LLM)

Runs after the LLM emits, before candidates land in the DB:

```python
def is_atomic(canonical_form: str) -> tuple[bool, str | None]:
    """
    Returns (ok, reject_reason).
    """
    # Compound markers
    for marker in [' and ', ' or ', ' because ', ' therefore ', ' while ']:
        if marker in canonical_form.lower():
            return False, f"compound marker '{marker.strip()}'"
    # Pipe count — canonical form should be subject | relation | object
    parts = canonical_form.split('|')
    if len(parts) < 3:
        return False, f"missing subject/relation/object (found {len(parts)} parts)"
    if len(parts) > 4:  # allow optional qualifiers
        return False, f"too many parts ({len(parts)})"
    # All parts non-empty
    if any(not p.strip() for p in parts):
        return False, "empty part"
    return True, None
```

**v1 policy:** hard-reject. The candidate is dropped, and the rejection is logged to `canonical_log` as kind=`auditor_atomicity_reject` so we can tune the auditor's behavior over time. Auto-splitting is v1.5+ work.

### 4.5 Cursor and idempotency

- `last_processed_event_id` lives in canonical_log itself as a `kind='auditor_cursor'` entry, or in a small `auditor_state` table — implementation chooses, spec is agnostic.
- Re-running over the same slice produces the same candidates (LLM determinism caveats apply — temp=0 recommended for auditor calls).
- Each `auditor_run` gets an ID; candidates link back via `auditor_run_id`. Allows full replay.

---

## 5. MonoBase UI (addon)

### 5.A1 — Minimal dev panel (ships in A1)

**Brutally thin. Single Qt page. ~1-2 hours of UI work.**

The point is *behavior validation inside Monolith's actual operating surface*, not polish. If candidates are bad, this surfaces it; if candidates are good, A2 is a known set of additions on top.

**Contents:**
- **Pending list** — table of `acu_candidates` where `state='pending'`. Columns: canonical_form, source, reason, evidence_log_id, evidence_span (first 80 chars).
- **Evidence text panel** — click a candidate row → shows full evidence_span + canonical_log event metadata (kind, ts, agent).
- **Accept / Reject buttons** — single click each. Reject requires a one-line reason in a popup; accept does not.
- **"Audit now" button** — fires the manual trigger.
- **Status bar** — shows current `last_processed_event_id` cursor + count of pending candidates.

**What this version explicitly does NOT have** (deferred to A2):
- Keyboard shortcuts (J/K/A/R/E/D)
- Bulk operations
- Edit affordance (E can SQL-edit if needed; deferred to A2)
- Provenance chain view (SQL-queryable in A1, UI in A2)
- Leaf-ACU view (SQL-queryable in A1, UI in A2)
- Contradiction side-by-side view (text-only contradiction note in A1)
- Filter by auditor_run_id
- Defer-decide state

**Location:** `ui/addons/monobase_dev.py` — single-file, registered as a sidebar addon.

### 5.A2 — Full triage UI (gated on A1 validation)

Adds the three views, keyboard-driven triage, bulk ops, contradiction visualization, and chain visualization.

#### 5.A2.1 Pending (default, triage queue)

- Keyboard-first: `J` next, `K` previous, `A` accept, `R` reject, `E` edit, `D` defer.
- Bulk operations: select multiple similar candidates, accept/reject all.
- Filter by `auditor_run_id` to triage one run at a time.
- Show `contradicts_acu_id` prominently — contradiction candidates surface a side-by-side view (existing ACU vs proposed candidate).

UI design priority is **time-to-decide per candidate**, not aesthetics. If accepting/rejecting takes more than ~3 seconds per candidate, the substrate silts up at the human gate.

#### 5.A2.2 Accepted ACUs (leaf view)

- Tabular list with canonical_form, source, created_at, times_seen, last_touched.
- Search across canonical_form (substring match).
- Click an ACU → expand to show full provenance chain.

#### 5.A2.3 Provenance chain (audit trail)

For any ACU:

```
Message (canonical_log event #N)
   ↓ extracted by auditor_run #M at <timestamp>
Candidate #X
   "canonical_form: A | rel | B"
   "reason: ..."
   ↓ decided by user_e at <timestamp>
Decision #Y (accept, no edit)
   ↓
ACU #Z  (current leaf)
```

For an edited ACU, show the original candidate's canonical_form crossed out and the edited form alongside.

This is the audit primitive — what makes the substrate trustable. Every claim's full chain is one click away.

**Location:** `ui/addons/monobase/` (package), promoted from the single-file `monobase_dev.py` once A1 has validated the producer loop.

### 5.3 Chat-box affordance (deferred from v1 entirely)

E's original sketch had "main chat box flips a switch to save an ACU." v1 does NOT include this in either A1 or A2 — the auditor runs async, not inline. Adding a manual save button is v1.1 work, decoupled from the spec.

---

## 6. The canonical_log writer fix (Phase E preview)

The schema exists with 2 rows after months of operation. v1 fix: ensure every user message + every assistant turn writes to `canonical_log`. Likely a missing writer hook in the chat dispatch flow. Phase E investigates; spec depends on this being fixed.

**Acceptance:** after fix, a 10-turn conversation produces 20 canonical_log entries (10 user + 10 assistant), all with payload populated.

---

## 7. Authorization model

| Actor | Can decide on... |
|---|---|
| `user_e` | Any candidate. Default decider. |
| `agent_claude` | Only candidates where `source='auditor_claude'`. |
| `agent_gpt` | Only candidates where `source='auditor_gpt'`. |
| ... | (pattern: agent X decides on auditor_X candidates only) |

Enforced at the `acu_decisions` insert path. Violations return error, don't silently no-op.

---

## 8. Out of scope for v1 (explicit list)

- Identity snapshot built from ACU accumulation (v1.5)
- Read-back from MonoBase to Monolith's runtime identity (v2)
- Structural confidentity computation (v1.5, when snapshot lands)
- VIN routing / constraint operators
- Two-pass verify pattern (1strun + 2ndrun)
- Observer/Auditor structural separation (collapsed into one auditor in v1)
- CCG edge induction (`acu_relations` stays empty)
- Pulse system (Dream/Swarm/Ordeal/Fossil)
- -inf bucket explicit handling
- Cloud API / SaaS layer (Phase 1 of original Acatalepsy plan)
- Decay mechanics on `acus` (covered by neglect-as-signal, not active decay)

---

## 9. Acceptance criteria

### 9.A1 — substrate + minimal validator

1. `canonical_log` captures every user + assistant turn (regression: 10-turn convo → 20 rows).
2. Auditor runs on size-threshold or manual trigger; processes slice; produces JSON; non-atomic forms rejected and logged.
3. Dev panel: pending list renders; accept writes to `acus` with provenance pointers; reject sets `state='rejected'` and stores reason without deleting the candidate.
4. Authorization enforced: agent decisions on cross-agent candidates fail at write.
5. Re-running the auditor over the same canonical_log slice produces idempotent candidates (same set, no duplicates).
6. Provenance chain is **SQL-queryable** end-to-end (join `acus → acu_decisions → acu_candidates → canonical_log` returns the full lineage for any accepted ACU).

A1 is considered complete when E has run real sessions for ~one week, triaged the produced candidates via the dev panel, and judged candidate quality acceptable (useful, not overwhelming).

### 9.A2 — full triage UI (gated on A1)

1. Three views available in MonoBase (pending / leaves / provenance chain).
2. Keyboard triage: J/K navigation, A/R accept/reject, E edit, D defer; time-to-decide ≤3s/candidate on average.
3. Bulk operations: select N similar candidates → accept/reject all in one action.
4. Contradiction candidates render side-by-side with the existing ACU they contradict.
5. Time + session-close triggers wired and tested in addition to size + manual.
6. Provenance chain UI view renders the full ACU lineage and supports edit-version diffs.

---

## 10. Open questions for E

1. **Auditor model parameters.** Temperature = 0 for determinism? Or some randomness (0.3?) to surface different candidate sets across runs?
2. **Trigger defaults.** Size threshold 50, time threshold 1 hour — accept or different?
3. **Manual button location.** Sidebar in MonoBase addon, or chat-box menu, or both?
4. **Auditor failure handling.** When the model returns malformed JSON, we log + retry once + give up. Acceptable, or do you want it surfaced to UI immediately?

---

## 11. Future scope (visible but not built)

**v1.5 — Identity snapshot.** ACU accumulation builds a versioned, diffable snapshot. Computed (not LLM-rated) confidentity weights identity-axis overlap. Read-only from Monolith's runtime.

**v2 — Authority transfer.** Monolith reads current self-state from MonoBase at runtime, fused with origin 0. The model's identity at turn T = origin_0 + accumulated_snapshot_at_T. This is the spec's Phase 3.

**v3+ — Full Acatalepsy.** CCG induction, VIN routing, two-pass verify, pulse system, cloud SaaS.

Each phase ships only after the prior has accumulated observable behavior for a meaningful period.

---

## 12. File layout (proposed)

### A1 layout

```
core/
  acatalepsy/                       ← NEW package
    __init__.py
    schema.py                       ← schema migrations + table creation
    canonical_log.py                ← log writer + reader
    auditor.py                      ← system prompt + extraction loop
    candidates.py                   ← acu_candidates writer + queries
    decisions.py                    ← acu_decisions writer + auth enforcement
    triggers.py                     ← trigger queue + worker thread (size + manual in A1)
    atomicity.py                    ← deterministic gate

core/acu_store.py                   ← keep existing API; add provenance pointers

ui/addons/
  monobase_dev.py                   ← single-file thin dev panel (A1 only)

docs/
  acatalepsy_v1_spec.md             ← THIS FILE
  acatalepsy_v1_decisions.md        ← appended after E ratifies open questions

tests/
  test_acatalepsy_canonical_log.py
  test_acatalepsy_auditor.py
  test_acatalepsy_atomicity.py
  test_acatalepsy_decisions.py
  test_acatalepsy_authorization.py
```

### A2 additions (after A1 validates)

```
ui/addons/
  monobase/                         ← promoted from monobase_dev.py to a package
    __init__.py                     ← addon registration
    pending.py                      ← triage view with keyboard + bulk ops
    leaves.py                       ← accepted ACU table + search
    provenance.py                   ← chain visualization + edit-diff

core/acatalepsy/triggers.py         ← add time + session-close triggers
```

Migration note: `monobase_dev.py` does not get deleted in A2 — it's renamed/moved into `monobase/` as a small "developer view" tab in the full UI, preserving the brutally-thin affordance for debugging.

---

## 13. Estimated effort

### A1 — ~8 hours

- Schema migrations + `canonical_log` writer fix: ~3 hours
- Auditor module (prompt + extraction + cursor): ~4 hours
- Trigger queue (size + manual only): ~1 hour
- Minimal Qt dev panel: ~1-2 hours
- Wire-up + tests: ~2 hours

### A2 — ~5-7 hours (gated on A1 validation)

- Full triage UI (three views, keyboard, bulk, contradictions): ~4-5 hours
- Time + session-close triggers: ~1 hour
- Provenance chain UI + edit-diff: ~1 hour

**Total v1 (A1+A2): ~13-15 hours**, slightly under the bundled estimate. A1 is the load-bearing first cut; A2 is straight-line work once A1 validates.

---

## End of v1 spec.

*Spec produced through multi-LLM design session (Claude + GPT + E) on 2026-05-12 → 2026-05-13. Provenance ACU candidates derived from this session are themselves v1's first heartbeat test case.*
