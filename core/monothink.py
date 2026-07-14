"""MonoThink — the only model-tended scaffold in Monolith's substrate.

The model authors and evolves its own thinking scaffold through bounded,
small-step evolutions triggered by rating signals. Garden, not architecture.

Evolution mechanism:
  * Trigger: a rating outcome (kind="rating") is recorded for a turn that ran
    on reasoning_mode="monothink".
  * Hook fires from ``core/turn_trace.py:record_outcome`` after the rating
    is persisted.
  * The hook composes a prompt with the current scaffold + rating value +
    rating reason, calls the same LLM backend the runtime is bound to, and
    asks for ONE small change or "NO_CHANGE".
  * The response is validated against hard caps: total size <= 12000 chars,
    diff <= 300 chars. Out-of-cap responses are rejected (journaled).
  * On accept: the scaffold file is rewritten atomically and a journal entry
    is appended.

Bounded autonomy (per S12 boundary preservation):
  * The scaffold path is hardcoded — the module writes to
    ``prompts/reasoning/monothink.md`` only. The journal is also hardcoded
    to ``prompts/reasoning/monothink.journal.jsonl``. No other file can be
    modified through this mechanism.
  * **Per-file boundary, NOT per-directory.** Bounded autonomy applies only
    to these two specific files. Other files that may land under
    ``prompts/reasoning/`` in the future (a future regulator-loop file, say)
    do NOT inherit MonoThink's mutation rights by virtue of sharing this
    directory. Each such file would need its own bounded-autonomy contract
    (or have none). Do not relax this boundary by generalizing to the
    directory level.
  * One evolution attempt PER TURN, ever. The journal is the dedup table —
    a re-rating on the same turn returns an ``already_processed`` sentinel
    instead of burning another LLM call.
  * The journal is append-only. Rollback is manual (E reads the journal,
    pastes the old text back into the scaffold file; delete the journal
    line to allow a fresh evolution attempt on that turn).
  * Conservative bias in the evolution prompt: NO_CHANGE is the safe default.

Plane: reasoning (moved here from effort during the plane-separation refactor).
The scaffold loads through ``core/reasoning.py`` (reasoning_interceptor) and
the evolution trigger gates on ``frame_traces.reasoning_mode`` (populated
from ``config["_resolved_reasoning_mode"]`` by ``engine/llm.py``).

Flag: MONOLITH_MONOTHINK_EVOLVE_V1 (default ON). Set =0 to freeze evolution
while keeping the mode usable (the scaffold still loads via reasoning_interceptor).

Files (per-file bounded autonomy — only these two are writable):
  * prompts/reasoning/monothink.md           — scaffold text (read by reasoning_interceptor)
  * prompts/reasoning/monothink.journal.jsonl — append-only diff history
"""
from __future__ import annotations

import difflib
import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

from core import failure_tags as _failure_tags

# Hard paths. Intentionally not configurable — this is bounded autonomy.
# Per S12: these two paths are the entire write surface. Do not refactor toward
# a configurable _PROMPTS_DIR that other files in the same dir could exploit.
_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts" / "reasoning"
_SCAFFOLD_PATH = _PROMPTS_DIR / "monothink.md"
_JOURNAL_PATH = _PROMPTS_DIR / "monothink.journal.jsonl"

_FLAG_ENV = "MONOLITH_MONOTHINK_EVOLVE_V1"
# Run the LLM call (and the journal write that follows it) in a daemon
# thread so the caller — typically the chat UI thread routing a /rating —
# isn't blocked on generation. Default ON; tests set =0 to keep the call
# synchronous so they can assert on the returned journal entry.
_ASYNC_ENV = "MONOLITH_MONOTHINK_ASYNC"

# Raised 8000 -> 12000 (2026-06-25): under training the scaffold reached 8051 chars and
# edits began rejecting on size alone (size_cap_exceeded). The prune-vs-raise call was
# raise (E), to keep training unblocked; scaffold-pruning stays a later option.
_SIZE_CAP = 12000
_DIFF_CAP = 300

# Tail-cap on the model's reasoning trace when injecting into the evolution
# prompt: keep the latest 4000 chars (the conclusion is usually at the end).
# Generous enough to carry a full <think> block on most turns without
# blowing through the LLM context budget at max_tokens=2048.
_THINK_TAIL_CAP = 4000

_NO_CHANGE_TOKEN = "NO_CHANGE"

# Phase 1a schema. Values match the prompt's required output format.
_SCOPE_ENUM = frozenset({"line", "section", "structural", "none"})
_HOT_SCOPE_ENUM = frozenset({"line", "section"})
_TAG_MAX_LEN = 32

# Phase 2a: protected scaffold sections. Edits that modify the body of any
# section in this tuple, OR remove the section entirely, are rejected before
# size/diff cap checks. The protection lives in code (not in the scaffold
# prose) on purpose: the patcher is the enforcer; mixing enforcement into
# the model-mutable substrate would let the model edit its own constraints
# away. Per-file bounded autonomy stays at the file level; intra-file
# protection lives here.
#
# Conflict Resolution is intentionally NOT protected — it documents
# downstream patterns that may legitimately shift over time.
_PROTECTED_SECTIONS: tuple[str, ...] = (
    "Core invariant",
    "Scope boundary",
    "Grounding cite",
)

_LLM_CALL_HEADER_RE_PREFIX = "[llm_call:"

# Process-wide gate so concurrent rating clicks for the same turn can't both
# pass the journal check, race the LLM call, and clobber each other's
# scaffold writes. Evolutions are rare; serializing them costs nothing.
_evolve_lock = threading.Lock()


# ── flag / time helpers ──────────────────────────────────────────────


def _flag_enabled() -> bool:
    raw = str(os.environ.get(_FLAG_ENV, "1")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _async_enabled() -> bool:
    raw = str(os.environ.get(_ASYNC_ENV, "1")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── scaffold + journal IO ────────────────────────────────────────────


def read_scaffold() -> str:
    """Return current scaffold text (empty string if missing/unreadable)."""
    try:
        return _SCAFFOLD_PATH.read_text(encoding="utf-8")
    except Exception:
        return ""


def _write_scaffold(text: str) -> None:
    """Atomic write to the scaffold file."""
    tmp = _SCAFFOLD_PATH.with_suffix(_SCAFFOLD_PATH.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, _SCAFFOLD_PATH)


def _append_journal(entry: dict) -> None:
    """Append-only JSONL append. Best-effort — never raises."""
    try:
        with _JOURNAL_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        # Journal failure must never break the rating loop.
        pass


def _journal_has_turn(turn_id: str) -> bool:
    """Return True if any journal entry already exists for *turn_id*.

    Drives per-turn idempotence: one evolution attempt per turn_id, period.
    Even a rejected entry (LLM failure, cap exceeded, NO_CHANGE) counts —
    re-rating the same turn is NOT a re-evaluation opportunity. This is
    bounded autonomy: the model sees each turn's rating signal once.

    Cheap streaming scan with a literal pre-filter; short-circuits on first
    hit. Defensive: any IO/parse error returns False (better to risk a
    duplicate evolution than to silently freeze on a corrupted journal).
    """
    if not _JOURNAL_PATH.exists() or not turn_id:
        return False
    tid = str(turn_id)
    try:
        with _JOURNAL_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or tid not in line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                if entry.get("turn_id") == tid:
                    return True
    except Exception:
        return False
    return False


def read_journal(limit: int = 20) -> list[dict]:
    """Return the last *limit* journal entries (newest first)."""
    if not _JOURNAL_PATH.exists():
        return []
    try:
        lines = _JOURNAL_PATH.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    out: list[dict] = []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    out.reverse()
    return out


# ── diff measurement ─────────────────────────────────────────────────


def _diff_chars(a: str, b: str) -> int:
    """Size of the GENUINE change between ``a`` and ``b`` — the larger of the
    inserted vs deleted character counts, ignoring unchanged spans even when
    they sit BETWEEN two edits.

    Uses difflib so an edit touching both ends of a region (e.g. a header word
    changed AND an item appended) measures only the changed text, not the whole
    bracketed span. The previous single prefix/suffix strip counted the entire
    span between the first and last difference, which inflated a ~180-char rule
    addition to ~955 and mis-routed line-scale edits into the section path
    (journal entry 27); see tests/test_monothink.py.

    ``max(inserted, deleted)`` — not the sum — so a full rewrite of disjoint
    text still measures the longer side (preserving "whole-file rewrites stay
    large"), while a pure insertion measures only what was inserted."""
    inserted = deleted = 0
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(
        None, a, b, autojunk=False
    ).get_opcodes():
        if op in ("replace", "delete"):
            deleted += i2 - i1
        if op in ("replace", "insert"):
            inserted += j2 - j1
    return max(inserted, deleted)


# ── prompt + LLM call ────────────────────────────────────────────────

# Sentinel the decider emits between its deliberation prose and the schema
# block. The parser isolates the schema by taking only what follows the LAST
# sentinel — deliberation prose (STEELMAN/ADJUDICATION) can contain
# field-looking lines, and without this the first-occurrence-wins field scan
# would grab them. Absent (test fixtures / legacy responses) the whole text is
# parsed, so existing fixtures are unaffected.
_SCHEMA_SENTINEL = "=== PATCH ==="
_DECISION_SENTINEL = "=== DECISION ==="


def _clip_think_block(think_block: str | None) -> str | None:
    """Tail-cap the reasoning trace at _THINK_TAIL_CAP chars.

    Tail-keep, not head-keep: the conclusion of a <think> block is where the
    decision crystallizes, so the last N chars carry more signal than the
    first N when the trace is long. Returns None when there's no usable
    content to inject.
    """
    if not think_block:
        return None
    text = str(think_block).strip()
    if not text:
        return None
    if len(text) <= _THINK_TAIL_CAP:
        return text
    return "…(trace truncated, latest tail shown)…\n" + text[-_THINK_TAIL_CAP:]


def _compose_prompt_legacy_unused(
    current: str,
    rating_value: int,
    failure_tags: list[str],
    think_block: str | None = None,
    contrast_block: str | None = None,
) -> str:
    """Build the evolution prompt as a three-stage adversarial deliberation.

    The rater's failure_tags are a PRIOR, not a command. The decider steelmans the
    tag, then positionally steelmans the alternative, then adjudicates — emitting the
    adjudicated verdict as PRIMARY_FAILURE_TAG. The directional signal is the canonical
    descriptive-only gloss of each tag (composed here); no rater free text enters this
    prompt, and surface_note never reaches it (the monothink_visible boundary).
    """
    scaffold_block = (
        current if current.strip()
        else "(empty — first evolution; seed it from your own observed posture)"
    )
    signal = _failure_tags.compose_monothink_signal(failure_tags)
    if not signal:
        signal = "(no failure tag supplied)"
    contrast_section = ""
    if contrast_block and str(contrast_block).strip():
        contrast_section = f"\n{str(contrast_block).strip()}\n"
    # The adjudicated verdict must be drawn from the closed vocabulary so that
    # divergence (rater_tag vs adjudicated_tag) and concern_repeated match exactly
    # — a failure outside monothink's mutation vocabulary isn't actionable anyway.
    vocab = ", ".join(_failure_tags.FAILURE_TAGS)
    clipped_think = _clip_think_block(think_block)
    think_section = ""
    if clipped_think:
        think_section = (
            "\nYour reasoning trace from that turn (between <think> tags — "
            "this is the path your thinking took under the current scaffold):\n"
            "```\n"
            f"{clipped_think}\n"
            "```\n\n"
            "Use the trace to locate the scaffold's actual effect: the rating "
            "judges where the response landed; the trace shows how you got "
            "there. The scaffold is the lever between them.\n"
        )
    return (
        "You are tending your own thinking scaffold — the only scaffold in this "
        "substrate that you are permitted to modify. Garden-mode: small "
        "evolutions, not redesigns.\n\n"
        "Current scaffold (prompts/reasoning/monothink.md):\n"
        "```\n"
        f"{scaffold_block}\n"
        "```\n\n"
        f"Last turn ran on this scaffold and received rating {int(rating_value)}/100.\n"
        f"The rater flagged this reasoning failure (a PRIOR, not a verdict):\n{signal}\n"
        f"{contrast_section}"
        f"{think_section}\n"
        "── Deliberate IN YOUR THINKING before you patch (three stages, all mandatory) ──\n\n"
        "The rater's tag is a hypothesis. Before editing anything you must argue both "
        "sides and reach your own verdict — but do all of this IN YOUR THINKING, not in "
        "your output. Keep your reasoning focused: your thinking and the patch share one "
        "token budget, so don't burn it all reasoning or you'll be cut off before the "
        "patch lands.\n\n"
        "STEELMAN_TAG: construct the strongest case that the flagged failure IS the "
        "load-bearing failure on this turn — the one whose correction would most change "
        "the answer.\n"
        "STEELMAN_ALTERNATIVE: construct the strongest case that the flagged failure is "
        "wrong or not load-bearing, and name the failure you would diagnose instead. "
        "Mandatory and positional: produce the best disagreement that exists EVEN IF you "
        "believe the tag is correct. Do not hedge; argue the alternative as if it were "
        "true.\n"
        "ADJUDICATION: weigh both cases and name the load-bearing failure you will act "
        "on. It may be the rater's tag or the alternative.\n\n"
        "── How to patch ──\n\n"
        "You are EDITING a scaffold, not REWRITING it. Find the SMALLEST span (1-3 lines "
        "preferred, full section maximum) whose change addresses the ADJUDICATED failure. "
        "Leave everything else BYTE-IDENTICAL. If multiple changes seem warranted, patch "
        "ONE and put the rest in DEFERRED_CONCERNS. A good patch is boring.\n\n"
        "DELETION is a legal patch: if a span is restating, redundant, or not "
        "load-bearing, leave AFTER empty (PATCH_MODE stays replace_lines or "
        "replace_section) and the span is removed. The scaffold's own invariant applies "
        "to itself — text that wouldn't change any conclusion if removed should be "
        "removed. Near the size cap, a deletion or net-shrinking patch is often the "
        "ONLY patch that can land.\n\n"
        "── Required output format ──\n\n"
        "Output ONLY the patch — the exact sentinel line below, then these field labels "
        "in this order, and NOTHING else. Do NOT write the deliberation stages, the "
        "steelman prose, or any commentary in your output; that reasoning belongs in your "
        "thinking. No code fences. The patch is INVALID without its AFTER block — you MUST "
        "reach and fill AFTER; if you are running long, shrink BEFORE/AFTER to a single "
        "line rather than omit AFTER.\n\n"
        "=== PATCH ===\n"
        "PRIMARY_FAILURE_TAG: <your ADJUDICATED failure — EXACTLY one tag from this set, "
        f"copied verbatim (do not invent or rephrase): {vocab}. It may be the rater's tag "
        "or another from the set; choose the one your deliberation concluded is "
        "load-bearing.>\n"
        "PRIMARY_FAILURE: <one sentence naming the adjudicated failure, ≤200 chars>\n"
        "TARGET_SECTION: <scaffold section header (without ## prefix), or NONE if no change>\n"
        "PROPOSED_SCOPE: line | section | structural | none\n"
        "DEFERRED_CONCERNS:\n"
        "- <other concern 1, ≤200 chars>\n"
        "- <other concern 2, ≤200 chars>\n"
        "PATCH_MODE: replace_lines | replace_section | no_change\n"
        "TARGET_LINES: <line range like '47-49', OR 'section: <name>', OR NONE if no_change>\n"
        "BEFORE:\n"
        "<the exact current text being replaced — must match the current scaffold "
        "byte-for-byte, whitespace and line endings included. Leave empty when "
        "PATCH_MODE=no_change.>\n"
        "AFTER:\n"
        "<the replacement text. Leave empty when PATCH_MODE=no_change.>\n\n"
        "── BEFORE/AFTER mechanics ──\n\n"
        "The patcher locates the BEFORE block as a literal substring of the current "
        "scaffold and replaces it with AFTER. Copy BEFORE from the scaffold block above; "
        "don't paraphrase. If BEFORE appears more than once, extend it until the match is "
        "unique. For no_change: both BEFORE and AFTER empty, PROPOSED_SCOPE=none, "
        "TARGET_SECTION=NONE, TARGET_LINES=NONE.\n\n"
        "── PROPOSED_SCOPE semantics ──\n\n"
        "STRONGLY PREFER the smallest change that fixes the adjudicated failure. A change of "
        f"{_DIFF_CAP} characters or fewer to a single section is measured as `line` and "
        "auto-applies immediately — that is the target. Make ONE surgical edit (append ONE "
        "rule, tighten ONE sentence); do not rewrite a whole section when a line will do.\n"
        "PROPOSED_SCOPE is the size of the CHANGE, NOT the size of the BEFORE block you "
        "quote for a unique match. The patcher measures only the genuine inserted/deleted "
        "text (an unchanged middle between two edits does NOT count). The CODE is the "
        "authority: it measures the actual scope and decides from THAT, logging any "
        "disagreement with your stated label. A `line` label on a large change does NOT make "
        "it apply as a line — so be honest, and above all keep the change small.\n"
        f"- line: a small change (≤{_DIFF_CAP} chars) — add, fix, or remove up to a few "
        "lines (e.g. appending ONE numbered rule, tightening a sentence). STILL `line` even "
        "if BEFORE quotes the surrounding paragraph to anchor the match. Adding one rule is "
        "the common case — label it `line`. THIS IS THE PREFERRED SHAPE.\n"
        "- section: most of a section's body genuinely rewritten — several rules/sentences "
        "changed at once — header set unchanged. Use only when a line edit genuinely cannot "
        "carry the fix; section edits auto-apply only when the same concern has recurred.\n"
        "- structural: adds, removes, or renames a ## header\n"
        "- none: no change (PATCH_MODE=no_change)\n\n"
        "Constraints (hard ceilings):\n"
        f"- Total scaffold (after applying AFTER) must stay <= {_SIZE_CAP} chars. It is "
        f"currently {len(scaffold_block)} chars — additions beyond that headroom WILL be "
        "rejected; shrink elsewhere in the same patch or delete instead.\n"
        "- Don't add ornament. The scaffold is pressure on HOW TO THINK, not how to "
        "sound. Precision over fluency.\n"
        "- Conservative bias: prefer no_change over change unless the adjudicated failure "
        "clearly indicates direction."
    )


# ── section-aware helpers (Phase 2b) ──────────────────────────────────


def _compose_prompt(
    current: str,
    rating_value: int,
    failure_tags: list[str],
    think_block: str | None = None,
    contrast_block: str | None = None,
    rater_note: str | None = None,
    ledger_block: str | None = None,
) -> str:
    """Build the permission-gated MonoThink decider prompt."""
    scaffold_block = (
        current if current.strip()
        else "(empty - first evolution; seed it from your own observed posture)"
    )
    signal = _failure_tags.compose_monothink_signal(failure_tags)
    if not signal:
        signal = "(no failure tag supplied)"
    vocab = ", ".join(_failure_tags.FAILURE_TAGS)

    note_section = ""
    if rater_note and str(rater_note).strip():
        note_section = (
            "\nRATER_NOTE_NONCANONICAL:\n"
            f"{str(rater_note).strip()}\n"
            "Use this only to locate evidence in the trace or to support reject/reserve. "
            "It may not be the source of FAILURE_SIGNATURE, may not satisfy promotion, "
            "and may not appear in DECISION_REASON for apply.\n"
        )

    ledger_section = f"\n{str(ledger_block).strip()}\n" if ledger_block and str(ledger_block).strip() else ""
    contrast_section = f"\n{str(contrast_block).strip()}\n" if contrast_block and str(contrast_block).strip() else ""

    clipped_think = _clip_think_block(think_block)
    think_section = ""
    if clipped_think:
        think_section = (
            "\nYour reasoning trace from that turn (between <think> tags - "
            "this is the path your thinking took under the current scaffold):\n"
            "```\n"
            f"{clipped_think}\n"
            "```\n\n"
            "Use the trace to locate the scaffold's actual effect. The rating "
            "judges where the response landed; the trace shows how you got there. "
            "The scaffold is the lever between them.\n"
        )

    return (
        "You are tending your own reasoning scaffold. Editing it is the exception, "
        "not the task. Default: PATCH_MODE: no_change.\n\n"
        "Current scaffold (prompts/reasoning/monothink.md):\n"
        "```\n"
        f"{scaffold_block}\n"
        "```\n\n"
        f"Last turn ran on this scaffold and received rating {int(rating_value)}/100.\n"
        f"The rater flagged this reasoning failure (a PRIOR, not a verdict):\n{signal}\n"
        f"{note_section}{ledger_section}{contrast_section}{think_section}\n"
        "-- Permission gate --\n\n"
        "A rating row is evidence for a reservation, not permission to mutate the scaffold. "
        "You may move from reservation to edit ONLY when one of these holds:\n\n"
        "(A) Repeated pattern. The SAME (tag, section, failure_signature) has recurred "
        "on at least one PRIOR distinct turn citing a distinct trace span; the ledger "
        "shows count >= 1 before this turn, so this turn makes it >= 2, and the "
        "signature is canonical.\n\n"
        "(B) Critical invariant break. The trace directly violates an explicit scaffold "
        "invariant, with all three of: INVARIANT_CITE verbatim from the scaffold, "
        "TRACE_SPAN verbatim from the trace, and VIOLATION_FORM. If any is missing "
        "or fails verbatim match, this is NOT an invariant break.\n\n"
        "Before proposing any edit, also confirm all three: the failure is structurally "
        "locatable in the current scaffold; it is not already covered by an existing "
        "rule; and the edit targets reasoning behavior, not surface wording. If the "
        "existing rule covers it, the fix is ENFORCEMENT, not a new rule. If any check "
        "fails: MUTATION_DECISION: reserve or reject, PATCH_MODE: no_change, and write "
        "the concern to the ledger as a reservation unless it is rating_misapplied.\n\n"
        "Before minting FAILURE_SIGNATURE, check the LEDGER rows above for the same "
        "(tag, section). Reuse an existing signature if the mechanism matches. New "
        "signatures must be short mechanism labels: lowercase words, hyphen joined, "
        "not prose.\n\n"
        "-- Deliberate IN YOUR THINKING before deciding (three stages, all mandatory) --\n\n"
        "STEELMAN_TAG: construct the strongest case that the flagged failure IS the "
        "load-bearing failure on this turn.\n"
        "STEELMAN_ALTERNATIVE: construct the strongest case that the flagged failure is "
        "wrong or not load-bearing, and name the failure you would diagnose instead. "
        "Mandatory and positional: argue the alternative as if it were true.\n"
        "ADJUDICATION: weigh both cases and name the load-bearing failure you will act on.\n\n"
        "-- Required output format --\n\n"
        "Output ONLY the decision and, only when applying, the patch. No commentary, "
        "no code fences. Do not write deliberation prose; it belongs in your thinking. "
        "On apply, output only the patch after the decision block. The no_change path "
        "is intentionally short; the apply path is intentionally laborious. A good "
        "patch is boring.\n\n"
        "=== DECISION ===\n"
        "PRIMARY_FAILURE_TAG: <exactly one tag from this set: "
        f"{vocab}>\n"
        "PRIMARY_FAILURE: <one sentence>\n"
        "FAILURE_SIGNATURE: <short mechanism label>\n"
        "SIGNATURE_STATUS: provisional | canonical\n"
        "LEDGER_LOOKUP: <(tag, section, signature) -> prior count, or NONE>\n"
        "EVIDENCE_CLASS: single_instance | repeated_pattern | critical_invariant_break\n"
        "ALREADY_COVERED: yes(<section>) | no\n"
        "FIX_KIND: scaffold_gap | enforcement_gap | rating_misapplied\n"
        "MUTATION_DECISION: reserve | apply | reject\n"
        "DECISION_REASON: <one sentence>\n\n"
        "Decision constraints: if ALREADY_COVERED=yes, FIX_KIND cannot be scaffold_gap. "
        "FIX_KIND=enforcement_gap must reserve and escalate; never apply as a new rule. "
        "FIX_KIND=rating_misapplied must reject. RATER_NOTE_NONCANONICAL may not appear "
        "in DECISION_REASON for apply.\n\n"
        "If MUTATION_DECISION != apply, STOP after this:\n\n"
        "PATCH_MODE: no_change\n"
        "DEFERRED_CONCERN: <concern written to ledger, or NONE if reject>\n\n"
        "DEFERRED_CONCERNS: is the old plural field and must not be emitted; the v2 "
        "reserve path uses DEFERRED_CONCERN singular.\n\n"
        "Only if MUTATION_DECISION = apply, continue:\n\n"
        "=== PATCH ===\n"
        "TARGET_SECTION: <section>\n"
        "TARGET_LINES: <range>\n"
        "PROPOSED_SCOPE: line | section\n"
        "STRUCTURAL_LOCATION: <which current scaffold text produced the failure, and how>\n"
        "PREDICTED_EFFECT: <behavioral change this edit should cause>\n"
        "FALSIFIER: <future signal showing the edit failed / should be reverted>\n"
        "APPLIED_AT_TURN: <rated turn id, or UNKNOWN>\n"
        "PATCH_MODE: replace_lines | replace_section\n"
        "BEFORE:\n"
        "<exact current text being replaced>\n"
        "AFTER:\n"
        "<replacement text>\n\n"
        "-- BEFORE/AFTER mechanics --\n"
        "The patcher locates BEFORE as a literal substring of the current scaffold "
        "and replaces it with AFTER. Copy BEFORE from the scaffold block above; do "
        "not paraphrase; it must match byte-for-byte. Leave all unrelated text "
        "BYTE-IDENTICAL. Make the smallest edit that can change reasoning behavior. "
        "Deletion is legal: leave AFTER empty when deleting the matched span. "
        "A line edit should stay within "
        f"{_DIFF_CAP} changed characters; section edits carry higher burden. "
        "The old hot-path scopes structural and none are not valid apply scopes; "
        "structural changes route to consolidation, and no_change is represented "
        "by PATCH_MODE: no_change.\n\n"
        "If EVIDENCE_CLASS=critical_invariant_break, the PATCH must also include:\n"
        "INVARIANT_CITE: <exact scaffold text>\n"
        "TRACE_SPAN: <exact trace span violating it>\n"
        "VIOLATION_FORM: contradiction | omission | forbidden_action\n\n"
        "BEFORE must be a literal substring of the current scaffold. INVARIANT_CITE "
        "and TRACE_SPAN are mechanically checked. Structural scope is not available "
        "to this hot-path decider.\n\n"
        "Constraints:\n"
        f"- Total scaffold after apply must stay <= {_SIZE_CAP} chars. Current scaffold "
        f"is {len(scaffold_block)} chars.\n"
        "- Do not add ornament. The scaffold is pressure on how to think, not how to sound.\n"
        "- Conservative bias: prefer reserve or reject unless promotion rule A or B is met."
    )


# A line is a `##` (h2) section header if, after lstrip, it starts with
# exactly `## ` (two hashes + space), NOT `### ` or deeper. The fenced-
# code state check is separate — fences are tracked while iterating.
_H2_HEADER_RE = re.compile(r"^##\s+(\S.*?)\s*$")
_CODE_FENCE_RE = re.compile(r"^\s*```")


def _parse_sections(text: str) -> dict[str, str]:
    """Parse markdown sections delimited by ``## Header`` lines.

    Returns an insertion-ordered dict mapping section header text (without
    the ``## `` prefix) to section body. ``### Subheader`` lines and deeper
    belong to their parent ``##`` section's body. The H1 preamble (anything
    before the first ``##``) is NOT a section.

    Code fences (``` lines) are tracked so a ``##`` inside a fenced block
    is treated as content, not as a section break — that prevents false
    splits on scaffold prose that demonstrates markdown structure inside a
    code example.

    Bodies are right-stripped of trailing whitespace so comparison is
    robust to incidental newline drift between old and new scaffolds.
    """
    sections: dict[str, str] = {}
    current_header: str | None = None
    current_body: list[str] = []
    in_fence = False

    def commit() -> None:
        if current_header is not None:
            sections[current_header] = "\n".join(current_body).rstrip()

    for line in text.splitlines():
        if _CODE_FENCE_RE.match(line):
            in_fence = not in_fence
            if current_header is not None:
                current_body.append(line)
            continue
        if not in_fence:
            m = _H2_HEADER_RE.match(line)
            if m:
                commit()
                current_header = m.group(1).strip()
                current_body = []
                continue
        if current_header is not None:
            current_body.append(line)

    commit()
    return sections


def _diff_touches_protected(old: str, new: str) -> list[tuple[str, str]]:
    """Return ``[(section_name, kind), ...]`` for every protected section
    whose state differs between ``old`` and ``new``.

    ``kind`` is one of:
      * ``"modified"`` — present in both, body differs
      * ``"removed"`` — present in old, absent in new (header gone entirely)
      * ``"added"`` — absent in old, present in new (defensive — shouldn't
        normally happen since protected sections start present)

    Empty list when nothing protected was touched. Caller uses the first
    entry to form the reject_reason (``protected_section:<name>`` for
    modified/added, ``protected_section_removed:<name>`` for removed).
    """
    old_sections = _parse_sections(old)
    new_sections = _parse_sections(new)
    touched: list[tuple[str, str]] = []
    for name in _PROTECTED_SECTIONS:
        old_body = old_sections.get(name)
        new_body = new_sections.get(name)
        if old_body is None and new_body is None:
            continue
        if old_body is None:
            touched.append((name, "added"))
        elif new_body is None:
            touched.append((name, "removed"))
        elif old_body.strip() != new_body.strip():
            touched.append((name, "modified"))
    return touched


def _diff_section_span(old: str, new: str) -> int:
    """Count sections whose bodies differ between ``old`` and ``new``.

    Includes additions and removals. Used by Phase 2d's empty-reason guard:
    a span > 1 means the proposed change crosses multiple section
    boundaries, which is a structural-scope move that empty-reason ratings
    (e.g. thumbs from Phase 0b) shouldn't authorize.
    """
    old_sections = _parse_sections(old)
    new_sections = _parse_sections(new)
    all_names = set(old_sections) | set(new_sections)
    span = 0
    for name in all_names:
        if old_sections.get(name, "").strip() != new_sections.get(name, "").strip():
            span += 1
    return span


# ── BEFORE/AFTER patcher (Phase 3.5) ──────────────────────────────────


# Allowed values for PATCH_MODE.
_PATCH_MODE_ENUM = frozenset({"replace_lines", "replace_section", "no_change"})


def _apply_patch(current: str, before: str, after: str) -> tuple[str | None, str | None]:
    """Apply a BEFORE → AFTER substitution to ``current``.

    Returns ``(new_scaffold, None)`` on success or ``(None, error_str)`` on
    failure. Error strings used as journal reject_reasons:

      * ``"before_block_empty"`` — BEFORE was empty but PATCH_MODE wasn't
        no_change. Caller should use no_change to signal "don't patch."
      * ``"before_block_mismatch:not_found"`` — BEFORE doesn't appear in
        the current scaffold. Usually means the model hallucinated about
        the current text.
      * ``"before_block_mismatch:ambiguous"`` — BEFORE appears more than
        once. The patch location isn't uniquely identified; the model
        should extend BEFORE with surrounding context.

    Matching is two-tier (Phase 3.6 deterministic anchor resolution):

    1. byte-exact — whitespace, line endings, indentation all count.
    2. whitespace-normalized line sequence — CRLF/CR unified, trailing
       whitespace per line ignored, leading/trailing blank lines of the
       BEFORE block ignored. The normalized sequence must still resolve to
       EXACTLY ONE span of whole lines in the current scaffold; the splice
       happens on the original lines, so text outside the span is preserved
       byte-exactly.

    Tier 2 exists because the decider copies BEFORE from its prompt and
    drifts on invisible whitespace — observed live as
    ``before_block_mismatch:not_found`` on otherwise-verbatim blocks. The
    Phase 3.5 design property survives: the BEFORE block's (normalized)
    size still physically bounds the change; hallucinated text still
    rejects loudly.
    """
    if not before:
        return None, "before_block_empty"
    count = current.count(before)
    if count == 1:
        return current.replace(before, after, 1), None
    if count > 1:
        return None, "before_block_mismatch:ambiguous"

    # Tier 2: whitespace-normalized whole-line resolution.
    cur_lines = current.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    cur_norm = [ln.rstrip() for ln in cur_lines]
    bef_norm = [ln.rstrip() for ln in
                before.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    while bef_norm and bef_norm[0] == "":
        bef_norm.pop(0)
    while bef_norm and bef_norm[-1] == "":
        bef_norm.pop()
    if not bef_norm:
        return None, "before_block_empty"
    span = len(bef_norm)
    matches = [i for i in range(len(cur_norm) - span + 1)
               if cur_norm[i:i + span] == bef_norm]
    if not matches:
        return None, "before_block_mismatch:not_found"
    if len(matches) > 1:
        return None, "before_block_mismatch:ambiguous"
    i = matches[0]
    after_text = after.replace("\r\n", "\n").replace("\r", "\n")
    if after_text.endswith("\n"):
        after_text = after_text[:-1]
    segment = after_text.split("\n") if after_text else []
    new_lines = cur_lines[:i] + segment + cur_lines[i + span:]
    return "\n".join(new_lines), None


# ── bounded BEFORE-anchor repair (Phase 3.7) ──────────────────────────


def _section_excerpt(current: str, target_section: str | None) -> str:
    """Exact current text of the region the model targeted.

    Resolves an H2 section via :func:`_parse_sections`, an H3 subsection via
    a header slice (the model's TARGET_SECTION frequently names ``Hardening``
    etc., which live under ``## Audit``), and falls back to the whole
    scaffold when the name resolves to nothing.
    """
    if target_section:
        secs = _parse_sections(current)
        if target_section in secs:
            return f"## {target_section}\n{secs[target_section]}".rstrip()
        lines = current.splitlines()
        for i, ln in enumerate(lines):
            if ln.strip() in (f"### {target_section}", f"## {target_section}"):
                body = []
                for nxt in lines[i + 1:]:
                    if nxt.startswith("## ") or nxt.startswith("### "):
                        break
                    body.append(nxt)
                return "\n".join([ln] + body).rstrip()
    return current


def _repair_before_block(current, parsed, call_llm):
    """ONE bounded retry when the model hallucinated its BEFORE anchor.

    Observed live (journal, 5×): the decider names the right TARGET_SECTION
    but mis-quotes the text — merging its intended edit into BEFORE, or
    inventing section adjacency. Rather than trusting model memory, hand it
    the targeted region's CURRENT text verbatim and demand a re-emit with
    BEFORE copied exactly. Deterministic tooling stays the judge: the caller
    re-runs :func:`_apply_patch` on the result.

    Returns ``(reparsed, raw)`` — ``reparsed`` is None when the retry could
    not produce a schema-valid patch (caller journals the original reject).
    """
    excerpt = _section_excerpt(current, parsed.get("target_section"))
    prompt = (
        "Your previous patch was REJECTED: the BEFORE block does not appear in "
        "the current scaffold (you mis-quoted it). Here is the CURRENT text of "
        "the region you targeted — copy BEFORE from it character-for-character; "
        "do not edit, trim, or merge your change into BEFORE (your change goes "
        "in AFTER only).\n\n"
        "Current text:\n"
        "```\n"
        f"{excerpt}\n"
        "```\n\n"
        "Your intended replacement (AFTER) was:\n"
        "```\n"
        f"{parsed.get('after', '')}\n"
        "```\n\n"
        "Re-emit the FULL patch in the exact same format as before (=== PATCH === "
        "sentinel, all field labels, BEFORE copied exactly from the text above, "
        "AFTER carrying your change). If the change no longer makes sense against "
        "the current text, use PATCH_MODE: no_change."
    )
    raw = call_llm(prompt)
    if raw is None:
        return None, None
    reparsed, parse_err = _parse_evolution_response(raw)
    if parse_err is not None:
        return None, raw
    return reparsed, raw


# ── diff classifier (Phase 3b) ────────────────────────────────────────


def _classify_diff(old: str, new: str) -> str:
    """Measure the actual scope of a proposed change.

    Hybrid classifier — uses Phase 2b's section parser AND a char-diff
    threshold. Returns one of: ``"none" | "line" | "section" | "structural"``.

    Rules (first match wins):
      1. ``old.strip() == new.strip()``                     → ``"none"``
      2. Header set differs (added/removed/renamed)         → ``"structural"``
      3. Multiple sections' bodies changed (span > 1)       → ``"section"``
      4. Single section's body changed, char_diff ≤ _DIFF_CAP → ``"line"``
      5. Single section's body changed, char_diff > _DIFF_CAP → ``"section"``

    Rule 5 is what catches the empirical pattern from real ratings: the
    model produces 3000+ char diffs while self-labeling them "line".
    Those reclassify as section before the policy table runs, so the
    `proposed=line × actual=section` cell fires and the rejection is
    scope_mismatch, not a misleading diff_cap_exceeded.

    No-section corner case: if both old and new have no `##` headers at all
    (no sections to compare), span is 0 and the char-diff threshold decides
    between line and section. This mainly affects test scaffolds; real
    monothink scaffolds always carry sections.
    """
    if old.strip() == new.strip():
        return "none"

    old_sections = _parse_sections(old)
    new_sections = _parse_sections(new)

    if set(old_sections.keys()) != set(new_sections.keys()):
        return "structural"

    span = _diff_section_span(old, new)
    if span > 1:
        return "section"

    char_diff = _diff_chars(old, new)
    if char_diff <= _DIFF_CAP:
        return "line"
    return "section"


# ── typed mutation authority (Phase 3a) ──────────────────────────────


# Decision tokens. Strings, not enums, so they show up directly in journal
# reject_reason fields without serialization gymnastics.
DECISION_APPLY = "APPLY"
DECISION_PROPOSE = "PROPOSE"
DECISION_NO_CHANGE = "NO_CHANGE"
# REJECT_CAP / REJECT_SCOPE_MISMATCH were removed (E, 2026-06-25): the policy
# decides from ACTUAL scope, so a proposed/actual disagreement no longer
# DISCARDS an edit — it is logged as a mismatch and the edit is governed by
# its actual scope (apply if it clears a gate, else propose). Less wasted
# evidence; the gates themselves are unchanged.


def _decide_policy(actual_scope: str, concern_repeated: bool) -> str:
    """Map the code-measured ACTUAL scope → decision. The model's
    PROPOSED_SCOPE is NOT an input (E, 2026-06-25): the code is the
    authority — it measures actual scope via :func:`_classify_diff` and
    decides from that. A wrong proposed label is logged as a mismatch by the
    caller, never used to discard a valid catch.

    | actual classifier says | gate             | decision  |
    |------------------------|------------------|-----------|
    | none                   | —                | NO_CHANGE |
    | line                   | —                | APPLY     |  (line ⇔ char_diff ≤ _DIFF_CAP)
    | section                | concern_repeated | APPLY     |
    | section                | not repeated     | PROPOSE   |
    | structural             | —                | PROPOSE   |

    Not more permissive: the gates are unchanged — `line` already means
    char_diff ≤ _DIFF_CAP (the classifier's definition), `section` still
    requires recurrence evidence to auto-apply, and `structural` never
    auto-applies. Deciding from actual only stops a wrong PROPOSED_SCOPE
    label from discarding an edit whose actual scope already clears a gate.
    The "model proposes, code decides" asymmetry is preserved and in fact
    sharpened: the model cannot influence the decision by mislabeling — only
    by changing what it actually writes, which the classifier measures.
    """
    if actual_scope == "none":
        return DECISION_NO_CHANGE
    if actual_scope == "line":
        return DECISION_APPLY
    if actual_scope == "section":
        return DECISION_APPLY if concern_repeated else DECISION_PROPOSE
    if actual_scope == "structural":
        return DECISION_PROPOSE
    # Defensive: unknown actual_scope (the classifier only emits the four above).
    return DECISION_PROPOSE


# ── response parser (Phase 1b) ──────────────────────────────────────


# Single-line fields. Match label at start of line, capture the value, stop
# at end of line. Defensive about trailing whitespace.
_SINGLE_LINE_FIELD_RE = re.compile(
    r"^\s*(PRIMARY_FAILURE_TAG|PRIMARY_FAILURE|TARGET_SECTION|PROPOSED_SCOPE|PATCH_MODE|TARGET_LINES)\s*:\s*(.*?)\s*$",
    flags=re.MULTILINE,
)

# DEFERRED_CONCERNS spans from its label to the start of PATCH_MODE (or EOF).
# Body may be empty — that's "no deferred concerns". The label itself MUST
# be present (we want explicit empty list, not implicit missing field).
_DEFERRED_BLOCK_RE = re.compile(
    r"^DEFERRED_CONCERNS\s*:\s*(.*?)(?=^PATCH_MODE\s*:|\Z)",
    flags=re.MULTILINE | re.DOTALL,
)

# BEFORE block: from `BEFORE:\n` to `AFTER:` at start of line (or EOF).
# AFTER block: from `AFTER:\n` to end of input. No stripping — the model is
# expected to produce byte-exact content; trailing newlines are preserved
# so they can match the scaffold's actual newline structure.
_BEFORE_BLOCK_RE = re.compile(
    r"^BEFORE\s*:\s*\n?(.*?)(?=^AFTER\s*:|\Z)",
    flags=re.MULTILINE | re.DOTALL,
)
_AFTER_BLOCK_RE = re.compile(
    r"^AFTER\s*:\s*\n?(.*)",
    flags=re.MULTILINE | re.DOTALL,
)

_DECISION_FIELD_RE = re.compile(
    r"^\s*(PRIMARY_FAILURE_TAG|PRIMARY_FAILURE|FAILURE_SIGNATURE|SIGNATURE_STATUS|"
    r"LEDGER_LOOKUP|EVIDENCE_CLASS|ALREADY_COVERED|FIX_KIND|MUTATION_DECISION|"
    r"DECISION_REASON|PATCH_MODE|DEFERRED_CONCERN|TARGET_SECTION|TARGET_LINES|"
    r"PROPOSED_SCOPE|STRUCTURAL_LOCATION|PREDICTED_EFFECT|FALSIFIER|APPLIED_AT_TURN|"
    r"INVARIANT_CITE|TRACE_SPAN|VIOLATION_FORM)\s*:\s*(.*?)\s*$",
    flags=re.MULTILINE,
)
_DECISION_BEFORE_RE = re.compile(
    r"^BEFORE\s*:\s*\n?(.*?)(?=^AFTER\s*:|\Z)",
    flags=re.MULTILINE | re.DOTALL,
)
_DECISION_AFTER_RE = re.compile(
    r"^AFTER\s*:\s*\n?(.*?)(?=^INVARIANT_CITE\s*:|^TRACE_SPAN\s*:|^VIOLATION_FORM\s*:|\Z)",
    flags=re.MULTILINE | re.DOTALL,
)


def _covered_section(value: str | None) -> str | None:
    text = str(value or "").strip()
    if not text.lower().startswith("yes"):
        return None
    m = re.search(r"\((.*?)\)", text)
    if not m:
        return ""
    return m.group(1).strip()


def _parse_decision_response(raw: str) -> tuple[dict | None, str | None]:
    text = _strip_llm_call_header(raw)
    text = _strip_code_fences(text).strip()
    if _DECISION_SENTINEL not in text:
        return None, "schema_violation:decision_sentinel"
    decision_text = text.rsplit(_DECISION_SENTINEL, 1)[1].strip()
    fields: dict[str, str] = {}
    for match in _DECISION_FIELD_RE.finditer(decision_text):
        fields.setdefault(match.group(1), match.group(2).strip())

    required = (
        "PRIMARY_FAILURE_TAG",
        "PRIMARY_FAILURE",
        "FAILURE_SIGNATURE",
        "SIGNATURE_STATUS",
        "LEDGER_LOOKUP",
        "EVIDENCE_CLASS",
        "ALREADY_COVERED",
        "FIX_KIND",
        "MUTATION_DECISION",
        "DECISION_REASON",
        "PATCH_MODE",
    )
    for label in required:
        if not fields.get(label):
            return None, f"schema_violation:{label.lower()}"

    tag = fields["PRIMARY_FAILURE_TAG"]
    if len(tag) > _TAG_MAX_LEN:
        return None, "schema_violation:tag_too_long"
    if tag not in _failure_tags.FAILURE_TAGS:
        return None, "schema_violation:tag_invalid"

    signature_status = fields["SIGNATURE_STATUS"].lower()
    if signature_status not in {"provisional", "canonical"}:
        return None, "schema_violation:signature_status"
    evidence_class = fields["EVIDENCE_CLASS"].lower()
    if evidence_class not in {"single_instance", "repeated_pattern", "critical_invariant_break"}:
        return None, "schema_violation:evidence_class"
    fix_kind = fields["FIX_KIND"].lower()
    if fix_kind not in {"scaffold_gap", "enforcement_gap", "rating_misapplied"}:
        return None, "schema_violation:fix_kind"
    mutation = fields["MUTATION_DECISION"].lower()
    if mutation not in {"reserve", "apply", "reject"}:
        return None, "schema_violation:mutation_decision"

    patch_mode = fields["PATCH_MODE"].lower()
    if mutation != "apply":
        if patch_mode != "no_change":
            return None, "schema_violation:patch_mode_invalid"
        return {
            "schema_version": "decision_v2",
            "primary_failure_tag": tag,
            "adjudicated_tag": tag,
            "primary_failure": fields["PRIMARY_FAILURE"],
            "failure_signature": fields["FAILURE_SIGNATURE"],
            "signature_status": signature_status,
            "ledger_lookup": fields["LEDGER_LOOKUP"],
            "evidence_class": evidence_class,
            "already_covered": fields["ALREADY_COVERED"],
            "covered_section": _covered_section(fields["ALREADY_COVERED"]),
            "fix_kind": fix_kind,
            "mutation_decision": mutation,
            "decision_reason": fields["DECISION_REASON"],
            "deferred_concern": fields.get("DEFERRED_CONCERN", "NONE"),
            "target_section": _covered_section(fields["ALREADY_COVERED"]),
            "proposed_scope": "none",
            "deferred_concerns": [fields.get("DEFERRED_CONCERN", "")],
            "patch_mode": "no_change",
            "target_lines": None,
            "before": "",
            "after": "",
        }, None

    # Apply path.
    for label in (
        "TARGET_SECTION", "TARGET_LINES", "PROPOSED_SCOPE", "STRUCTURAL_LOCATION",
        "PREDICTED_EFFECT", "FALSIFIER", "APPLIED_AT_TURN",
    ):
        if not fields.get(label):
            return None, f"schema_violation:{label.lower()}"
    scope = fields["PROPOSED_SCOPE"].lower()
    if scope not in _HOT_SCOPE_ENUM:
        return None, "schema_violation:scope_invalid"
    if patch_mode not in {"replace_lines", "replace_section"}:
        return None, "schema_violation:patch_mode_invalid"
    bm = _DECISION_BEFORE_RE.search(decision_text)
    am = _DECISION_AFTER_RE.search(decision_text)
    if bm is None:
        return None, "schema_violation:before"
    if am is None:
        return None, "schema_violation:after"
    before = bm.group(1)
    after = am.group(1)
    if before.endswith("\n"):
        before = before[:-1]
    if after.endswith("\n"):
        after = after[:-1]
    if not before.strip():
        return None, "schema_violation:before"

    return {
        "schema_version": "decision_v2",
        "primary_failure_tag": tag,
        "adjudicated_tag": tag,
        "primary_failure": fields["PRIMARY_FAILURE"],
        "failure_signature": fields["FAILURE_SIGNATURE"],
        "signature_status": signature_status,
        "ledger_lookup": fields["LEDGER_LOOKUP"],
        "evidence_class": evidence_class,
        "already_covered": fields["ALREADY_COVERED"],
        "covered_section": _covered_section(fields["ALREADY_COVERED"]),
        "fix_kind": fix_kind,
        "mutation_decision": mutation,
        "decision_reason": fields["DECISION_REASON"],
        "deferred_concern": fields.get("DEFERRED_CONCERN", ""),
        "target_section": fields["TARGET_SECTION"],
        "proposed_scope": scope,
        "deferred_concerns": [],
        "patch_mode": patch_mode,
        "target_lines": fields["TARGET_LINES"],
        "structural_location": fields["STRUCTURAL_LOCATION"],
        "predicted_effect": fields["PREDICTED_EFFECT"],
        "falsifier": fields["FALSIFIER"],
        "applied_at_turn": fields["APPLIED_AT_TURN"],
        "invariant_cite": fields.get("INVARIANT_CITE", ""),
        "trace_span": fields.get("TRACE_SPAN", ""),
        "violation_form": fields.get("VIOLATION_FORM", ""),
        "before": before,
        "after": after,
    }, None


def _parse_evolution_response(raw: str) -> tuple[dict | None, str | None]:
    """Parse the LLM's response per the Phase 1a schema.

    Returns ``(parsed, None)`` on success, ``(None, "schema_violation:<...>")``
    on failure. The error string is the journal ``reject_reason`` directly.

    Strict about:
      * All four single-line fields present and non-empty
        (PRIMARY_FAILURE_TAG, PRIMARY_FAILURE, TARGET_SECTION, PROPOSED_SCOPE).
      * DEFERRED_CONCERNS label present (its body can be empty).
      * PATCH label present with non-empty body.
      * PROPOSED_SCOPE is one of {line, section, structural, none}.
      * PRIMARY_FAILURE_TAG ≤32 chars.

    Lenient about:
      * Field order (single-line fields can interleave).
      * Outer code fences wrapping the whole response.
      * Trailing/leading whitespace on field values.
      * TARGET_SECTION="NONE" (case-insensitive) → normalized to ``None``.
      * PATCH value on same line as label OR on the next line.

    The parser exists to give downstream code structured data instead of a
    raw blob. If parsing fails, ``raw_response_full`` is still journaled
    (forensic backup); if parsing succeeds, the structured fields are the
    artifact and ``raw_response_full`` becomes the audit trail of "what the
    model literally said before we structured it."
    """
    if not raw:
        return None, "schema_violation:empty"

    text = _strip_llm_call_header(raw)
    text = _strip_code_fences(text).strip()
    if not text:
        return None, "schema_violation:empty"

    if _DECISION_SENTINEL in text:
        return _parse_decision_response(text)

    # Isolate the schema from any preceding deliberation prose: parse only what
    # follows the LAST sentinel. Deliberation prose can contain field-looking
    # lines (it argues about scope, names tags); without this the field scan
    # below would grab them. Absent (fixtures/legacy) the whole text is parsed.
    if _SCHEMA_SENTINEL in text:
        text = text.rsplit(_SCHEMA_SENTINEL, 1)[1].strip()
        if not text:
            return None, "schema_violation:empty"

    fields: dict[str, str] = {}
    for m in _SINGLE_LINE_FIELD_RE.finditer(text):
        label, value = m.group(1), m.group(2).strip()
        # First occurrence wins — defensive against the model echoing labels.
        fields.setdefault(label, value)

    for required in (
        "PRIMARY_FAILURE_TAG",
        "PRIMARY_FAILURE",
        "TARGET_SECTION",
        "PROPOSED_SCOPE",
        "PATCH_MODE",
        "TARGET_LINES",
    ):
        if required not in fields or not fields[required]:
            return None, f"schema_violation:{required.lower()}"

    tag = fields["PRIMARY_FAILURE_TAG"]
    if len(tag) > _TAG_MAX_LEN:
        return None, "schema_violation:tag_too_long"

    scope = fields["PROPOSED_SCOPE"].lower()
    if scope not in _SCOPE_ENUM:
        return None, "schema_violation:scope_invalid"

    patch_mode = fields["PATCH_MODE"].lower()
    if patch_mode not in _PATCH_MODE_ENUM:
        return None, "schema_violation:patch_mode_invalid"

    target_raw = fields["TARGET_SECTION"]
    target_section = None if target_raw.upper() == "NONE" else target_raw

    target_lines_raw = fields["TARGET_LINES"]
    target_lines = None if target_lines_raw.upper() == "NONE" else target_lines_raw

    dm = _DEFERRED_BLOCK_RE.search(text)
    if dm is None:
        return None, "schema_violation:deferred_concerns"
    deferred: list[str] = []
    for line in dm.group(1).splitlines():
        line = line.strip()
        if line.startswith("- "):
            deferred.append(line[2:].strip())
        elif line.startswith("-") and len(line) > 1 and line[1] != "-":
            # Bare "- item" without space — be tolerant.
            deferred.append(line[1:].strip())

    # BEFORE/AFTER blocks. Both must be present (the label, at least) even on
    # no_change — keeps the parser contract uniform; values may be empty.
    bm = _BEFORE_BLOCK_RE.search(text)
    if bm is None:
        return None, "schema_violation:before"
    am = _AFTER_BLOCK_RE.search(text)
    if am is None:
        return None, "schema_violation:after"

    # Capture content as-is, then ONE-pass rstrip to drop only the
    # newline immediately preceding the next field label (or EOF). The
    # trailing-newline drop matters because the regex always captures a
    # final \n before the lookahead; failing to drop it would make
    # BEFORE blocks fail to match scaffolds whose target text doesn't
    # end with a newline.
    before = bm.group(1)
    after = am.group(1)
    if before.endswith("\n"):
        before = before[:-1]
    if after.endswith("\n"):
        after = after[:-1]

    # On no_change, BEFORE and AFTER may be empty; on any other mode they
    # must carry content (the worker's _apply_patch enforces this with
    # before_block_empty too, but the parser short-circuits earlier).
    if patch_mode != "no_change":
        if not before.strip():
            return None, "schema_violation:before"
        # AFTER may legitimately be empty (deletion of a line/section).
        # Distinguish "empty" (intentional) from "missing" (schema bug)
        # by trusting the label being present — handled above.

    return {
        "primary_failure_tag": tag,
        "adjudicated_tag": tag,
        "primary_failure": fields["PRIMARY_FAILURE"],
        "target_section": target_section,
        "proposed_scope": scope,
        "deferred_concerns": deferred,
        "patch_mode": patch_mode,
        "target_lines": target_lines,
        "before": before,
        "after": after,
    }, None


def _call_llm(prompt: str) -> str | None:
    """Direct, STATELESS, COMPLETE (non-streaming) LLM call for scaffold edits.
    Returns raw text or None on failure. Broad except — evolution must never
    break the rating loop.

    STATELESS: sends ONLY the self-contained evolution prompt as a single user
    message — no Monolith system prompt, identity, or history. We read llm
    config directly and deliberately skip ``build_system_prompt`` (the old
    ``load_config()`` assembled a ~51k-char system prompt this path never sent —
    pure waste; there's no KV-cache prefix to preserve on a cloud backend).

    COMPLETE (``stream=False``): one atomic read, hard-bounded by the client
    socket timeout, so it cannot trickle-hang the way the in-process STREAMED
    iterator did — a live evolution streamed inside the app hung >10min with no
    terminal entry, while this complete call returns in ~150s (measured). The
    non-stream path also reads only ``message.content`` (the final patch),
    DROPPING ``reasoning_content``: thinking backends (DeepSeek V4 ignores
    ``enable_thinking=False`` and emits ~20k chars of <think>) otherwise let the
    reasoning compete for the token budget and truncate the patch
    (schema_violation:after / empty_response on past live ratings). max_tokens
    stays generous so the content patch always fits after the model's think.
    """
    try:
        from core.config import get_config
        from engine.llm import OpenAICompatLLM
        cfg = get_config().llm.model_dump()
        api_base = str(cfg.get("api_base", "") or "").strip()
        api_model = str(cfg.get("api_model", "") or "").strip()
        if not api_base or not api_model:
            return None
        client = OpenAICompatLLM(api_base, str(cfg.get("api_key", "") or ""), api_model)
        parts: list[str] = []
        for chunk in client.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=16384,
            stream=False,
        ):
            delta = (chunk.get("choices", [{}]) or [{}])[0].get("delta", {})
            if delta.get("content"):
                parts.append(delta["content"])
        return "".join(parts)
    except Exception:
        return None


def _strip_llm_call_header(text: str) -> str:
    """If the response starts with the [llm_call: ...] envelope header, strip
    it. Defensive — direct generate_sync_from_config shouldn't add the header,
    but some routing paths might."""
    if text.startswith(_LLM_CALL_HEADER_RE_PREFIX):
        nl = text.find("\n")
        if nl > 0:
            return text[nl + 1:].lstrip()
    return text


def _is_no_change(text: str) -> bool:
    r"""Robust NO_CHANGE detector. The model may decorate, downcase, or append
    reasoning; treat any first-line variant of the token as NO_CHANGE.

    Accepts (all return True):
      NO_CHANGE, no_change, **NO_CHANGE**, `NO_CHANGE`, "NO_CHANGE",
      NO_CHANGE., NO_CHANGE:, NO_CHANGE\nbecause... (token + newline + explanation).
    """
    if not text:
        return False
    first_line = text.splitlines()[0] if text else ""
    canonical = first_line.strip()
    # Strip common wrapping characters (markdown bold, code spans, quotes).
    canonical = canonical.strip("*`\"'")
    # Strip trailing punctuation/whitespace that might follow the token.
    canonical = canonical.rstrip(".,:;!? \t-—")
    canonical = canonical.strip()
    return canonical.upper() == _NO_CHANGE_TOKEN


def _strip_code_fences(text: str) -> str:
    """If the model wrapped the scaffold in ``` fences, strip them."""
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        # Drop the opening fence (and any language tag on the same line) and
        # the closing fence.
        first_nl = stripped.find("\n")
        if first_nl > 0:
            inner = stripped[first_nl + 1:]
            if inner.endswith("```"):
                inner = inner[:-3]
            return inner.strip()
    return text


# ── lookup: turn → monothink active ──────────────────────────────────


def _lookup_turn_monothink_active(turn_id: str) -> bool:
    """Check whether monothink was active for the given turn.

    Tries the new monothink_active column first (from the /prompt
    consolidation); falls back to the legacy reasoning_mode == "monothink"
    check for rows written before the migration.
    """
    try:
        from core.turn_trace import get_turn_monothink_active
        if get_turn_monothink_active(turn_id):
            return True
    except Exception:
        pass
    # Legacy fallback: pre-migration rows have reasoning_mode but not
    # monothink_active. Check the old column so evolution still works
    # on turns recorded before the schema migration.
    try:
        from core.turn_trace import get_turn_reasoning_mode
        return get_turn_reasoning_mode(turn_id) == "monothink"
    except Exception:
        return False


# ── journal entry construction ───────────────────────────────────────


def _make_journal_entry(turn_id, rating_value, rating_reason, old, new,
                         applied, reject_reason, *,
                         raw_response_full: str | None = None,
                         parsed: dict | None = None,
                         applied_scaffold_full: str | None = None,
                         parent_scaffold_version: str | None = None,
                         failure_tags: list[str] | None = None,
                         actual_scope: str | None = None,
                         contrast_summary: dict | None = None):
    """Build a journal entry.

    ``raw_response_full`` (when provided) carries the FULL LLM output for the
    evolution call. Set on every rejection path so cap-tuning and post-hoc
    analysis can inspect what the model actually produced. Bounded by
    ``max_tokens=8192`` ≈ ~32KB per journal line worst-case; multi-MB
    journal growth over hundreds of rejections is acceptable for the
    forensic value it buys.

    A derived ``raw_preview`` field (last 240 chars with leading "…" when
    truncated) is also written for quick scan-by-eye and back-compat with
    the pre-Phase-0a readers that only knew about that field.

    Phase 0a of the MonoThink evolution plan introduced ``raw_response_full``
    after two consecutive rejections journaled `empty_response` with only
    240 chars of tail visible — too little to tell whether the LLM
    truly returned nothing or returned something the stripping logic ate.
    """
    entry = {
        "ts": _now_iso(),
        "turn_id": str(turn_id),
        "rating_value": (int(rating_value)
                         if isinstance(rating_value, (int, float))
                         else None),
        "rating_reason": str(rating_reason) if rating_reason is not None else "",
        "old_chars": len(old),
        "new_chars": len(new),
        "diff_chars": _diff_chars(old, new),
        "applied": bool(applied),
        "reject_reason": reject_reason,
    }
    if raw_response_full is not None:
        full = str(raw_response_full)
        entry["raw_response_full"] = full
        # Derive the preview: full text when short, ellipsized tail when long.
        # Kept identical to the prior `raw_preview` semantics so existing
        # readers (forensic scripts, eye-scans of the journal) don't break.
        if len(full) > 240:
            entry["raw_preview"] = "…" + full[-239:]
        else:
            entry["raw_preview"] = full
    # Phase 1b: structured fields from the parsed response. Pass the whole
    # ``parsed`` dict (or None) — not individual kwargs — so we can
    # distinguish "parser hasn't run / failed" from "parser succeeded with
    # target_section=None (model wrote NONE)". With a per-field None
    # sentinel those two states are indistinguishable; with a dict-or-None
    # gate they're clean. Once parsing succeeded, ALL five fields land in
    # the entry, even when their values are None or [] — that's what makes
    # the journal shape uniform past the parser.
    if parsed is not None:
        if parsed.get("schema_version"):
            entry["schema_version"] = parsed.get("schema_version")
        entry["primary_failure_tag"] = parsed["primary_failure_tag"]
        entry["primary_failure"] = parsed["primary_failure"]
        entry["target_section"] = parsed["target_section"]  # may be None
        entry["proposed_scope"] = parsed["proposed_scope"]
        entry["deferred_concerns"] = list(parsed["deferred_concerns"])
        for key in (
            "failure_signature",
            "signature_status",
            "ledger_lookup",
            "evidence_class",
            "already_covered",
            "covered_section",
            "fix_kind",
            "mutation_decision",
            "decision_reason",
            "deferred_concern",
            "structural_location",
            "predicted_effect",
            "falsifier",
            "applied_at_turn",
            "invariant_cite",
            "trace_span",
            "violation_form",
        ):
            if key in parsed:
                entry[key] = parsed.get(key)
        # Decide-from-actual telemetry (E, 2026-06-25): record the code-measured
        # actual scope and whether the model's proposed label disagreed. The
        # mismatch is LOGGED here, never used as a discard reason.
        if actual_scope is not None:
            entry["actual_scope"] = actual_scope
            entry["scope_mismatch"] = parsed["proposed_scope"] != actual_scope
    # Phase 3e: rollback substrate. applied_scaffold_full carries the new
    # scaffold text on every applied=True entry so revert can restore the
    # previous version without reading the on-disk file (which may have
    # been further modified by a later evolution). parent_scaffold_version
    # forms the chain — the turn_id of the most recent prior applied=True
    # entry (including bootstrap), null only on the very first applied
    # entry post-bootstrap.
    if applied_scaffold_full is not None:
        entry["applied_scaffold_full"] = applied_scaffold_full
    if parent_scaffold_version is not None:
        entry["parent_scaffold_version"] = parent_scaffold_version
    entry["failure_tags"] = list(failure_tags) if failure_tags else []
    entry["rater_tag"] = failure_tags[0] if failure_tags else None
    if parsed is not None:
        adjudicated = parsed["primary_failure_tag"]
        entry["adjudicated_tag"] = adjudicated
        entry["divergent"] = bool(failure_tags) and adjudicated not in failure_tags
    if contrast_summary:
        entry.update(dict(contrast_summary))
    return entry


# ── canary fast-check (Phase 3d) ──────────────────────────────────────


def _fast_check_scaffold(text: str) -> str | None:
    """Pre-apply structural sanity check. Returns ``None`` on pass, or a
    short error string suitable for ``reject_reason``.

    Checks (cheap, deterministic):
      * size cap not exceeded
      * triple-backtick fences are balanced (no half-fenced code block)
      * scaffold parses into at least one ``##`` section
      * preamble (anything before the first ``##``) is allowed; only the
        first non-blank line may be an ``#`` heading, otherwise it's orphan
      * each parsed section's body is non-empty after strip

    NOT enforced here: section-membership of non-protected sections (per
    G6 of the v1→v2 review — the model is allowed to remove or rename
    non-protected sections; only `_PROTECTED_SECTIONS` are content-locked,
    which is enforced separately in the worker's protected_section check).

    Background LLM canary is deliberately NOT in this function. That's
    the next ship (Phase 3d.2) and runs asynchronously after a successful
    APPLY; it can revert via ``rollback_last_apply`` if predicates regress
    against the baseline. The fast check here is what gates the apply
    itself.
    """
    if len(text) > _SIZE_CAP:
        return f"canary_fast_fail:size_cap_exceeded:{len(text)}>{_SIZE_CAP}"

    # Triple-backtick parity. Each opening fence has a closing fence.
    fence_count = 0
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            fence_count += 1
    if fence_count % 2 != 0:
        return "canary_fast_fail:half_fenced_code_block"

    sections = _parse_sections(text)
    if not sections:
        return "canary_fast_fail:no_sections"

    # Empty section bodies are suspicious (model dropped content under a
    # header). `[tbd]` style placeholders are fine — they're not empty.
    for name, body in sections.items():
        if not body.strip():
            return f"canary_fast_fail:empty_section:{name}"

    return None


# ── repeated-concern detector (Phase 3c) ──────────────────────────────


def _concern_repeated(
    tag: str | None,
    current_turn_id: str,
    lookback_n: int = 5,
) -> bool:
    """True iff at least one prior journal entry within the last
    ``lookback_n`` DISTINCT turn_ids has matching ``primary_failure_tag``
    AND a turn_id different from ``current_turn_id``.

    Strict equality on the tag — no fuzzy matching, no deferred-concerns
    fallback. If the gate never fires across 20+ post-ship turns, that's
    evidence the assumption ("concerns recur on distinct turns") doesn't
    match how the user uses monothink, and the design needs to change —
    NOT be patched with similarity matching that masks the signal.

    The "distinct turn_ids" constraint matters because debug-retry
    sequences (re-rating the same prompt over and over) would otherwise
    appear to "repeat" the concern and unjustifiedly promote
    section-scope to APPLY.
    """
    if not tag or not _JOURNAL_PATH.exists():
        return False
    try:
        lines = _JOURNAL_PATH.read_text(encoding="utf-8").splitlines()
    except Exception:
        return False
    distinct_seen: list[str] = []  # most-recent first, dedup'd
    matching_turn_ids: set[str] = set()
    for line in reversed(lines):
        try:
            entry = json.loads(line)
        except Exception:
            continue
        # Skip bootstrap/rollback rows — they don't carry primary_failure_tag.
        if entry.get("kind") in ("bootstrap", "rollback"):
            continue
        tid = entry.get("turn_id")
        if not tid or str(tid) == str(current_turn_id):
            continue
        if tid not in distinct_seen:
            distinct_seen.append(tid)
        if len(distinct_seen) > lookback_n:
            break
        if (entry.get("adjudicated_tag") or entry.get("primary_failure_tag")) == tag:
            matching_turn_ids.add(tid)
    return len(matching_turn_ids) >= 1


# ── rollback substrate (Phase 3e) ─────────────────────────────────────


_BOOTSTRAP_KIND = "bootstrap"


def _journal_has_bootstrap_entry() -> bool:
    """True iff the journal already contains an entry with ``kind=bootstrap``.

    The bootstrap entry is the rollback floor — the recorded state of the
    scaffold before any evolution attempts. We need exactly one. The check
    scans the journal because the in-memory state isn't authoritative;
    multiple processes could in principle race on first-import.
    """
    if not _JOURNAL_PATH.exists():
        return False
    try:
        with _JOURNAL_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                if '"kind"' not in line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                if entry.get("kind") == _BOOTSTRAP_KIND:
                    return True
    except Exception:
        return False
    return False


def _ensure_bootstrap_entry() -> None:
    """Write a single bootstrap journal entry capturing the current on-disk
    scaffold. Idempotent — re-running is a no-op if one already exists.

    Called lazily at the start of an evolution attempt (not at module import)
    so test environments with monkeypatched paths still work cleanly, and
    so import-time has no IO side effects.
    """
    if _journal_has_bootstrap_entry():
        return
    try:
        scaffold = read_scaffold()
        ts = _now_iso()
        # Bootstrap entry has a synthesized turn_id of the form
        # "bootstrap-<isoformat>". Distinguishable from real UUIDs by
        # the literal prefix.
        entry = {
            "ts": ts,
            "turn_id": f"bootstrap-{ts}",
            "kind": _BOOTSTRAP_KIND,
            "applied": True,
            "applied_scaffold_full": scaffold,
            "parent_scaffold_version": None,
            "reject_reason": None,
            "old_chars": 0,
            "new_chars": len(scaffold),
            "diff_chars": 0,
            "rating_value": None,
            "rating_reason": "",
        }
        _append_journal(entry)
    except Exception:
        # Best-effort — never break the evolution path on bootstrap failure.
        pass


def rollback_last_apply(reason: str = "manual") -> dict | None:
    """Restore the scaffold to the state it was in before the last applied
    evolution. Used by Phase 3d's background canary on regression detection,
    and available as a manual operator override.

    Algorithm: find the most recent applied=True entry, follow its
    parent_scaffold_version pointer back to the prior applied entry, and
    restore that entry's applied_scaffold_full. Journal a new
    ``rollback_from:<turn_id>`` entry to record the action.

    Returns the rollback entry on success, ``None`` if there's nothing to
    roll back to (e.g., the journal only contains bootstrap or no entries).
    """
    try:
        if not _JOURNAL_PATH.exists():
            return None
        lines = _JOURNAL_PATH.read_text(encoding="utf-8").splitlines()
        # Find most recent applied=True entry that isn't a bootstrap.
        target = None
        for line in reversed(lines):
            if '"applied": true' not in line and '"applied":true' not in line:
                continue
            try:
                entry = json.loads(line)
            except Exception:
                continue
            if entry.get("applied") is True and entry.get("kind") != _BOOTSTRAP_KIND:
                target = entry
                break
        if target is None:
            return None  # nothing applied; nothing to roll back
        # Resolve parent
        parent_tid = target.get("parent_scaffold_version")
        if not parent_tid:
            return None
        # Find parent entry's applied_scaffold_full
        parent_scaffold = None
        for line in lines:
            if parent_tid not in line:
                continue
            try:
                entry = json.loads(line)
            except Exception:
                continue
            if entry.get("turn_id") == parent_tid and entry.get("applied") is True:
                parent_scaffold = entry.get("applied_scaffold_full")
                break
        if parent_scaffold is None:
            return None
        # Write parent scaffold back.
        _write_scaffold(parent_scaffold)
        rollback_entry = {
            "ts": _now_iso(),
            "turn_id": f"rollback-{_now_iso()}",
            "kind": "rollback",
            "applied": True,
            "applied_scaffold_full": parent_scaffold,
            "parent_scaffold_version": parent_tid,
            "rolled_back_from": target.get("turn_id"),
            "reject_reason": f"rollback_from:{target.get('turn_id')}:{reason}",
            "old_chars": len(target.get("applied_scaffold_full") or ""),
            "new_chars": len(parent_scaffold),
            "diff_chars": _diff_chars(
                target.get("applied_scaffold_full") or "", parent_scaffold,
            ),
            "rating_value": None,
            "rating_reason": "",
        }
        _append_journal(rollback_entry)
        return rollback_entry
    except Exception:
        return None


def revert_to_version(turn_id: str, reason: str = "manual") -> dict | None:
    """Restore the scaffold to the exact state captured at applied entry *turn_id*
    (git "checkout this version") and append a rollback record. Returns the rollback
    entry, or ``None`` if *turn_id* is not an applied entry carrying a snapshot.

    Unlike :func:`rollback_last_apply` (which steps back one applied entry), this
    restores to ANY point in the ledger — the per-row Revert the omnibar undo addon
    exposes. Append-only: the revert is itself journaled, so history is never
    rewritten. Never raises.

    Acquires ``_evolve_lock`` so the revert is serialized against the synchronous
    evolution path and against other reverts. CAVEAT: the production async
    evolution path writes the scaffold from a daemon thread WITHOUT holding this
    lock (a known scaffold-write race flagged in ``_run_evolution_blocking``), so
    a revert issued mid-flight during an unattended run can still be clobbered by a
    later evolution write. Precondition for a guaranteed-clean restore: pause the
    training loop (``/monothink off``) before reverting. A shared scaffold-write
    lock across both paths is the full fix (deferred follow-up).
    """
    if not turn_id:
        return None
    if not _evolve_lock.acquire(timeout=60.0):
        return None
    try:
        if not _JOURNAL_PATH.exists():
            return None
        target_scaffold = None
        for line in _JOURNAL_PATH.read_text(encoding="utf-8").splitlines():
            if turn_id not in line:
                continue
            try:
                entry = json.loads(line)
            except Exception:
                continue
            if entry.get("turn_id") == turn_id and entry.get("applied") is True:
                snap = entry.get("applied_scaffold_full")
                if isinstance(snap, str):
                    target_scaffold = snap
                    break
        if target_scaffold is None:
            return None
        current = read_scaffold()
        _write_scaffold(target_scaffold)
        rollback_entry = {
            "ts": _now_iso(),
            "turn_id": f"revert-{_now_iso()}",
            "kind": "rollback",
            "applied": True,
            "applied_scaffold_full": target_scaffold,
            "parent_scaffold_version": turn_id,
            "reverted_to": turn_id,
            "reject_reason": f"revert_to:{turn_id}:{reason}",
            "old_chars": len(current),
            "new_chars": len(target_scaffold),
            "diff_chars": _diff_chars(current, target_scaffold),
            "rating_value": None,
            "rating_reason": "",
        }
        _append_journal(rollback_entry)
        return rollback_entry
    except Exception:
        return None
    finally:
        try:
            _evolve_lock.release()
        except RuntimeError:
            pass


def list_ledger(limit: int = 50, applied_only: bool = False) -> list[dict]:
    """Return the monothink evolution ledger for display (newest first) — the data
    the omnibar undo addon renders.

    Each row: ``turn_id``, ``ts``, ``kind`` (evolution | bootstrap | rollback),
    ``applied``, ``tag`` (adjudicated, else primary), ``rating_value``,
    ``diff_chars``, ``reject_reason``, ``is_current`` (its snapshot equals the live
    scaffold), ``revertable`` (applied + carries a snapshot), and ``reverted_to``
    for rollback rows. ``applied_only`` drops rejected attempts.
    """
    if not _JOURNAL_PATH.exists():
        return []
    try:
        lines = _JOURNAL_PATH.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    current = read_scaffold().strip()
    rows: list[dict] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        # Skip async reservation placeholders — transient, superseded by the
        # turn's terminal entry; they are not ledger events the user acts on.
        if str(e.get("reject_reason") or "") == "scheduled_async":
            continue
        applied = bool(e.get("applied"))
        if applied_only and not applied:
            continue
        snap = e.get("applied_scaffold_full")
        reverted_to = e.get("reverted_to") or e.get("rolled_back_from")
        kind = e.get("kind") or ("rollback" if reverted_to else "evolution")
        rows.append({
            "turn_id": e.get("turn_id"),
            "ts": e.get("ts"),
            "kind": kind,
            "applied": applied,
            "tag": e.get("adjudicated_tag") or e.get("primary_failure_tag"),
            "rating_value": e.get("rating_value"),
            "diff_chars": e.get("diff_chars"),
            "reject_reason": e.get("reject_reason"),
            "is_current": bool(isinstance(snap, str) and snap.strip() == current),
            "revertable": bool(applied and isinstance(snap, str)),
            "reverted_to": reverted_to,
        })
    rows.reverse()  # newest first
    return rows[:limit]


def _most_recent_applied_turn_id() -> str | None:
    """Return the turn_id of the most recent ``applied=True`` journal entry,
    or ``None`` if no journal exists.

    Includes the bootstrap entry (which is always ``applied=True``), so this
    never returns ``None`` in practice once ``_ensure_bootstrap_entry`` has
    run. The defensive ``None`` covers the race window between bootstrap
    write attempt and read.
    """
    if not _JOURNAL_PATH.exists():
        return None
    try:
        lines = _JOURNAL_PATH.read_text(encoding="utf-8").splitlines()
    except Exception:
        return None
    for line in reversed(lines):
        if '"applied": true' not in line and '"applied":true' not in line:
            continue
        try:
            entry = json.loads(line)
        except Exception:
            continue
        if entry.get("applied") is True:
            tid = entry.get("turn_id")
            if tid:
                return str(tid)
    return None


def _contrast_prompt_block(tag: str | None) -> tuple[object | None, str]:
    """Return the prompt-visible contrast block for *tag* when shadow is on.

    The returned store is passed back into case-build/evaluation paths so tests
    can monkeypatch the module paths without changing global MonoThink state.
    """
    if not tag:
        return None, ""
    try:
        from core import monothink_contrast as _contrast
        if not _contrast.enabled():
            return None, ""
        store = _contrast.ContrastStore()
        cases = _contrast.select_for_prompt(store, tag)
        return store, _contrast.render_contrast_block(cases)
    except Exception:
        return None, ""


def _record_contrast_case(
    *,
    store: object | None,
    turn_id: str,
    tag: str | None,
    rating_value,
    current_scaffold: str,
    replay_input: str | None,
    think_block: str | None,
) -> None:
    if store is None or not tag:
        return
    try:
        from core import monothink_contrast as _contrast
        if not _contrast.enabled():
            return
        gloss = _failure_tags.FAILURE_TAGS.get(tag, "")
        _contrast.build_case_with_default_profile(
            store=store,
            turn_id=str(turn_id),
            tag=tag,
            rating=int(rating_value) if isinstance(rating_value, (int, float)) else 0,
            base_scaffold_sha=_contrast.sha(current_scaffold),
            minimized_input=replay_input,
            failed_trace=think_block,
            tag_gloss=gloss,
        )
    except Exception:
        pass


def _contrast_shadow_summary(
    *,
    tag: str | None,
    old_scaffold: str,
    candidate_scaffold: str,
) -> dict | None:
    if not tag:
        return None
    try:
        from core import monothink_contrast as _contrast
        if not _contrast.enabled():
            return None
        store = _contrast.ContrastStore()
        verdict = _contrast.evaluate_shadow_default(
            tag=tag,
            old_scaffold=old_scaffold,
            candidate_scaffold=candidate_scaffold,
            store=store,
        )
        _contrast.record_verdict(verdict)
        return verdict.journal_summary()
    except Exception as exc:
        try:
            from core import monothink_contrast as _contrast
            _contrast.record_event({
                "kind": "separation_verdict_failed",
                "tag": tag,
                "reason": f"{type(exc).__name__}:{exc}",
            })
            return {
                "contrast_shadow": True,
                "contrast_would_admit": False,
                "contrast_target_gain": 0.0,
                "contrast_worst_invariant_regression": 0.0,
                "contrast_case_count": 0,
                "contrast_reason": f"exception:{type(exc).__name__}",
            }
        except Exception:
            return None


def _ledger_context_block(tag: str | None) -> tuple[object | None, str, int]:
    try:
        from core import monothink_deferred_ledger as _ledger
        store = _ledger.DeferredLedger()
        rated_index = _ledger.next_rated_index()
        if tag:
            return store, store.lookup_block(tag), rated_index
        return store, "LEDGER: no tag selected.", rated_index
    except Exception:
        return None, "LEDGER: unavailable.", 0


def _ledger_trace_span(parsed: dict | None, think_block: str | None) -> str:
    if parsed and parsed.get("trace_span"):
        return str(parsed.get("trace_span") or "")
    text = str(think_block or "").strip()
    if not text:
        return ""
    return text[-240:]


def _record_ledger_reservation(
    *,
    store: object | None,
    parsed: dict,
    turn_id: str,
    think_block: str | None,
    rater_note: str | None,
    rated_index: int,
) -> object | None:
    if store is None or parsed.get("schema_version") != "decision_v2":
        return None
    if parsed.get("mutation_decision") == "reject":
        return None
    try:
        section = (
            parsed.get("target_section")
            or parsed.get("covered_section")
            or "NONE"
        )
        return store.record_reservation(
            tag=parsed["primary_failure_tag"],
            section=str(section),
            failure_signature=str(parsed.get("failure_signature") or ""),
            signature_status=str(parsed.get("signature_status") or "provisional"),
            turn_id=str(turn_id),
            trace_span=_ledger_trace_span(parsed, think_block),
            rater_note=rater_note,
            rated_index=int(rated_index or 0),
        )
    except Exception:
        return None


def _validate_critical_invariant(parsed: dict, current: str, think_block: str | None) -> str | None:
    if parsed.get("evidence_class") != "critical_invariant_break":
        return None
    invariant = str(parsed.get("invariant_cite") or "").strip()
    trace_span = str(parsed.get("trace_span") or "").strip()
    form = str(parsed.get("violation_form") or "").strip()
    if not invariant:
        return "critical_invariant_invalid:missing_invariant_cite"
    if not trace_span:
        return "critical_invariant_invalid:missing_trace_span"
    if form not in {"contradiction", "omission", "forbidden_action"}:
        return "critical_invariant_invalid:violation_form"
    if invariant not in current:
        return "critical_invariant_invalid:invariant_cite_not_found"
    if trace_span not in str(think_block or ""):
        return "critical_invariant_invalid:trace_span_not_found"
    return None


def _promotion_allowed(
    *,
    store: object | None,
    parsed: dict,
    current: str,
    think_block: str | None,
) -> tuple[bool, str]:
    if parsed.get("schema_version") != "decision_v2":
        return True, "legacy_schema"
    if parsed.get("mutation_decision") != "apply":
        return False, "not_apply"
    if parsed.get("fix_kind") == "rating_misapplied":
        return False, "rating_misapplied"
    if parsed.get("covered_section") is not None or str(parsed.get("already_covered") or "").lower().startswith("yes"):
        return False, "already_covered_enforcement_gap"
    if parsed.get("fix_kind") == "enforcement_gap":
        return False, "enforcement_gap"

    crit_error = _validate_critical_invariant(parsed, current, think_block)
    if crit_error:
        return False, crit_error
    if parsed.get("evidence_class") == "critical_invariant_break":
        return True, "critical_invariant_break"

    if parsed.get("evidence_class") != "repeated_pattern":
        return False, "single_instance_reserved"
    if parsed.get("signature_status") != "canonical":
        return False, "signature_not_canonical"
    if store is None:
        return False, "ledger_unavailable"
    try:
        row = store.find(
            parsed["primary_failure_tag"],
            str(parsed.get("target_section") or "NONE"),
            str(parsed.get("failure_signature") or ""),
        )
        if row is None:
            return False, "ledger_key_missing"
        if row.signature_status != "canonical":
            return False, "ledger_signature_not_canonical"
        if row.count < 1:
            return False, "ledger_prior_count_missing"
        prior_turns = {str(s).split(":", 1)[0] for s in row.trace_spans}
        if len(prior_turns) < 1:
            return False, "ledger_prior_span_missing"
    except Exception:
        return False, "ledger_check_failed"
    return True, "repeated_pattern"


def _record_apply_monitor(
    *,
    parsed: dict,
    turn_id: str,
    actual_scope: str,
    result_hash: str,
) -> None:
    if parsed.get("schema_version") != "decision_v2":
        return
    try:
        from core import monothink_deferred_ledger as _ledger
        _ledger.record_monitor_event({
            "kind": "applied_edit",
            "turn_id": str(turn_id),
            "tag": parsed.get("primary_failure_tag"),
            "section": parsed.get("target_section"),
            "failure_signature": parsed.get("failure_signature"),
            "predicted_effect": parsed.get("predicted_effect"),
            "falsifier": parsed.get("falsifier"),
            "actual_scope": actual_scope,
            "result_hash": result_hash,
        })
    except Exception:
        pass


# ── the hook entry point ─────────────────────────────────────────────


def maybe_evolve_after_rating(
    turn_id: str,
    rating_value,
    failure_tags,
    think_block: str | None = None,
    replay_input: str | None = None,
    rater_note: str | None = None,
) -> dict | None:
    """Called from record_outcome when a kind=='rating' outcome is persisted.

    Returns the journal entry (whether applied or rejected) for visibility,
    a tiny "already_processed" sentinel dict when this turn has already had
    its one evolution attempt, or None when no action was taken at all
    (e.g., flag off, turn wasn't monothink).

    When ``think_block`` is provided (the rated turn's <think>...</think>
    reasoning trace, extracted at /rating time), it is injected into the
    evolution prompt so the model can see *how* it thought on that turn,
    not just *what* the rating said. The scaffold is the lever between
    those two — seeing both lets the model edit at the right altitude.

    Multiple-ratings-per-turn policy:
      Each turn gets exactly ONE evolution attempt. The journal IS the dedup
      table — :func:`_journal_has_turn` checks it before any LLM call. Re-rating
      the same turn (UI double-click, deliberate re-evaluation, schema retry)
      returns the "already_processed" sentinel without spending another LLM
      call. The first attempt wins, whether it was applied or rejected. This is
      deliberate: the model gets one shot to evolve from a given turn's rating
      signal. Operator override is manual (delete the journal entry to retry).

    Guarantees:
      * Never raises (broad except wrap).
      * Only writes to monothink.md and monothink.journal.jsonl.
      * At most one LLM call per turn_id, ever.
    """
    if not _flag_enabled():
        return None
    failure_tags = _failure_tags.normalize_tags(failure_tags)
    if not failure_tags:
        return None
    # Serialize evolutions process-wide: closes the journal-check / LLM-call
    # race window so two concurrent ratings on the same turn can't both pass
    # _journal_has_turn and write conflicting scaffold versions.
    if not _evolve_lock.acquire(timeout=60.0):
        return {
            "applied": False,
            "reject_reason": "lock_timeout",
            "turn_id": str(turn_id),
        }
    try:
        if _journal_has_turn(str(turn_id)):
            return {
                "applied": False,
                "reject_reason": "already_processed",
                "turn_id": str(turn_id),
            }

        if not _lookup_turn_monothink_active(turn_id):
            return None

        # Phase 3e: ensure the bootstrap journal entry exists. Placed AFTER
        # the reasoning_mode gate so non-monothink turns don't write a
        # bootstrap row (preserves "journal doesn't exist when monothink
        # isn't engaged" invariant). Idempotent — re-running is a no-op.
        _ensure_bootstrap_entry()

        # Async path: spawn a daemon thread for the LLM call so the calling
        # UI thread isn't blocked on generation (the freeze observed on the
        # first live /rating). The journal-reservation entry written below
        # closes the race window — concurrent ratings on the same turn see
        # it via _journal_has_turn and return already_processed.
        if _async_enabled():
            reserved = _make_journal_entry(
                turn_id, rating_value,
                _failure_tags.compose_reasoning_why(failure_tags),
                "", "", applied=False,
                reject_reason="scheduled_async",
                failure_tags=failure_tags,
            )
            _append_journal(reserved)
            import threading as _threading
            thread_kwargs = (
                {
                    k: v for k, v in {
                        "replay_input": replay_input,
                        "rater_note": rater_note,
                    }.items()
                    if v is not None
                }
            )
            thread = _threading.Thread(
                target=_run_evolution_blocking,
                args=(turn_id, rating_value, failure_tags, think_block),
                kwargs=thread_kwargs,
                daemon=True,
                name=f"monothink-evolve-{str(turn_id)[:8]}",
            )
            thread.start()
            return reserved

        return _run_evolution_blocking(
            turn_id, rating_value, failure_tags, think_block,
            replay_input=replay_input,
            rater_note=rater_note,
        )
    except Exception as exc:
        # Sync path exception handler. The async path's thread has its own
        # try/except in _run_evolution_blocking. Reaching here means the
        # gating logic (journal scan, frame lookup) raised.
        try:
            entry = {
                "ts": _now_iso(),
                "turn_id": str(turn_id),
                "rating_value": (int(rating_value)
                                 if isinstance(rating_value, (int, float))
                                 else None),
                "rating_reason": _failure_tags.compose_reasoning_why(failure_tags),
                "failure_tags": list(failure_tags) if failure_tags else [],
                "applied": False,
                "reject_reason": f"exception:{type(exc).__name__}",
            }
            _append_journal(entry)
            return entry
        except Exception:
            return None
    finally:
        try:
            _evolve_lock.release()
        except RuntimeError:
            pass


def _run_evolution_blocking(
    turn_id: str,
    rating_value,
    failure_tags,
    think_block: str | None,
    *,
    replay_input: str | None = None,
    rater_note: str | None = None,
) -> dict | None:
    """The original LLM-call + journal-write path. Runs either inline (sync
    test path; under the caller's lock) or inside a daemon thread (production
    async path; the caller already released the lock before spawning).

    Concurrency note: in the async path, two threads serving two different
    rated turns back-to-back could in principle race on ``_write_scaffold``.
    The atomic write (os.replace) prevents corruption but the LATER write
    wins both edits. /rating workload makes this extremely rare; a dedicated
    scaffold-write lock is a known follow-up. Same-turn ratings can't reach
    this function — the reservation entry in the journal blocks them at the
    caller's gating step.

    Never raises — failures fold into a journal entry. The async thread is
    daemon, so a stuck LLM call doesn't keep the process alive at shutdown.
    """
    try:
        rating_reason = _failure_tags.compose_reasoning_why(failure_tags)
        current = read_scaffold()
        prompt_tag = failure_tags[0] if failure_tags else None
        contrast_store, contrast_block = _contrast_prompt_block(prompt_tag)
        ledger_store, ledger_block, rated_index = _ledger_context_block(prompt_tag)
        try:
            if ledger_store is not None:
                ledger_store.expire_stale(rated_index)
        except Exception:
            pass
        prompt = _compose_prompt(
            current, rating_value, failure_tags,
            think_block=think_block,
            contrast_block=contrast_block,
            rater_note=rater_note,
            ledger_block=ledger_block,
        )
        _record_contrast_case(
            store=contrast_store,
            turn_id=turn_id,
            tag=prompt_tag,
            rating_value=rating_value,
            current_scaffold=current,
            replay_input=replay_input,
            think_block=think_block,
        )
        raw = _call_llm(prompt)
        if raw is None:
            entry = _make_journal_entry(turn_id, rating_value, rating_reason,
                                         current, current, applied=False,
                                         reject_reason="llm_call_failed",
                                         raw_response_full="",
                                         failure_tags=failure_tags)
            _append_journal(entry)
            return entry

        # Phase 1b: parse the model's response per the schema before any
        # mutation logic. Schema-violating responses get journaled with
        # raw_response_full as forensic backup, then we return — none of the
        # cap/accept logic runs on unstructured blobs.
        parsed, parse_err = _parse_evolution_response(raw)
        if parse_err is not None:
            entry = _make_journal_entry(turn_id, rating_value, rating_reason,
                                         current, current, applied=False,
                                         reject_reason=parse_err,
                                         raw_response_full=raw,
                                         failure_tags=failure_tags)
            _append_journal(entry)
            return entry

        # Phase 3.5: PATCH_MODE=no_change short-circuits to the no_change
        # branch. The literal PATCH=NO_CHANGE string from the old schema is
        # gone; PATCH_MODE is the structured signal now.
        if parsed["patch_mode"] == "no_change":
            ledger_row = _record_ledger_reservation(
                store=ledger_store,
                parsed=parsed,
                turn_id=turn_id,
                think_block=think_block,
                rater_note=rater_note,
                rated_index=rated_index,
            )
            reject_reason = "no_change_requested"
            if parsed.get("schema_version") == "decision_v2":
                decision = parsed.get("mutation_decision")
                if decision == "reserve":
                    reject_reason = "reserved_to_ledger"
                elif decision == "reject":
                    reject_reason = "decision_rejected"
            entry = _make_journal_entry(turn_id, rating_value, rating_reason,
                                         current, current, applied=False,
                                         reject_reason=reject_reason,
                                         raw_response_full=raw, parsed=parsed,
                                         failure_tags=failure_tags)
            if ledger_row is not None:
                entry["ledger_count"] = getattr(ledger_row, "count", None)
                entry["ledger_status"] = getattr(ledger_row, "status", None)
            _append_journal(entry)
            return entry

        allowed, allow_reason = _promotion_allowed(
            store=ledger_store,
            parsed=parsed,
            current=current,
            think_block=think_block,
        )
        if not allowed:
            ledger_row = _record_ledger_reservation(
                store=ledger_store,
                parsed=parsed,
                turn_id=turn_id,
                think_block=think_block,
                rater_note=rater_note,
                rated_index=rated_index,
            )
            entry = _make_journal_entry(turn_id, rating_value, rating_reason,
                                         current, current, applied=False,
                                         reject_reason=f"promotion_gate:{allow_reason}",
                                         raw_response_full=raw, parsed=parsed,
                                         failure_tags=failure_tags)
            if ledger_row is not None:
                entry["ledger_count"] = getattr(ledger_row, "count", None)
                entry["ledger_status"] = getattr(ledger_row, "status", None)
            _append_journal(entry)
            return entry

        # Apply the BEFORE → AFTER substitution against the current scaffold.
        # The byte-exact match is the substrate's correctness check: the
        # model can't propose a 4000-char rewrite without also producing a
        # 4000-char BEFORE block that exists in the current scaffold — same
        # impossibility as the rewrite itself. Phase 3.5's central
        # mechanism.
        proposed, apply_err = _apply_patch(current, parsed["before"], parsed["after"])
        if (
            apply_err is not None
            and apply_err.startswith("before_block_mismatch")
            and parsed.get("schema_version") != "decision_v2"
        ):
            # Phase 3.7: ONE bounded repair. The decider names TARGET_SECTION
            # reliably but mis-quotes BEFORE (live journal: 5×). Hand it the
            # region's current text verbatim; deterministic tooling re-judges.
            reparsed, repair_raw = _repair_before_block(current, parsed, _call_llm)
            if reparsed is not None and reparsed["patch_mode"] != "no_change":
                proposed2, err2 = _apply_patch(
                    current, reparsed["before"], reparsed["after"])
                if err2 is None:
                    parsed, proposed, apply_err = reparsed, proposed2, None
                    raw = f"{raw}\n\n=== REPAIR ===\n{repair_raw}"
            if apply_err is not None:
                apply_err = f"{apply_err}:repair_failed"
        if apply_err is not None:
            entry = _make_journal_entry(turn_id, rating_value, rating_reason,
                                         current, current, applied=False,
                                         reject_reason=apply_err,
                                         raw_response_full=raw, parsed=parsed,
                                         failure_tags=failure_tags)
            _append_journal(entry)
            return entry

        # ── Phase 3b hook: scope_mismatch enforcement ─────────────────────
        # Order rationale: scope_mismatch is the fastest semantic check —
        # a string comparison between the model's stated PROPOSED_SCOPE and
        # the patcher's measured scope. Runs BEFORE protected-section
        # parsing (which is more expensive) and BEFORE cap checks (which
        # are content-dependent). When `_classify_diff` lands in Phase 3b,
        # plug in here:
        #
        #   actual_scope = _classify_diff(current, proposed)
        #   if actual_scope != parsed["proposed_scope"]:
        #       reject = (f"scope_mismatch:proposed={parsed['proposed_scope']}"
        #                 f",actual={actual_scope}")
        #       entry = _make_journal_entry(turn_id, rating_value, rating_reason,
        #                                    current, proposed, applied=False,
        #                                    reject_reason=reject,
        #                                    raw_response_full=raw, parsed=parsed)
        #       _append_journal(entry)
        #       return entry
        #
        # Until then PROPOSED_SCOPE is captured in the journal but not used
        # as a gate. Data accumulates with the field populated from day one.

        # Phase 2c: protected-section enforcement. Body change to a protected
        # section, OR removal of one, rejects before any cap check. The
        # patcher owns this — putting the protection list in the scaffold
        # itself would let the model edit its own constraints away.
        protected_touched = _diff_touches_protected(current, proposed)
        if protected_touched:
            name, kind = protected_touched[0]
            label = (
                f"protected_section_removed:{name}"
                if kind == "removed"
                else f"protected_section:{name}"
            )
            entry = _make_journal_entry(turn_id, rating_value, rating_reason,
                                         current, proposed, applied=False,
                                         reject_reason=label,
                                         raw_response_full=raw, parsed=parsed,
                                         failure_tags=failure_tags)
            _append_journal(entry)
            return entry

        # Size cap (hard ceiling — fires before policy dispatch).
        if len(proposed) > _SIZE_CAP:
            entry = _make_journal_entry(turn_id, rating_value, rating_reason,
                                         current, proposed, applied=False,
                                         reject_reason=f"size_cap_exceeded:{len(proposed)}>{_SIZE_CAP}",
                                         raw_response_full=raw, parsed=parsed,
                                         failure_tags=failure_tags)
            _append_journal(entry)
            return entry

        # Identical-to-current — pre-policy fast branch. Could fold into the
        # policy via actual_scope=="none", but keeping it as an explicit
        # branch preserves the `proposed_equals_current` reject_reason that
        # existing tooling and journal queries already key on.
        if proposed.strip() == current.strip():
            entry = _make_journal_entry(turn_id, rating_value, rating_reason,
                                         current, current, applied=False,
                                         reject_reason="proposed_equals_current",
                                         raw_response_full=raw, parsed=parsed,
                                         failure_tags=failure_tags)
            _append_journal(entry)
            return entry

        # ── Phase 3b: classify the ACTUAL change; PROPOSED_SCOPE is advisory.
        # The code is the authority (E, 2026-06-25): it decides from the
        # measured actual scope and LOGS any proposed/actual mismatch rather
        # than discarding a valid catch on a wrong label.
        actual_scope = _classify_diff(current, proposed)
        # Phase 3c: section-scope auto-apply gated on whether the same
        # primary_failure_tag has appeared on a prior DISTINCT turn_id
        # within the lookback window.
        concern_repeated = _concern_repeated(
            parsed["primary_failure_tag"], turn_id,
        )
        decision = _decide_policy(actual_scope, concern_repeated)

        if decision == DECISION_NO_CHANGE:
            # Unreachable in practice — proposed_equals_current is caught
            # above — but keep the branch explicit so an actual=="none" can
            # never fall through to the APPLY write.
            entry = _make_journal_entry(turn_id, rating_value, rating_reason,
                                         current, current, applied=False,
                                         reject_reason="proposed_equals_current",
                                         raw_response_full=raw, parsed=parsed,
                                         failure_tags=failure_tags,
                                         actual_scope=actual_scope)
            _append_journal(entry)
            return entry

        if decision == DECISION_PROPOSE:
            # No write. Journal the proposal so operators can review and
            # manually apply if desired. The reject_reason carries the actual
            # scope so journal queries can find proposals by shape; the
            # proposed/actual mismatch (if any) is logged on the entry too.
            entry = _make_journal_entry(turn_id, rating_value, rating_reason,
                                         current, proposed, applied=False,
                                         reject_reason=f"proposed_only:{actual_scope}",
                                         raw_response_full=raw, parsed=parsed,
                                         failure_tags=failure_tags,
                                         actual_scope=actual_scope)
            _append_journal(entry)
            return entry

        # DECISION_APPLY — gate through the fast-check, then write scaffold
        # + append journal. Phase 3d's fast-check catches structural breakage
        # the policy/classifier paths can't see: half-fenced code blocks,
        # empty section bodies, parses-to-no-sections. Failure downgrades
        # apply → propose, journaling canary_fast_fail:<what>.
        fast_fail = _fast_check_scaffold(proposed)
        if fast_fail is not None:
            entry = _make_journal_entry(turn_id, rating_value, rating_reason,
                                         current, proposed, applied=False,
                                         reject_reason=fast_fail,
                                         raw_response_full=raw, parsed=parsed,
                                         failure_tags=failure_tags)
            _append_journal(entry)
            return entry

        # Phase 3e: capture parent turn_id BEFORE the write so the rollback
        # chain forms cleanly even under concurrent writes (different turns
        # writing back-to-back).
        contrast_summary = _contrast_shadow_summary(
            tag=parsed["primary_failure_tag"],
            old_scaffold=current,
            candidate_scaffold=proposed,
        )
        ledger_row = _record_ledger_reservation(
            store=ledger_store,
            parsed=parsed,
            turn_id=turn_id,
            think_block=think_block,
            rater_note=rater_note,
            rated_index=rated_index,
        )
        try:
            if parsed.get("schema_version") == "decision_v2" and ledger_store is not None:
                ledger_store.mark_promoted(
                    parsed["primary_failure_tag"],
                    str(parsed.get("target_section") or "NONE"),
                    str(parsed.get("failure_signature") or ""),
                    turn_id=str(turn_id),
                    rated_index=int(rated_index or 0),
                )
        except Exception:
            pass
        parent_tid = _most_recent_applied_turn_id()
        _write_scaffold(proposed)
        result_hash = str(__import__("hashlib").sha256(proposed.encode("utf-8")).hexdigest())
        _record_apply_monitor(
            parsed=parsed,
            turn_id=turn_id,
            actual_scope=actual_scope,
            result_hash=result_hash,
        )
        entry = _make_journal_entry(turn_id, rating_value, rating_reason,
                                     current, proposed, applied=True,
                                     reject_reason=None,
                                     raw_response_full=raw, parsed=parsed,
                                     applied_scaffold_full=proposed,
                                     parent_scaffold_version=parent_tid,
                                     failure_tags=failure_tags,
                                     actual_scope=actual_scope,
                                     contrast_summary=contrast_summary)
        if ledger_row is not None:
            entry["ledger_count"] = getattr(ledger_row, "count", None)
            entry["ledger_status"] = "promoted"
        _append_journal(entry)
        return entry

    except Exception as exc:
        # Last-ditch: never break the rating loop. Record minimal failure entry.
        # No lock release here — _run_evolution_blocking does not own the
        # lock. In the sync path, the caller (maybe_evolve_after_rating)
        # owns and releases it. In the async path, the lock was already
        # released by the caller before this thread was spawned.
        try:
            entry = {
                "ts": _now_iso(),
                "turn_id": str(turn_id),
                "rating_value": (int(rating_value)
                                 if isinstance(rating_value, (int, float))
                                 else None),
                "rating_reason": _failure_tags.compose_reasoning_why(failure_tags),
                "failure_tags": list(failure_tags) if failure_tags else [],
                "applied": False,
                "reject_reason": f"exception:{type(exc).__name__}",
            }
            _append_journal(entry)
            return entry
        except Exception:  # noqa: BLE001 — last-ditch safety net
            return None


# ── standalone interceptor (replaces reasoning_interceptor) ─────────────

_monothink_world_state: Any = None


def set_monothink_world_state(ws: Any) -> None:
    """Wire the world_state so monothink_interceptor can read the toggle."""
    global _monothink_world_state
    _monothink_world_state = ws


def set_monothink_toggle(mode: str) -> dict:
    """Set (or read) the monothink toggle from a non-UI caller (the agent server).

    Mirrors the UI ``/monothink`` command against the same wired world_state, so an
    external rater can turn monothink on for a training session without the GUI.
    ``mode`` is one of ``on`` | ``off`` | ``once`` | ``status``. Returns a small
    dict ``{"ok", "monothink", ...}``. Never raises.
    """
    ws = _monothink_world_state
    if ws is None:
        return {"ok": False, "error": "world_state not wired"}
    m = str(mode or "status").strip().lower()
    try:
        if m == "on":
            ws.set_monothink(True)
            return {"ok": True, "monothink": "on"}
        if m == "off":
            ws.set_monothink(False)
            return {"ok": True, "monothink": "off"}
        if m == "once":
            ws.set_monothink_once(True)
            return {"ok": True, "monothink": "once"}
        if m == "status":
            state = getattr(ws, "state", {}) or {}
            return {
                "ok": True,
                "monothink": "on" if bool(state.get("monothink_enabled")) else "off",
                "once_pending": bool(state.get("monothink_once")),
            }
        return {"ok": False, "error": f"unknown mode: {mode!r} (use on|off|once|status)"}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def monothink_interceptor(messages: list[dict], config: dict) -> list[dict] | None:
    """Inject the monothink scaffold when the monothink toggle is active.

    Standalone interceptor registered in bootstrap.py — replaces the
    reasoning plane's interceptor for monothink injection. The evolution
    mechanism (maybe_evolve_after_rating) is unchanged; only the injection
    path is new.

    Resolution layers:
      1. config["force_monothink"]  — testing/automation
      2. world_state monothink_once (consumed)
      3. world_state monothink_enabled
      4. False (opt-in default)
    """
    if not _flag_enabled():
        return None

    active = False

    if isinstance(config, dict):
        forced = config.get("force_monothink")
        if isinstance(forced, bool):
            active = forced

    if not active and _monothink_world_state is not None:
        try:
            active = _monothink_world_state.consume_monothink()
        except Exception:
            pass

    if isinstance(config, dict):
        config["_resolved_monothink"] = active

    if not active:
        return None

    for msg in messages:
        if msg.get("source") == "monothink":
            return None

    content = read_scaffold()
    if not content or not content.strip():
        return None
    # Envelope added at injection time, never written into monothink.md
    # (the scaffold file is protected/self-evolving). Without a header this
    # is the only ephemeral lane with no boundary marker — cloud backends
    # merge consecutive user-role messages, so the bare scaffold read as
    # the user's own document.
    content = (
        "[MONOTHINK] — reasoning scaffold for this turn; "
        "runtime-injected, NOT the user's message.\n" + content
    )

    last_user_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") == "user" and not msg.get("ephemeral"):
            last_user_idx = i
            break
    if last_user_idx < 0:
        return None

    result = list(messages)
    result.insert(
        last_user_idx,
        {
            "role": "user",
            "content": content,
            "ephemeral": True,
            "source": "monothink",
            "reasoning_mode": "monothink",
        },
    )
    return result
