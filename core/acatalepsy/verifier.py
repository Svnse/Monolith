"""Truth-branch Verifier (Phase 3) — grounds checkable claims and records a verdict.

For a ``world-fact`` (or ``causal``) ACU:
  1. **Deterministic internal contradiction:** does it conflict with a stored ACU
     on a functional (single-valued) relation? -> verdict ``contradicted`` (method
     ``internal``) + a typed ``contradicts`` CCG edge. No network.
  2. **Grounding:** else query Tavily; an injected judge reads the evidence ->
     ``{confirmed | contradicted | unverifiable | contested}`` + confidence.
  3. **Record:** write the Truth columns; a high-confidence ``contradicted`` flips
     ``state='-inf'`` (kept for audit, excluded from recall). ``causal`` confirmations
     are capped at ``contested`` (the hypothesis ceiling — "facts murder theories").

Mad Cow: the verdict is grounded in EXTERNAL evidence (Tavily/world), not the
model's own parametric memory — so it may confirm. ``self``/``meta``/``emotional``
kinds are NOT truth-checked. ``search_fn``/``judge_fn`` are injectable; production
wires ``grounding.tavily_search`` + an LLM judge.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone

from core.db_connect import authorized_write, connect_acatalepsy
from core.acatalepsy import canonical_log as _canonical_log
from core.acatalepsy import grounding
from core.acatalepsy.normalize import parse_triple

__all__ = ("Verdict", "verify_acu", "run_verifier", "VerifierWorker")

_VERDICTS = ("confirmed", "contradicted", "unverifiable", "contested")
_CHECKABLE = ("world-fact", "causal")
_INF_THRESHOLD = 0.7

# Functional (single-valued) relations: a subject has essentially one value, so a
# second stored value is a contradiction. Conservative, hand-curated floor.
_FUNCTIONAL = frozenset({
    "capital_of", "president_of", "born_in", "died_in", "located_in",
    "author_of", "ceo_of", "invented_by", "founded_by", "headquartered_in",
    "currency_of", "atomic_number", "president", "capital",
})


@dataclass(frozen=True)
class Verdict:
    verdict: str
    confidence: float
    method: str          # internal | tavily


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _query_from(canonical: str, triple) -> str:
    return str(canonical or "").replace("|", " ").strip()


def _detect_internal_contradiction(acu_id: int, triple, conn) -> int | None:
    if triple is None or triple.relation not in _FUNCTIONAL:
        return None
    rows = conn.execute(
        "SELECT id, canonical_triple FROM acus "
        "WHERE merged_into IS NULL AND state='active' AND id<>? "
        "AND json_extract(canonical_triple,'$.entity_a')=? "
        "AND json_extract(canonical_triple,'$.relation')=?",
        (int(acu_id), triple.entity_a, triple.relation),
    ).fetchall()
    for r in rows:
        try:
            ct = json.loads(r["canonical_triple"]) if r["canonical_triple"] else {}
        except (TypeError, ValueError):
            continue
        if ct.get("entity_b") != triple.entity_b:
            return int(r["id"])
    return None


def _write_verdict(conn, acu_id, verdict, confidence, method, *, evidence, contra_id=None):
    now = _now()
    ev = list(evidence or [])
    ev_json = json.dumps([{"url": e.url, "snippet": e.snippet, "score": e.score} for e in ev])
    ev_url = ev[0].url if ev else None
    conn.execute(
        "UPDATE acus SET truth=?, truth_confidence=?, truth_method=?, truth_checked_at=?, "
        "evidence_url=?, evidence_json=? WHERE id=?",
        (verdict, float(confidence), method, now, ev_url, ev_json, int(acu_id)),
    )
    if contra_id is not None:
        # Idempotent: re-verifying must not write a duplicate contradicts edge.
        conn.execute(
            "INSERT INTO acu_relations(source_id, target_id, relation, score, created_at, updated_at) "
            "SELECT ?,?,?,?,?,? WHERE NOT EXISTS ("
            "  SELECT 1 FROM acu_relations WHERE source_id=? AND target_id=? AND relation='contradicts')",
            (int(acu_id), int(contra_id), "contradicts", float(confidence), now, now,
             int(acu_id), int(contra_id)),
        )
    _canonical_log.append_on(
        conn, "truth_verdict",
        {"acu_id": int(acu_id), "verdict": verdict, "confidence": float(confidence),
         "method": method, "contradicts_acu_id": contra_id, "evidence_count": len(ev)},
        acu_id=int(acu_id),
    )


def verify_acu(acu_id: int, *, conn, search_fn=None, judge_fn=None) -> Verdict | None:
    """Verify one ACU. Returns a Verdict, or None if the claim's Kind is not
    truth-checkable. The caller owns the transaction (commit/sentinel)."""
    row = conn.execute(
        "SELECT canonical, kind FROM acus WHERE id=?", (int(acu_id),)
    ).fetchone()
    if row is None:
        return None
    kind = str(row["kind"] or "").strip()
    if kind not in _CHECKABLE:
        return None

    triple = parse_triple(row["canonical"])

    # 1. deterministic internal contradiction (no network)
    contra_id = _detect_internal_contradiction(int(acu_id), triple, conn)
    if contra_id is not None:
        # An internal conflict FLAGS both claims as contested — it does not
        # unilaterally declare the newcomer false with no external evidence
        # (that would be recency bias). Tavily/governance adjudicates which is
        # actually false; only then does a claim flip to -inf.
        _write_verdict(conn, int(acu_id), "contested", 0.5, "internal",
                       evidence=[], contra_id=contra_id)
        return Verdict("contested", 0.5, "internal")

    # 2. ground externally
    search_fn = search_fn or grounding.tavily_search
    judge_fn = judge_fn or _llm_judge
    try:
        evidence = search_fn(_query_from(row["canonical"], triple))
    except grounding.GroundingUnavailable:
        _write_verdict(conn, int(acu_id), "unverifiable", 0.0, "tavily", evidence=[])
        return Verdict("unverifiable", 0.0, "tavily")

    verdict, confidence = judge_fn(row["canonical"], evidence)
    verdict = verdict if verdict in _VERDICTS else "unverifiable"
    confidence = float(confidence or 0.0)
    if verdict == "confirmed" and not evidence:
        verdict = "unverifiable"   # never confirm without external evidence (Mad Cow)
    if kind == "causal" and verdict == "confirmed":
        verdict = "contested"   # causal hypothesis ceiling

    _write_verdict(conn, int(acu_id), verdict, confidence, "tavily", evidence=evidence)
    if verdict == "contradicted" and confidence >= _INF_THRESHOLD:
        conn.execute("UPDATE acus SET state='-inf' WHERE id=?", (int(acu_id),))
    elif verdict == "confirmed":
        # A claim grounded as true AND already reinforced is trusted (L2->L3).
        from core.acatalepsy import crystallize as _cryst
        _cryst.maybe_promote_l3(int(acu_id), conn)
    return Verdict(verdict, confidence, "tavily")


def _llm_judge(claim: str, evidence) -> tuple[str, float]:
    """Production default judge: a local LLM reads the retrieved evidence and
    returns a verdict. Defensive — any failure (no model, bad JSON) yields
    ('unverifiable', 0.0) so verification never breaks the caller."""
    try:
        from core.acatalepsy.llm_sidecar import make_auditor_llm
        llm = make_auditor_llm()
        if llm is None:
            return ("unverifiable", 0.0)
        ev_text = "\n".join(f"- {e.snippet} ({e.url})" for e in list(evidence)[:5]) or "(no evidence)"
        system = (
            "You are a fact-checker. Given a CLAIM and EVIDENCE, judge ONLY from the "
            "evidence. Respond with STRICT JSON, no prose: "
            '{"verdict": "confirmed|contradicted|unverifiable|contested", "confidence": 0.0-1.0}'
        )
        user = f"CLAIM: {claim}\n\nEVIDENCE:\n{ev_text}"
        raw = llm(system_prompt=system, user_content=user)
        start, end = raw.find("{"), raw.rfind("}")
        data = json.loads(raw[start:end + 1]) if start >= 0 and end > start else {}
        verdict = str(data.get("verdict", "unverifiable"))
        if verdict not in _VERDICTS:
            verdict = "unverifiable"
        return (verdict, float(data.get("confidence", 0.0) or 0.0))
    except Exception:
        return ("unverifiable", 0.0)


def run_verifier(*, limit: int = 20, search_fn=None, judge_fn=None) -> dict:
    """Batch-verify un-checked, active, checkable ACUs. Returns a {verdict: count}
    summary. Manages its own write transaction."""
    counts: dict[str, int] = {v: 0 for v in _VERDICTS}
    counts["skipped"] = 0
    conn = connect_acatalepsy(role="memory_writer")
    try:
        rows = conn.execute(
            "SELECT id FROM acus WHERE merged_into IS NULL AND state='active' "
            "AND kind IN ('world-fact','causal') AND truth IS NULL ORDER BY id LIMIT ?",
            (int(limit),),
        ).fetchall()
        with authorized_write("verifier"):
            for r in rows:
                try:
                    v = verify_acu(int(r["id"]), conn=conn, search_fn=search_fn, judge_fn=judge_fn)
                except Exception:
                    counts["skipped"] += 1
                    continue   # one claim's failure must not abort the batch
                if v is None:
                    counts["skipped"] += 1
                else:
                    counts[v.verdict] = counts.get(v.verdict, 0) + 1
            conn.commit()
        return counts
    finally:
        conn.close()


class VerifierWorker:
    """Background daemon that periodically grounds un-verified world-fact/causal
    ACUs (Tavily + LLM judge). Flag-gated at bootstrap; needs a Tavily key.

    Mirrors AuditorWorker's lifecycle (start/stop) but is a simple periodic loop
    rather than a trigger queue — verification is not latency-sensitive.
    """

    def __init__(self, *, poll_interval_secs: float = 300.0, batch_limit: int = 10,
                 search_fn=None, judge_fn=None) -> None:
        self._poll = float(poll_interval_secs)
        self._limit = int(batch_limit)
        self._search_fn = search_fn
        self._judge_fn = judge_fn
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_summary: dict | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="acatalepsy-verifier", daemon=True)
        self._thread.start()

    def stop(self, timeout: float | None = 5.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
        self._thread = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            if self._stop_event.wait(self._poll):  # interruptible sleep
                return
            try:
                self.last_summary = run_verifier(
                    limit=self._limit, search_fn=self._search_fn, judge_fn=self._judge_fn)
            except Exception:
                pass  # never let the worker thread die
