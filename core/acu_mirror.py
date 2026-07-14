"""Read-only ACU Mirror.

ACU Mirror is an inspection surface over the Acatalepsy substrate. It computes
identity-facing signals in memory and opens the DB with the reader role, so a
mirror run cannot promote, score-persist, retire, or otherwise mutate claims.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from core import identity_emergence as _emergence
from core.acatalepsy import decay
from core.acatalepsy.normalize import parse_triple
from core.db_connect import connect_acatalepsy
from core.identity_alignment import (
    STABILITY_THRESHOLD,
    score_confidentity,
    stability_score,
)


DEFAULT_THRESHOLD = _emergence._DEFAULT_THRESHOLD
DEFAULT_NEAR_BAND = 0.05
DEFAULT_LIMIT = 8
SCAN_CAP = 500
PENDING_CAP = 50

_FUNCTIONAL_RELATIONS = {
    "is",
    "equals",
    "located_in",
    "born_in",
    "died_in",
    "capital_of",
    "currency_of",
    "atomic_number",
    "president",
    "capital",
}


def _coerce_float(value: Any, default: float, *, min_value: float | None = None) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        out = default
    if min_value is not None:
        out = max(min_value, out)
    return out


def _coerce_int(value: Any, default: int, *, min_value: int = 1, max_value: int = 50) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        out = default
    return max(min_value, min(max_value, out))


def _parse_ts(value: Any) -> datetime | None:
    try:
        ts = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts


def _age_days(row: dict[str, Any], now: datetime) -> float | None:
    ts = (
        _parse_ts(row.get("last_touched_ts"))
        or _parse_ts(row.get("last_seen"))
        or _parse_ts(row.get("created_at"))
    )
    if ts is None:
        return None
    return max(0.0, (now - ts).total_seconds() / 86400.0)


def _round(value: Any, digits: int = 4) -> float:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return 0.0


def _scope(row: dict[str, Any]) -> str:
    domain = str(row.get("domain") or row.get("scope") or "").strip().lower()
    if domain in {"self", "world", "meta"}:
        return domain
    kind = str(row.get("kind") or "").strip().lower()
    provenance = str(row.get("provenance") or "").strip().lower()
    source = str(row.get("source") or "").strip().lower()
    if kind in {"world-fact", "world_fact", "causal"} or provenance in {"world", "tool"}:
        return "world"
    if kind in {"self", "identity"} or provenance == "self" or source in {"model", "assistant"}:
        return "self"
    if "meta" in kind or "meta" in domain:
        return "meta"
    return "meta"


def _canonical(row: dict[str, Any]) -> str:
    return str(row.get("canonical") or row.get("canonical_form") or "").strip()


def _read_active_acus(conn, scan_cap: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT
            id, canonical, source, provenance, kind, domain, l_level, reinforcement,
            state, truth, truth_confidence, truth_checked_at, created_at, last_seen,
            last_touched_ts, confidentity, locked, lock_reason, canonical_triple,
            candidate_id, decision_id, merged_into
        FROM acus
        WHERE merged_into IS NULL AND COALESCE(state, 'active') = 'active'
        ORDER BY
            COALESCE(confidentity, 0) DESC,
            reinforcement DESC,
            COALESCE(last_touched_ts, last_seen, created_at, '') DESC
        LIMIT ?
        """,
        (int(scan_cap),),
    ).fetchall()
    return [dict(r) for r in rows]


def _read_pending_candidates(conn, limit: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT
            id, canonical_form, source, reason, reinforcement_count,
            contradicts_acu_id, state, created_at, auditor_run_id
        FROM acu_candidates
        WHERE state = 'pending'
        ORDER BY id DESC
        LIMIT ?
        """,
        (int(limit),),
    ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        item = dict(r)
        item["canonical"] = item.pop("canonical_form", "")
        item["scope"] = _scope(item)
        out.append(item)
    return out


def _surface_reason(
    row: dict[str, Any],
    *,
    threshold: float,
    corpus: str,
    confidentity: float,
    stability: float,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    canonical = _canonical(row)
    if int(row.get("locked", 0) or 0):
        reasons.append("locked/origin material")
    if not canonical:
        reasons.append("empty canonical")
    elif canonical in corpus:
        reasons.append("already reflected in identity corpus")
    if confidentity < threshold:
        reasons.append(f"below threshold by {_round(threshold - confidentity)}")
    if stability < STABILITY_THRESHOLD:
        reasons.append(f"below stability gate by {_round(STABILITY_THRESHOLD - stability)}")
    surfaceable = not reasons
    if surfaceable:
        reasons.append("would surface through identity_review")
    return surfaceable, reasons


def _mirror_item(
    row: dict[str, Any],
    *,
    corpus: str,
    threshold: float,
    now: datetime,
    backend: str | None,
) -> dict[str, Any]:
    conf = score_confidentity(row, corpus, backend=backend)
    stability = stability_score(row)
    surfaceable, reasons = _surface_reason(
        row,
        threshold=threshold,
        corpus=corpus,
        confidentity=conf,
        stability=stability,
    )
    raw = float(row.get("reinforcement", 0) or 0)
    effective = decay.effective_reinforcement(row, now=now)
    age = _age_days(row, now)
    item = {
        "id": row.get("id"),
        "canonical": _canonical(row),
        "scope": _scope(row),
        "provenance": row.get("provenance"),
        "source": row.get("source"),
        "kind": row.get("kind"),
        "l_level": row.get("l_level") or "L1",
        "reinforcement": int(raw),
        "effective_reinforcement": _round(effective),
        "decay_factor": _round(effective / raw) if raw > 0 else 0.0,
        "age_days": _round(age, 2) if age is not None else None,
        "stored_confidentity": _round(row.get("confidentity")),
        "confidentity": _round(conf),
        "stability": _round(stability),
        "truth": row.get("truth"),
        "locked": bool(int(row.get("locked", 0) or 0)),
        "surfaceable": surfaceable,
        "surface_reasons": reasons,
    }
    if row.get("lock_reason"):
        item["lock_reason"] = row.get("lock_reason")
    return item


def _relation_contradictions(conn, limit: int) -> list[dict[str, Any]]:
    try:
        rows = conn.execute(
            """
            SELECT
                r.source_id, r.target_id, r.score,
                a.canonical AS source_canonical,
                b.canonical AS target_canonical
            FROM acu_relations r
            LEFT JOIN acus a ON a.id = r.source_id
            LEFT JOIN acus b ON b.id = r.target_id
            WHERE r.relation = 'contradicts'
            ORDER BY COALESCE(r.updated_at, r.created_at, '') DESC, r.id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    except Exception:
        return []
    return [
        {
            "type": "relation",
            "source_id": r["source_id"],
            "target_id": r["target_id"],
            "score": _round(r["score"]),
            "source": r["source_canonical"],
            "target": r["target_canonical"],
        }
        for r in rows
    ]


def _candidate_contradictions(conn, limit: int) -> list[dict[str, Any]]:
    try:
        rows = conn.execute(
            """
            SELECT
                c.id AS candidate_id,
                c.canonical_form,
                c.contradicts_acu_id,
                a.canonical AS target_canonical
            FROM acu_candidates c
            LEFT JOIN acus a ON a.id = c.contradicts_acu_id
            WHERE c.state = 'pending' AND c.contradicts_acu_id IS NOT NULL
            ORDER BY c.id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    except Exception:
        return []
    return [
        {
            "type": "pending_candidate",
            "candidate_id": r["candidate_id"],
            "candidate": r["canonical_form"],
            "target_id": r["contradicts_acu_id"],
            "target": r["target_canonical"],
        }
        for r in rows
    ]


def _functional_conflicts(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[tuple[dict[str, Any], Any]]] = {}
    for row in rows:
        triple = parse_triple(_canonical(row))
        if triple is None or triple.relation not in _FUNCTIONAL_RELATIONS:
            continue
        buckets.setdefault((triple.entity_a, triple.relation), []).append((row, triple))

    conflicts: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for (entity, relation), items in buckets.items():
        for i, (left, lt) in enumerate(items):
            for right, rt in items[i + 1:]:
                if lt.entity_b == rt.entity_b:
                    continue
                pair = tuple(sorted((int(left["id"]), int(right["id"]))))
                if pair in seen:
                    continue
                seen.add(pair)
                conflicts.append({
                    "type": "functional_conflict",
                    "relation": relation,
                    "entity": entity,
                    "left_id": left["id"],
                    "left": _canonical(left),
                    "right_id": right["id"],
                    "right": _canonical(right),
                })
                if len(conflicts) >= limit:
                    return conflicts
    return conflicts


def _scope_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"self": 0, "world": 0, "meta": 0}
    for item in items:
        scope = item.get("scope")
        counts[scope if scope in counts else "meta"] += 1
    return counts


def build_snapshot(
    *,
    threshold: float = DEFAULT_THRESHOLD,
    near_band: float = DEFAULT_NEAR_BAND,
    limit: int = DEFAULT_LIMIT,
    scan_cap: int = SCAN_CAP,
    backend: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Return a read-only mirror snapshot.

    This function never calls the identity emergence detector because that path
    persists confidentity and updates the milestone ledger. The mirror computes
    its scores in memory from a reader connection.
    """
    threshold = _coerce_float(threshold, DEFAULT_THRESHOLD, min_value=0.0)
    near_band = _coerce_float(near_band, DEFAULT_NEAR_BAND, min_value=0.0)
    limit = _coerce_int(limit, DEFAULT_LIMIT, min_value=1, max_value=50)
    scan_cap = _coerce_int(scan_cap, SCAN_CAP, min_value=limit, max_value=5000)
    now_dt = now or datetime.now(timezone.utc)
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=timezone.utc)

    conn = connect_acatalepsy(role="reader")
    try:
        rows = _read_active_acus(conn, scan_cap)
        pending = _read_pending_candidates(conn, min(PENDING_CAP, max(limit * 2, limit)))
        contradictions = (
            _relation_contradictions(conn, limit)
            + _candidate_contradictions(conn, limit)
            + _functional_conflicts(rows, limit)
        )
    finally:
        conn.close()

    corpus = _emergence.compute_identity_corpus()
    items = [
        _mirror_item(row, corpus=corpus, threshold=threshold, now=now_dt, backend=backend)
        for row in rows
    ]
    unlocked = [i for i in items if not i.get("locked")]

    surfaceable = [i for i in unlocked if i["surfaceable"]]
    surfaceable.sort(key=lambda i: (i["confidentity"], i["stability"], i["reinforcement"]), reverse=True)

    near_floor = max(0.0, threshold - near_band)
    near_threshold = [
        i for i in unlocked
        if near_floor <= i["confidentity"] < threshold
    ]
    near_threshold.sort(key=lambda i: (i["confidentity"], i["stability"]), reverse=True)

    blocked = [
        i for i in unlocked
        if i["confidentity"] >= threshold and not i["surfaceable"]
    ]
    blocked.sort(key=lambda i: (i["confidentity"], i["stability"]), reverse=True)

    decayed = sorted(
        items,
        key=lambda i: (float(i["reinforcement"]) - float(i["effective_reinforcement"]), i["age_days"] or 0),
        reverse=True,
    )

    return {
        "ok": True,
        "mode": "read_only",
        "threshold": threshold,
        "near_band": near_band,
        "stability_threshold": STABILITY_THRESHOLD,
        "scanned_active_acus": len(rows),
        "scope_counts": _scope_counts(items),
        "surfaceable": surfaceable[:limit],
        "near_threshold": near_threshold[:limit],
        "blocked": blocked[:limit],
        "contradictions": contradictions[:limit],
        "pending_candidates": pending[:limit],
        "pending_scope_counts": _scope_counts(pending),
        "decay": {
            "active": decay.decay_enabled(),
            "mode": "active" if decay.decay_enabled() else "preview_only",
            "items": decayed[:limit],
        },
    }


def _one_line(item: dict[str, Any]) -> str:
    bits = [
        f"#{item.get('id')}",
        str(item.get("canonical") or ""),
        f"conf={item.get('confidentity')}",
        f"stored={item.get('stored_confidentity')}",
        f"stab={item.get('stability')}",
        f"{item.get('scope')}/{item.get('provenance') or 'unknown'}",
        f"{item.get('l_level')}",
    ]
    reasons = item.get("surface_reasons") or []
    if reasons:
        bits.append("reason=" + "; ".join(str(r) for r in reasons[:2]))
    return "  - " + " | ".join(bits)


def _section(title: str, items: list[dict[str, Any]], *, empty: str) -> list[str]:
    lines = [title]
    if not items:
        lines.append(f"  ({empty})")
        return lines
    lines.extend(_one_line(i) for i in items)
    return lines


def format_snapshot(snapshot: dict[str, Any]) -> str:
    counts = snapshot.get("scope_counts") or {}
    pending_counts = snapshot.get("pending_scope_counts") or {}
    lines = [
        (
            "[acu_mirror: read-only snapshot; "
            f"scanned={snapshot.get('scanned_active_acus')} active ACUs; "
            f"threshold={snapshot.get('threshold')}; band={snapshot.get('near_band')}]"
        ),
        (
            "scopes: "
            f"self={counts.get('self', 0)} world={counts.get('world', 0)} meta={counts.get('meta', 0)}; "
            f"pending self={pending_counts.get('self', 0)} "
            f"world={pending_counts.get('world', 0)} meta={pending_counts.get('meta', 0)}"
        ),
    ]
    lines.extend(_section("would_surface_identity_review", snapshot.get("surfaceable") or [], empty="none"))
    lines.extend(_section("near_threshold", snapshot.get("near_threshold") or [], empty="none"))
    lines.extend(_section("blocked_above_threshold", snapshot.get("blocked") or [], empty="none"))

    lines.append("contradictions")
    contradictions = snapshot.get("contradictions") or []
    if not contradictions:
        lines.append("  (none detected in scanned rows)")
    else:
        for c in contradictions:
            ctype = c.get("type")
            if ctype == "functional_conflict":
                lines.append(
                    f"  - functional #{c.get('left_id')} vs #{c.get('right_id')}: "
                    f"{c.get('left')} <> {c.get('right')}"
                )
            elif ctype == "pending_candidate":
                lines.append(
                    f"  - pending candidate #{c.get('candidate_id')} contradicts "
                    f"#{c.get('target_id')}: {c.get('candidate')} <> {c.get('target')}"
                )
            else:
                lines.append(
                    f"  - relation #{c.get('source_id')} -> #{c.get('target_id')}: "
                    f"{c.get('source')} <> {c.get('target')}"
                )

    decay_info = snapshot.get("decay") or {}
    lines.append(f"decay ({decay_info.get('mode', 'preview_only')})")
    for item in (decay_info.get("items") or [])[:DEFAULT_LIMIT]:
        lines.append(
            "  - "
            f"#{item.get('id')} {item.get('canonical')} | raw={item.get('reinforcement')} "
            f"effective={item.get('effective_reinforcement')} factor={item.get('decay_factor')} "
            f"age_days={item.get('age_days')}"
        )
    if not decay_info.get("items"):
        lines.append("  (no active ACUs scanned)")

    lines.append("pending_candidates")
    pending = snapshot.get("pending_candidates") or []
    if not pending:
        lines.append("  (none)")
    else:
        for p in pending:
            parts = [
                f"#{p.get('id')}",
                str(p.get("canonical") or ""),
                f"{p.get('scope')}/{p.get('source') or 'unknown'}",
            ]
            if p.get("contradicts_acu_id"):
                parts.append(f"contradicts=#{p.get('contradicts_acu_id')}")
            if p.get("reason"):
                parts.append(f"reason={p.get('reason')}")
            lines.append("  - " + " | ".join(parts))

    return "\n".join(lines)


def snapshot_json(snapshot: dict[str, Any]) -> str:
    return json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True)
