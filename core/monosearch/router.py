"""The MonoSearch router (spec §6). Fans a query out to all registered adapters,
merges results, ranks with evidence_tier as a HARD primary ordering (a lower
tier can never outrank a higher one), then relevance+salience within a tier.
Owns no content.
"""
from __future__ import annotations

import re
from datetime import datetime

from core.monosearch import registry, salience
from core.monosearch.record import Record


def _iso_to_epoch(s: str | None) -> float | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s).timestamp()
    except (ValueError, TypeError):
        return None


def _now_epoch() -> float:
    return datetime.now().timestamp()


def _salience_boost(rec: Record, now: float) -> float:
    if rec.recurrence_key is None:
        return 0.0
    row = salience.get_row(rec.recurrence_key, rec.source)
    if row is None:
        return 0.0
    return salience._salience(row["count"], row["last_seen"], now)


# Friendly source aliases so the model can target a store by concept rather than
# memorizing internal adapter names: `search "X" source=knowledge`. Passthrough
# for exact adapter names.
SOURCE_ALIASES: dict[str, str] = {
    "faults": "fault_traces", "mistakes": "fault_traces", "failures": "fault_traces",
    "verdicts": "fault_traces", "fault_traces": "fault_traces",
    "ratings": "outcome_traces", "grades": "outcome_traces", "outcomes": "outcome_traces",
    "rated": "outcome_traces", "outcome_traces": "outcome_traces",
    "knowledge": "acatalepsy-acus", "claims": "acatalepsy-acus", "facts": "acatalepsy-acus",
    "acus": "acatalepsy-acus", "acatalepsy-acus": "acatalepsy-acus",
    "warrants": "acatalepsy-warrants", "warrant": "acatalepsy-warrants",
    "claim_graph": "acatalepsy-warrants", "proof": "acatalepsy-warrants",
    "rationale": "acatalepsy-warrants", "defeaters": "acatalepsy-warrants",
    "acatalepsy-warrants": "acatalepsy-warrants",
    "conversation": "canonical_log", "messages": "canonical_log", "log": "canonical_log",
    "said": "canonical_log", "history": "canonical_log", "canonical_log": "canonical_log",
    "turns": "turn_trace", "turn_trace": "turn_trace",
    "memory": "continuity", "pins": "continuity", "continuity": "continuity",
    "notes": "mononote", "note": "mononote", "mononote": "mononote",
    "vault": "mononote", "markdown_notes": "mononote",
    "bearing": "bearing", "goals": "bearing", "coherence": "bearing",
    "identity": "identity", "self": "identity",
    "curiosity": "identity_signals", "pulls": "identity_signals", "signals": "identity_signals",
    "identity_signals": "identity_signals",
    "stages": "stage_traces", "stage_trace": "stage_traces", "stage_traces": "stage_traces",
    "reminders": "plan_reminders", "plan_reminders": "plan_reminders",
    "investigations": "investigations", "research": "investigations",
    "lag": "lag_watch", "lag_watch": "lag_watch", "turn_shapes": "lag_watch",
    "health": "runtime_health", "runtime_health": "runtime_health",
    "tools": "tools", "tool": "tools", "catalog": "tools", "tool_catalog": "tools",
    "skills": "skills", "skill": "skills", "skill_catalog": "skills",
}

_SOURCE_FAMILIES: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (
        re.compile(r"\b(tools?|tool[\s_-]*catalog|schemas?|executable[\s_-]*tools?)\b"),
        "tools",
        "please use source='tools' for executable tool schemas",
    ),
    (
        re.compile(r"\b(skills?|skill[\s_-]*catalog|procedures?|capabilit(?:y|ies))\b"),
        "skills",
        "please use source='skills' for skill cards, or meta='capabilities' for tools+skills",
    ),
    (
        re.compile(r"\b(faults?|failures?|mistakes?|verdicts?|failing)\b"),
        "fault_traces",
        "please use source='faults'",
    ),
    (
        re.compile(r"\b(ratings?|grades?|outcomes?|rated)\b"),
        "outcome_traces",
        "please use source='ratings'",
    ),
    (
        re.compile(r"\b(knowledge|facts?|acus?|acu[\s_-]*store)\b|\bclaims?\b(?![\s_/-]*(?:graph|evidence))"),
        "acatalepsy-acus",
        "please use source='knowledge' or source='claims'",
    ),
    (
        re.compile(
            r"(?:\bwarrants?\b|\bclaim[\s_/-]*graph\b|\bclaim[\s_/-]*evidence\b|"
            r"\bevidence[\s_/-]*graph\b|\bproof[\s_/-]*graph\b|\brationale\b|\bdefeaters?\b)"
        ),
        "acatalepsy-warrants",
        "please use source='warrants' or source='claim_graph'",
    ),
    (
        re.compile(r"\b(conversation|messages?|history|canonical[\s_-]*log|timeline|chat[\s_-]*log)\b"),
        "canonical_log",
        "please use source='history' or source='canonical_log'",
    ),
    (
        re.compile(r"\b(turns?|turn[\s_-]*trace|frames?|frame[\s_-]*trace|run[\s_-]*trace)\b"),
        "turn_trace",
        "please use source='turns'",
    ),
    (
        re.compile(r"\b(memory|memories|pins?|continuity|scratchpad)\b"),
        "continuity",
        "please use source='memory' or source='pins'",
    ),
    (
        re.compile(r"\b(notes?|mononote|markdown[\s_-]*notes?|vault)\b"),
        "mononote",
        "please use source='notes' or source='mononote'",
    ),
    (
        re.compile(r"\b(bearing|bearings|goals?|coherence)\b"),
        "bearing",
        "please use source='bearing'",
    ),
    (
        re.compile(r"\b(identity|self[\s_-]*knowledge|origin[\s_-]*0|emergent)\b"),
        "identity",
        "please use source='identity'",
    ),
    (
        re.compile(r"\b(curiosity|pulls?|signals?|identity[\s_-]*signals?)\b"),
        "identity_signals",
        "please use source='curiosity'",
    ),
    (
        re.compile(r"\b(stages?|stage[\s_-]*trace|interceptor|prompt[\s_-]*build)\b"),
        "stage_traces",
        "please use source='stages'",
    ),
    (
        re.compile(r"\b(reminders?|plans?|plan[\s_-]*reminders?)\b"),
        "plan_reminders",
        "please use source='reminders'",
    ),
    (
        re.compile(r"\b(investigations?|research|explorations?)\b"),
        "investigations",
        "please use source='investigations'",
    ),
    (
        re.compile(r"\b(lag|turn[\s_-]*shapes?|latency|slowdowns?)\b"),
        "lag_watch",
        "please use source='lag'",
    ),
    (
        re.compile(r"\b(health|runtime[\s_-]*health|diagnostics?)\b"),
        "runtime_health",
        "please use source='health'",
    ),
)


def _source_key(name: str | None) -> str:
    return str(name or "").strip().lower()


def _split_source_combo(key: str) -> list[str]:
    return [
        part.strip()
        for part in re.split(r"\s*(?:/|\||,|;|\bor\b)\s*", key)
        if part.strip()
    ]


def _source_family(key: str) -> tuple[str | None, str | None]:
    if not key:
        return None, None
    for pattern, source, hint in _SOURCE_FAMILIES:
        if pattern.search(key):
            return source, hint
    return None, None


def _canonical_intent_text(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", value.lower())).strip("_")


def _skill_source_match(key: str, *, exact_only: bool = False) -> tuple[str | None, str | None]:
    """Detect source values that are actually skill/tool discovery intents."""
    canonical = _canonical_intent_text(key)
    if not canonical:
        return None, None
    try:
        from core.skill_registry import get_tool, list_tools
    except Exception:
        return None, None

    exact = get_tool(canonical) or get_tool(key)
    if exact is not None:
        return (
            exact.name,
            f"source={key!r} looks like the {exact.name!r} skill/tool; "
            f"please use source='skills' query='{exact.name}' or "
            f"meta='capabilities' query='{exact.name}'",
        )
    if exact_only:
        return None, None

    tokens = [t for t in re.findall(r"[a-z0-9]+", key.lower()) if len(t) > 2]
    if not tokens:
        return None, None
    best_name: str | None = None
    best_score = 0
    for spec in list_tools():
        blob = " ".join((
            spec.name,
            spec.name.replace("_", " "),
            spec.description,
            " ".join(spec.legacy_ops),
        )).lower()
        score = 0
        if canonical and canonical in spec.name:
            score += 60
        if key.lower() in blob:
            score += 40
        for token in tokens:
            if token in blob:
                score += 8
        if score > best_score:
            best_name = spec.name
            best_score = score
    if best_name and best_score >= max(16, min(len(tokens), 3) * 8):
        return (
            f"{best_name} {key}",
            f"source={key!r} looks like skill/tool discovery; "
            f"please use source='skills' query='{key}' or "
            f"meta='capabilities' query='{key}'",
        )
    return None, None


def _combo_resolution(key: str) -> tuple[str | None, str | None]:
    parts = _split_source_combo(key)
    if len(parts) < 2:
        return None, None
    resolved = [resolve_source(part) for part in parts]
    if not all(item is not None for item in resolved):
        return None, None
    unique = set(resolved)
    if len(unique) == 1:
        choices = " or ".join(f"source='{part}'" for part in parts[:3])
        return resolved[0], f"source={key!r} combines aliases for one store; please use {choices}"
    if unique.issubset({"tools", "skills"}):
        return "skills", (
            f"source={key!r} names catalog stores; please use meta='capabilities', "
            "or source='tools' / source='skills' separately"
        )
    return None, (
        f"source={key!r} names multiple stores; please use one source per call, "
        "or omit source for cross-store search"
    )


def resolve_source(name: str | None) -> str | None:
    """Friendly alias / exact name -> adapter name, or None if not given/unknown."""
    key = _source_key(name)
    if not key:
        return None
    exact = SOURCE_ALIASES.get(key)
    if exact is not None:
        return exact
    combo, combo_hint = _combo_resolution(key)
    if combo is not None:
        return combo
    if combo_hint:
        return None
    skill_query, _ = _skill_source_match(key, exact_only=True)
    if skill_query is not None:
        return "skills"
    family, _ = _source_family(key)
    if family is not None:
        return family
    skill_query, _ = _skill_source_match(key)
    if skill_query is not None:
        return "skills"
    return None


def source_usage_hint(name: str | None) -> str | None:
    """Return a terse correction for recognizable but non-canonical source names."""
    key = _source_key(name)
    if not key or key in SOURCE_ALIASES:
        return None
    _, combo_hint = _combo_resolution(key)
    if combo_hint:
        return combo_hint
    _, skill_hint = _skill_source_match(key, exact_only=True)
    if skill_hint:
        return skill_hint
    _, family_hint = _source_family(key)
    if family_hint:
        return f"source={key!r} recognized; {family_hint}"
    _, skill_hint = _skill_source_match(key)
    if skill_hint:
        return skill_hint
    return None


def source_query_hint(name: str | None) -> str | None:
    """Return extra query text when source was really a skill/tool intent."""
    key = _source_key(name)
    if not key or key in SOURCE_ALIASES:
        return None
    combo, combo_hint = _combo_resolution(key)
    if combo is not None or combo_hint:
        return None
    family, _ = _source_family(key)
    if family is not None:
        return None
    skill_query, _ = _skill_source_match(key)
    return skill_query


def _query_with_source_hint(query: str, source: object) -> str:
    hint = source_query_hint(str(source or ""))
    if not hint:
        return query
    text = str(query or "").strip()
    if hint.lower() in text.lower():
        return text
    return " ".join(part for part in (text, hint) if part).strip()


def search(query: str, filters: dict | None, limit: int = 20) -> list[Record]:
    filters = filters or {}
    cutoff = _iso_to_epoch(filters.get("since"))
    now = _now_epoch()
    limit = max(1, int(limit))

    # Source targeting: if a source is named, fan out ONLY to it (the model wants
    # that store — no diversity cap). Otherwise fan out to all.
    src = resolve_source(filters.get("source"))
    query = _query_with_source_hint(query, filters.get("source"))
    if src is not None:
        adapter = registry.get_adapter(src)
        adapters = [adapter] if adapter is not None else []
    else:
        adapters = registry.all_adapters()

    collected: list[Record] = []
    for adapter in adapters:
        try:
            collected.extend(adapter.search(query, filters, limit))
        except Exception:
            continue  # one adapter failing must not break the others
    if cutoff is not None:
        collected = [r for r in collected if r.ts is None or r.ts >= cutoff]
    # Primary: evidence_tier (IntEnum, lower = higher priority — the hard guard).
    # Secondary: salience boost (recurrence × decay), then recency.
    collected.sort(key=lambda r: (
        int(r.evidence_tier),
        -_salience_boost(r, now),
        -(r.ts or 0.0),
    ))

    if src is not None:
        return collected[:limit]  # scoped: deep dive, no cap

    # Un-scoped default: per-source diversity cap so one high-volume LITERAL store
    # (canonical_log) can't fill every slot and bury the other stores. Backfill
    # from the capped overflow only if we're short — at that point no other source
    # is being crowded out, so returning extras from a single source is fine.
    cap = max(2, (limit + 2) // 3)
    out: list[Record] = []
    overflow: list[Record] = []
    counts: dict[str, int] = {}
    for r in collected:
        if counts.get(r.source, 0) >= cap:
            overflow.append(r)
            continue
        out.append(r)
        counts[r.source] = counts.get(r.source, 0) + 1
        if len(out) >= limit:
            return out
    for r in overflow:
        if len(out) >= limit:
            break
        out.append(r)
    return out[:limit]


def get(namespaced_id: str) -> Record | None:
    prefix = namespaced_id.split(":", 1)[0]
    for adapter in registry.all_adapters():
        # adapter.name is the source; ids are source-prefixed by convention
        try:
            rec = adapter.get(namespaced_id)
        except Exception:
            rec = None
        if rec is not None:
            return rec
    return None
