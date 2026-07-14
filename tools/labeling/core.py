"""V1 labeling pass — pure core (no Monolith substrate imports).

Hard constraints (ratified 2026-06-11):
- READ-ONLY against outcome_traces: sqlite opened with mode=ro; no UPDATE,
  no failure_tags writes, no /rating calls, no metadata backfill.
- Labels live in a separate append-only JSONL ledger keyed by outcome_id.
  Exact row fields: outcome_id, label, species, labeled_at, batch_id, seed.
- No timers/daemons. No write path to monothink.md. surface_note content
  never enters any prompt assembly — this module renders to a human only.
- Curiosity canonicals are a separate partition (own ledger namespace,
  never pooled with surface_note stats); duds become a kill-candidate
  list FILE for E — kill_pull is never called from here.
"""
from __future__ import annotations

import json
import random
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

LABELS = ("hit", "interesting", "dud")

APPDATA_LOGS = Path.home() / "AppData" / "Roaming" / "Monolith" / "logs"
DB_PATH = APPDATA_LOGS / "turn_trace.sqlite3"
MILESTONES_PATH = Path.home() / "AppData" / "Roaming" / "Monolith" / "config" / "identity_milestones.json"
LEDGER_SN = APPDATA_LOGS / "labeling_ledger.jsonl"
LEDGER_CUR = APPDATA_LOGS / "labeling_ledger_curiosity.jsonl"
KILL_CANDIDATES = APPDATA_LOGS / "curiosity_kill_candidates.txt"


# ── population ────────────────────────────────────────────────────────

def load_population(db_path: Path) -> list[dict]:
    """All kind='rating' rows with a non-empty surface_note, read-only.

    species: 'tagged' if the row also carries >=1 failure_tag, else
    'note_only'. Verdict fields (rating_value/failure_tags/reason) ride
    along for post-label reveal but are excluded from blind_view().
    """
    uri = f"file:{Path(db_path).as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        rows = conn.execute(
            "SELECT id, recorded_at, rating_value, reason, metadata_json"
            " FROM outcome_traces WHERE kind='rating' ORDER BY recorded_at"
        ).fetchall()
    finally:
        conn.close()
    items = []
    for oid, ts, rv, reason, meta_raw in rows:
        try:
            meta = json.loads(meta_raw or "{}")
        except json.JSONDecodeError:
            continue
        note = str(meta.get("surface_note") or "").strip()
        if not note:
            continue
        tags = meta.get("failure_tags")
        tags = [str(t) for t in tags] if isinstance(tags, list) else []
        items.append({
            "outcome_id": oid,
            "recorded_at": ts,
            "note": note,
            "species": "tagged" if tags else "note_only",
            "rating_value": rv,
            "failure_tags": tags,
            "reason": reason,
        })
    return items


def blind_view(item: dict) -> dict:
    """What the labeler sees before submitting a label. Nothing else."""
    return {k: item[k] for k in ("outcome_id", "note", "recorded_at", "species")}


# ── sampler ───────────────────────────────────────────────────────────

def sample_batch(items: list[dict], *, labeled_ids: set, seed: int,
                 size: int = 20) -> list[dict]:
    """Stratified random batch: proportional by species, spread across
    date terciles within each species, deterministic under seed."""
    pool = [it for it in items if it["outcome_id"] not in labeled_ids]
    if not pool:
        return []
    size = min(size, len(pool))
    rng = random.Random(seed)

    by_species: dict[str, list[dict]] = {}
    for it in pool:
        by_species.setdefault(it["species"], []).append(it)

    # proportional quotas, rounding fixed up/down to hit `size` exactly
    species = sorted(by_species)
    quotas = {s: round(size * len(by_species[s]) / len(pool)) for s in species}
    quotas = {s: min(q, len(by_species[s])) for s, q in quotas.items()}
    drift = size - sum(quotas.values())
    for s in species:
        while drift > 0 and quotas[s] < len(by_species[s]):
            quotas[s] += 1
            drift -= 1
        while drift < 0 and quotas[s] > 0:
            quotas[s] -= 1
            drift += 1

    batch: list[dict] = []
    for s in species:
        members = sorted(by_species[s], key=lambda it: it["recorded_at"])
        n = len(members)
        terciles = [members[: n // 3], members[n // 3: 2 * n // 3], members[2 * n // 3:]]
        terciles = [t for t in terciles if t]
        picked: list[dict] = []
        # round-robin the quota across terciles so every era is represented
        want = [0] * len(terciles)
        for i in range(quotas[s]):
            want[i % len(terciles)] += 1
        for t, w in zip(terciles, want):
            picked.extend(rng.sample(t, min(w, len(t))))
        # tercile shortfall (tiny terciles) -> top up from leftovers
        if len(picked) < quotas[s]:
            left = [it for it in members if it not in picked]
            picked.extend(rng.sample(left, quotas[s] - len(picked)))
        batch.extend(picked)
    rng.shuffle(batch)
    return batch


# ── ledger (append-only JSONL) ────────────────────────────────────────

def append_label(ledger_path: Path, *, outcome_id, label: str, species: str,
                 batch_id: str, seed: int) -> dict:
    if label not in LABELS:
        raise ValueError(f"label must be one of {LABELS}, got {label!r}")
    row = {
        "outcome_id": outcome_id,
        "label": label,
        "species": species,
        "labeled_at": datetime.now(timezone.utc).isoformat(),
        "batch_id": batch_id,
        "seed": seed,
    }
    ledger_path = Path(ledger_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def read_ledger(ledger_path: Path) -> list[dict]:
    p = Path(ledger_path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def labeled_ids(ledger_path: Path) -> set:
    return {r["outcome_id"] for r in read_ledger(ledger_path)}


def next_batch_id(ledger_path: Path, namespace: str) -> str:
    seen = {r.get("batch_id") for r in read_ledger(ledger_path)}
    n = 1
    while f"{namespace}-{n:03d}" in seen:
        n += 1
    return f"{namespace}-{n:03d}"


# ── scoreboard (the ledger's first reader) ────────────────────────────

def tally(ledger_path: Path) -> dict:
    """Running totals over a partition's ledger: counts, hit-rate, and the
    trailing dud streak the 40-dud kill rule watches."""
    rows = read_ledger(ledger_path)
    counts = {label: 0 for label in LABELS}
    for r in rows:
        if r.get("label") in counts:
            counts[r["label"]] += 1
    total = sum(counts.values())
    streak = 0
    for r in reversed(rows):
        if r.get("label") == "dud":
            streak += 1
        else:
            break
    return {
        "counts": counts,
        "total": total,
        "hit_rate": (counts["hit"] / total) if total else 0.0,
        "dud_streak": streak,
    }


def scoreboard_line(t: dict, *, batch_pos: int, batch_size: int) -> str:
    c = t["counts"]
    return (f"tally {c['hit']}h/{c['interesting']}i/{c['dud']}d"
            f" | hit-rate {round(t['hit_rate'] * 100)}%"
            f" | dud streak {t['dud_streak']}"
            f" | batch {batch_pos}/{batch_size}")


# ── curiosity partition ───────────────────────────────────────────────

def load_curiosity(milestones_path: Path) -> list[dict]:
    """Surfaced curiosity canonicals as a separate labeling partition."""
    try:
        data = json.loads(Path(milestones_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    surfaced = data.get("curiosity_surfaced") or {}
    return [{
        "outcome_id": f"curiosity:{canonical}",
        "recorded_at": "",
        "note": canonical,
        "species": "curiosity",
        "rating_value": None,
        "failure_tags": [],
        "reason": f"surfaced {count}x",
    } for canonical, count in sorted(surfaced.items())]


def write_kill_candidates(ledger_path: Path, out_path: Path) -> int:
    """Dud-labeled curiosity canonicals -> a list file for E.
    Deliberately does NOT call kill_pull — E decides."""
    duds = [r["outcome_id"] for r in read_ledger(ledger_path)
            if r.get("species") == "curiosity" and r.get("label") == "dud"]
    canonicals = [d.removeprefix("curiosity:") for d in duds]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(canonicals) + ("\n" if canonicals else ""),
                        encoding="utf-8")
    return len(canonicals)
