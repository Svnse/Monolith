"""V1 labeling pass — tests for tools/labeling/core.py.

Hard constraints under test (ratified 2026-06-11):
- read-only against outcome_traces (DB byte-identical after load)
- labels in a separate append-only JSONL ledger, exact field set
- stratified sampling: 20/batch, proportional by species, date terciles, seeded
- curiosity partition: own namespace, never pooled; dud -> kill-candidate list
- no imports of substrate modules (turn_trace/monothink/acatalepsy/intake/acu)
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from tools.labeling import core as lab


# ── fixtures ──────────────────────────────────────────────────────────

def _mk_db(path: Path, rows: list[tuple]) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE outcome_traces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            turn_id TEXT NOT NULL,
            recorded_at TEXT NOT NULL,
            kind TEXT NOT NULL,
            rating_value INTEGER,
            reason TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}'
        )
        """
    )
    conn.executemany(
        "INSERT INTO outcome_traces(turn_id, recorded_at, kind, rating_value, reason, metadata_json)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


def _rating(ts: str, note: str | None, tags: list[str] | None, rv: int = 80) -> tuple:
    meta: dict = {}
    if note is not None:
        meta["surface_note"] = note
    if tags is not None:
        meta["failure_tags"] = tags
    return (f"T-{ts}", ts, "rating", rv, "reason text", json.dumps(meta))


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    """30 surface_note ratings: 10 tagged, 20 note_only, spread over 30 days.
    Plus noise rows that must be excluded (no note / wrong kind)."""
    rows = []
    for i in range(30):
        ts = f"2026-06-{(i % 30) + 1:02d}T12:00:00+00:00"
        tags = ["restatement_unpruned"] if i < 10 else []
        rows.append(_rating(ts, f"note {i}", tags))
    rows.append(_rating("2026-05-16T00:00:00+00:00", None, ["x"]))      # no note
    rows.append(("T-th", "2026-06-01T00:00:00+00:00", "thumbs_up", None, None, "{}"))
    p = tmp_path / "turn_trace.sqlite3"
    _mk_db(p, rows)
    return p


# ── population / species ─────────────────────────────────────────────

def test_population_is_note_rows_only(db: Path):
    items = lab.load_population(db)
    assert len(items) == 30
    assert all(it["note"].strip() for it in items)


def test_species_classification(db: Path):
    items = lab.load_population(db)
    tagged = [it for it in items if it["species"] == "tagged"]
    note_only = [it for it in items if it["species"] == "note_only"]
    assert len(tagged) == 10
    assert len(note_only) == 20


def test_load_is_read_only(db: Path):
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    lab.load_population(db)
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before


def test_blind_view_hides_verdict_fields(db: Path):
    it = lab.load_population(db)[0]
    blind = lab.blind_view(it)
    assert set(blind) == {"outcome_id", "note", "recorded_at", "species"}


# ── sampler ───────────────────────────────────────────────────────────

def test_batch_size_and_proportionality(db: Path):
    items = lab.load_population(db)
    batch = lab.sample_batch(items, labeled_ids=set(), seed=7, size=20)
    assert len(batch) == 20
    n_tagged = sum(1 for it in batch if it["species"] == "tagged")
    # population is 10/30 tagged -> quota round(20*10/30) = 7
    assert n_tagged == 7


def test_sampler_excludes_labeled_and_is_deterministic(db: Path):
    items = lab.load_population(db)
    b1 = lab.sample_batch(items, labeled_ids=set(), seed=7, size=20)
    b2 = lab.sample_batch(items, labeled_ids=set(), seed=7, size=20)
    assert [i["outcome_id"] for i in b1] == [i["outcome_id"] for i in b2]
    labeled = {it["outcome_id"] for it in b1}
    b3 = lab.sample_batch(items, labeled_ids=labeled, seed=8, size=20)
    assert not labeled & {it["outcome_id"] for it in b3}
    assert len(b3) == 10  # only 10 unlabeled remain


def test_sampler_spreads_across_date_terciles(db: Path):
    items = lab.load_population(db)
    batch = lab.sample_batch(items, labeled_ids=set(), seed=7, size=20)
    dates = sorted(it["recorded_at"] for it in items)
    t1, t2 = dates[len(dates) // 3], dates[2 * len(dates) // 3]
    buckets = [0, 0, 0]
    for it in batch:
        buckets[0 if it["recorded_at"] < t1 else 1 if it["recorded_at"] < t2 else 2] += 1
    assert all(b >= 3 for b in buckets)


# ── ledger ────────────────────────────────────────────────────────────

def test_ledger_append_only_and_exact_fields(tmp_path: Path):
    led = tmp_path / "ledger.jsonl"
    lab.append_label(led, outcome_id=5, label="hit", species="note_only",
                     batch_id="sn-001", seed=7)
    first = led.read_text(encoding="utf-8")
    lab.append_label(led, outcome_id=6, label="dud", species="tagged",
                     batch_id="sn-001", seed=7)
    text = led.read_text(encoding="utf-8")
    assert text.startswith(first)  # append-only: first row untouched
    rows = [json.loads(l) for l in text.splitlines()]
    assert len(rows) == 2
    assert set(rows[0]) == {"outcome_id", "label", "species", "labeled_at", "batch_id", "seed"}
    assert rows[1]["label"] == "dud"


def test_ledger_rejects_unknown_label(tmp_path: Path):
    with pytest.raises(ValueError):
        lab.append_label(tmp_path / "l.jsonl", outcome_id=1, label="meh",
                         species="note_only", batch_id="sn-001", seed=1)


def test_labeled_ids_roundtrip(tmp_path: Path):
    led = tmp_path / "ledger.jsonl"
    lab.append_label(led, outcome_id=5, label="hit", species="note_only",
                     batch_id="sn-001", seed=7)
    assert lab.labeled_ids(led) == {5}
    assert lab.labeled_ids(tmp_path / "missing.jsonl") == set()


# ── curiosity partition ───────────────────────────────────────────────

def test_curiosity_population_from_milestones(tmp_path: Path):
    ms = tmp_path / "identity_milestones.json"
    ms.write_text(json.dumps({"curiosity_surfaced": {"a | is | b": 3, "c | is | d": 1}}),
                  encoding="utf-8")
    items = lab.load_curiosity(ms)
    assert {it["outcome_id"] for it in items} == {"curiosity:a | is | b", "curiosity:c | is | d"}
    assert all(it["species"] == "curiosity" for it in items)


def test_curiosity_duds_become_kill_candidates_no_pooling(tmp_path: Path):
    led = tmp_path / "ledger_curiosity.jsonl"
    lab.append_label(led, outcome_id="curiosity:a | is | b", label="dud",
                     species="curiosity", batch_id="cur-001", seed=1)
    lab.append_label(led, outcome_id="curiosity:c | is | d", label="hit",
                     species="curiosity", batch_id="cur-001", seed=1)
    out = tmp_path / "kill_candidates.txt"
    n = lab.write_kill_candidates(led, out)
    assert n == 1
    assert out.read_text(encoding="utf-8").strip() == "a | is | b"


# ── constraint: no substrate imports ──────────────────────────────────

def test_v1_code_imports_no_substrate_modules():
    root = Path(__file__).resolve().parent.parent / "tools" / "labeling"
    forbidden = ("turn_trace", "monothink", "acatalepsy", "intake", "acu_store",
                 "acu_retrieval", "kill_pull")
    for py in root.glob("*.py"):
        src = py.read_text(encoding="utf-8")
        for line in src.splitlines():
            ls = line.strip()
            if ls.startswith(("import ", "from ")):
                assert not any(f in ls for f in forbidden), f"{py.name}: {ls}"


# ── scoreboard (the ledger's first reader) ────────────────────────────

def test_tally_counts_hit_rate_and_dud_streak(tmp_path: Path):
    led = tmp_path / "ledger.jsonl"
    for oid, label in [(1, "hit"), (2, "dud"), (3, "interesting"), (4, "dud"), (5, "dud")]:
        lab.append_label(led, outcome_id=oid, label=label, species="note_only",
                         batch_id="sn-001", seed=7)
    t = lab.tally(led)
    assert t["counts"] == {"hit": 1, "interesting": 1, "dud": 3}
    assert t["total"] == 5
    assert t["hit_rate"] == pytest.approx(1 / 5)
    assert t["dud_streak"] == 2  # trailing consecutive duds only


def test_tally_streak_resets_on_non_dud(tmp_path: Path):
    led = tmp_path / "ledger.jsonl"
    for oid, label in [(1, "dud"), (2, "dud"), (3, "hit")]:
        lab.append_label(led, outcome_id=oid, label=label, species="note_only",
                         batch_id="sn-001", seed=7)
    assert lab.tally(led)["dud_streak"] == 0


def test_tally_empty_or_missing_ledger(tmp_path: Path):
    t = lab.tally(tmp_path / "missing.jsonl")
    assert t == {"counts": {"hit": 0, "interesting": 0, "dud": 0},
                 "total": 0, "hit_rate": 0.0, "dud_streak": 0}


def test_scoreboard_line_renders_all_signals(tmp_path: Path):
    led = tmp_path / "ledger.jsonl"
    for oid, label in [(1, "hit"), (2, "dud"), (3, "dud")]:
        lab.append_label(led, outcome_id=oid, label=label, species="note_only",
                         batch_id="sn-001", seed=7)
    line = lab.scoreboard_line(lab.tally(led), batch_pos=3, batch_size=20)
    for must in ("1h", "0i", "2d", "33%", "streak 2", "3/20"):
        assert must in line, line
