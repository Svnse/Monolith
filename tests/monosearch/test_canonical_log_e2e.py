"""Real end-to-end for the canonical_log adapter — no mocks.

The unit tests (test_canonical_log_adapter.py) stub `read_one` / hand-build
Events. This one exercises the REAL path: connect_acatalepsy(reader/writer) + the
authorizer + thread-local connections + the actual read_recent/search SQL. Uses
the same DB-isolation pattern as the acatalepsy suites.
"""
import threading


def test_adapter_searches_a_real_canonical_log_row(tmp_path, monkeypatch):
    from core import db_connect as _dbc
    db_path = tmp_path / "test_acatalepsy.sqlite3"
    monkeypatch.setattr(_dbc, "DB_PATH", db_path, raising=True)
    monkeypatch.setenv("MONOLITH_DB_AUTHORIZER_STRICT", "1")

    # Minimal real schema with the canonical_log table.
    conn = _dbc.connect_acatalepsy(role="migration")
    conn.executescript(
        "CREATE TABLE canonical_log ("
        " event_id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, kind TEXT NOT NULL,"
        " session_id TEXT, acu_id INTEGER, payload TEXT);"
    )
    conn.commit()
    conn.close()

    from core.acatalepsy import canonical_log as cl
    # Fresh thread-local connection pool so reader/writer re-open at the temp DB.
    monkeypatch.setattr(cl, "_tl", threading.local(), raising=True)

    # Write a real event through the real authorized write path.
    eid = cl.append("user_message", {"text": "the quick brown fox"}, session_id="ui:test")

    from core.monosearch.adapters.canonical_log import CanonicalLogAdapter
    a = CanonicalLogAdapter()

    # search() goes through the real cl.search() (LIKE on payload) -> _to_record.
    hits = a.search("quick brown", {}, 10)
    assert any(r.namespaced_id == f"clog:{eid}" for r in hits), "real search missed the seeded row"

    # get() goes through the real cl.read_one().
    r = a.get(f"clog:{eid}")
    assert r is not None
    assert r.text == "the quick brown fox"
    assert r.provenance.value == "user"
    assert r.recurrence_key is None  # search/lookup source, not a recurrence source
