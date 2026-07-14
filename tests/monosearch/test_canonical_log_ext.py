import threading

from core.acatalepsy import canonical_log as cl


def test_read_recent_and_search_exist_and_return_lists(tmp_path, monkeypatch):
    """Readers return empty lists after the required boot migration.

    Read connections deliberately cannot run DDL.  Reproduce the real startup
    precondition here instead of making a read helper mutate an uninitialized
    database.
    """
    from core import db_connect
    from core.acatalepsy import schema

    monkeypatch.setattr(db_connect, "DB_PATH", tmp_path / "acatalepsy.sqlite3")
    thread_state = threading.local()
    monkeypatch.setattr(cl, "_tl", thread_state)
    schema.migrate()

    try:
        assert cl.read_recent(limit=5) == []
        assert cl.search("anything", limit=5) == []
    finally:
        for name in ("reader_conn", "writer_conn"):
            conn = getattr(thread_state, name, None)
            if conn is not None:
                conn.close()
