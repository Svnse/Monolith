"""Test isolation for the MonoSearch suite.

Mirrors the autouse fixture in tests/test_fault_response.py: each test gets its
own turn_trace SQLite DB so reads against fault_traces start empty, independent
of any populated dev DB on the machine. Scoped to tests/monosearch/ only so it
cannot perturb the rest of the test tree.
"""
from __future__ import annotations

import os

import pytest

import core.turn_trace as tt


@pytest.fixture(autouse=True)
def isolated_turn_trace_db(tmp_path):
    """Give each MonoSearch test its own turn_trace SQLite database."""
    db = tmp_path / "test_monosearch_turn_trace.sqlite3"
    tt.set_db_path(db)
    os.environ["MONOLITH_TURN_TRACE_V1"] = "1"
    yield db
    tt.set_db_path(None)


@pytest.fixture(autouse=True)
def isolated_salience_db(tmp_path):
    """Give each MonoSearch test its own (empty) salience ledger, and CLOSE the
    connection in teardown so tmp_path can be removed on Windows."""
    from core.monosearch import salience
    salience.set_db_path(tmp_path / "test_monosearch_salience.sqlite3")
    salience.ensure_schema()
    yield
    salience.close()
