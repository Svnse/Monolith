"""Tests for core/irp_chat_annotator.py — IRP annotations on assistant chat emissions."""
from __future__ import annotations

from pathlib import Path

import pytest

from core.irp_chat_annotator import annotate_assistant_payload
from core.acu_store import ACUStore


@pytest.fixture
def _isolate_acu_store(tmp_path: Path, monkeypatch):
    from core import db_connect as _dbc
    monkeypatch.setattr(_dbc, "DB_PATH", tmp_path / "acatalepsy.sqlite3", raising=True)
    monkeypatch.setenv("MONOLITH_DB_AUTHORIZER_STRICT", "1")
    from core.acatalepsy import schema, intake, canonical_log
    schema.migrate()
    for mod in (intake, canonical_log):
        tl = getattr(mod, "_tl", None)
        if tl is not None:
            for a in ("writer_conn", "reader_conn", "writer", "reader"):
                if hasattr(tl, a):
                    delattr(tl, a)


def test_annotates_matching_acus(_isolate_acu_store):
    store = ACUStore()
    try:
        store.ingest("substrate | carries | operator frame state across turns", source="model")
        store.ingest("precision over fluency | is | core value", source="model")
    finally:
        store.close()

    text = (
        "The substrate is designed to carry state that the operator frame "
        "cannot hold across turns. This is a core architectural choice that "
        "prioritizes precision over fluency in how the system maintains "
        "its internal representation."
    )
    payload = {"text": text, "agent": "assistant"}
    result = annotate_assistant_payload(text, payload)

    assert "irp_annotations" in result
    annotations = result["irp_annotations"]
    assert len(annotations) >= 1
    assert all("acu_id" in a for a in annotations)
    assert all("label" in a for a in annotations)
    assert all("overlap" in a for a in annotations)


def test_no_annotation_when_no_overlap(_isolate_acu_store):
    store = ACUStore()
    try:
        store.ingest("quantum computing uses qubits", source="model")
    finally:
        store.close()

    text = (
        "The weather forecast for tomorrow shows clear skies with a high "
        "of twenty degrees. Wind will be light from the northwest."
    )
    payload = {"text": text, "agent": "assistant"}
    result = annotate_assistant_payload(text, payload)

    assert "irp_annotations" not in result


def test_no_annotation_when_text_too_short(_isolate_acu_store):
    payload = {"text": "hi", "agent": "assistant"}
    result = annotate_assistant_payload("hi", payload)
    assert "irp_annotations" not in result


def test_no_annotation_when_store_empty(_isolate_acu_store):
    text = "A" * 100
    payload = {"text": text, "agent": "assistant"}
    result = annotate_assistant_payload(text, payload)
    assert "irp_annotations" not in result


def test_locked_acus_get_locked_label(_isolate_acu_store):
    store = ACUStore()
    try:
        store.ingest_locked(
            "Origin 0 / What I value: Precision over fluency",
            source="identity_origin_0",
        )
    finally:
        store.close()

    text = (
        "I value precision over fluency. This is an origin commitment "
        "that the substrate carries as immutable identity."
    )
    payload = {"text": text, "agent": "assistant"}
    result = annotate_assistant_payload(text, payload)

    assert "irp_annotations" in result
    labels = [a["label"] for a in result["irp_annotations"]]
    assert "LOCKED" in labels


def test_caps_at_max_annotations(_isolate_acu_store):
    store = ACUStore()
    try:
        for i in range(20):
            store.ingest(f"component {i} | handles | routing in the system", source="model")
    finally:
        store.close()

    text = (
        "The system has many components. Each component handles routing "
        "in its own way. Component number 1 through 20 all participate "
        "in the routing layer of the system architecture."
    )
    payload = {"text": text, "agent": "assistant"}
    result = annotate_assistant_payload(text, payload)

    if "irp_annotations" in result:
        assert len(result["irp_annotations"]) <= 5


def test_never_raises_on_error(monkeypatch):
    """Annotator must never break the chat path."""
    def boom(*a, **kw):
        raise RuntimeError("store crash")

    monkeypatch.setattr(ACUStore, "__init__", boom)

    text = "A" * 100
    payload = {"text": text, "agent": "assistant"}
    result = annotate_assistant_payload(text, payload)
    assert result is payload
