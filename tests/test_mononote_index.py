from __future__ import annotations

from pathlib import Path


def test_mononote_index_refreshes_note_metadata(tmp_path: Path, monkeypatch) -> None:
    from core import paths
    from core.mononote import index, store

    monkeypatch.setattr(paths, "NOTES_DIR", tmp_path / "notes", raising=False)
    monkeypatch.setattr(paths, "LOG_DIR", tmp_path / "logs", raising=False)
    store.write_note("Alpha", "one")

    notes = index.refresh_notes()
    rows = index.list_indexed_notes()

    assert len(notes) == 1
    assert len(rows) == 1
    assert rows[0]["safe_title"] == "Alpha"
    assert rows[0]["sha256"] == notes[0].sha256
    assert (tmp_path / "logs" / "mononote.sqlite3").exists()


def test_mononote_index_persists_canvas_positions(tmp_path: Path, monkeypatch) -> None:
    from core import paths
    from core.mononote import index

    monkeypatch.setattr(paths, "LOG_DIR", tmp_path / "logs", raising=False)

    index.write_canvas_position("note-1", 12.5, 33.0)

    assert index.read_canvas_position("note-1") == (12.5, 33.0)


def test_mononote_index_extracts_links_tags_and_backlinks(tmp_path: Path, monkeypatch) -> None:
    from core import paths
    from core.mononote import index, store

    monkeypatch.setattr(paths, "NOTES_DIR", tmp_path / "notes", raising=False)
    monkeypatch.setattr(paths, "LOG_DIR", tmp_path / "logs", raising=False)
    alpha = store.write_note(
        "Alpha",
        "---\ntags: [project, graph]\n---\n# Alpha\n\nLinks to [[Beta]] and [Gamma](Gamma.md).\nAlso #daily.",
    )
    beta = store.write_note("Beta", "Back target")
    gamma = store.write_note("Gamma", "Other target")

    index.refresh_notes()

    backlinks = index.read_backlinks(beta.note_id)
    assert [item.source_title for item in backlinks] == ["Alpha"]
    assert backlinks[0].kind == "wikilink"
    graph = index.read_note_graph()
    node_by_title = {node.safe_title: node for node in graph.nodes}
    assert node_by_title["Alpha"].tags == ("daily", "graph", "project")
    assert node_by_title["Alpha"].degree == 2
    assert {edge.target for edge in graph.edges} >= {beta.note_id, gamma.note_id}
    assert index.resolve_note_ref("Gamma")["note_id"] == gamma.note_id
    assert index.resolve_note_ref("Gamma.md")["note_id"] == gamma.note_id
    assert alpha.note_id in {edge.source for edge in graph.edges}


def test_mononote_graph_keeps_unresolved_refs_and_local_depth(tmp_path: Path, monkeypatch) -> None:
    from core import paths
    from core.mononote import index, store

    monkeypatch.setattr(paths, "NOTES_DIR", tmp_path / "notes", raising=False)
    monkeypatch.setattr(paths, "LOG_DIR", tmp_path / "logs", raising=False)
    alpha = store.write_note("Alpha", "[[Beta]] [[Missing Note]]")
    beta = store.write_note("Beta", "[[Gamma]]")
    gamma = store.write_note("Gamma", "leaf")
    store.write_note("Delta", "outside")

    index.refresh_notes()

    local = index.read_note_graph(alpha.note_id, depth=1)
    titles = {node.safe_title for node in local.nodes}
    assert {"Alpha", "Beta", "Missing-Note"} <= titles
    assert "Gamma" not in titles
    assert "Delta" not in titles
    assert any(not node.resolved for node in local.nodes)
    deeper = index.read_note_graph(alpha.note_id, depth=2)
    assert "Gamma" in {node.safe_title for node in deeper.nodes}
    assert index.read_backlinks(gamma.note_id)[0].source_title == "Beta"
