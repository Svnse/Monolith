"""Tests for ui.components.blob_tray — BlobChip, BlobTray, helpers."""
from __future__ import annotations

import textwrap
from pathlib import Path

from PySide6.QtWidgets import QApplication

from ui.components.blob_tray import (
    BlobChip,
    BlobTray,
    _PASTE_BLOB_THRESHOLD,
    format_attached_blocks,
    is_text_file,
    is_zip_file,
)


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


# ── is_text_file / is_zip_file ──────────────────────────────────────


def test_is_text_file_returns_true_for_known_text_extensions() -> None:
    for ext in (".py", ".json", ".md", ".txt", ".yaml", ".yml", ".csv",
                ".html", ".css", ".js", ".ts", ".toml", ".rs", ".go"):
        assert is_text_file(f"some/path/file{ext}"), f"Expected True for {ext}"


def test_is_text_file_returns_false_for_non_text() -> None:
    for ext in (".zip", ".png", ".jpg", ".exe", ".bin", ".pdf"):
        assert not is_text_file(f"some/path/file{ext}"), f"Expected False for {ext}"


def test_is_text_file_is_case_insensitive() -> None:
    assert is_text_file("readme.MD")
    assert is_text_file("script.PY")


def test_is_zip_file_returns_true_for_archives() -> None:
    for ext in (".zip", ".tar", ".gz", ".tgz", ".7z", ".rar"):
        assert is_zip_file(f"archive{ext}"), f"Expected True for {ext}"


def test_is_zip_file_returns_false_for_non_archives() -> None:
    for ext in (".py", ".txt", ".png", ".exe"):
        assert not is_zip_file(f"file{ext}"), f"Expected False for {ext}"


# ── _PASTE_BLOB_THRESHOLD sentinel ──────────────────────────────────


def test_paste_blob_threshold_value() -> None:
    assert _PASTE_BLOB_THRESHOLD == 1000


# ── BlobChip ─────────────────────────────────────────────────────────


class TestBlobChipMetadata:
    def test_file_chip_stores_metadata(self) -> None:
        _app()
        chip = BlobChip(
            index=0, kind="file", label="notes.txt",
            size=512, content="hello world", path="/tmp/notes.txt",
        )
        assert chip.index == 0
        assert chip.kind == "file"
        assert chip.label == "notes.txt"
        assert chip.size == 512
        assert chip.content == "hello world"
        assert chip.path == "/tmp/notes.txt"

    def test_paste_chip_stores_metadata(self) -> None:
        _app()
        chip = BlobChip(
            index=1, kind="paste", label="pasted text",
            size=2000, content="x" * 2000,
        )
        assert chip.index == 1
        assert chip.kind == "paste"
        assert chip.label == "pasted text"
        assert chip.size == 2000
        assert chip.content == "x" * 2000
        assert chip.path is None

    def test_zip_chip_stores_metadata(self) -> None:
        _app()
        chip = BlobChip(
            index=2, kind="zip", label="data.zip",
            size=99999, path="/tmp/data.zip",
        )
        assert chip.index == 2
        assert chip.kind == "zip"
        assert chip.label == "data.zip"
        assert chip.size == 99999
        assert chip.content is None
        assert chip.path == "/tmp/data.zip"


class TestBlobChipSignal:
    def test_sig_removed_fires_with_correct_index(self) -> None:
        _app()
        chip = BlobChip(index=7, kind="file", label="f.py", size=10)
        received: list[int] = []
        chip.sig_removed.connect(received.append)

        chip._on_remove_clicked()

        assert received == [7]

    def test_sig_removed_fires_updated_index(self) -> None:
        """After externally changing chip.index, the emitted value follows."""
        _app()
        chip = BlobChip(index=0, kind="paste", label="p", size=5)
        received: list[int] = []
        chip.sig_removed.connect(received.append)

        chip.index = 3
        chip._on_remove_clicked()

        assert received == [3]


# ── BlobTray ─────────────────────────────────────────────────────────


class TestBlobTrayInitialState:
    def test_starts_hidden(self) -> None:
        _app()
        tray = BlobTray()
        assert not tray.isVisible()

    def test_has_blobs_is_false(self) -> None:
        _app()
        tray = BlobTray()
        assert not tray.has_blobs()

    def test_blobs_is_empty(self) -> None:
        _app()
        tray = BlobTray()
        assert tray.blobs() == []


class TestBlobTrayAddPaste:
    def test_add_paste_shows_tray_and_stores_content(self) -> None:
        _app()
        tray = BlobTray()
        tray.add_paste("hello world this is pasted text")

        assert tray.isVisible()
        assert tray.has_blobs()
        blobs = tray.blobs()
        assert len(blobs) == 1
        assert blobs[0]["kind"] == "paste"
        assert blobs[0]["label"] == "pasted text"
        assert blobs[0]["content"] == "hello world this is pasted text"
        assert blobs[0]["size"] == len("hello world this is pasted text")
        assert blobs[0]["path"] is None

    def test_add_paste_accepts_custom_label(self) -> None:
        _app()
        tray = BlobTray()
        tray.add_paste("body", label="MonoNote")

        blobs = tray.blobs()
        assert blobs[0]["label"] == "MonoNote"
        assert blobs[0]["content"] == "body"


class TestBlobTrayAddFile:
    def test_add_text_file_reads_content(self, tmp_path: Path) -> None:
        _app()
        f = tmp_path / "example.py"
        f.write_text("print('hi')", encoding="utf-8")

        tray = BlobTray()
        tray.add_file(str(f))

        assert tray.isVisible()
        blobs = tray.blobs()
        assert len(blobs) == 1
        assert blobs[0]["kind"] == "file"
        assert blobs[0]["label"] == "example.py"
        assert blobs[0]["content"] == "print('hi')"
        assert blobs[0]["path"] == str(f)
        assert blobs[0]["size"] == f.stat().st_size

    def test_add_binary_file_stores_path_only(self, tmp_path: Path) -> None:
        _app()
        f = tmp_path / "image.png"
        f.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        tray = BlobTray()
        tray.add_file(str(f))

        blobs = tray.blobs()
        assert len(blobs) == 1
        assert blobs[0]["kind"] == "file"
        assert blobs[0]["content"] is None
        assert blobs[0]["path"] == str(f)

    def test_add_text_file_json_extension(self, tmp_path: Path) -> None:
        _app()
        f = tmp_path / "data.json"
        f.write_text('{"key": "value"}', encoding="utf-8")

        tray = BlobTray()
        tray.add_file(str(f))

        blobs = tray.blobs()
        assert blobs[0]["content"] == '{"key": "value"}'

    def test_add_text_file_md_extension(self, tmp_path: Path) -> None:
        _app()
        f = tmp_path / "readme.md"
        f.write_text("# Hello", encoding="utf-8")

        tray = BlobTray()
        tray.add_file(str(f))

        blobs = tray.blobs()
        assert blobs[0]["content"] == "# Hello"


class TestBlobTrayAddZip:
    def test_add_zip_stores_path_content_is_none(self, tmp_path: Path) -> None:
        _app()
        f = tmp_path / "archive.zip"
        f.write_bytes(b"PK\x03\x04" + b"\x00" * 50)

        tray = BlobTray()
        tray.add_zip(str(f))

        blobs = tray.blobs()
        assert len(blobs) == 1
        assert blobs[0]["kind"] == "zip"
        assert blobs[0]["label"] == "archive.zip"
        assert blobs[0]["content"] is None
        assert blobs[0]["path"] == str(f)
        assert blobs[0]["size"] == f.stat().st_size


class TestBlobTrayClear:
    def test_clear_removes_all_chips_and_hides(self) -> None:
        _app()
        tray = BlobTray()
        tray.add_paste("a")
        tray.add_paste("b")
        tray.add_paste("c")
        assert tray.has_blobs()
        assert tray.isVisible()

        tray.clear()

        assert not tray.has_blobs()
        assert tray.blobs() == []
        assert not tray.isVisible()


class TestBlobTrayRemoveChip:
    def test_remove_chip_removes_specific_chip(self) -> None:
        _app()
        tray = BlobTray()
        tray.add_paste("first")
        tray.add_paste("second")
        tray.add_paste("third")

        # Remove the middle chip (index 1)
        tray._remove_chip(1)

        blobs = tray.blobs()
        assert len(blobs) == 2
        assert blobs[0]["content"] == "first"
        assert blobs[1]["content"] == "third"

    def test_remove_chip_reindexes_remaining(self) -> None:
        _app()
        tray = BlobTray()
        tray.add_paste("a")
        tray.add_paste("b")
        tray.add_paste("c")

        tray._remove_chip(0)

        # After removing index 0, remaining chips should be re-indexed 0,1
        assert tray._chips[0].index == 0
        assert tray._chips[1].index == 1
        assert tray._chips[0].content == "b"
        assert tray._chips[1].content == "c"

    def test_remove_last_chip_hides_tray(self) -> None:
        _app()
        tray = BlobTray()
        tray.add_paste("only")
        assert tray.isVisible()

        tray._remove_chip(0)

        assert not tray.has_blobs()
        assert not tray.isVisible()

    def test_remove_out_of_range_is_noop(self) -> None:
        _app()
        tray = BlobTray()
        tray.add_paste("x")

        tray._remove_chip(99)
        tray._remove_chip(-1)

        assert len(tray.blobs()) == 1


# ── "Move to chat input" (paste chip un-attach) ──────────────────────


class TestBlobChipToInput:
    def test_paste_chip_has_to_input_button(self) -> None:
        _app()
        chip = BlobChip(index=0, kind="paste", label="pasted text", size=5, content="hi")
        # the paste chip renders as a card with a "move to input" button
        assert getattr(chip, "_to_input_btn", None) is not None

    def test_file_chip_has_no_to_input_button(self) -> None:
        _app()
        chip = BlobChip(index=0, kind="file", label="f.py", size=10, content="x")
        # only paste blobs get the un-attach affordance (their content is inline)
        assert getattr(chip, "_to_input_btn", None) is None

    def test_to_input_click_emits_index(self) -> None:
        _app()
        chip = BlobChip(index=2, kind="paste", label="pasted text", size=5, content="hello")
        received: list[int] = []
        chip.sig_to_input.connect(received.append)

        chip._on_to_input_clicked()

        assert received == [2]


class TestBlobTrayToInput:
    def test_to_input_emits_content_and_removes_chip(self) -> None:
        _app()
        tray = BlobTray()
        tray.add_paste("the pasted body")
        got: list[str] = []
        tray.sig_blob_to_input.connect(got.append)

        tray._chips[0]._on_to_input_clicked()

        assert got == ["the pasted body"]   # content handed to the composer
        assert not tray.has_blobs()          # chip removed from the tray
        assert not tray.isVisible()

    def test_to_input_only_moves_the_clicked_chip(self) -> None:
        _app()
        tray = BlobTray()
        tray.add_paste("first")
        tray.add_paste("second")
        got: list[str] = []
        tray.sig_blob_to_input.connect(got.append)

        tray._chips[1]._on_to_input_clicked()  # move the second one

        assert got == ["second"]
        assert len(tray.blobs()) == 1
        assert tray.blobs()[0]["content"] == "first"


# ── format_attached_blocks ───────────────────────────────────────────


class TestFormatAttachedBlocks:
    def test_empty_blobs_returns_user_text_unchanged(self) -> None:
        assert format_attached_blocks([], "hello") == "hello"

    def test_empty_blobs_with_empty_text(self) -> None:
        assert format_attached_blocks([], "") == ""

    def test_text_file_includes_content_inline(self) -> None:
        blobs = [
            {"kind": "file", "label": "code.py", "size": 42,
             "content": "print('hi')", "path": "/tmp/code.py"},
        ]
        result = format_attached_blocks(blobs, "user question")

        assert "[ATTACHED: code.py" in result
        assert "text)" in result  # type_tag = text when content present
        assert "print('hi')" in result
        assert "[/ATTACHED]" in result
        assert result.endswith("user question")

    def test_zip_includes_path_and_open_file_instruction(self) -> None:
        blobs = [
            {"kind": "zip", "label": "data.zip", "size": 8000,
             "content": None, "path": "/tmp/data.zip"},
        ]
        result = format_attached_blocks(blobs, "extract this")

        assert "[ATTACHED: data.zip" in result
        assert "archive)" in result
        assert "Path: /tmp/data.zip" in result
        assert "open_file" in result
        assert result.endswith("extract this")

    def test_paste_includes_content(self) -> None:
        blobs = [
            {"kind": "paste", "label": "pasted text", "size": 5,
             "content": "hello", "path": None},
        ]
        result = format_attached_blocks(blobs, "what is this?")

        assert "[ATTACHED: pasted text" in result
        assert "paste)" in result
        assert "hello" in result
        assert "[/ATTACHED]" in result
        assert result.endswith("what is this?")

    def test_binary_file_without_content_uses_open_file(self) -> None:
        blobs = [
            {"kind": "file", "label": "image.png", "size": 5000,
             "content": None, "path": "/tmp/image.png"},
        ]
        result = format_attached_blocks(blobs, "describe this")

        assert "binary)" in result
        assert "Path: /tmp/image.png" in result
        assert "open_file" in result

    def test_multiple_blobs_all_present_with_separator_and_user_text_at_end(self) -> None:
        blobs = [
            {"kind": "file", "label": "a.py", "size": 10,
             "content": "code_a", "path": "/a.py"},
            {"kind": "zip", "label": "b.zip", "size": 2000,
             "content": None, "path": "/b.zip"},
            {"kind": "paste", "label": "pasted text", "size": 3,
             "content": "xyz", "path": None},
        ]
        result = format_attached_blocks(blobs, "do stuff")

        # All three blocks present
        assert "a.py" in result
        assert "b.zip" in result
        assert "pasted text" in result
        assert "code_a" in result
        assert "xyz" in result

        # Separator before user text
        assert "\n\n---\n" in result
        assert result.endswith("do stuff")

        # Blocks are separated by double newlines
        blocks_part = result.split("\n\n---\n")[0]
        assert "\n\n" in blocks_part

    def test_blob_with_no_content_and_no_path(self) -> None:
        blobs = [
            {"kind": "file", "label": "mystery", "size": 0,
             "content": None, "path": None},
        ]
        result = format_attached_blocks(blobs, "huh")

        assert "(no content available)" in result
        assert result.endswith("huh")

    def test_format_size_display(self) -> None:
        """Size is rendered via _human_size inside the header."""
        blobs = [
            {"kind": "file", "label": "big.py", "size": 2048,
             "content": "x", "path": "/big.py"},
        ]
        result = format_attached_blocks(blobs, "q")
        # 2048 bytes = 2.0KB
        assert "2.0KB" in result

    def test_format_large_size_in_mb(self) -> None:
        blobs = [
            {"kind": "file", "label": "huge.bin", "size": 5 * 1024 * 1024,
             "content": None, "path": "/huge.bin"},
        ]
        result = format_attached_blocks(blobs, "q")
        assert "5.0MB" in result

    def test_format_small_size_in_bytes(self) -> None:
        blobs = [
            {"kind": "paste", "label": "pasted text", "size": 42,
             "content": "a" * 42, "path": None},
        ]
        result = format_attached_blocks(blobs, "q")
        assert "42B" in result
