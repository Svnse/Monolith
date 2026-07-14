"""Tests for core.attached_blocks — split [ATTACHED:...]...[/ATTACHED] blocks
out of a message's DISPLAY text while leaving the model/stored text untouched.

The block format is produced by ui.components.blob_tray.format_attached_blocks:
    [ATTACHED: label (size, type)]
    {content}            # OR  "Path: {path}\nUse open_file ..."  OR  "(no content available)"
    [/ATTACHED]
joined by "\n\n", then "\n\n---\n" + user_text.
"""
from __future__ import annotations

from core.attached_blocks import split_attached, Attachment


def test_no_blocks_returns_text_unchanged():
    clean, atts = split_attached("just a normal message")
    assert clean == "just a normal message"
    assert atts == []


def test_single_text_block_extracted_and_stripped():
    msg = (
        "[ATTACHED: notes.md (1.2KB, text)]\n"
        "# Title\n"
        "body line\n"
        "[/ATTACHED]\n"
        "\n---\n"
        "what do you think?"
    )
    clean, atts = split_attached(msg)
    assert clean == "what do you think?"
    assert len(atts) == 1
    a = atts[0]
    assert isinstance(a, Attachment)
    assert a.label == "notes.md"
    assert a.size == "1.2KB"
    assert a.type == "text"
    assert a.content == "# Title\nbody line"
    assert a.path is None


def test_file_block_with_path_has_no_inline_content():
    msg = (
        "[ATTACHED: archive.zip (3.0MB, archive)]\n"
        "Path: C:\\x\\archive.zip\n"
        "Use open_file tool to list or preview archive contents.\n"
        "[/ATTACHED]\n"
        "\n---\n"
        "unzip this"
    )
    clean, atts = split_attached(msg)
    assert clean == "unzip this"
    assert len(atts) == 1
    assert atts[0].type == "archive"
    assert atts[0].path == "C:\\x\\archive.zip"
    assert atts[0].content is None


def test_multiple_blocks_all_extracted():
    msg = (
        "[ATTACHED: a.txt (10B, text)]\naaa\n[/ATTACHED]\n"
        "\n"
        "[ATTACHED: b.txt (10B, text)]\nbbb\n[/ATTACHED]\n"
        "\n---\n"
        "both"
    )
    clean, atts = split_attached(msg)
    assert clean == "both"
    assert [a.label for a in atts] == ["a.txt", "b.txt"]
    assert [a.content for a in atts] == ["aaa", "bbb"]


def test_no_content_block_yields_empty_attachment():
    msg = (
        "[ATTACHED: weird.bin (5B, binary)]\n"
        "(no content available)\n"
        "[/ATTACHED]\n"
        "\n---\n"
        "hi"
    )
    clean, atts = split_attached(msg)
    assert clean == "hi"
    assert atts[0].path is None
    assert atts[0].content is None


def test_unclosed_block_is_left_alone():
    # A malformed (never-closed) block must not be silently swallowed.
    msg = "[ATTACHED: x.txt (1B, text)]\noops no close\nstill talking"
    clean, atts = split_attached(msg)
    assert atts == []
    assert clean == msg
