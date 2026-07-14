"""Parse [ATTACHED:...]...[/ATTACHED] blocks out of a message's DISPLAY text.

These blocks are produced by ``ui.components.blob_tray.format_attached_blocks``
and prepended to the user's message before it is sent to the model and stored
in history. The model SHOULD see them (they carry the attached content); the
human SHOULD NOT see the raw block in the chat bubble — it renders as a chip
instead. This module is the render-side split: it never mutates stored/model
text, it only tells the renderer what to show.

Block shape (one per attachment, joined by "\\n\\n", then "\\n\\n---\\n" + user_text):

    [ATTACHED: label (size, type)]
    {body}
    [/ATTACHED]

where ``body`` is the inline content (text/paste), or ``Path: {path}`` plus a
tool hint (zip/binary files), or ``(no content available)``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Attachment:
    """One parsed attachment. ``content`` is set for inline text/paste blobs;
    ``path`` is set for file/zip blobs referenced by path. At most one is set."""
    label: str
    size: str
    type: str
    path: str | None = None
    content: str | None = None


# A well-formed block: header line, body (DOTALL, non-greedy), close tag.
_BLOCK_RE = re.compile(
    r"\[ATTACHED:(?P<head>[^\]]*)\]\n(?P<body>.*?)\n\[/ATTACHED\]",
    re.DOTALL,
)
# Header inner: " label (size, type)" — label greedy-minimal up to the size/type paren.
_HEAD_RE = re.compile(r"^\s*(?P<label>.*?)\s*\((?P<size>[^()]*),\s*(?P<type>[^()]*)\)\s*$")
_NO_CONTENT = "(no content available)"


def _parse_block(head: str, body: str) -> Attachment:
    m = _HEAD_RE.match(head)
    if m:
        label = m.group("label").strip()
        size = m.group("size").strip()
        type_ = m.group("type").strip()
    else:
        label, size, type_ = head.strip(), "", ""

    path: str | None = None
    content: str | None = None
    first_line = body.split("\n", 1)[0]
    if first_line.startswith("Path:"):
        path = first_line[len("Path:"):].strip() or None
    elif body.strip() == _NO_CONTENT:
        pass  # both None
    else:
        content = body
    return Attachment(label=label, size=size, type=type_, path=path, content=content)


def split_attached(text: str) -> tuple[str, list[Attachment]]:
    """Return ``(display_text, attachments)``.

    ``display_text`` is *text* with every well-formed [ATTACHED] block removed
    and the leading ``---`` separator cleaned up. ``attachments`` is the parsed
    list (empty when there are none). When there are no blocks the text is
    returned verbatim — separators are only touched when attachments exist, so a
    normal message that happens to start with ``---`` is left alone.
    """
    if not text or "[ATTACHED:" not in text:
        return text, []

    attachments = [
        _parse_block(m.group("head"), m.group("body"))
        for m in _BLOCK_RE.finditer(text)
    ]
    if not attachments:
        return text, []

    clean = _BLOCK_RE.sub("", text).lstrip()
    # Drop the leading "---" separator line that joined the blocks to user_text.
    clean = re.sub(r"^-{3,}[ \t]*\n?", "", clean).lstrip()
    return clean, attachments
