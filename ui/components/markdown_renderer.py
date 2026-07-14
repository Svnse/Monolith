"""Themed markdown -> HTML rendering for assistant messages.

Replaces Qt's built-in `setMarkdown` (CommonMark only, no syntax highlight)
with a python-markdown + pygments pipeline whose output respects the active
theme. Renders fenced code blocks with a language label header and a body
styled by pygments. Inline code, blockquotes, tables, and task lists get
theme-coherent CSS. A small Monolith enrichment layer upgrades portable
markdown patterns into runtime-native cards (callouts, signal reports, and
compact key/value ledgers) without asking the model to emit raw HTML.

Public API:
    render_message_html(text: str) -> str

The returned string is a single HTML document fragment safe to feed to
QTextDocument.setHtml(). Qt's HTML subset is limited (no flexbox, no JS),
so all styling is plain inline / class-based CSS that QTextDocument
understands.

Performance: the renderer is called ONCE at stream finalize per message.
During streaming we still cursor-insert plain text into the widget; the
HTML rerender at finalize is what produces the polished view.
"""
from __future__ import annotations

import html as _html
import json as _json
import re
from functools import lru_cache

import markdown as _md
from pygments.formatters import HtmlFormatter

import core.style as _s


# Match a fenced code block so we can wrap it with a language-label header
# BEFORE handing the body to python-markdown. We do this pre-processing
# because Qt's QTextDocument doesn't honor most CSS selectors that would
# otherwise let us style headers via attribute matching alone.
_FENCE_RE = re.compile(r"^```([^\n`]*)\n(.*?)^```\s*$", re.MULTILINE | re.DOTALL)
_JSON_BLOCK_TYPES = (dict, list)
_CALLOUT_HEAD_RE = re.compile(r"^>\s*\[!(?P<kind>[A-Za-z0-9_-]+)\]\s*(?P<title>.*?)\s*$")
_RUNTIME_HEAD_RE = re.compile(
    r"^\[(?P<name>[A-Za-z_][\w-]*)(?::(?P<verb>[^\]\s]+))?(?P<meta>(?:\s+[^\]]+)*)\]\s*$"
)
_RUNTIME_ROW_RE = re.compile(r"^\s*(?P<severity>FAIL|ERROR|WARN|STALE|OK|PASS|INFO)\s+(?P<body>.+?)\s*$", re.IGNORECASE)
_KV_LINE_RE = re.compile(r"^(?P<key>[A-Za-z][A-Za-z0-9 _./-]{1,30}):\s+(?P<value>.+?)\s*$")
_RUNTIME_BLOCK_NAMES = frozenset({
    "monopulse",
    "monosearch",
    "inspect_trace",
    "inspect_pipeline",
    "runtime_health",
    "get_budget_score",
    "get_context_summary",
})
_CALLOUT_LABELS = {
    "ACU": "ACU",
    "BEARING": "BEARING",
    "CHECK": "CHECK",
    "FAIL": "FAULT",
    "FAULT": "FAULT",
    "INFO": "INFO",
    "MONOLITH": "MONOLITH",
    "MONOPULSE": "PULSE",
    "NOTE": "NOTE",
    "RUNTIME": "RUNTIME",
    "WARN": "WARN",
    "WARNING": "WARN",
}


def _balance_markdown(text: str) -> str:
    """Close any unclosed inline markdown markers at end-of-text.

    A truncated stream (network drop, max_tokens hit, model stop) often
    leaves dangling syntax like ``**The `system_prompt`` -- without
    closers, python-markdown swallows everything after the opener and
    the rendered message looks broken or empty. We detect odd-count
    markers in the safe set and append closers so the visible text
    survives even when generation cuts mid-formatting.

    Conservative: only handles the markers most likely to truncate
    mid-stream. Doesn't touch raw newlines, heading lines, list items,
    or HTML -- those are robust to truncation by python-markdown.
    """
    if not text:
        return text

    # Triple-backtick fences first (must close before single backticks).
    # Count fences; an odd count means an unclosed code block.
    fences = text.count("```")
    if fences % 2 == 1:
        # Append a newline + close so the closing ``` lands on its own line.
        suffix = "\n```" if not text.endswith("\n") else "```"
        text = text + suffix

    # Strip out fenced regions before counting inline markers so backticks
    # inside a closed fence don't pollute the count.
    scrub = _FENCE_RE.sub("", text)

    # Single backtick (inline code). Don't touch text inside fences.
    if scrub.count("`") % 2 == 1:
        text = text + "`"

    # Bold: ** must come in pairs. Count non-overlapping occurrences.
    # Use a literal scan so a single * in prose doesn't get mistaken for **.
    if scrub.count("**") % 2 == 1:
        text = text + "**"

    return text


def _format_json_for_display(value: object) -> str:
    compact = _json.dumps(value, ensure_ascii=False, separators=(", ", ": "))
    if len(compact) <= 180:
        return compact
    return _json.dumps(value, indent=2, ensure_ascii=False)


def _maybe_json_fence(block: str) -> str:
    stripped = block.strip()
    if not stripped or stripped[0] not in "{[":
        return block
    try:
        value = _json.loads(stripped)
    except Exception:
        return block
    if not isinstance(value, _JSON_BLOCK_TYPES):
        return block
    body = _format_json_for_display(value)
    trailing = "\n" if block.endswith("\n") else ""
    return f"```json\n{body}\n```{trailing}"


def _auto_fence_json_blocks(text: str) -> str:
    """Turn standalone JSON paragraphs into themed code blocks.

    Models often emit raw pretty JSON as a normal paragraph. With hard
    newline rendering that becomes a tall, unstyled line-per-property slab.
    We only touch complete paragraphs that parse as JSON objects/arrays, and
    we skip existing fenced regions so authored code blocks remain intact.
    """
    if not text:
        return text

    def _plain_segment(segment: str) -> str:
        out: list[str] = []
        block: list[str] = []

        def flush_block() -> None:
            if not block:
                return
            out.append(_maybe_json_fence("".join(block)))
            block.clear()

        for line in segment.splitlines(keepends=True):
            if line.strip():
                if line.lstrip().startswith(("{", "[")) and block:
                    flush_block()
                block.append(line)
                continue
            flush_block()
            out.append(line)
        flush_block()
        return "".join(out)

    chunks: list[str] = []
    pos = 0
    for match in _FENCE_RE.finditer(text):
        chunks.append(_plain_segment(text[pos:match.start()]))
        chunks.append(match.group(0))
        pos = match.end()
    chunks.append(_plain_segment(text[pos:]))
    return "".join(chunks)


def _transform_plain_segments(text: str, fn) -> str:
    """Apply *fn* outside fenced code blocks only."""
    if not text:
        return text
    chunks: list[str] = []
    pos = 0
    for match in _FENCE_RE.finditer(text):
        chunks.append(fn(text[pos:match.start()]))
        chunks.append(match.group(0))
        pos = match.end()
    chunks.append(fn(text[pos:]))
    return "".join(chunks)


def _class_token(value: str) -> str:
    token = re.sub(r"[^a-z0-9_-]+", "-", str(value or "").strip().lower()).strip("-")
    return token or "note"


def _paragraphs_html(lines: list[str]) -> str:
    paragraphs: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if str(line).strip():
            current.append(str(line))
            continue
        if current:
            paragraphs.append(current)
            current = []
    if current:
        paragraphs.append(current)
    if not paragraphs:
        return ""
    parts = []
    for para in paragraphs:
        body = "<br/>".join(_html.escape(line) for line in para)
        parts.append(f"<p>{body}</p>")
    return "".join(parts)


def _render_callout(kind: str, title: str, body_lines: list[str]) -> str:
    normalized = str(kind or "note").strip().upper()
    label = _CALLOUT_LABELS.get(normalized, normalized.replace("_", " "))
    suffix = _class_token(label)
    title = str(title or "").strip()
    title_html = _html.escape(title) if title else _html.escape(label.title())
    body_html = _paragraphs_html(body_lines) or "<p></p>"
    return (
        f'<div class="mono-callout mono-callout-{suffix}">'
        f'<div class="mono-callout-title">'
        f'<span class="mono-callout-badge">{_html.escape(label)}</span>'
        f'<span>{title_html}</span>'
        f"</div>"
        f'<div class="mono-callout-body">{body_html}</div>'
        f"</div>\n"
    )


def _enhance_callouts(segment: str) -> str:
    lines = segment.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.rstrip("\r\n")
        match = _CALLOUT_HEAD_RE.match(stripped)
        if match is None:
            out.append(raw)
            i += 1
            continue

        i += 1
        body_lines: list[str] = []
        while i < len(lines):
            body_raw = lines[i]
            body = body_raw.rstrip("\r\n")
            if body.startswith(">"):
                body_lines.append(re.sub(r"^>\s?", "", body))
                i += 1
                continue
            break
        out.append(_render_callout(match.group("kind"), match.group("title"), body_lines))
    return "".join(out)


def _parse_meta_tokens(meta: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for token in str(meta or "").split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        key = key.strip()
        if not key:
            continue
        out[key] = value.strip()
    return out


def _render_runtime_report(name: str, verb: str, meta: str, row_lines: list[str]) -> str:
    meta_values = _parse_meta_tokens(meta)
    source = _class_token(name).replace("-", "_")
    title = str(name or "runtime").replace("_", " ").title()
    if verb:
        title += f": {verb.replace('_', ' ')}"
    chips = []
    for key, value in meta_values.items():
        chips.append(
            f'<span class="mono-chip mono-chip-{_class_token(value)}">'
            f'{_html.escape(key)}={_html.escape(value)}</span>'
        )
    chip_html = "".join(chips)
    rows_html: list[str] = []
    for line in row_lines:
        match = _RUNTIME_ROW_RE.match(line)
        if match is not None:
            severity = match.group("severity").upper()
            body = match.group("body")
        else:
            severity = "SIGNAL"
            body = line.strip()
        severity_class = _class_token(severity)
        if not body:
            continue
        rows_html.append(
            f'<div class="mono-signal-row mono-sev-{severity_class}">'
            f'<span class="mono-severity">{_html.escape(severity)}</span>'
            f'<span class="mono-signal-text">{_html.escape(body)}</span>'
            f"</div>"
        )
    if not rows_html:
        rows_html.append(
            '<div class="mono-signal-row mono-sev-info">'
            '<span class="mono-severity">INFO</span>'
            '<span class="mono-signal-text">No row detail.</span>'
            "</div>"
        )
    return (
        f'<div class="mono-runtime-card mono-runtime-{_class_token(source)}">'
        f'<div class="mono-runtime-head">'
        f'<span class="mono-runtime-title">{_html.escape(title)}</span>'
        f'<span class="mono-runtime-source">Monolith signal</span>'
        f"</div>"
        f'<div class="mono-chip-row">{chip_html}</div>'
        f'<div class="mono-signal-list">{"".join(rows_html)}</div>'
        f"</div>\n"
    )


def _enhance_runtime_reports(segment: str) -> str:
    lines = segment.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.rstrip("\r\n")
        match = _RUNTIME_HEAD_RE.match(stripped)
        name = (match.group("name") if match is not None else "").strip().lower()
        if match is None or name not in _RUNTIME_BLOCK_NAMES:
            out.append(raw)
            i += 1
            continue

        i += 1
        rows: list[str] = []
        while i < len(lines):
            row = lines[i].rstrip("\r\n")
            if not row.strip():
                break
            if _RUNTIME_HEAD_RE.match(row):
                break
            if row.startswith((" ", "\t")) or _RUNTIME_ROW_RE.match(row):
                rows.append(row.strip())
                i += 1
                continue
            break
        out.append(_render_runtime_report(name, match.group("verb") or "", match.group("meta") or "", rows))
    return "".join(out)


def _render_kv_table(rows: list[tuple[str, str]]) -> str:
    body = []
    for key, value in rows:
        body.append(
            "<tr>"
            f'<th>{_html.escape(key.strip())}</th>'
            f'<td>{_html.escape(value.strip())}</td>'
            "</tr>"
        )
    return f'<table class="mono-kv-table">{"".join(body)}</table>\n'


def _enhance_key_value_blocks(segment: str) -> str:
    lines = segment.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    while i < len(lines):
        rows: list[tuple[str, str]] = []
        j = i
        while j < len(lines):
            raw = lines[j]
            stripped = raw.rstrip("\r\n")
            match = _KV_LINE_RE.match(stripped)
            if match is None:
                break
            rows.append((match.group("key"), match.group("value")))
            j += 1
        if len(rows) >= 3:
            out.append(_render_kv_table(rows))
            i = j
            continue
        out.append(lines[i])
        i += 1
    return "".join(out)


def _apply_monolith_enrichment(text: str) -> str:
    """Upgrade portable Markdown idioms into Monolith-native render blocks."""
    text = _transform_plain_segments(text, _enhance_callouts)
    text = _transform_plain_segments(text, _enhance_runtime_reports)
    text = _transform_plain_segments(text, _enhance_key_value_blocks)
    return text


def _is_dark_theme() -> bool:
    """Detect dark/light theme from BG_MAIN luminance.

    Used to pick a pygments style that visually belongs to the rest of the
    UI rather than always rendering monokai on a light surface (or vice
    versa).
    """
    bg = (_s.BG_MAIN or "#000000").lstrip("#")
    if len(bg) != 6:
        return True
    try:
        r, g, b = int(bg[0:2], 16), int(bg[2:4], 16), int(bg[4:6], 16)
    except ValueError:
        return True
    # Rec. 709 luma; <128 = dark.
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) < 128


@lru_cache(maxsize=1)
def _pygments_css() -> str:
    """Pygments-generated CSS for code-block tokens, theme-tinted.

    Picks a built-in pygments style that matches the active theme's
    luminance so dark themes get a dark code palette and light themes
    get a light one. Cached; invalidate via reset_renderer_cache().
    """
    style = "one-dark" if _is_dark_theme() else "friendly"
    formatter = HtmlFormatter(style=style, nowrap=False)
    return formatter.get_style_defs(".codehilite")


@lru_cache(maxsize=1)
def _document_css() -> str:
    """Theme-coherent CSS applied to every rendered assistant message."""
    return f"""
    body {{
        color: {_s.FG_TEXT};
        font-family: 'Segoe UI', system-ui, sans-serif;
        font-size: 12px;
        line-height: 1.55;
    }}
    p {{ margin: 0 0 8px 0; }}
    h1, h2, h3, h4, h5, h6 {{
        color: {_s.FG_TEXT};
        font-weight: 600;
        margin: 14px 0 6px 0;
        line-height: 1.3;
    }}
    h1 {{
        font-size: 18px;
        color: {_s.FG_ACCENT};
        padding-bottom: 4px;
        border-bottom: 1px solid {_s.BORDER_SUBTLE};
    }}
    h2 {{
        font-size: 15px;
        color: {_s.FG_ACCENT};
        padding-left: 8px;
        border-left: 2px solid {_s.ACCENT_PRIMARY};
    }}
    h3 {{ font-size: 13px; }}
    h4, h5, h6 {{ font-size: 12px; color: {_s.FG_SECONDARY}; }}
    a {{ color: {_s.ACCENT_PRIMARY}; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    ul, ol {{ margin: 4px 0 8px 0; padding-left: 22px; }}
    li {{ margin: 3px 0; }}
    strong {{ color: {_s.FG_ACCENT}; font-weight: 600; }}
    em {{ color: {_s.FG_TEXT}; font-style: italic; }}
    blockquote {{
        border-left: 3px solid {_s.ACCENT_PRIMARY};
        margin: 8px 0;
        padding: 6px 14px;
        color: {_s.FG_SECONDARY};
        background: {_s.BG_SURFACE_1};
        border-radius: 0 6px 6px 0;
    }}
    table {{
        border-collapse: collapse;
        margin: 8px 0;
        background: {_s.BG_SURFACE_1};
        border-radius: 6px;
    }}
    th, td {{
        border-bottom: 1px solid {_s.BORDER_SUBTLE};
        padding: 6px 12px;
        text-align: left;
    }}
    th {{
        background: {_s.BG_SURFACE_2};
        color: {_s.FG_TEXT};
        font-weight: 600;
        font-size: 11px;
        letter-spacing: 0.3px;
        text-transform: uppercase;
    }}
    tr:last-child td {{ border-bottom: none; }}
    code {{
        font-family: 'Cascadia Code', 'JetBrains Mono', Consolas, monospace;
        font-size: 11px;
        background: {_s.BG_SURFACE_1};
        color: {_s.FG_TEXT};
        padding: 2px 6px;
        border-radius: 5px;
    }}
    pre {{
        background: {_s.BG_SURFACE_1};
        border: 1px solid {_s.BORDER_SUBTLE};
        border-radius: 8px;
        padding: 12px 14px;
        margin: 6px 0 10px 0;
        font-family: 'Cascadia Code', 'JetBrains Mono', Consolas, monospace;
        font-size: 11px;
        line-height: 1.5;
    }}
    pre code {{
        background: transparent;
        color: {_s.FG_TEXT};
        padding: 0;
        border: none;
        border-radius: 0;
        font-size: 11px;
    }}
    .code-fence-header {{
        background: {_s.BG_SURFACE_2};
        color: {_s.FG_DIM};
        font-family: 'Cascadia Code', Consolas, monospace;
        font-size: 9px;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        padding: 4px 12px;
        border: 1px solid {_s.BORDER_SUBTLE};
        border-bottom: none;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
    }}
    .code-fence-body pre {{
        margin-top: 0;
        border-top-left-radius: 0;
        border-top-right-radius: 0;
        background: {_s.BG_SURFACE_2};
    }}
    hr {{
        border: none;
        border-top: 1px solid {_s.BORDER_SUBTLE};
        margin: 12px 0;
    }}
    .mono-callout {{
        margin: 9px 0 11px 0;
        padding: 0;
        background: {_s.BG_SURFACE_1};
        border: 1px solid {_s.BORDER_SUBTLE};
        border-left: 3px solid {_s.ACCENT_PRIMARY};
        border-radius: 8px;
    }}
    .mono-callout-title {{
        color: {_s.FG_TEXT};
        background: {_s.BG_SURFACE_2};
        padding: 6px 10px;
        font-size: 11px;
        font-weight: 600;
        border-bottom: 1px solid {_s.BORDER_SUBTLE};
    }}
    .mono-callout-badge {{
        color: {_s.ACCENT_PRIMARY};
        background: {_s.BG_MAIN};
        border: 1px solid {_s.BORDER_SUBTLE};
        border-radius: 5px;
        padding: 1px 6px;
        margin-right: 8px;
        font-size: 9px;
        font-weight: 700;
        letter-spacing: 0.6px;
    }}
    .mono-callout-body {{
        color: {_s.FG_SECONDARY};
        padding: 8px 10px 2px 10px;
    }}
    .mono-callout-body p {{
        margin: 0 0 7px 0;
    }}
    .mono-callout-fault, .mono-callout-fail, .mono-callout-warn {{
        border-left-color: {_s.FG_WARN};
    }}
    .mono-callout-bearing, .mono-callout-runtime, .mono-callout-pulse {{
        border-left-color: {_s.FG_ACCENT};
    }}
    .mono-runtime-card {{
        margin: 9px 0 11px 0;
        background: {_s.BG_SURFACE_1};
        border: 1px solid {_s.BORDER_SUBTLE};
        border-left: 3px solid {_s.ACCENT_PRIMARY};
        border-radius: 8px;
    }}
    .mono-runtime-head {{
        background: {_s.BG_SURFACE_2};
        padding: 7px 10px;
        border-bottom: 1px solid {_s.BORDER_SUBTLE};
    }}
    .mono-runtime-title {{
        color: {_s.FG_TEXT};
        font-weight: 700;
        font-size: 12px;
    }}
    .mono-runtime-source {{
        color: {_s.FG_DIM};
        font-size: 9px;
        margin-left: 10px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }}
    .mono-chip-row {{
        padding: 6px 10px 0 10px;
    }}
    .mono-chip {{
        color: {_s.FG_SECONDARY};
        background: {_s.BG_MAIN};
        border: 1px solid {_s.BORDER_SUBTLE};
        border-radius: 5px;
        padding: 2px 6px;
        margin-right: 5px;
        font-size: 9px;
        font-family: 'Cascadia Code', Consolas, monospace;
    }}
    .mono-chip-fail, .mono-chip-error {{
        color: {_s.FG_ERROR};
        border-color: {_s.FG_ERROR};
    }}
    .mono-chip-warn, .mono-chip-stale {{
        color: {_s.FG_WARN};
        border-color: {_s.FG_WARN};
    }}
    .mono-chip-ok, .mono-chip-pass {{
        color: {_s.FG_ACCENT};
        border-color: {_s.FG_ACCENT};
    }}
    .mono-signal-list {{
        padding: 6px 8px 8px 8px;
    }}
    .mono-signal-row {{
        background: {_s.BG_MAIN};
        border: 1px solid {_s.BORDER_SUBTLE};
        border-radius: 6px;
        padding: 5px 7px;
        margin: 4px 0;
    }}
    .mono-severity {{
        color: {_s.FG_DIM};
        background: {_s.BG_SURFACE_2};
        border: 1px solid {_s.BORDER_SUBTLE};
        border-radius: 4px;
        padding: 1px 5px;
        margin-right: 8px;
        font-size: 9px;
        font-weight: 700;
        font-family: 'Cascadia Code', Consolas, monospace;
    }}
    .mono-signal-text {{
        color: {_s.FG_TEXT};
        font-size: 11px;
    }}
    .mono-sev-fail .mono-severity, .mono-sev-error .mono-severity {{
        color: {_s.FG_ERROR};
        border-color: {_s.FG_ERROR};
    }}
    .mono-sev-warn .mono-severity, .mono-sev-stale .mono-severity {{
        color: {_s.FG_WARN};
        border-color: {_s.FG_WARN};
    }}
    .mono-sev-ok .mono-severity, .mono-sev-pass .mono-severity {{
        color: {_s.FG_ACCENT};
        border-color: {_s.FG_ACCENT};
    }}
    .mono-kv-table {{
        border-collapse: collapse;
        margin: 8px 0 10px 0;
        background: {_s.BG_SURFACE_1};
        border: 1px solid {_s.BORDER_SUBTLE};
        border-left: 3px solid {_s.ACCENT_PRIMARY};
        border-radius: 8px;
    }}
    .mono-kv-table th {{
        color: {_s.FG_DIM};
        background: {_s.BG_SURFACE_2};
        border-bottom: 1px solid {_s.BORDER_SUBTLE};
        padding: 5px 10px;
        text-align: left;
        font-size: 9px;
        letter-spacing: 0.4px;
        text-transform: uppercase;
    }}
    .mono-kv-table td {{
        color: {_s.FG_TEXT};
        border-bottom: 1px solid {_s.BORDER_SUBTLE};
        padding: 5px 12px;
        font-size: 11px;
    }}
    """


def reset_renderer_cache() -> None:
    """Drop cached CSS so the next render picks up a theme swap."""
    _pygments_css.cache_clear()
    _document_css.cache_clear()


def _wrap_fence(match: re.Match) -> str:
    """Replace a ``` fence with a header + fenced body so the language is shown.

    The header is a separate block that python-markdown will pass through
    unchanged because it's already HTML.
    """
    lang = (match.group(1) or "").strip()
    body = match.group(2) or ""
    label = lang.lower() if lang else "text"
    header = f'<div class="code-fence-header">{_html.escape(label)}</div>'
    # Re-emit the fence so codehilite still does syntax highlighting.
    fence = f"```{lang}\n{body}```"
    return f"{header}\n<div class=\"code-fence-body\">\n\n{fence}\n\n</div>"


def render_message_html(text: str) -> str:
    """Render markdown source to a themed HTML fragment."""
    if not text:
        return ""
    # Close any dangling **/`/``` so a truncated stream doesn't render as
    # a broken cascade where everything after the unclosed marker vanishes.
    text = _balance_markdown(text)
    text = _auto_fence_json_blocks(text)
    text = _apply_monolith_enrichment(text)
    pre = _FENCE_RE.sub(_wrap_fence, text)
    body_html = _md.markdown(
        pre,
        extensions=[
            "fenced_code",
            "codehilite",
            "tables",
            "sane_lists",
        ],
        extension_configs={
            "codehilite": {
                "noclasses": False,
                "css_class": "codehilite",
                "guess_lang": False,
            },
        },
        output_format="html",
    )
    style_block = f"<style>{_document_css()}{_pygments_css()}</style>"
    return f"{style_block}<body>{body_html}</body>"
