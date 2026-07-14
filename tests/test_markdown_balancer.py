"""Robustness checks for truncated markdown rendering.

A streamed message can end mid-syntax (network drop, max_tokens, model
stop). Without balancing, python-markdown swallows everything after an
unclosed marker -- the user sees a broken cascade or empty bubble.
"""
from __future__ import annotations

from ui.components.markdown_renderer import (
    _apply_monolith_enrichment,
    _auto_fence_json_blocks,
    _balance_markdown,
    render_message_html,
)


def test_unclosed_inline_code_gets_closed():
    out = _balance_markdown("Look at `system_prompt")
    assert out.endswith("`")
    # Closer added exactly once.
    assert out.count("`") == 2


def test_unclosed_bold_gets_closed():
    out = _balance_markdown("This is **important")
    assert out.endswith("**")
    assert out.count("**") == 2


def test_combined_unclosed_bold_and_code_both_close():
    out = _balance_markdown("- **The `system_prompt")
    assert "`" in out
    assert "**" in out
    # Both pairs balanced.
    assert out.count("`") % 2 == 0
    assert out.count("**") % 2 == 0


def test_unclosed_fenced_code_block_gets_closed():
    src = "Here is code:\n\n```python\ndef hello():\n    print('hi')"
    out = _balance_markdown(src)
    # Fence count should be even after balancing.
    assert out.count("```") % 2 == 0


def test_balanced_markdown_passes_through_unchanged():
    src = "Normal **bold** and `code` here."
    assert _balance_markdown(src) == src


def test_render_truncated_message_does_not_eat_visible_text():
    truncated = "Notes:\n- **The `field"
    html = render_message_html(truncated)
    assert "The" in html
    assert "field" in html


def test_empty_input_is_safe():
    assert _balance_markdown("") == ""
    assert render_message_html("") == ""


def test_standalone_json_gets_fenced_for_final_render():
    src = 'Result:\n\n{\n  "mode": "snapshot_only",\n  "ok": true\n}'
    out = _auto_fence_json_blocks(src)

    assert "```json" in out
    assert '"mode": "snapshot_only"' in out


def test_json_after_label_gets_fenced_without_blank_line():
    src = 'Result:\n{\n  "mode": "snapshot_only",\n  "ok": true\n}'
    out = _auto_fence_json_blocks(src)

    assert out.startswith("Result:\n```json")
    assert '"mode": "snapshot_only"' in out


def test_small_json_is_compacted_inside_fence():
    src = '{\n  "ok": true,\n  "count": 2\n}'
    out = _auto_fence_json_blocks(src)

    assert out == '```json\n{"ok": true, "count": 2}\n```'


def test_final_render_does_not_force_plain_newlines_to_breaks():
    html = render_message_html("alpha\nbeta")

    assert "<br" not in html.lower()


def test_existing_fenced_json_is_preserved():
    src = '```json\n{\n  "ok": true\n}\n```'

    assert _auto_fence_json_blocks(src) == src


def test_monolith_callout_gets_themed_block():
    html = render_message_html("> [!BEARING] Coherence\n> Keep referents grounded.")

    assert "mono-callout" in html
    assert "mono-callout-bearing" in html
    assert "BEARING" in html
    assert "Keep referents grounded." in html


def test_runtime_report_gets_signal_card():
    src = (
        "[monopulse:pulse count=2 status=warn]\n"
        "  WARN Bearing update rejection pending - D1 failed [bearing:turn-1]\n"
        "  INFO Curiosity pull - inspect renderer [identity:1]\n"
    )

    html = render_message_html(src)

    assert "mono-runtime-card" in html
    assert "Monopulse: pulse" in html
    assert "mono-chip-warn" in html
    assert "mono-sev-warn" in html
    assert "Bearing update rejection pending" in html


def test_key_value_block_gets_signal_ledger_table():
    html = render_message_html("Status: WARN\nSource: MonoSearch\nScore: 72")

    assert "mono-kv-table" in html
    assert "<th>Status</th>" in html
    assert "<td>WARN</td>" in html


def test_monolith_enrichment_skips_fenced_code_blocks():
    src = "```md\n> [!BEARING]\n> do not transform\n```"

    enriched = _apply_monolith_enrichment(src)

    assert "mono-callout" not in enriched
    assert enriched == src
