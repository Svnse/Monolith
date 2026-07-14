"""Genesis-card build, Slice piece 1: the reflection scanner that projects a plain
Monolith function into a Monoline-block spec WITHOUT a hand-written registry ('show up
naturally'). See docs/reports/GENESIS_CARD_BUILD_LOG.md. Pure/model-free; new module, touches no
live path. Flag-independent (projection is inert until wired into the palette later)."""
from __future__ import annotations

import pytest

from core.organ_discovery import OrganSpec, ProjectionError, project_organ


def test_typed_function_projects_to_block_spec():
    def recall(query: str, k: int = 5) -> str:
        """Pull the top-k ACUs for a query (relevance x decay)."""
        return ""

    spec = project_organ(recall)

    assert spec.name == "recall"
    # params with no default -> input ports; type hint -> port type
    assert spec.input_ports == [("query", "text")]
    # params WITH a default -> config (not a wired port), typed, default carried
    assert spec.config == [("k", "number", 5)]
    # return hint -> the single output port type
    assert spec.output_type == "text"
    # docstring -> description (palette search text)
    assert "top-k ACUs" in spec.description
    # capability not inferable from a signature -> least-privilege default
    assert spec.capability == "pure"
    # the function itself is the handler ("send it to the right area" == call it)
    assert spec.handler is recall


def test_port_type_mapping():
    def organ(a: str, b: dict, c: list, d) -> dict:
        return {}

    spec = project_organ(organ)
    # str->text, dict/list->json; an UNTYPED param on a pure organ is allowed as 'any'
    assert dict(spec.input_ports) == {"a": "text", "b": "json", "c": "json", "d": "any"}
    assert spec.output_type == "json"


def test_unprojectable_fails_loud_not_silent():
    # A non-callable can't be projected -> LOUD error, never a silent skip
    # (E's observability contract: discoverable-but-inert is the bug).
    with pytest.raises(ProjectionError):
        project_organ("not a function")
