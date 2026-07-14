"""Reflection scanner: project a plain Monolith function into a Monoline-block spec by
introspection, so capabilities "show up naturally" — no hand-written registry (E, 2026-06-14).

A registry OBJECT still exists downstream (the validator/governor need something to check),
but it is GENERATED from this projection, never hand-maintained: add a function (in core/organs/
or any reflected registry), and the block appears. Anything that cannot be cleanly projected
raises ProjectionError so it surfaces LOUDLY in the palette rather than silently vanishing
(observability contract). Pure + model-free: this module touches no live path and is inert
until an adapter wires OrganSpecs into the Monoline palette (later slice step).

Part of the Genesis-card build — see docs/reports/GENESIS_CARD_BUILD_LOG.md.
"""
from __future__ import annotations

import inspect
import typing
from dataclasses import dataclass
from typing import Any, Callable

_EMPTY = inspect.Parameter.empty  # is inspect.Signature.empty too (same sentinel)


class ProjectionError(Exception):
    """A capability could not be projected into a block spec. Raised, never swallowed,
    so the palette shows a loud error entry instead of a silent omission."""


@dataclass(frozen=True)
class OrganSpec:
    name: str
    input_ports: list[tuple[str, str]]          # (port_name, port_type in text|json|any)
    config: list[tuple[str, str, Any]]          # (key, type in string|number|bool|enum, default)
    output_type: str                            # text|json|any
    description: str
    capability: str                             # least-privilege default 'pure'; observed/blessed later
    handler: Callable[..., Any]


def _base_name(annotation: str) -> str:
    # "dict[str, str]" / "typing.Dict" -> "dict"; handles PEP 563 string annotations
    return annotation.split("[", 1)[0].strip().rsplit(".", 1)[-1].lower()


def _port_type(annotation: Any) -> str:
    if annotation is _EMPTY:
        return "any"
    if isinstance(annotation, str):                 # `from __future__ import annotations`
        base = _base_name(annotation)
        if base in ("str", "text"):
            return "text"
        if base in ("dict", "list", "json"):
            return "json"
        return "any"
    origin = typing.get_origin(annotation) or annotation
    if origin is str:
        return "text"
    if origin in (dict, list):
        return "json"
    return "any"


def _config_type(annotation: Any, default: Any) -> str:
    if isinstance(annotation, str):                 # PEP 563 string annotation
        base = _base_name(annotation)
        if base == "bool":
            return "bool"
        if base in ("int", "float"):
            return "number"
        if base in ("str", "string"):
            return "string"
        annotation = _EMPTY                          # unknown string -> infer from default
    origin = typing.get_origin(annotation) or annotation
    if origin is _EMPTY:
        origin = type(default)
    if origin is bool:          # bool before int (bool subclasses int)
        return "bool"
    if origin in (int, float):
        return "number"
    if origin is str:
        return "string"
    return "string"


def project_organ(fn: Callable[..., Any]) -> OrganSpec:
    """Introspect a function into an OrganSpec. Params without a default become wired input
    ports (typed from the hint); params with a default become typed config; the return hint
    is the single output port; the docstring is the description; the function is the handler."""
    if not callable(fn):
        raise ProjectionError(f"not projectable (not callable): {fn!r}")
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError) as exc:
        raise ProjectionError(f"cannot introspect {fn!r}: {exc}") from exc

    input_ports: list[tuple[str, str]] = []
    config: list[tuple[str, str, Any]] = []
    for name, p in sig.parameters.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue  # *args/**kwargs have no stable port shape
        if p.default is _EMPTY:
            input_ports.append((name, _port_type(p.annotation)))
        else:
            config.append((name, _config_type(p.annotation, p.default), p.default))

    return OrganSpec(
        name=getattr(fn, "__name__", "organ"),
        input_ports=input_ports,
        config=config,
        output_type=_port_type(sig.return_annotation),
        description=(inspect.getdoc(fn) or "").strip(),
        capability="pure",
        handler=fn,
    )
