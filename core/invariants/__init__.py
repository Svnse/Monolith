"""Monolith's system-wide invariant taxonomy + companion schema enums.

Two contracts live here, deliberately separate:
  * :mod:`core.invariants.taxonomy` — the closed, system-wide registry of invariant-break
    (``kind="failure"``) tags across every layer. The monothink decider reads only the
    scoped PROJECTION of it (``monothink_decider_visible`` + ``owner_layer == "monothink"``);
    the
    live projection equals :data:`core.failure_tags.FAILURE_TAGS`.
  * :mod:`core.invariants.schema` — companion DIMENSIONS (intent / state_change / verdict
    / source). Not failure tags; never reach the decider.

Hard rule: tag existence is system-wide; tag visibility is layer-local.
"""
