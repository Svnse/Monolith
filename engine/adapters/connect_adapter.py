"""Remote CONNECT producer adapter.

Wraps the HTTP/CONNECT client side that talks to a remote model (Claude
via agent_server.py, or any other remote provider) so the Turn Pipeline
can consume its stream through ProducerAdapter.

Phase 2 scope: protocol-compliant stub. Phase 4 wires the actual HTTP
streaming. The independence direction — this file imports pipeline, the
pipeline does not import this file — is the key fact established here.
"""
from __future__ import annotations

import queue
from dataclasses import dataclass
from typing import Any, Iterable

from monokernel.turn_pipeline import ProducerChunk
from core.turn_pipeline_events import ProducerKind


_END_SENTINEL = object()


@dataclass
class ConnectAdapter:
    """ProducerAdapter for a remote model reached via CONNECT/HTTP.

    Holds the client transport (URL, auth token, model name) and bridges
    streamed responses into ProducerChunk objects the pipeline can consume.
    """
    base_url: str = ""
    model_name: str = ""
    producer_kind: str = ProducerKind.REMOTE_CONNECT.value

    def __post_init__(self) -> None:
        self._chunk_q: queue.Queue[Any] = queue.Queue()

    # ── public Protocol surface ────────────────────────────────────

    def stream(self, turn_id: str, context: dict[str, Any]) -> Iterable[ProducerChunk]:
        """Stream chunks from the remote endpoint for *turn_id*.

        Phase 2: returns an empty iterable when no transport is wired.
        Phase 4 migration plugs in the actual HTTP streaming client.
        """
        if not self.base_url:
            return iter(())
        return self._drain_queue()

    def supports_continuation(self) -> bool:
        return True

    def continue_with_tool_result(
        self,
        turn_id: str,
        tool_result_summary: dict[str, Any],
        hint: str | None,
    ) -> Iterable[ProducerChunk]:
        """Re-invoke the remote with the latest tool result appended."""
        if not self.base_url:
            return iter(())
        return self._drain_queue()

    # ── chunk pipe — used by Phase 4 wiring ───────────────────────

    def push_chunk(self, text: str, meta: dict[str, Any] | None = None) -> None:
        self._chunk_q.put(ProducerChunk(text=text, meta=dict(meta or {})))

    def push_end(self) -> None:
        self._chunk_q.put(_END_SENTINEL)

    def _drain_queue(self) -> Iterable[ProducerChunk]:
        while True:
            item = self._chunk_q.get()
            if item is _END_SENTINEL:
                return
            yield item
