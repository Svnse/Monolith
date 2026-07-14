"""Local llama.cpp producer adapter.

Wraps engine/llm.py:LLMEngine so the Turn Pipeline can consume its stream
through the ProducerAdapter protocol. The adapter does NOT alter how the
engine streams; it observes the existing Qt-signal stream and republishes
it as a synchronous iterable of ProducerChunk objects.

Phase 2 scope: protocol-compliant stub. The full bridge (subscribing to
LLMEngine.token signals and bridging them to chunks the pipeline can pull)
lands in Phase 4 migration, when chat.py is wired to call the pipeline
instead of consuming the engine signal directly.

Independence direction: this file imports pipeline; the pipeline does not
import this file.
"""
from __future__ import annotations

import queue
from dataclasses import dataclass
from typing import Any, Iterable

from monokernel.turn_pipeline import ProducerChunk
from core.turn_pipeline_events import ProducerKind


_END_SENTINEL = object()


@dataclass
class LocalLLMAdapter:
    """ProducerAdapter for the local llama.cpp engine.

    Holds a reference to an LLMEngine and exposes stream() / continuation
    methods that the pipeline can drive. Streaming uses an internal queue
    fed by the engine's existing token signal callbacks.
    """
    engine: Any  # engine.llm.LLMEngine — typed as Any to keep the adapter
                 # importable without engine being constructed (helpful in
                 # tests; production wiring always passes a real engine).
    producer_kind: str = ProducerKind.LOCAL_LLM.value

    def __post_init__(self) -> None:
        self._chunk_q: queue.Queue[Any] = queue.Queue()

    # ── public Protocol surface ────────────────────────────────────

    def stream(self, turn_id: str, context: dict[str, Any]) -> Iterable[ProducerChunk]:
        """Yield chunks for *turn_id* until the engine signals end.

        Phase 2: returns an empty iterable when no engine is wired. Phase 4
        migration plugs in the actual signal->queue bridge.
        """
        if self.engine is None:
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
        """Re-invoke the engine with the latest tool result appended.

        Phase 2: protocol-compliant stub. Phase 4 wires this to invoke
        LLMEngine.generate with the prior history plus the tool result and
        the optional hint injected as an assistant-side guidance message.
        """
        if self.engine is None:
            return iter(())
        return self._drain_queue()

    # ── chunk pipe — used by Phase 4 wiring ───────────────────────

    def push_chunk(self, text: str, meta: dict[str, Any] | None = None) -> None:
        """Called by the signal handler when a new token chunk arrives."""
        self._chunk_q.put(ProducerChunk(text=text, meta=dict(meta or {})))

    def push_end(self) -> None:
        """Called when the engine finishes the stream for this turn."""
        self._chunk_q.put(_END_SENTINEL)

    def _drain_queue(self) -> Iterable[ProducerChunk]:
        while True:
            item = self._chunk_q.get()
            if item is _END_SENTINEL:
                return
            yield item
