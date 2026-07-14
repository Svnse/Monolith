"""The unified run view's backend contract (Workshop Pane v2, spec 2026-06-14).

PURE — imports nothing from Monolith's UI/engine and NO Qt. A live run streams
RunEvents through RunModelBuilder; a historical run rehydrates the SAME way (one
fold path → identical RunModel). RunView (ui/) is the only Qt boundary; it
subscribes to a model via plain observer callbacks.

The bridge (engine/monoline_bridge) emits the event stream today; Monoline-v2's
organ layer emits the same stream later (drop-in). Inputs are DERIVED from
wires + upstream outputs + user_input, because today's StepResult carries no
inputs; the native runtime can supply them directly (preferred when present).
"""
from __future__ import annotations

from dataclasses import dataclass, field


# --- events (the seam) -------------------------------------------------------
@dataclass(frozen=True)
class RunBlockSpec:
    id: str
    label: str
    kind: str


@dataclass(frozen=True)
class RunStarted:
    run_id: str
    flow_id: str
    name: str
    user_input: str
    graph: list          # list[RunBlockSpec], in display order
    wires: list          # list[str] "src.port -> dst.port"


@dataclass(frozen=True)
class BlockFinished:
    run_id: str
    block_id: str
    label: str
    kind: str
    outputs: dict        # {port: value}
    started_at: float
    completed_at: float
    status: str = "done"          # "done" | "error"
    error: str = ""
    verdict: dict | None = None
    detectors: list | None = None
    inputs: dict | None = None    # native runtime may supply; else derived


@dataclass(frozen=True)
class RunFinished:
    run_id: str
    output: str = ""
    error: str = ""
    stopped: bool = False   # user STOP — a clean halt, not an error (status -> "stopped")


@dataclass(frozen=True)
class RunSummary:
    """A row in the companion run browser's recent-runs list (turn_trace.list_recent_runs)."""
    run_id: str
    flow_id: str
    name: str
    captured_at: str


# --- model -------------------------------------------------------------------
@dataclass
class RunBlock:
    id: str
    label: str
    kind: str
    status: str = "pending"       # pending | running | done | error
    outputs: dict = field(default_factory=dict)
    inputs: dict | None = None    # set only if the runtime supplied them
    started_at: float | None = None
    completed_at: float | None = None
    verdict: dict | None = None
    detectors: list | None = None
    error: str = ""

    def duration_ms(self) -> float | None:
        if self.started_at is None or self.completed_at is None:
            return None
        return (self.completed_at - self.started_at) * 1000.0


class RunModel:
    """Observable (plain callbacks; no Qt). Built only by RunModelBuilder."""

    def __init__(self, run_id, flow_id, name, user_input, blocks, wires):
        self.run_id = run_id
        self.flow_id = flow_id
        self.name = name
        self.user_input = user_input
        self._blocks: dict = blocks        # insertion-ordered {id: RunBlock}
        self.wires = list(wires or [])
        self.status = "running"
        self.final_output = ""
        self.error = ""
        self._observers: list = []

    # -- access --
    def block_list(self) -> list:
        """Blocks in DATAFLOW order (topological by wires: sources first, sinks last), so the
        view reads input → … → output regardless of the file's declaration order."""
        return [self._blocks[bid] for bid in self._ordered_ids()]

    def _ordered_ids(self) -> list:
        ids = list(self._blocks.keys())
        pos = {bid: i for i, bid in enumerate(ids)}   # stable tie-break: declaration order
        indeg = {bid: 0 for bid in ids}
        adj: dict = {bid: [] for bid in ids}
        for wire in self.wires:
            src, dst = self._parse_wire(wire)
            if dst is None:
                continue
            sb, db = src[0], dst[0]
            if sb in indeg and db in indeg:
                adj[sb].append(db)
                indeg[db] += 1
        ready = sorted([b for b in ids if indeg[b] == 0], key=lambda b: pos[b])
        out: list = []
        seen: set = set()
        while ready:
            n = ready.pop(0)
            if n in seen:
                continue
            seen.add(n)
            out.append(n)
            newly = []
            for m in adj[n]:
                indeg[m] -= 1
                if indeg[m] == 0:
                    newly.append(m)
            if newly:
                ready = sorted(ready + newly, key=lambda b: pos[b])
        for b in ids:                 # cycles / unreached -> declaration order, never dropped
            if b not in seen:
                out.append(b)
        return out

    def block(self, block_id: str):
        return self._blocks.get(block_id)

    # -- observability --
    def subscribe(self, cb) -> None:
        self._observers.append(cb)

    def unsubscribe(self, cb) -> None:
        try:
            self._observers.remove(cb)
        except ValueError:
            pass

    def _notify(self) -> None:
        for cb in list(self._observers):
            try:
                cb(self)
            except Exception:
                pass

    # -- input derivation (wires + upstream outputs + user_input) --
    def inputs_for(self, block_id: str) -> dict:
        blk = self._blocks.get(block_id)
        if blk is not None and blk.inputs is not None:
            return dict(blk.inputs)   # runtime supplied them; trust over derivation
        out: dict = {}
        for src, dst in (self._parse_wire(w) for w in self.wires):
            if dst is None:
                continue
            dst_block, dst_port = dst
            if dst_block != block_id:
                continue
            out[dst_port] = self._value_at(*src)
        return out

    def _value_at(self, src_block: str, src_port: str) -> str:
        blk = self._blocks.get(src_block)
        if blk is None:
            return ""
        if src_port in (blk.outputs or {}):
            return blk.outputs[src_port]
        if blk.kind == "port":            # the input port carries the user prompt
            return self.user_input
        return ""                          # not yet produced / unresolvable

    @staticmethod
    def _parse_wire(wire: str):
        if "->" not in wire:
            return (RunModel._split(wire.strip()), None)
        left, right = wire.split("->", 1)
        return (RunModel._split(left.strip()), RunModel._split(right.strip()))

    @staticmethod
    def _split(side: str):
        if "." in side:
            block, port = side.rsplit(".", 1)
            return (block, port)
        return (side, "")


# --- next-turn context summary -----------------------------------------------
# Kinds whose output is worth summarizing into the chat's next-turn context. Port/text/
# trigger/input/output blocks are scaffolding (routing + static prompt text), not
# model-authored content, so they are skipped.
_TRACE_KINDS = frozenset({"llm", "synthesis_bus", "trace_port", "merge", "extract", "transform"})


def _primary_output_text(block) -> str:
    """The block's main output: the longest string value across its output ports."""
    best = ""
    for value in (getattr(block, "outputs", {}) or {}).values():
        s = str(value)
        if len(s) > len(best):
            best = s
    return best


def build_workshop_trace_attachment(model, *, max_chars_per_block: int = 200) -> str:
    """Compact, model-facing summary of a finished workshop run's blocks, wrapped as an
    ``[ATTACHED]`` block. This rides on the existing display-vs-model seam: core.attached_blocks
    strips it from the chat bubble, while it stays in the stored message content so the model
    sees it via build_engine_history. One line per meaningful completed block:
    ``- <label>: <single-lined, truncated output>``. Returns ``""`` when the run produced no
    summarizable blocks."""
    lines: list[str] = []
    for blk in model.block_list():
        if getattr(blk, "status", "") != "done":
            continue
        if str(getattr(blk, "kind", "")) not in _TRACE_KINDS:
            continue
        text = " ".join(_primary_output_text(blk).split())  # collapse newlines/runs to one line
        if not text:
            continue
        if len(text) > max_chars_per_block:
            text = text[:max_chars_per_block] + "…"
        label = str(getattr(blk, "label", "") or getattr(blk, "id", "") or "block")
        lines.append(f"- {label}: {text}")
    if not lines:
        return ""
    summary = "\n".join(lines)
    return f"[ATTACHED: workshop trace ({len(summary)} chars, trace)]\n{summary}\n[/ATTACHED]"


# --- builder (the single fold path) ------------------------------------------
class RunModelBuilder:
    def __init__(self):
        self.model: RunModel | None = None

    def apply(self, event) -> None:
        if isinstance(event, RunStarted):
            blocks = {
                spec.id: RunBlock(id=spec.id, label=spec.label, kind=spec.kind)
                for spec in event.graph
            }
            self.model = RunModel(event.run_id, event.flow_id, event.name,
                                  event.user_input, blocks, event.wires)
            self.model._notify()
            return
        if self.model is None:
            return  # events before run_started are ignored
        if isinstance(event, BlockFinished):
            blk = self.model.block(event.block_id)
            if blk is None:
                # composite sub-step / dynamic block not in the declared graph
                blk = RunBlock(id=event.block_id, label=event.label, kind=event.kind)
                self.model._blocks[event.block_id] = blk
            blk.status = event.status
            blk.outputs = dict(event.outputs or {})
            blk.inputs = dict(event.inputs) if event.inputs is not None else None
            blk.started_at = event.started_at
            blk.completed_at = event.completed_at
            blk.verdict = event.verdict
            blk.detectors = event.detectors
            blk.error = event.error or ""
            self.model._notify()
            return
        if isinstance(event, RunFinished):
            self.model.final_output = event.output or ""
            if event.stopped:
                self.model.error = ""               # a stop is not an error — carry no error text
                self.model.status = "stopped"
            else:
                self.model.error = event.error or ""
                self.model.status = "error" if event.error else "done"
            self.model._notify()
            return


# --- live-run registry (shared instance for chat + browser) ------------------
class RunRegistry:
    _CAP = 100   # bound session growth: keep the most recent runs, evict the oldest

    def __init__(self):
        self._runs: dict = {}   # run_id -> RunModel, insertion order

    def register(self, model: RunModel) -> None:
        self._runs[model.run_id] = model
        while len(self._runs) > self._CAP:
            del self._runs[next(iter(self._runs))]   # drop the oldest insertion

    def get(self, run_id: str):
        return self._runs.get(run_id)

    def list_runs(self) -> list:
        return list(reversed(self._runs.values()))   # newest-first

    def drop(self, run_id: str) -> None:
        self._runs.pop(run_id, None)


# Shared, process-wide registry of this session's live/just-finished runs.
live_runs = RunRegistry()
