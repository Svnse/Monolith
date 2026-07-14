"""Thinkpad — the reasoning-branch generator (Seed A).

Fans out N independent candidate branches within one turn (``run_subagent`` in
prod; an injected ``generate`` in tests), splits each into answer/reasoning, and
surfaces ALL branches with an ADVISORY grounded-verdict ranking.

Two decisions, both forced by the live gates this was built against:

1. **cite-parse runs on the split-out ANSWER, never the raw output.** The
   selection gate caught a branch that *mentioned* a handle while weighing it in
   deliberation ("[R1] says reject, [R2] says accept…") being scored as grounded
   on it. Splitting the reasoning off first means a cite counts only when it's in
   the committed answer, not in the thinking.

2. **the verdict is ADVISORY, not the committed selector.** The recall lane's
   [LOCKED]/[VERIFIED] deference contract already converges branches onto the
   well-grounded answer in-context, and contested cite-emission is ~0%, so
   cite-ranking-over-branches has not earned commit-authority. Thinkpad returns
   the ranking for display; what (if anything) it commits is the caller's call.
   TODO(commit-authority): revisit only with a tool-grounded selection test that
   doesn't yet exist — do not promote the verdict to auto-picker without it.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from core import grounded_verdict, recall_handles

# Mirrors engine.matrix_bridge.split_thinking (kept local so core/ doesn't import
# the matrix-nio-laden sidecar). Covers <think>/<analysis>/<reasoning> pairs.
_THINK_RE = re.compile(r"<(think|analysis|reasoning)>(.*?)</\1>", re.IGNORECASE | re.DOTALL)


def _split_answer(raw: str) -> tuple[str, str]:
    """Return (committed_answer, reasoning) — reasoning lifted out of the answer."""
    reasoning = "\n".join(m.group(2).strip() for m in _THINK_RE.finditer(raw)).strip()
    answer = _THINK_RE.sub("", raw).strip()
    return answer, reasoning


@dataclass(frozen=True)
class Branch:
    id: str
    raw: str                # full branch output (the MonoTrace source)
    answer: str             # committed answer, reasoning stripped (cite-parse target)
    think: str              # the reasoning lane (the MonoTrace's trace)
    cites: tuple[str, ...]  # ground handles cited IN THE ANSWER (not the deliberation)
    trace_id: str = ""      # run_subagent child_turn_id = this branch's MonoTrace key


@dataclass(frozen=True)
class ThinkpadResult:
    branches: tuple[Branch, ...]
    advisory: tuple[grounded_verdict.Ranked, ...]   # display-only ranking; NOT a commit


def run_thinkpad(
    messages: list[dict],
    base_config: dict,
    *,
    n: int = 2,
    generate: Callable[[list[dict]], str] | None = None,
    resolve: Callable[[str], "int | None"] | None = None,
) -> ThinkpadResult:
    """Generate ``n`` branches, surface all, rank advisorily."""
    if n < 1:
        return ThinkpadResult((), ())
    if generate is None:
        from engine.sync_bridge import generate_sync_from_config

        def generate(msgs: list[dict]) -> str:
            return generate_sync_from_config(base_config, msgs)
    if resolve is None:
        resolve = recall_handles.resolve

    branches: list[Branch] = []
    for i in range(n):
        try:
            out = generate(messages)
        except Exception:
            continue  # one failed branch must not sink the fan-out
        text, trace_id = out if isinstance(out, tuple) else (out, "")
        text, trace_id = str(text or ""), str(trace_id or "")
        if not text:
            continue  # a refused/empty branch is dropped, not surfaced
        answer, think = _split_answer(text)
        cites = grounded_verdict.parse_cites(answer).handles
        branches.append(Branch(id=f"c{i}", raw=text, answer=answer, think=think, cites=cites, trace_id=trace_id))

    candidates = [grounded_verdict.Candidate(b.id, b.cites) for b in branches]
    advisory = tuple(grounded_verdict.rank_candidates(candidates, resolve))
    return ThinkpadResult(branches=tuple(branches), advisory=advisory)


_THINK_CONVENTION = (
    "REASONING FORMAT: put all reasoning inside a single <think>…</think> block; your "
    "final answer goes AFTER </think>. In the ANSWER, cite the recalled-belief handle for "
    "any load-bearing conclusion (e.g. [cite: R1]) per the Grounding cite rule — a cite in "
    "the reasoning does not count, only a cite in the answer does."
)


def build_branch_frame(
    messages: list[dict],
    *,
    recall_lane: str = "",
    scaffold: str = "",
    think_convention: bool = True,
) -> list[dict]:
    """Assemble one branch's message frame: a system prompt carrying the reasoning
    scaffold (cite-slot included), the render-once recall lane (the [R#] handles —
    rendered ONCE by the caller and handed to every branch so they share one map), and
    the <think> answer/reasoning convention (so :func:`_split_answer` can lift the
    reasoning and cite-parse sees only the committed answer). The conversation follows;
    any inbound main system message is dropped — a branch is a fenced reasoner that gets
    the branch frame, not the full app system prompt.
    """
    parts = [p for p in (scaffold, recall_lane, _THINK_CONVENTION if think_convention else "") if p]
    system = "\n\n".join(parts)
    convo = [m for m in messages if m.get("role") != "system"]
    return ([{"role": "system", "content": system}] if system else []) + convo


def _load_scaffold() -> str:
    from pathlib import Path
    try:
        return (Path(__file__).resolve().parent.parent / "prompts" / "reasoning" / "monothink.md").read_text(encoding="utf-8")
    except Exception:
        return ""


def run_thinkpad_live(
    messages: list[dict],
    base_config: dict,
    *,
    n: int,
    parent_turn_id: str | None,
    scaffold: str | None = None,
    is_busy: Callable[[], bool] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> tuple[ThinkpadResult, tuple[str, ...]]:
    """Live Thinkpad fan-out over ``run_subagent``.

    Renders the recall lane ONCE (so every branch shares one orchestrator-owned
    handle map — branches must not re-render), builds the branch frame, then spawns
    ``n`` fenced reasoning branches (level=3, no tools, single inference). Returns
    ``(result, trace_ids)`` where trace_ids are the per-branch MonoTrace keys (the
    run_subagent child_turn_ids, persisted in turn_trace.sqlite3).
    """
    from core import runtime_state_projection as rsp
    from core.subagent import run_subagent

    if scaffold is None:
        scaffold = _load_scaffold()
    lane = rsp.render_recall_lane(messages)  # render ONCE -> shared map
    frame_msgs = build_branch_frame(messages, recall_lane=lane, scaffold=scaffold)

    counter = {"i": 0}

    def gen(m: list[dict]):
        i = counter["i"]
        counter["i"] += 1
        r = run_subagent(
            m, base_config, level=3, frame=f"thinkpad:c{i}",
            parent_turn_id=parent_turn_id, allowed_tools=frozenset(),
            should_cancel=should_cancel, is_busy=is_busy, max_followups=0,
            # run_subagent's 1024-token default is eaten by the think phase
            # alone (observed live: raw empty, ~4.5k chars of thinking) —
            # a reasoning branch needs answer headroom past its trace.
            llm_config={"max_tokens": 4096, "temp": 0.4},
        )
        text = r.text if r.ok else ""
        # The engine separates native reasoning out of r.text; without this
        # weave a branch that ignored the <think> convention loses its trace —
        # and the trace IS what a /rating's think_block trains on.
        if text and r.thinking and not _THINK_RE.search(text):
            text = f"<think>{r.thinking}</think>\n{text}"
        return (text, r.child_turn_id)

    res = run_thinkpad(frame_msgs, base_config, n=n, generate=gen)
    return res, tuple(b.trace_id for b in res.branches)
