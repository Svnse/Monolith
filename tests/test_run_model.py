from __future__ import annotations

from core.run_model import (
    BlockFinished,
    RunBlockSpec,
    RunFinished,
    RunModelBuilder,
    RunRegistry,
    RunStarted,
)


def _started(**kw) -> RunStarted:
    base = dict(
        run_id="r1", flow_id="two-step", name="Two-Step", user_input="hello",
        graph=[
            RunBlockSpec(id="input", label="request", kind="port"),
            RunBlockSpec(id="draft", label="draft", kind="llm"),
            RunBlockSpec(id="polish", label="polish", kind="llm"),
            RunBlockSpec(id="output", label="response", kind="port"),
        ],
        wires=[
            "input.value -> draft.prompt",
            "draft.response -> polish.prompt",
            "polish.response -> output.value",
        ],
    )
    base.update(kw)
    return RunStarted(**base)


def _block_done(bid="draft", outputs=None, started=1.0, completed=2.0,
                status="done", error="") -> BlockFinished:
    return BlockFinished(
        run_id="r1", block_id=bid, label=bid, kind="llm",
        outputs=outputs if outputs is not None else {"response": "X"},
        started_at=started, completed_at=completed, status=status, error=error)


def test_run_started_creates_running_model_with_pending_blocks():
    b = RunModelBuilder()
    b.apply(_started())
    m = b.model
    assert m.run_id == "r1"
    assert m.flow_id == "two-step"
    assert m.name == "Two-Step"
    assert m.status == "running"
    assert [blk.id for blk in m.block_list()] == ["input", "draft", "polish", "output"]
    assert all(blk.status == "pending" for blk in m.block_list())


def test_block_finished_marks_block_done_with_outputs_and_timing():
    b = RunModelBuilder()
    b.apply(_started())
    b.apply(_block_done(outputs={"response": "DRAFTED"}, started=1.0, completed=2.5))
    blk = b.model.block("draft")
    assert blk.status == "done"
    assert blk.outputs == {"response": "DRAFTED"}
    assert blk.started_at == 1.0 and blk.completed_at == 2.5
    assert blk.duration_ms() == 1500.0


def test_block_finished_with_error_marks_block_error():
    b = RunModelBuilder()
    b.apply(_started())
    b.apply(_block_done(outputs={}, status="error", error="boom"))
    blk = b.model.block("draft")
    assert blk.status == "error"
    assert blk.error == "boom"


def test_run_finished_sets_final_output_and_done_status():
    b = RunModelBuilder()
    b.apply(_started())
    b.apply(RunFinished(run_id="r1", output="FINAL", error=""))
    assert b.model.status == "done"
    assert b.model.final_output == "FINAL"


def test_run_finished_with_error_sets_error_status():
    b = RunModelBuilder()
    b.apply(_started())
    b.apply(RunFinished(run_id="r1", output="", error="kaboom"))
    assert b.model.status == "error"
    assert b.model.error == "kaboom"


def test_run_finished_stopped_sets_stopped_status_and_clears_error():
    # A user STOP is a first-class state, not an error: status="stopped", no error text carried
    # (the runtime's "Activation stopped." sentinel must not surface as a failure).
    b = RunModelBuilder()
    b.apply(_started())
    b.apply(RunFinished(run_id="r1", output="", error="Activation stopped.", stopped=True))
    assert b.model.status == "stopped"
    assert b.model.error == ""


def test_inputs_derived_from_upstream_output():
    b = RunModelBuilder()
    b.apply(_started())
    b.apply(_block_done(bid="draft", outputs={"response": "DRAFTED"}))
    assert b.model.inputs_for("polish") == {"prompt": "DRAFTED"}


# --- build_workshop_trace_attachment (Feature 1: workshop output -> next-turn context) ---


def test_workshop_trace_summarizes_done_blocks_as_attachment():
    from core.run_model import build_workshop_trace_attachment
    b = RunModelBuilder()
    b.apply(_started())
    b.apply(_block_done(bid="draft", outputs={"response": "DRAFTED text"}))
    b.apply(_block_done(bid="polish", outputs={"response": "POLISHED FINAL"}))

    block = build_workshop_trace_attachment(b.model)

    assert block.startswith("[ATTACHED: workshop trace (")
    assert ", trace)]" in block                  # header type tag -> split_attached strips it
    assert block.rstrip().endswith("[/ATTACHED]")
    assert "- draft: DRAFTED text" in block
    assert "- polish: POLISHED FINAL" in block
    # port scaffolding (input/output ports) is excluded
    assert "request" not in block
    assert "- response:" not in block


def test_workshop_trace_empty_when_no_done_meaningful_blocks():
    from core.run_model import build_workshop_trace_attachment
    b = RunModelBuilder()
    b.apply(_started())  # everything still pending; the only finished kinds would be ports anyway
    assert build_workshop_trace_attachment(b.model) == ""


def test_workshop_trace_truncates_and_single_lines_output():
    from core.run_model import build_workshop_trace_attachment
    b = RunModelBuilder()
    b.apply(_started())
    b.apply(_block_done(bid="draft", outputs={"response": "line one\n" + "x" * 500}))

    block = build_workshop_trace_attachment(b.model, max_chars_per_block=200)
    draft_line = next(l for l in block.splitlines() if l.startswith("- draft:"))

    assert "line one" in draft_line and "x" in draft_line  # multiline collapsed onto one bullet
    assert draft_line.rstrip().endswith("…")               # truncated with an ellipsis
    assert len(draft_line) <= len("- draft: ") + 200 + 3


def test_input_port_block_input_derived_from_user_input():
    b = RunModelBuilder()
    b.apply(_started(user_input="hello world"))
    # draft's only upstream is the input port block, which carries the user prompt
    assert b.model.inputs_for("draft") == {"prompt": "hello world"}


def test_model_notifies_observers_on_apply():
    b = RunModelBuilder()
    b.apply(_started())
    seen: list[str] = []
    b.model.subscribe(lambda m: seen.append(m.run_id))
    b.apply(_block_done())
    assert seen == ["r1"]


def test_registry_register_get_list_by_run_id():
    reg = RunRegistry()
    b = RunModelBuilder()
    b.apply(_started())
    reg.register(b.model)
    assert reg.get("r1") is b.model
    assert [m.run_id for m in reg.list_runs()] == ["r1"]
    assert reg.get("nope") is None


def test_block_finished_before_run_started_is_ignored():
    b = RunModelBuilder()
    b.apply(_block_done())  # no model yet
    assert b.model is None


def test_block_list_is_topologically_ordered_by_wires():
    # blocks declared OUT of dataflow order (llm, output, input) must render in dataflow
    # order: input (source) -> llm -> output (sink), following the wires.
    b = RunModelBuilder()
    b.apply(RunStarted(
        run_id="r", flow_id="f", name="F", user_input="hi",
        graph=[RunBlockSpec(id="assistant", label="Assistant", kind="llm"),
               RunBlockSpec(id="output", label="Output", kind="port"),
               RunBlockSpec(id="input", label="Input", kind="port")],
        wires=["input.value -> assistant.prompt", "assistant.response -> output.value"]))
    assert [blk.id for blk in b.model.block_list()] == ["input", "assistant", "output"]


def test_block_list_appends_unwired_blocks_in_declaration_order():
    # a block with no wires (and a cycle-safe fallback) still appears, after the ordered ones.
    b = RunModelBuilder()
    b.apply(RunStarted(
        run_id="r", flow_id="f", name="F", user_input="hi",
        graph=[RunBlockSpec(id="a", label="A", kind="llm"),
               RunBlockSpec(id="lonely", label="L", kind="text")],
        wires=[]))
    assert [blk.id for blk in b.model.block_list()] == ["a", "lonely"]


def test_registry_bounds_session_growth_evicting_oldest():
    reg = RunRegistry()
    for i in range(RunRegistry._CAP + 20):
        b = RunModelBuilder()
        b.apply(_started(run_id=f"r{i}"))
        reg.register(b.model)
    assert len(reg.list_runs()) == RunRegistry._CAP
    assert reg.get("r0") is None                                   # oldest evicted
    assert reg.get(f"r{RunRegistry._CAP + 19}") is not None        # newest retained
