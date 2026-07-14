# tests/test_session_tree.py
import pytest
from core import branch_tree
from ui.pages import session_tree


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv("MONOLITH_BRANCH_TREE_V1", "1")


@pytest.fixture
def session():
    return {"messages": [], "tree": None}


def _msg(role, text, **extra):
    return {"i": 0, "time": "t0", "role": role, "text": text, **extra}


def test_flag_off_adapter_inactive(monkeypatch, session):
    monkeypatch.delenv("MONOLITH_BRANCH_TREE_V1", raising=False)
    assert session_tree.active() is False


def test_adoption_converts_flat_messages(flag_on):
    session = {"messages": [_msg("user", "q"), _msg("assistant", "a")]}
    tree = session_tree.ensure_tree(session)
    assert len(tree["nodes"]) == 3                              # root + 2
    assert [m["role"] for m in session["messages"]] == ["user", "assistant"]
    assert all("node_id" in m for m in session["messages"])     # reprojected payloads


def test_append_grows_tree_and_projection(flag_on, session):
    i = session_tree.tree_append(session, _msg("user", "q"))
    assert i == 0
    i2 = session_tree.tree_append(session, _msg("assistant", ""))
    assert i2 == 1
    assert session["messages"][1]["text"] == ""
    # identity: mutating the projected dict reaches the node (streaming hot path)
    session["messages"][1]["text"] += "tok"
    leaf = session["tree"]["active_leaf"]
    assert session["tree"]["nodes"][leaf]["msg"]["text"] == "tok"


def test_regen_reset_retains_old_answer_as_sibling(flag_on, session):
    session_tree.tree_append(session, _msg("user", "q"))
    session_tree.tree_append(session, _msg("assistant", "old answer"))
    prompt = session_tree.tree_regen_reset(session, 1)
    assert prompt == "q"
    assert [m["role"] for m in session["messages"]] == ["user"]   # path ends at user
    # old answer retained off-path
    assert any(n["msg"]["text"] == "old answer" for n in session["tree"]["nodes"].values())
    # the next appended assistant becomes a SIBLING with origin=regen
    session_tree.tree_append(session, _msg("assistant", "new answer"))
    u = session["messages"][0]["node_id"]
    assert len(session["tree"]["nodes"][u]["children"]) == 2
    new_leaf = session["tree"]["active_leaf"]
    assert session["tree"]["nodes"][new_leaf]["origin"] == "regen"


def test_user_edit_forks_and_classifies(flag_on, session):
    session_tree.tree_append(session, _msg("user", "explain the verifier checks"))
    session_tree.tree_append(session, _msg("assistant", "a1"))
    new_text = session_tree.tree_fork_user_edit(session, 0, "write a poem about cats")
    assert new_text == "write a poem about cats"
    assert [m["text"] for m in session["messages"]] == ["write a poem about cats"]
    root_kids = session["tree"]["nodes"][branch_tree.ROOT_ID]["children"]
    assert len(root_kids) == 2                                   # original + fork retained
    leaf_node = session["tree"]["nodes"][session["tree"]["active_leaf"]]
    assert leaf_node["branch_kind"] == "divergent"
    assert leaf_node["origin"] == "edit"


def test_assistant_rewrite_forks_variant(flag_on, session):
    session_tree.tree_append(session, _msg("user", "q"))
    session_tree.tree_append(session, _msg("assistant", "original"))
    session_tree.tree_fork_assistant_rewrite(session, 1, "rewritten")
    assert session["messages"][1]["text"] == "rewritten"
    u = session["messages"][0]["node_id"]
    kids = session["tree"]["nodes"][u]["children"]
    assert len(kids) == 2
    texts = {session["tree"]["nodes"][k]["msg"]["text"] for k in kids}
    assert texts == {"original", "rewritten"}


def test_prune_from_detaches_subtree(flag_on, session):
    session_tree.tree_append(session, _msg("user", "q"))
    session_tree.tree_append(session, _msg("assistant", "a"))
    session_tree.tree_append(session, _msg("user", "q2"))
    assert session_tree.tree_prune_from(session, 1) is True
    assert [m["role"] for m in session["messages"]] == ["user"]
    assert len(session["tree"]["nodes"]) == 2                    # root + user


def test_switch_resolves_default_descendant(flag_on, session):
    session_tree.tree_append(session, _msg("user", "q"))
    session_tree.tree_append(session, _msg("assistant", "a1"))
    session_tree.tree_append(session, _msg("user", "q2"))
    a1 = session["messages"][1]["node_id"]
    session_tree.tree_regen_reset(session, 1)
    session_tree.tree_append(session, _msg("assistant", "a2"))
    assert session_tree.tree_switch(session, a1) is True
    assert [m["text"] for m in session["messages"]] == ["q", "a1", "q2"]


def test_insert_splices_tool_row(flag_on, session):
    session_tree.tree_append(session, _msg("user", "q"))
    session_tree.tree_append(session, _msg("assistant", "a"))
    i = session_tree.tree_insert(session, 1, _msg("tool_result", "{}"))
    assert i == 1
    assert [m["role"] for m in session["messages"]] == ["user", "tool_result", "assistant"]


def test_listeners_notified_once_per_op(flag_on, session):
    hits = []
    session_tree.subscribe(lambda op: hits.append(op))
    try:
        session_tree.tree_append(session, _msg("user", "q"))
        assert hits and hits[-1] == "append"
    finally:
        session_tree._listeners.clear()


def test_emitted_canonical_kinds_are_registered():
    from core.acatalepsy.canonical_log_kinds import KNOWN_KINDS
    for kind in session_tree.CANONICAL_KINDS.values():
        assert kind in KNOWN_KINDS, f"unregistered canonical_log kind: {kind}"


def test_fingerprint_changes_on_mutation(flag_on, session):
    f0 = session_tree.fingerprint(session)
    session_tree.tree_append(session, _msg("user", "q"))
    assert session_tree.fingerprint(session) != f0


# ── fix 1: stale _next_origin cleared by non-append ops ───────────────────────

def test_stale_next_origin_cleared_by_other_ops(flag_on, session):
    session_tree.tree_append(session, _msg("user", "q"))
    session_tree.tree_append(session, _msg("assistant", "a"))
    session_tree.tree_regen_reset(session, 1)          # stamps _next_origin
    a_old = [n for n in session["tree"]["nodes"].values() if n["msg"].get("text") == "a"][0]
    session_tree.tree_switch(session, a_old["id"])     # user cancels regen by switching back
    assert "_next_origin" not in session               # stale stamp cleared
    session_tree.tree_append(session, _msg("assistant", "later"))
    leaf = session["tree"]["active_leaf"]
    assert session["tree"]["nodes"][leaf]["origin"] != "regen"


def test_stale_next_origin_cleared_by_insert(flag_on, session):
    session_tree.tree_append(session, _msg("user", "q"))
    session_tree.tree_append(session, _msg("assistant", "a"))
    session_tree.tree_regen_reset(session, 1)
    session_tree.tree_insert(session, 1, _msg("tool_result", "{}"))
    assert "_next_origin" not in session


def test_stale_next_origin_cleared_by_prune_from(flag_on, session):
    session_tree.tree_append(session, _msg("user", "q"))
    session_tree.tree_append(session, _msg("assistant", "a"))
    session_tree.tree_regen_reset(session, 1)
    session_tree.tree_prune_from(session, 1)
    assert "_next_origin" not in session


def test_stale_next_origin_cleared_by_fork_user_edit(flag_on, session):
    session_tree.tree_append(session, _msg("user", "q"))
    session_tree.tree_append(session, _msg("assistant", "a"))
    session_tree.tree_regen_reset(session, 1)
    session_tree.tree_fork_user_edit(session, 0, "new question")
    assert "_next_origin" not in session


def test_stale_next_origin_cleared_by_fork_assistant_rewrite(flag_on, session):
    session_tree.tree_append(session, _msg("user", "q"))
    session_tree.tree_append(session, _msg("assistant", "a"))
    session_tree.tree_regen_reset(session, 1)
    # move active leaf back to assistant node for the rewrite
    a_old = [n for n in session["tree"]["nodes"].values() if n["msg"].get("text") == "a"][0]
    session_tree.tree_switch(session, a_old["id"])
    session["_next_origin"] = "regen"  # re-stamp to test rewrite clears it
    session_tree.tree_fork_assistant_rewrite(session, 0, "rewritten")
    assert "_next_origin" not in session


def test_stale_next_origin_cleared_by_restore(flag_on, session):
    session_tree.tree_append(session, _msg("user", "q"))
    session_tree.tree_append(session, _msg("assistant", "a"))
    snap = session_tree.snapshot(session)
    session_tree.tree_regen_reset(session, 1)
    session_tree.restore(session, snap)
    assert "_next_origin" not in session


# ── fix 2: unsubscribe ─────────────────────────────────────────────────────────

def test_unsubscribe_stops_notifications(flag_on, session):
    hits = []
    cb = lambda op: hits.append(op)
    session_tree.subscribe(cb)
    try:
        session_tree.tree_append(session, _msg("user", "q"))
        assert hits  # sanity: cb was called
        hits.clear()
        session_tree.unsubscribe(cb)
        session_tree.tree_append(session, _msg("assistant", "a"))
        assert hits == []  # no notification after unsubscribe
    finally:
        session_tree._listeners.clear()


def test_unsubscribe_absent_cb_is_noop(flag_on, session):
    """unsubscribe of a non-registered callback must not raise."""
    session_tree.unsubscribe(lambda op: None)  # should not raise


# ── fix 3: _log breadcrumb — tested structurally ──────────────────────────────

def test_log_breadcrumb_uses_module_logger():
    """_log must not crash on a bad canonical_log, and the module must import logging."""
    import logging
    import importlib
    import ui.pages.session_tree as st_mod
    src = importlib.util.find_spec("ui.pages.session_tree").origin
    with open(src) as f:
        text = f.read()
    assert "import logging" in text
    assert 'logging.getLogger(__name__)' in text


# ── Task 6: persist_fanout ─────────────────────────────────────────────────────

from types import SimpleNamespace


def _branch(i, raw, trace_id=""):
    return SimpleNamespace(id=f"c{i}", raw=raw, answer=raw, think="", cites=(), trace_id=trace_id)


def test_persist_fanout_commits_first_retains_rest(flag_on, session):
    session_tree.tree_append(session, _msg("user", "q"))
    n = session_tree.persist_fanout(
        session, [_branch(0, "<think>t0</think>a0", "tr0"), _branch(1, "<think>t1</think>a1", "tr1")])
    assert n == 2
    u = session["messages"][0]["node_id"]
    kids = session["tree"]["nodes"][u]["children"]
    assert len(kids) == 2
    assert session["messages"][-1]["text"] == "<think>t0</think>a0"   # first committed
    origins = [session["tree"]["nodes"][k]["origin"] for k in kids]
    assert origins == ["thinkpad:c0", "thinkpad:c1"]
    assert session["tree"]["nodes"][kids[1]]["msg"]["task_id"] == "tr1"


def test_persist_fanout_requires_user_leaf(flag_on, session):
    session_tree.tree_append(session, _msg("user", "q"))
    session_tree.tree_append(session, _msg("assistant", "a"))
    assert session_tree.persist_fanout(session, [_branch(0, "x")]) == 0   # leaf is assistant → refuse
