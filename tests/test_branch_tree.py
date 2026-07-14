"""Increment 1 — the pure reasoning branch tree (core/branch_tree.py).

The load-bearing invariant: messages[] is a READ-ONLY projection of the active
path (root -> active_leaf); off-active-path siblings are retained in the tree but
NEVER appear in the projection.
"""
from core import branch_tree as bt


def test_append_child_builds_linear_path_projected_in_order():
    t = bt.new_tree()
    bt.append_child(t, t["root_id"], role="user", text="explain X")
    a = bt.append_child(t, bt.active_path(t)[-1], role="assistant", text="X is ...")
    msgs = bt.project_to_messages(t)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert [m["text"] for m in msgs] == ["explain X", "X is ..."]
    assert [m["i"] for m in msgs] == [1, 2]
    assert t["nodes"][a]["msg"]["role"] == "assistant"


def test_fork_sibling_retains_original_and_switches_projection():
    t = bt.new_tree()
    u = bt.append_child(t, t["root_id"], role="user", text="explain X")
    a1 = bt.append_child(t, u, role="assistant", text="answer one")
    a2 = bt.fork_sibling(t, a1, role="assistant", text="answer two", origin="regen")
    # the original is RETAINED, not discarded
    assert a1 in t["nodes"]
    assert t["nodes"][a1]["msg"]["text"] == "answer one"
    # both are children of the same user node
    assert set(t["nodes"][u]["children"]) == {a1, a2}
    # active path ends at the new sibling; the old one is off-path and NOT projected
    assert [m["text"] for m in bt.project_to_messages(t)] == ["explain X", "answer two"]


def test_active_path_walks_parent_links_excluding_root():
    t = bt.new_tree()
    u = bt.append_child(t, t["root_id"], role="user", text="q")
    a = bt.append_child(t, u, role="assistant", text="r")
    assert bt.active_path(t) == [u, a]


def test_switch_active_leaf_reprojects_to_chosen_sibling():
    t = bt.new_tree()
    u = bt.append_child(t, t["root_id"], role="user", text="q")
    a1 = bt.append_child(t, u, role="assistant", text="first")
    bt.fork_sibling(t, a1, role="assistant", text="second")
    bt.set_active_leaf(t, a1)  # navigate back to the first take
    assert [m["text"] for m in bt.project_to_messages(t)] == ["q", "first"]


def test_prune_removes_node_and_all_descendants():
    t = bt.new_tree()
    u = bt.append_child(t, t["root_id"], role="user", text="q")
    a1 = bt.append_child(t, u, role="assistant", text="keep")
    a2 = bt.fork_sibling(t, a1, role="assistant", text="drop")
    follow = bt.append_child(t, a2, role="user", text="under drop")
    bt.prune(t, a2)
    assert a2 not in t["nodes"]
    assert follow not in t["nodes"]  # descendant removed too
    assert a2 not in t["nodes"][u]["children"]
    assert t["active_leaf"] in t["nodes"]  # relocated to a survivor


def test_siblings_lists_the_branch_set():
    t = bt.new_tree()
    u = bt.append_child(t, t["root_id"], role="user", text="q")
    a1 = bt.append_child(t, u, role="assistant", text="one")
    a2 = bt.fork_sibling(t, a1, role="assistant", text="two")
    assert set(bt.siblings(t, a1)) == {a1, a2}


def test_branch_kind_and_divergence_fields_are_carried():
    t = bt.new_tree()
    u = bt.append_child(t, t["root_id"], role="user", text="q")
    a1 = bt.append_child(t, u, role="assistant", text="one")
    a2 = bt.fork_sibling(t, a1, role="assistant", text="two",
                         origin="regen", branch_kind="variant", divergence=0.0)
    assert t["nodes"][a2]["branch_kind"] == "variant"
    assert t["nodes"][a2]["divergence"] == 0.0
    assert t["nodes"][a1]["branch_kind"] == "variant"  # default


def test_fork_sibling_of_root_is_rejected():
    t = bt.new_tree()
    import pytest
    with pytest.raises(ValueError):
        bt.fork_sibling(t, t["root_id"], role="user", text="nope")


def test_project_passes_through_task_id_and_node_id():
    t = bt.new_tree()
    u = bt.append_child(t, t["root_id"], role="user", text="q")
    a = bt.append_child(t, u, role="assistant", text="r", task_id="turn-123")
    msg = bt.project_to_messages(t)[-1]
    assert msg["task_id"] == "turn-123"
    assert msg["node_id"] == a  # the UI maps a rendered message back to its node


def test_projection_is_identity_preserving_so_inplace_writes_survive_reproject():
    # The hot streaming path mutates the projected message in place (msg["text"] += token).
    # For that to reach the tree node WITHOUT rerouting the streaming loop, the projection
    # must hand back the node's own payload by reference, not a copy.
    t = bt.new_tree()
    bt.append_child(t, t["root_id"], role="user", text="q")
    bt.append_child(t, bt.active_path(t)[-1], role="assistant", text="")
    bt.project_to_messages(t)[-1]["text"] += "streamed tokens"
    assert bt.project_to_messages(t)[-1]["text"] == "streamed tokens"


# --- 2a: divergence scorer (the variant/divergent edit classifier) ---

def test_score_divergence_is_zero_for_identical_text():
    assert bt.score_divergence("explain recursion", "explain recursion") == 0.0


def test_score_divergence_is_small_for_a_minor_edit():
    s = bt.score_divergence("explain recursion", "explain recursion please")
    assert 0.0 < s < 0.5


def test_score_divergence_is_large_for_a_total_rewrite():
    s = bt.score_divergence("explain recursion", "what is the capital of France")
    assert s > 0.5


def test_classify_edit_splits_variant_from_divergent_at_the_threshold():
    minor_kind, minor_score = bt.classify_edit("explain recursion", "explain recursion please")
    major_kind, major_score = bt.classify_edit("explain recursion", "what is the capital of France")
    assert minor_kind == "variant"
    assert major_kind == "divergent"
    assert minor_score < major_score


# --- payload-identity ops (append_child_payload, splice_child_payload,
#     default_descendant_leaf, serialize/deserialize) ---

def test_append_child_payload_wraps_existing_dict_by_identity():
    tree = bt.new_tree()
    msg = {"i": 1, "time": "t", "role": "user", "text": "hello", "kind": "x"}
    nid = bt.append_child_payload(tree, bt.ROOT_ID, msg, origin="user")
    node = tree["nodes"][nid]
    assert node["msg"] is msg
    assert msg["node_id"] == nid
    assert tree["active_leaf"] == nid
    assert node["origin"] == "user"


def test_splice_child_payload_reparents_displaced_child():
    tree = bt.new_tree()
    a = bt.append_child_payload(tree, bt.ROOT_ID, {"role": "user", "text": "u"})
    b = bt.append_child_payload(tree, a, {"role": "assistant", "text": "a"})
    t = bt.splice_child_payload(tree, a, b, {"role": "tool_result", "text": "{}"})
    assert tree["nodes"][a]["children"] == [t]
    assert tree["nodes"][t]["children"] == [b]
    assert tree["nodes"][b]["parent_id"] == t
    assert tree["active_leaf"] == b


def test_splice_child_payload_without_displaced_appends():
    tree = bt.new_tree()
    a = bt.append_child_payload(tree, bt.ROOT_ID, {"role": "user", "text": "u"})
    t = bt.splice_child_payload(tree, a, None, {"role": "tool_result", "text": "{}"})
    assert tree["nodes"][a]["children"] == [t]
    assert tree["active_leaf"] == t


def test_default_descendant_leaf_walks_first_children():
    tree = bt.new_tree()
    u = bt.append_child_payload(tree, bt.ROOT_ID, {"role": "user", "text": "u"})
    a1 = bt.append_child_payload(tree, u, {"role": "assistant", "text": "1"})
    u2 = bt.append_child_payload(tree, a1, {"role": "user", "text": "next"})
    a2 = bt.append_child_payload(tree, u, {"role": "assistant", "text": "2"})
    assert bt.default_descendant_leaf(tree, a1) == u2
    assert bt.default_descendant_leaf(tree, a2) == a2


def test_serialize_deserialize_round_trip():
    tree = bt.new_tree()
    u = bt.append_child_payload(tree, bt.ROOT_ID, {"role": "user", "text": "u", "task_id": "tt1"})
    blob = bt.serialize(tree)
    assert blob["schema_version"] == 1
    restored = bt.deserialize(blob)
    assert restored["active_leaf"] == u
    assert restored["nodes"][u]["msg"]["task_id"] == "tt1"
    assert restored["nodes"][bt.ROOT_ID]["children"] == [u]


def test_deserialize_rejects_garbage():
    import pytest
    with pytest.raises(ValueError):
        bt.deserialize({"schema_version": 1})
    with pytest.raises(ValueError):
        bt.deserialize({"schema_version": 99, "root_id": "root", "active_leaf": "root", "nodes": {}})


def test_deserialize_rejects_dangling_refs():
    import pytest
    import json
    tree = bt.new_tree()
    u = bt.append_child_payload(tree, bt.ROOT_ID, {"role": "user", "text": "u"})
    blob = bt.serialize(tree)
    bad_child = json.loads(json.dumps(blob))
    bad_child["nodes"][u]["children"] = ["ghost"]
    with pytest.raises(ValueError):
        bt.deserialize(bad_child)
    bad_parent = json.loads(json.dumps(blob))
    bad_parent["nodes"][u]["parent_id"] = "missingparent"
    with pytest.raises(ValueError):
        bt.deserialize(bad_parent)
