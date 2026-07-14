"""session_tree — the flag-gated chokepoint between the linear chat writers and
core.branch_tree.

Every STRUCTURAL session write (append / insert / regen / edit / delete / switch)
goes through here when MONOLITH_BRANCH_TREE_V1 is on: mutate the tree, reproject
``messages[]`` (identity-preserving — see core/branch_tree.py), emit a
canonical_log event, notify listeners (the ReasoningTreePanel). In-place content
writes (the streaming ``msg["text"] += token`` hot path, metadata stamps) never
come through here — they reach the node by identity, by design.

Flag OFF: ``active()`` is False and no caller routes here — writer behavior is
byte-identical to the legacy truncate-and-replace world.
"""
from __future__ import annotations

import copy
import logging
import os

from core import branch_tree

_FLAG_ENV = "MONOLITH_BRANCH_TREE_V1"

CANONICAL_KINDS = {
    "fork": "branch_forked",
    "switch": "branch_switched",
    "prune": "branch_pruned",
}

_listeners: list = []
_op_seq = 0


def flag_enabled() -> bool:
    return str(os.environ.get(_FLAG_ENV, "0")).strip().lower() in {"1", "true", "yes", "on"}


def active() -> bool:
    return flag_enabled()


def subscribe(cb) -> None:
    _listeners.append(cb)


def unsubscribe(cb) -> None:
    """Remove *cb* from listeners.  No-op if it was never registered."""
    try:
        _listeners.remove(cb)
    except ValueError:
        pass


def _notify(op: str) -> None:
    global _op_seq
    _op_seq += 1
    for cb in list(_listeners):
        try:
            cb(op)
        except Exception:
            pass  # a broken listener must never break a chat write


def _log(kind_key: str, payload: dict) -> None:
    try:
        from core.acatalepsy import canonical_log
        canonical_log.append(CANONICAL_KINDS[kind_key], payload=payload)
    except Exception:
        logging.getLogger(__name__).debug(
            "canonical_log append failed for %s", kind_key, exc_info=True
        )  # swallowed — telemetry must never crash a chat write


def fingerprint(session: dict) -> tuple:
    tree = session.get("tree")
    if not isinstance(tree, dict):
        return (0, "", 0)
    return (len(tree.get("nodes", {})), tree.get("active_leaf", ""), _op_seq)


# ── tree lifecycle ─────────────────────────────────────────────────────

def ensure_tree(session: dict) -> dict:
    """Return the session's tree, building it (and adopting any pre-existing flat
    messages as a linear chain) on first touch."""
    tree = session.get("tree")
    if isinstance(tree, dict) and "nodes" in tree:
        return tree
    tree = branch_tree.new_tree()
    for msg in list(session.get("messages", [])):
        if isinstance(msg, dict):
            branch_tree.append_child_payload(
                tree, tree["active_leaf"], msg, origin=str(msg.get("role", "user")))
    session["tree"] = tree
    _reproject(session)
    return tree


def _reproject(session: dict) -> None:
    session["messages"] = branch_tree.project_to_messages(session["tree"])


def _node_at(session: dict, index: int) -> dict | None:
    msgs = session.get("messages", [])
    if not (0 <= index < len(msgs)):
        return None
    nid = msgs[index].get("node_id")
    return session["tree"]["nodes"].get(nid) if nid else None


# ── structural ops (each: mutate → reproject → log → notify) ──────────

def tree_append(session: dict, message: dict) -> int:
    tree = ensure_tree(session)
    origin = session.pop("_next_origin", None) or str(message.get("role", "user"))
    branch_tree.append_child_payload(tree, tree["active_leaf"], message, origin=origin)
    _reproject(session)
    _notify("append")
    return len(session["messages"]) - 1


def tree_insert(session: dict, index: int, message: dict) -> int:
    session.pop("_next_origin", None)
    tree = ensure_tree(session)
    msgs = session["messages"]
    index = max(0, min(index, len(msgs)))
    parent = msgs[index - 1]["node_id"] if index > 0 else tree["root_id"]
    displaced = msgs[index]["node_id"] if index < len(msgs) else None
    branch_tree.splice_child_payload(tree, parent, displaced, message,
                                     origin=str(message.get("role", "user")))
    _reproject(session)
    _notify("insert")
    return index


def tree_regen_reset(session: dict, index: int) -> str | None:
    """Regen-as-fork: move the active leaf UP to the user node above ``index``.
    The old assistant subtree stays retained; the caller's normal generation
    append then lands as a sibling (origin stamped 'regen' via _next_origin)."""
    tree = ensure_tree(session)
    node = _node_at(session, index)
    if node is None or node["msg"].get("role") != "assistant":
        return None
    prompt = None
    cur = node
    while cur["parent_id"] is not None:
        cur = tree["nodes"][cur["parent_id"]]
        if cur["msg"].get("role") == "user":
            prompt = str(cur["msg"].get("text", ""))
            break
    if not prompt:
        return None
    branch_tree.set_active_leaf(tree, cur["id"])
    session["_next_origin"] = "regen"
    _reproject(session)
    _log("fork", {"op": "regen", "parent": cur["id"], "retained": node["id"]})
    _notify("regen")
    return prompt


def tree_fork_user_edit(session: dict, index: int, new_text: str,
                        now: str | None = None) -> str | None:
    """Fork the user node at *index* with *new_text*.

    Callers pass a fresh timestamp via *now* so the forked message carries the
    edit instant rather than the original node's time (legacy parity: legacy
    code stamped ``message["time"]`` on every edit).  Passing ``None`` inherits
    the sibling node's time, which is convenient for tests that don't care about
    timestamps.
    """
    session.pop("_next_origin", None)
    tree = ensure_tree(session)
    node = _node_at(session, index)
    new_text = str(new_text or "").strip()
    if node is None or node["msg"].get("role") != "user" or not new_text:
        return None
    kind, score = branch_tree.classify_edit(str(node["msg"].get("text", "")), new_text)
    fork_msg = {"i": 0, "time": now if now is not None else node["msg"].get("time", ""),
                "role": "user", "text": new_text}
    branch_tree.append_child_payload(tree, node["parent_id"], fork_msg,
                                     origin="edit", branch_kind=kind, divergence=score)
    _reproject(session)
    _log("fork", {"op": "edit", "kind": kind, "divergence": round(score, 3),
                  "parent": node["parent_id"], "retained": node["id"]})
    _notify("fork")
    return new_text


def tree_fork_assistant_rewrite(session: dict, index: int, new_text: str,
                                now: str | None = None) -> bool:
    """Fork the assistant node at *index* with *new_text*.

    Callers pass a fresh timestamp via *now* so the forked message carries the
    rewrite instant rather than the original node's time (legacy parity).
    Passing ``None`` inherits the sibling node's time, which is convenient for
    tests that don't care about timestamps.
    """
    session.pop("_next_origin", None)
    tree = ensure_tree(session)
    node = _node_at(session, index)
    if node is None or node["msg"].get("role") != "assistant":
        return False
    fork_msg = {"i": 0, "time": now if now is not None else node["msg"].get("time", ""),
                "role": "assistant",
                "text": str(new_text), "task_id": node["msg"].get("task_id", "")}
    branch_tree.append_child_payload(tree, node["parent_id"], fork_msg,
                                     origin="edit", branch_kind="variant")
    _reproject(session)
    _log("fork", {"op": "rewrite", "parent": node["parent_id"], "retained": node["id"]})
    _notify("fork")
    return True


def tree_prune_from(session: dict, index: int) -> bool:
    session.pop("_next_origin", None)
    tree = ensure_tree(session)
    node = _node_at(session, index)
    if node is None:
        return False
    branch_tree.prune(tree, node["id"])
    _reproject(session)
    _log("prune", {"node": node["id"]})
    _notify("prune")
    return True


def tree_prune_node(session: dict, node_id: str) -> bool:
    """Prune by node id (the empty-placeholder cleanup path)."""
    session.pop("_next_origin", None)
    tree = ensure_tree(session)
    if node_id not in tree["nodes"]:
        return False
    branch_tree.prune(tree, node_id)
    _reproject(session)
    _log("prune", {"node": node_id})
    _notify("prune")
    return True


def tree_switch(session: dict, node_id: str) -> bool:
    session.pop("_next_origin", None)
    tree = ensure_tree(session)
    if node_id not in tree["nodes"]:
        return False
    leaf = branch_tree.default_descendant_leaf(tree, node_id)
    branch_tree.set_active_leaf(tree, leaf)
    _reproject(session)
    _log("switch", {"node": node_id, "leaf": leaf})
    _notify("switch")
    return True


# ── in-chat ‹k/n› take switcher ────────────────────────────────────────

def take_info_for_index(session: dict, index: int) -> tuple[int, int] | None:
    """(k, n) sibling position for the message at ``index``; None when n == 1
    or the tree is absent — the ‹k/n› chip's data source. READ-ONLY: never
    mutates the session (must not pop ``_next_origin``)."""
    tree = session.get("tree")
    node = _node_at(session, index) if isinstance(tree, dict) else None
    if node is None or node["parent_id"] is None:
        return None
    sibs = tree["nodes"][node["parent_id"]]["children"]
    if len(sibs) < 2:
        return None
    return (sibs.index(node["id"]) + 1, len(sibs))


def switch_take(session: dict, index: int, direction: int) -> bool:
    """Step the message at ``index`` to its prev/next sibling take (the in-chat
    ‹k/n› control). Clamps at the ends. Delegates to ``tree_switch``, which
    already clears ``_next_origin``."""
    tree = session.get("tree")
    node = _node_at(session, index) if isinstance(tree, dict) else None
    if node is None or node["parent_id"] is None:
        return False
    sibs = tree["nodes"][node["parent_id"]]["children"]
    k = sibs.index(node["id"]) + (1 if direction > 0 else -1)
    if not (0 <= k < len(sibs)):
        return False
    return tree_switch(session, sibs[k])


# ── thinkpad fan-out ───────────────────────────────────────────────────

def persist_fanout(session: dict, branches, *, now: str | None = None) -> int:
    """Persist Thinkpad fan-out candidates as N sibling assistant nodes under the
    active user node. Commit the FIRST (active_leaf) — advisory rank never
    auto-commits (commit-authority gate, see core/thinkpad.py) — retain the rest.
    ``now`` stamps each sibling's display time (callers pass a fresh stamp, same
    convention as the fork fns). Returns the TOTAL sibling count under the parent
    after the write (pre-existing children included, e.g. a regen-retained
    answer); 0 if the leaf isn't a user node or no branches were given."""
    session.pop("_next_origin", None)
    branches = list(branches)          # materialise once so we can len() after iteration
    tree = ensure_tree(session)
    leaf = tree["nodes"][tree["active_leaf"]]
    if leaf["msg"].get("role") != "user":
        return 0
    parent_id = leaf["id"]
    first_id = None
    for i, br in enumerate(branches):
        msg = {"i": 0, "time": now or "", "role": "assistant",
               "text": str(getattr(br, "raw", "") or ""),
               "task_id": str(getattr(br, "trace_id", "") or "")}
        nid = branch_tree.append_child_payload(tree, parent_id, msg,
                                               origin=f"thinkpad:c{i}")
        if first_id is None:
            first_id = nid
    if first_id is None:
        return 0
    branch_tree.set_active_leaf(tree, first_id)
    _reproject(session)
    _log("fork", {"op": "fanout", "parent": parent_id, "n": len(branches)})
    _notify("fork")
    return len(tree["nodes"][parent_id]["children"])


# ── undo support ───────────────────────────────────────────────────────

def snapshot(session: dict) -> dict | None:
    tree = session.get("tree")
    return copy.deepcopy(tree) if isinstance(tree, dict) else None


def restore(session: dict, snap: dict | None) -> bool:
    session.pop("_next_origin", None)
    if not isinstance(snap, dict):
        return False
    session["tree"] = snap
    _reproject(session)
    _notify("restore")
    return True
