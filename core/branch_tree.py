"""core/branch_tree.py — the reasoning branch tree (pure node store).

The conversation as a tree: each message is a node with a stable id, a parent, and
possibly multiple children (= branches). Regen and Thinkpad fan-out add *variant*
siblings (same prompt); a drastic edit adds a *divergent* sibling (different prompt).

The flat ``messages[]`` the rest of Monolith reads is a READ-ONLY projection of the
active path (root -> active_leaf). Nothing mutates ``messages[]`` directly; every
writer mutates the tree here, then reprojects. Off-active-path siblings are retained
in the tree but never appear in the projection — that is what keeps abandoned
branches out of the engine history, the summary, and export.

Pure: no Qt, no IO, no clock. ``time`` is passed in by the caller so the module
stays deterministic and testable. MonoTrace content (think/cites) lives in
turn_trace, linked by each node's ``task_id``; this module holds only the shape.
"""
from __future__ import annotations

import difflib
import uuid

ROOT_ID = "root"

# Above this lexical-divergence score, an edit is a *different prompt* (divergent),
# not an alternative phrasing of the same one (variant). Tunable.
DIVERGENCE_THRESHOLD = 0.5

# A node separates TREE metadata from the MESSAGE payload:
#   node = {id, parent_id, children, origin, branch_kind, divergence, msg: {...}}
# project_to_messages hands back ``node["msg"]`` BY REFERENCE, so in-place content
# writes (the streaming `msg["text"] += token`, metadata stamps) reach the node for
# free without rerouting the hot path. The structural keys never leak into messages[]
# because they live on the node, outside ``msg``.


def _new_node(parent_id, *, role, text, time="", task_id="",
              origin="user", branch_kind="variant", divergence=None, **extra) -> dict:
    nid = uuid.uuid4().hex
    msg = {"node_id": nid, "i": 0, "role": role, "text": text, "time": time, "task_id": task_id}
    msg.update(extra)  # arbitrary message extras (kind, agent_approved, ...) ride in the payload
    return {
        "id": nid, "parent_id": parent_id, "children": [],
        "origin": origin, "branch_kind": branch_kind, "divergence": divergence,
        "msg": msg,
    }


def new_tree() -> dict:
    """An empty tree anchored by a synthetic root, so even the first user message
    has a parent (and can therefore be edit-forked into a sibling)."""
    root = {
        "id": ROOT_ID, "parent_id": None, "children": [],
        "origin": "root", "branch_kind": "variant", "divergence": None,
        "msg": {"node_id": ROOT_ID, "i": 0, "role": "root", "text": "", "time": "", "task_id": ""},
    }
    return {"root_id": ROOT_ID, "active_leaf": ROOT_ID, "nodes": {ROOT_ID: root}}


def append_child(tree: dict, parent_id: str, **fields) -> str:
    """Create a child under ``parent_id``, advance active_leaf to it, return its id."""
    if parent_id not in tree["nodes"]:
        raise KeyError(f"no such parent: {parent_id}")
    node = _new_node(parent_id, **fields)
    tree["nodes"][node["id"]] = node
    tree["nodes"][parent_id]["children"].append(node["id"])
    tree["active_leaf"] = node["id"]
    return node["id"]


def fork_sibling(tree: dict, node_id: str, **fields) -> str:
    """Create a sibling of ``node_id`` (a new child of its parent). This is the
    single operation behind both regen (a variant answer) and edit (a divergent
    prompt): same parent, new branch. active_leaf advances to the new sibling."""
    if node_id not in tree["nodes"]:
        raise KeyError(f"no such node: {node_id}")
    parent_id = tree["nodes"][node_id]["parent_id"]
    if parent_id is None:
        raise ValueError("cannot fork a sibling of the root")
    return append_child(tree, parent_id, **fields)


def set_active_leaf(tree: dict, node_id: str) -> None:
    """Switch the active path to end at ``node_id`` (the ‹k/n› sibling switch)."""
    if node_id not in tree["nodes"]:
        raise KeyError(f"no such node: {node_id}")
    tree["active_leaf"] = node_id


def active_path(tree: dict) -> list[str]:
    """Node ids from the first real message down to active_leaf (excludes root)."""
    path: list[str] = []
    cur = tree["active_leaf"]
    while cur is not None and cur != tree["root_id"]:
        path.append(cur)
        cur = tree["nodes"][cur]["parent_id"]
    path.reverse()
    return path


def siblings(tree: dict, node_id: str) -> list[str]:
    """The branch set ``node_id`` belongs to (its parent's children, in order)."""
    parent_id = tree["nodes"][node_id]["parent_id"]
    if parent_id is None:
        return [node_id]
    return list(tree["nodes"][parent_id]["children"])


def prune(tree: dict, node_id: str) -> None:
    """Detach ``node_id`` and all its descendants. If active_leaf was inside the
    pruned subtree, relocate it to the nearest surviving ancestor (the parent)."""
    if node_id == tree["root_id"]:
        raise ValueError("cannot prune the root")
    if node_id not in tree["nodes"]:
        return
    parent_id = tree["nodes"][node_id]["parent_id"]
    doomed: list[str] = []
    stack = [node_id]
    while stack:
        nid = stack.pop()
        doomed.append(nid)
        stack.extend(tree["nodes"][nid]["children"])
    doomed_set = set(doomed)
    for nid in doomed:
        del tree["nodes"][nid]
    if parent_id is not None and parent_id in tree["nodes"]:
        tree["nodes"][parent_id]["children"] = [
            c for c in tree["nodes"][parent_id]["children"] if c not in doomed_set
        ]
    if tree["active_leaf"] in doomed_set:
        tree["active_leaf"] = parent_id if parent_id in tree["nodes"] else tree["root_id"]


def score_divergence(old: str, new: str) -> float:
    """How much an edit changed the text, 0.0 (identical) .. 1.0 (nothing in common).

    Lexical proxy via difflib — cheap, deterministic, no model call. It is fooled by
    one-word meaning flips ("is X safe" -> "is X unsafe" reads as minor); semantic
    divergence (embedding distance) is the V2 upgrade. Good enough for V1's "is this
    a different prompt or a rephrase" split.
    """
    ratio = difflib.SequenceMatcher(None, old or "", new or "").ratio()
    return 1.0 - ratio


def classify_edit(old: str, new: str, threshold: float = DIVERGENCE_THRESHOLD) -> tuple[str, float]:
    """Classify a user-message edit: ('variant'|'divergent', divergence_score).

    Variant = same question, rephrased -> the verdict may rank its answer against
    siblings. Divergent = a different question -> the verdict must refuse to rank
    across it. The kind is what guards apples-vs-oranges ranking downstream.
    """
    score = score_divergence(old, new)
    kind = "divergent" if score > threshold else "variant"
    return kind, score


def project_to_messages(tree: dict) -> list[dict]:
    """The active path as a flat messages[] list — the projection every linear
    consumer reads. Each entry is the node's OWN ``msg`` payload BY REFERENCE
    (identity-preserving: in-place content writes reach the node), carrying node_id
    (for UI→node mapping) and task_id (the MonoTrace link). The list itself is
    replaced wholesale on every reproject — never cache it across a tree op."""
    out: list[dict] = []
    for i, nid in enumerate(active_path(tree), start=1):
        msg = tree["nodes"][nid]["msg"]  # by reference
        msg["i"] = i
        out.append(msg)
    return out


# ---------------------------------------------------------------------------
# Payload-identity ops — callers own the msg dict; these wrap it BY REFERENCE
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1


def append_child_payload(tree: dict, parent_id: str, msg: dict, *,
                         origin: str = "user",
                         branch_kind: str = "variant",
                         divergence=None) -> str:
    """Create a child under ``parent_id`` whose payload IS ``msg`` (by reference).

    Unlike ``append_child`` which constructs a fresh msg dict, this function
    wraps an EXISTING dict so callers can stream ``msg["text"] += token`` into
    the same object the projection hands back. Two stamps are injected in place:

    * ``msg["node_id"]`` is set to the new node id.
    * ``msg.setdefault("i", 0)`` ensures the sequence counter field exists
      (project_to_messages will overwrite it on the next reproject).

    ``tree["active_leaf"]`` advances to the new node.

    Raises KeyError if ``parent_id`` is not in the tree.
    """
    if parent_id not in tree["nodes"]:
        raise KeyError(f"no such parent: {parent_id}")
    nid = uuid.uuid4().hex
    msg["node_id"] = nid
    msg.setdefault("i", 0)
    node = {
        "id": nid, "parent_id": parent_id, "children": [],
        "origin": origin, "branch_kind": branch_kind, "divergence": divergence,
        "msg": msg,
    }
    tree["nodes"][nid] = node
    tree["nodes"][parent_id]["children"].append(nid)
    tree["active_leaf"] = nid
    return nid


def splice_child_payload(tree: dict, parent_id: str, displaced_id, msg: dict, *,
                         origin: str = "user") -> str:
    """Insert a new node between ``parent_id`` and (optionally) one of its children.

    This is the tool-row insert: a tool-result node must appear between the
    assistant turn that called the tool (``parent_id``) and the next message
    (``displaced_id``). The new node wraps ``msg`` BY REFERENCE (same identity
    guarantee as ``append_child_payload``).

    * If ``displaced_id`` is not None: the displaced child and its entire subtree
      are re-parented under the new node. The new node takes the displaced child's
      former positional index in ``parent_id``'s children list. After the structural
      surgery ``tree["active_leaf"]`` is RESTORED to its pre-call value — splice is
      a structural move, not a path switch.
    * If ``displaced_id`` is None: the new node is appended as a plain leaf and
      ``tree["active_leaf"]`` advances to it (same semantics as
      ``append_child_payload``).

    Raises KeyError if ``parent_id`` or ``displaced_id`` (when not None) is
    missing from the tree.
    """
    if parent_id not in tree["nodes"]:
        raise KeyError(f"no such parent: {parent_id}")
    if displaced_id is not None and displaced_id not in tree["nodes"]:
        raise KeyError(f"no such displaced node: {displaced_id}")

    saved_leaf = tree["active_leaf"]

    nid = uuid.uuid4().hex
    msg["node_id"] = nid
    msg.setdefault("i", 0)
    node = {
        "id": nid, "parent_id": parent_id, "children": [],
        "origin": origin, "branch_kind": "variant", "divergence": None,
        "msg": msg,
    }
    tree["nodes"][nid] = node

    if displaced_id is None:
        # Plain append — leaf advances.
        tree["nodes"][parent_id]["children"].append(nid)
        tree["active_leaf"] = nid
    else:
        # Structural insert: new node takes displaced child's position.
        kids = tree["nodes"][parent_id]["children"]
        idx = kids.index(displaced_id)
        kids[idx] = nid                       # replace displaced_id with new node at same index
        # Re-parent the displaced child under the new node.
        node["children"].append(displaced_id)
        tree["nodes"][displaced_id]["parent_id"] = nid
        # Restore active_leaf — this is a structural op, not a navigation.
        tree["active_leaf"] = saved_leaf

    return nid


def default_descendant_leaf(tree: dict, node_id: str) -> str:
    """Walk the FIRST child from ``node_id`` downward until a leaf; return its id.

    "Default" here means the eldest / first-created branch at every fork — the
    same path a linear reader would follow if it never noticed a branch existed.
    If ``node_id`` is itself a leaf (no children) it is returned immediately.

    Raises KeyError if ``node_id`` is not in the tree.
    """
    if node_id not in tree["nodes"]:
        raise KeyError(f"no such node: {node_id}")
    cur = node_id
    while tree["nodes"][cur]["children"]:
        cur = tree["nodes"][cur]["children"][0]
    return cur


def serialize(tree: dict) -> dict:
    """Snapshot the tree to a plain dict safe for JSON serialisation.

    Returns ``{"schema_version": SCHEMA_VERSION, "root_id": ...,
    "active_leaf": ..., "nodes": tree["nodes"]}``.  The nodes dict is shared
    by reference here; callers that need a deep copy must do so themselves
    (e.g. ``json.loads(json.dumps(serialize(tree)))``).
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "root_id": tree["root_id"],
        "active_leaf": tree["active_leaf"],
        "nodes": tree["nodes"],
    }


def deserialize(blob) -> dict:
    """Restore a tree from a blob produced by ``serialize``.

    Validates the following shape invariants and raises ``ValueError`` on any
    violation:

    * ``blob`` is a dict.
    * ``blob["schema_version"] == SCHEMA_VERSION`` (currently 1).
    * ``blob["nodes"]`` is a dict.
    * ``blob["root_id"]`` is present and exists as a key in nodes.
    * ``blob["active_leaf"]`` is present and exists as a key in nodes.
    * Every node in nodes is a dict with ``node["id"] == key`` and a ``"msg"``
      key.
    * Referential integrity: every ``parent_id`` and every entry in
      ``children`` points to a key that actually exists in nodes.  This makes
      ``deserialize`` the single chokepoint that converts ALL structural
      corruption (dangling refs, truncated blobs) into ``ValueError`` before
      any tree walker can raise ``KeyError`` against the caller's
      ``except ValueError`` fallback.

    Returns ``{"root_id": ..., "active_leaf": ..., "nodes": ...}``.
    """
    if not isinstance(blob, dict):
        raise ValueError("blob must be a dict")
    version = blob.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {version!r} (expected {SCHEMA_VERSION})")
    nodes = blob.get("nodes")
    if not isinstance(nodes, dict):
        raise ValueError("blob['nodes'] must be a dict")
    root_id = blob.get("root_id")
    if root_id not in nodes:
        raise ValueError(f"root_id {root_id!r} not found in nodes")
    active_leaf = blob.get("active_leaf")
    if active_leaf not in nodes:
        raise ValueError(f"active_leaf {active_leaf!r} not found in nodes")
    for key, node in nodes.items():
        if not isinstance(node, dict):
            raise ValueError(f"node at key {key!r} is not a dict")
        if node.get("id") != key:
            raise ValueError(f"node['id'] {node.get('id')!r} != key {key!r}")
        if "msg" not in node:
            raise ValueError(f"node at key {key!r} has no 'msg' field")
    for key, node in nodes.items():
        parent = node.get("parent_id")
        if parent is not None and parent not in nodes:
            raise ValueError(f"node {key!r} parent_id {parent!r} not in nodes")
        for child in node.get("children", []):
            if child not in nodes:
                raise ValueError(f"node {key!r} child {child!r} not in nodes")
    return {"root_id": root_id, "active_leaf": active_leaf, "nodes": nodes}
