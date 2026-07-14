import pytest
from ui.pages.chat_session import ChatSessionManager


@pytest.fixture
def mgr():
    return ChatSessionManager(master_prompt="sys", now_iso=lambda: "2026-06-10T00:00:00+00:00")


def test_flag_off_no_tree_block(monkeypatch, mgr):
    monkeypatch.delenv("MONOLITH_BRANCH_TREE_V1", raising=False)
    mgr.add_message("user", "hello")
    assert "tree" not in mgr.current
    assert mgr.current["messages"][0]["text"] == "hello"
    assert "node_id" not in mgr.current["messages"][0]      # byte-identical legacy shape


def test_flag_on_add_message_builds_tree(monkeypatch, mgr):
    monkeypatch.setenv("MONOLITH_BRANCH_TREE_V1", "1")
    idx = mgr.add_message("user", "hello")
    assert idx == 0
    assert isinstance(mgr.current.get("tree"), dict)
    assert mgr.current["messages"][0]["node_id"] in mgr.current["tree"]["nodes"]


def test_flag_on_streaming_token_reaches_node(monkeypatch, mgr):
    monkeypatch.setenv("MONOLITH_BRANCH_TREE_V1", "1")
    mgr.add_message("user", "q")
    idx = mgr.add_message("assistant", "")
    mgr.append_assistant_token("tok", idx)
    leaf = mgr.current["tree"]["active_leaf"]
    assert mgr.current["tree"]["nodes"][leaf]["msg"]["text"] == "tok"


def test_flag_on_cleanup_prunes_placeholder(monkeypatch, mgr):
    monkeypatch.setenv("MONOLITH_BRANCH_TREE_V1", "1")
    mgr.add_message("user", "q")
    idx = mgr.add_message("assistant", "")
    removed = mgr.cleanup_empty_assistant_if_needed(idx, True, 0)
    assert removed is True
    assert [m["role"] for m in mgr.current["messages"]] == ["user"]
    assert len(mgr.current["tree"]["nodes"]) == 2            # root + user


def test_flag_on_undo_restores_tree(monkeypatch, mgr):
    monkeypatch.setenv("MONOLITH_BRANCH_TREE_V1", "1")
    mgr.add_message("user", "q")
    mgr.snapshot()
    mgr.add_message("assistant", "a")
    assert mgr.undo() is True
    assert [m["role"] for m in mgr.current["messages"]] == ["user"]
    assert len(mgr.current["tree"]["nodes"]) == 2


def test_set_current_clears_tree_undo(monkeypatch, mgr):
    monkeypatch.setenv("MONOLITH_BRANCH_TREE_V1", "1")
    mgr.add_message("user", "A-question")
    mgr.snapshot()
    session_b = mgr.create_session()
    mgr.set_current(session_b)
    mgr.add_message("user", "B-question")
    assert mgr.undo() is False                      # no cross-session restore
    assert [m["text"] for m in mgr.current["messages"]] == ["B-question"]


# ── turn-box wiring (Task 4) ───────────────────────────────────────────

from ui.pages.assistant_turn_box import AssistantTurnBox


@pytest.fixture
def turn_box():
    sessions = ChatSessionManager("MASTER", now_iso=lambda: "2026-06-10T00:00:00+00:00")
    sessions.set_current(sessions.create_session())
    box = AssistantTurnBox(sessions)
    return box


def test_flag_on_regen_forks_instead_of_truncating(monkeypatch, turn_box):
    monkeypatch.setenv("MONOLITH_BRANCH_TREE_V1", "1")
    box = turn_box
    box.sessions.add_message("user", "q")
    box.sessions.add_message("assistant", "old")
    prompt = box.regen_from_index(1)
    assert prompt == "q"
    tree = box.current_session["tree"]
    assert any(n["msg"].get("text") == "old" for n in tree["nodes"].values())  # retained
    assert [m["role"] for m in box.current_session["messages"]] == ["user"]


def test_flag_on_commit_edit_forks_user_sibling(monkeypatch, turn_box):
    monkeypatch.setenv("MONOLITH_BRANCH_TREE_V1", "1")
    box = turn_box
    box.sessions.add_message("user", "original question")
    box.sessions.add_message("assistant", "a")
    out = box.commit_edit_from_index(0, "completely different question about cats")
    assert out == "completely different question about cats"
    tree = box.current_session["tree"]
    from core.branch_tree import ROOT_ID
    assert len(tree["nodes"][ROOT_ID]["children"]) == 2          # both prompts retained


def test_flag_on_truncate_prunes(monkeypatch, turn_box):
    monkeypatch.setenv("MONOLITH_BRANCH_TREE_V1", "1")
    box = turn_box
    box.sessions.add_message("user", "q")
    box.sessions.add_message("assistant", "a")
    assert box.truncate_from(1) is True
    assert [m["role"] for m in box.current_session["messages"]] == ["user"]


def test_flag_off_regen_still_truncates(monkeypatch, turn_box):
    monkeypatch.delenv("MONOLITH_BRANCH_TREE_V1", raising=False)
    box = turn_box
    box.sessions.add_message("user", "q")
    box.sessions.add_message("assistant", "old")
    prompt = box.regen_from_index(1)
    assert prompt == "q"
    assert len(box.current_session["messages"]) == 1
    assert "tree" not in box.current_session


def test_flag_on_edit_fork_gets_fresh_timestamp(monkeypatch, turn_box):

    monkeypatch.setenv("MONOLITH_BRANCH_TREE_V1", "1")
    box = turn_box
    # Use an advancing clock so the edit stamp differs from the original stamp.
    _tick = {"n": 0}

    def _advancing_clock():
        _tick["n"] += 1
        return f"2026-06-10T00:00:{_tick['n']:02d}+00:00"

    box.sessions.now_iso = _advancing_clock
    box.sessions.add_message("user", "original")
    box.sessions.add_message("assistant", "a")
    old_time = box.current_session["messages"][0]["time"]
    box.commit_edit_from_index(0, "edited question")
    new_time = box.current_session["messages"][0]["time"]
    assert new_time != old_time


# ── Task 5: archive persistence round-trip ────────────────────────────

from pathlib import Path
from ui.pages.chat_archive import ChatArchiveManager


def _archive(tmp_path):
    mgr = ChatSessionManager(master_prompt="sys", now_iso=lambda: "2026-06-10T00:00:00+00:00")
    return ChatArchiveManager(tmp_path, mgr), mgr


def test_tree_round_trips_through_archive(monkeypatch, tmp_path):
    monkeypatch.setenv("MONOLITH_BRANCH_TREE_V1", "1")
    arch, mgr = _archive(tmp_path)
    mgr.add_message("user", "q")
    mgr.add_message("assistant", "<think>secret reasoning</think>answer")
    mgr.current["messages"][1]["task_id"] = "trace-1"
    path = arch.save_session(mgr.current)
    loaded = arch.load_session(Path(path))
    assert isinstance(loaded.get("tree"), dict)
    assert [m["role"] for m in loaded["messages"]] == ["user", "assistant"]
    assert loaded["messages"][1]["task_id"] == "trace-1"          # persisted now
    assert "<think>" in loaded["messages"][1]["text"]             # think envelope survives


def test_flag_off_load_ignores_tree_block(monkeypatch, tmp_path):
    monkeypatch.setenv("MONOLITH_BRANCH_TREE_V1", "1")
    arch, mgr = _archive(tmp_path)
    mgr.add_message("user", "q")
    path = arch.save_session(mgr.current)
    monkeypatch.delenv("MONOLITH_BRANCH_TREE_V1", raising=False)
    loaded = arch.load_session(Path(path))
    assert "tree" not in loaded                                    # legacy shape
    assert loaded["messages"][0]["text"] == "q"


def test_corrupt_tree_block_falls_back_to_messages(monkeypatch, tmp_path):
    import json
    monkeypatch.setenv("MONOLITH_BRANCH_TREE_V1", "1")
    arch, mgr = _archive(tmp_path)
    mgr.add_message("user", "q")
    path = Path(arch.save_session(mgr.current))
    data = json.loads(path.read_text(encoding="utf-8"))
    data["tree"] = {"schema_version": 1}                           # garbage
    path.write_text(json.dumps(data), encoding="utf-8")
    loaded = arch.load_session(path)
    assert loaded["messages"][0]["text"] == "q"                    # fell back, no crash


def test_fork_sibling_survives_round_trip(monkeypatch, tmp_path):
    monkeypatch.setenv("MONOLITH_BRANCH_TREE_V1", "1")
    arch, mgr = _archive(tmp_path)
    mgr.add_message("user", "q")
    mgr.add_message("assistant", "<think>old reasoning</think>old answer")
    from ui.pages import session_tree
    session_tree.tree_regen_reset(mgr.current, 1)
    mgr.add_message("assistant", "new answer")
    path = arch.save_session(mgr.current)
    loaded = arch.load_session(Path(path))
    texts = [n["msg"].get("text", "") for n in loaded["tree"]["nodes"].values()]
    assert any("old answer" in t for t in texts)            # off-path sibling survived
    assert [m["text"] for m in loaded["messages"]][-1] == "new answer"   # active path correct
    leaf = loaded["tree"]["active_leaf"]
    assert loaded["tree"]["nodes"][leaf]["origin"] == "regen"
