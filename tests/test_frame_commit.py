"""Tests for the frame-commit fastpath (MONOLITH_FRAME_COMMIT_V1, dark).

Coverage:
  - kill_switch.frame_commit_is_enabled() — default OFF / ON.
  - updater.commit_frame — empty input, valid short frame, over-400-char frame,
    only current_frame mutated, bearing.is_empty() False after commit.
  - chat_finalize — flag OFF leaves bearing unchanged; flag ON + empty bearing +
    <frame> present → current_frame committed; bu_outcome "applied" → skipped.

Isolation mirrors test_bearing_updater.py: monkeypatch store._STORE_PATH and
audit._AUDIT_PATH to tmp_path so no real config dir is touched.
"""
from __future__ import annotations

import pytest

from addons.system.bearing import audit
from addons.system.bearing import schema as bs
from addons.system.bearing import store


# ── shared fixture ───────────────────────────────────────────────────


@pytest.fixture()
def tmp_bearing(monkeypatch, tmp_path):
    """Redirect the bearing store and audit JSONL to a temp dir."""
    monkeypatch.setattr(store, "_STORE_PATH", tmp_path / "bearing.json")
    monkeypatch.setattr(audit, "_AUDIT_PATH", tmp_path / "bearing.audit.jsonl")
    yield tmp_path


# ── kill_switch.frame_commit_is_enabled() ───────────────────────────


class TestFrameCommitFlag:
    def test_default_off(self, monkeypatch):
        monkeypatch.delenv("MONOLITH_FRAME_COMMIT_V1", raising=False)
        from addons.system.bearing import kill_switch
        assert kill_switch.frame_commit_is_enabled() is False

    def test_explicitly_zero_is_off(self, monkeypatch):
        monkeypatch.setenv("MONOLITH_FRAME_COMMIT_V1", "0")
        from addons.system.bearing import kill_switch
        assert kill_switch.frame_commit_is_enabled() is False

    def test_one_is_on(self, monkeypatch):
        monkeypatch.setenv("MONOLITH_FRAME_COMMIT_V1", "1")
        from addons.system.bearing import kill_switch
        assert kill_switch.frame_commit_is_enabled() is True

    def test_true_is_on(self, monkeypatch):
        monkeypatch.setenv("MONOLITH_FRAME_COMMIT_V1", "true")
        from addons.system.bearing import kill_switch
        assert kill_switch.frame_commit_is_enabled() is True

    def test_yes_is_on(self, monkeypatch):
        monkeypatch.setenv("MONOLITH_FRAME_COMMIT_V1", "yes")
        from addons.system.bearing import kill_switch
        assert kill_switch.frame_commit_is_enabled() is True

    def test_on_is_on(self, monkeypatch):
        monkeypatch.setenv("MONOLITH_FRAME_COMMIT_V1", "on")
        from addons.system.bearing import kill_switch
        assert kill_switch.frame_commit_is_enabled() is True

    def test_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("MONOLITH_FRAME_COMMIT_V1", "TRUE")
        from addons.system.bearing import kill_switch
        assert kill_switch.frame_commit_is_enabled() is True


# ── updater.commit_frame ─────────────────────────────────────────────


class TestCommitFrame:
    def test_empty_string_returns_false_no_write(self, tmp_bearing):
        from addons.system.bearing import updater
        result = updater.commit_frame("t1", "")
        assert result is False
        assert store.get_bearing().current_frame == ""

    def test_whitespace_only_returns_false_no_write(self, tmp_bearing):
        from addons.system.bearing import updater
        result = updater.commit_frame("t1", "   ")
        assert result is False
        assert store.get_bearing().current_frame == ""

    def test_none_returns_false_no_write(self, tmp_bearing):
        from addons.system.bearing import updater
        result = updater.commit_frame("t1", None)  # type: ignore[arg-type]
        assert result is False
        assert store.get_bearing().current_frame == ""

    def test_valid_short_frame_returns_true(self, tmp_bearing):
        from addons.system.bearing import updater
        result = updater.commit_frame("t1", "Helping user debug a Python error")
        assert result is True

    def test_valid_short_frame_updates_current_frame(self, tmp_bearing):
        from addons.system.bearing import updater
        frame_text = "Helping user debug a Python error"
        updater.commit_frame("t1", frame_text)
        assert store.get_bearing().current_frame == frame_text

    def test_valid_frame_audits_applied_with_source(self, tmp_bearing):
        from addons.system.bearing import updater
        updater.commit_frame("t1", "some frame text")
        rows = audit.read_recent()
        applied = [r for r in rows if r["kind"] == "applied"]
        assert len(applied) == 1
        assert applied[0]["turn_id"] == "t1"
        assert applied[0].get("source") == "frame_fastpath"
        assert applied[0].get("slots_changed") == ["current_frame"]

    def test_over_400_char_frame_returns_false(self, tmp_bearing):
        from addons.system.bearing import updater
        long_frame = "x" * 401
        result = updater.commit_frame("t1", long_frame)
        assert result is False

    def test_over_400_char_frame_leaves_current_frame_unchanged(self, tmp_bearing):
        from addons.system.bearing import updater
        store.set_bearing(bs.Bearing(current_frame="keep me"))
        long_frame = "x" * 401
        updater.commit_frame("t1", long_frame)
        assert store.get_bearing().current_frame == "keep me"

    def test_over_400_char_frame_audits_rejected_with_source(self, tmp_bearing):
        from addons.system.bearing import updater
        long_frame = "x" * 401
        updater.commit_frame("t1", long_frame)
        rows = audit.read_recent()
        rejected = [r for r in rows if r["kind"] == "rejected"]
        assert len(rejected) == 1
        assert rejected[0]["turn_id"] == "t1"
        assert rejected[0].get("source") == "frame_fastpath"

    def test_commit_sets_only_current_frame_other_slots_untouched(self, tmp_bearing):
        from addons.system.bearing import updater
        original = bs.Bearing(
            active_goal="keep goal",
            trajectory="keep traj",
            next_move="keep nm",
            open_tensions=(bs.Tension(text="keep tension", opened_at_turn="t0"),),
        )
        store.set_bearing(original)
        updater.commit_frame("t1", "new frame here")
        b = store.get_bearing()
        assert b.current_frame == "new frame here"
        assert b.active_goal == "keep goal"
        assert b.trajectory == "keep traj"
        assert b.next_move == "keep nm"
        assert len(b.open_tensions) == 1
        assert b.open_tensions[0].text == "keep tension"

    def test_after_commit_bearing_is_not_empty(self, tmp_bearing):
        """Loop-close: after commit, is_empty() is False → empty-bearing nudge stops."""
        from addons.system.bearing import updater
        assert store.get_bearing().is_empty() is True
        updater.commit_frame("t1", "discussing the refactor plan")
        assert store.get_bearing().is_empty() is False

    def test_exactly_400_char_frame_is_accepted(self, tmp_bearing):
        """Boundary: 400 chars should pass D5 (limit is MAX_CURRENT_FRAME = 400)."""
        from addons.system.bearing import updater
        frame_at_limit = "a" * 400
        result = updater.commit_frame("t1", frame_at_limit)
        assert result is True
        assert store.get_bearing().current_frame == frame_at_limit

    def test_commit_stamps_turn_id(self, tmp_bearing):
        from addons.system.bearing import updater
        updater.commit_frame("my-turn-uuid", "some frame")
        assert store.get_bearing().updated_at_turn == "my-turn-uuid"

    def test_commit_stamps_model_id(self, tmp_bearing):
        from addons.system.bearing import updater
        updater.commit_frame("t1", "some frame", model_id="deepseek-v3")
        assert store.get_bearing().last_writer_model_id == "deepseek-v3"

    def test_commit_stamps_turn_n(self, tmp_bearing):
        from addons.system.bearing import updater
        updater.commit_frame("t1", "some frame", turn_n=42)
        assert store.get_bearing().updated_at_turn_n == 42

    def test_commit_does_not_touch_rejection_streak(self, tmp_bearing):
        """commit_frame is not in the <bearing_update> path; streak is untouched."""
        from addons.system.bearing import updater
        store.increment_rejection_streak()
        store.increment_rejection_streak()
        updater.commit_frame("t1", "a valid frame")
        # streak must NOT be reset by the frame-fastpath
        assert store.get_rejection_streak() == 2

    def test_commit_does_not_touch_pending_rejection(self, tmp_bearing):
        """commit_frame must not clear a pending rejection meant for next-turn block."""
        from addons.system.bearing import updater
        store.set_pending_rejection(["D1"], turn_id="t0", ts="ts0", detail="missing reason")
        updater.commit_frame("t1", "a valid frame")
        assert store.get_pending_rejection() is not None


# ── chat_finalize integration ────────────────────────────────────────


class TestChatFinalizeFrameCommit:
    """Integration: verify the gate in finalize_assistant_turn.

    We need both MONOLITH_OBSERVABLE_FRAME_V0=1 (to enter the observe block)
    and MONOLITH_FRAME_COMMIT_V1=1 (for the commit). MONOLITH_BEARING_V1 is
    default ON (truthy), so we leave it unset.
    """

    @pytest.fixture()
    def _finalize_env(self, monkeypatch, tmp_bearing):
        """Common env: bearing on, observable frame on, verifier off, model-id stub."""
        monkeypatch.setenv("MONOLITH_BEARING_V1", "1")
        monkeypatch.setenv("MONOLITH_OBSERVABLE_FRAME_V0", "1")
        monkeypatch.setenv("MONOLITH_VERIFIER_V1", "0")
        monkeypatch.setenv("MONOLITH_SOURCE_TIER_V1", "0")
        monkeypatch.setenv("MONOLITH_GROUNDED_VERDICT_V1", "0")
        # Redirect frame ledger so it doesn't hit real CONFIG_DIR.
        from addons.system.bearing import frame_observe
        monkeypatch.setattr(frame_observe, "_LEDGER", tmp_bearing / "frame.ledger.jsonl")
        # Stub get_current_model_id to avoid config-dir reads.
        import core.llm_config as llm_cfg
        monkeypatch.setattr(llm_cfg, "get_current_model_id", lambda: "stub-model")
        return tmp_bearing

    def _call_finalize(self, raw: str, config: dict | None = None):
        from core.chat_finalize import finalize_assistant_turn
        finalize_assistant_turn(
            raw=raw,
            public=raw,
            config=config or {},
            emit_pipeline_ready=lambda r, p, t: None,
            record_verdict=lambda v: None,
        )

    # -- flag OFF: no commit regardless --------------------------------

    def test_flag_off_frame_turn_does_not_commit(self, monkeypatch, _finalize_env):
        monkeypatch.setenv("MONOLITH_FRAME_COMMIT_V1", "0")
        self._call_finalize("<frame>working on auth module</frame>")
        assert store.get_bearing().current_frame == ""

    def test_flag_off_bearing_is_still_empty(self, monkeypatch, _finalize_env):
        monkeypatch.setenv("MONOLITH_FRAME_COMMIT_V1", "0")
        self._call_finalize("<frame>working on auth module</frame>")
        assert store.get_bearing().is_empty() is True

    # -- flag ON: commit fires on empty bearing + frame present --------

    def test_flag_on_empty_bearing_frame_present_commits(self, monkeypatch, _finalize_env):
        monkeypatch.setenv("MONOLITH_FRAME_COMMIT_V1", "1")
        self._call_finalize("<frame>working on auth module</frame>", config={"_turn_id": "t-fin-1"})
        assert store.get_bearing().current_frame == "working on auth module"

    def test_flag_on_commit_makes_bearing_not_empty(self, monkeypatch, _finalize_env):
        monkeypatch.setenv("MONOLITH_FRAME_COMMIT_V1", "1")
        self._call_finalize("<frame>working on auth module</frame>", config={"_turn_id": "t-fin-2"})
        assert store.get_bearing().is_empty() is False

    def test_flag_on_commit_audited_applied(self, monkeypatch, _finalize_env):
        monkeypatch.setenv("MONOLITH_FRAME_COMMIT_V1", "1")
        self._call_finalize("<frame>working on auth module</frame>", config={"_turn_id": "t-fin-3"})
        rows = audit.read_recent()
        applied = [r for r in rows if r["kind"] == "applied" and r.get("source") == "frame_fastpath"]
        assert len(applied) == 1

    # -- bu_outcome "applied" takes precedence; commit skipped ---------

    def test_bu_applied_skips_frame_commit(self, monkeypatch, _finalize_env):
        """When a real <bearing_update> was applied this turn (bu_outcome=="applied"),
        the frame-commit fastpath must NOT fire and must not overwrite the richer form."""
        monkeypatch.setenv("MONOLITH_FRAME_COMMIT_V1", "1")
        # A valid <bearing_update> that sets current_frame via the real path.
        raw = (
            '<bearing_update>{"current_frame": {"new": "from bearing update", '
            '"previous": "", "reason": "context set", "trigger": "user ask"}}'
            "</bearing_update>"
            "<frame>a different frame text</frame>"
        )
        self._call_finalize(raw, config={"_turn_id": "t-fin-4"})
        # The bearing_update path sets current_frame; frame-commit must not override it.
        b = store.get_bearing()
        # Either the bearing_update committed "from bearing update"
        # (and frame-commit was skipped), OR both paths ran (wrong). Verify only the
        # bearing_update value can be present — never the "a different frame text" value
        # overwriting it.
        assert b.current_frame == "from bearing update"

    def test_no_frame_tag_does_not_commit(self, monkeypatch, _finalize_env):
        monkeypatch.setenv("MONOLITH_FRAME_COMMIT_V1", "1")
        self._call_finalize("plain answer with no frame tag", config={"_turn_id": "t-fin-5"})
        assert store.get_bearing().current_frame == ""

    def test_frame_matches_existing_bearing_no_spurious_audit(self, monkeypatch, _finalize_env):
        """disparity=='match' → commit does NOT fire (only 'empty_bearing'/'differ' trigger it)."""
        monkeypatch.setenv("MONOLITH_FRAME_COMMIT_V1", "1")
        # Pre-seed bearing with the same frame text.
        store.set_bearing(bs.Bearing(current_frame="already set frame"))
        self._call_finalize("<frame>already set frame</frame>", config={"_turn_id": "t-fin-6"})
        rows = audit.read_recent()
        frame_commit_rows = [r for r in rows if r.get("source") == "frame_fastpath"]
        assert frame_commit_rows == [], "match disparity must not trigger commit"

    def test_differ_disparity_commits_new_value(self, monkeypatch, _finalize_env):
        """When bearing has an old frame and model emits a different one, commit fires."""
        monkeypatch.setenv("MONOLITH_FRAME_COMMIT_V1", "1")
        store.set_bearing(bs.Bearing(current_frame="old frame"))
        self._call_finalize("<frame>updated frame text</frame>", config={"_turn_id": "t-fin-7"})
        assert store.get_bearing().current_frame == "updated frame text"
