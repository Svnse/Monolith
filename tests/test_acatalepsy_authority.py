"""Tests for the Authority branch (B4) — behavioral reach, gated by Kind/Truth.

AU1 stored-only -> AU2 recall-eligible -> AU3 behavior-shaping -> AU4 locked-rule.
Compute-on-read: a pure function of the ACU's resolved branches + provenance +
maturity. This is the real form of the deferred A2 recall ranking.
"""
from core.acatalepsy.authority import (
    compute_authority, AU_STORED, AU_RECALL, AU_BEHAVIOR, AU_LOCKED,
)


def _au(**row):
    return compute_authority(row)


def test_locked_is_au_locked():
    assert _au(locked=1) == AU_LOCKED


def test_inf_falsehood_is_stored_only():
    assert _au(state="-inf", truth="contradicted") == AU_STORED


def test_confirmed_fact_is_behavior_shaping():
    assert _au(state="active", truth="confirmed", l_level="L2") == AU_BEHAVIOR


def test_contradicted_is_stored_only():
    assert _au(state="active", truth="contradicted", l_level="L2") == AU_STORED


def test_contested_is_recall_eligible():
    assert _au(state="active", truth="contested", l_level="L2") == AU_RECALL


def test_reinforced_user_l2_is_behavior_shaping():
    assert _au(state="active", l_level="L2", provenance="user", reinforcement=3) == AU_BEHAVIOR


def test_self_l2_stays_recall_eligible_leash():
    assert _au(state="active", l_level="L2", provenance="self", reinforcement=5) == AU_RECALL


# ── Self-kind authority seal (fix #3) ─────────────────────────────────
# A claim ABOUT Monolith (kind=self) never reaches behavior authority via the
# maturity path, even when user/world-sourced and reinforced — identity goes
# through the projection channel, not AU3. Closes the latent bug where
# compute_authority never read `kind`.

def test_kind_self_user_l2_capped_at_recall_not_behavior():
    assert _au(state="active", l_level="L2", provenance="user",
               reinforcement=3, kind="self") == AU_RECALL


def test_kind_self_world_l2_capped_at_recall_not_behavior():
    assert _au(state="active", l_level="L2", provenance="world",
               reinforcement=9, kind="self") == AU_RECALL


def test_kind_world_fact_user_l2_still_behavior_shaping():
    # The seal is scoped to kind=self; a world-fact at L2 user/reinforced still steers.
    assert _au(state="active", l_level="L2", provenance="user",
               reinforcement=3, kind="world-fact") == AU_BEHAVIOR


def test_kind_self_locked_still_au_locked():
    # Origin-0 (kind=self + locked) is unaffected — locked short-circuits first.
    assert _au(locked=1, kind="self", provenance="user") == AU_LOCKED


def test_l1_stub_is_stored_only():
    assert _au(state="active", l_level="L1", provenance="user") == AU_STORED


# ── staleness (Phase 6): a confirmed fact decays out of behavior-shaping ──
# until it is re-confirmed. It stays recall-eligible (we don't bury it), but a
# fact not re-checked in a long time should not silently steer behavior.

def _iso_days_ago(days):
    from datetime import datetime, timezone, timedelta
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


def test_confirmed_without_check_timestamp_is_behavior_shaping():
    # No truth_checked_at -> not stale (backward-compatible with existing rows).
    assert _au(state="active", truth="confirmed", l_level="L2") == AU_BEHAVIOR


def test_confirmed_recently_checked_is_behavior_shaping():
    assert _au(state="active", truth="confirmed", l_level="L2",
               truth_checked_at=_iso_days_ago(5)) == AU_BEHAVIOR


def test_confirmed_but_stale_drops_to_recall_only():
    assert _au(state="active", truth="confirmed", l_level="L2",
               truth_checked_at=_iso_days_ago(200)) == AU_RECALL


def test_staleness_window_is_tunable():
    # A 30-day-old confirmation is fresh under the default 90d window...
    assert _au(state="active", truth="confirmed", l_level="L2",
               truth_checked_at=_iso_days_ago(30)) == AU_BEHAVIOR
    # ...but stale under a 7-day window.
    assert compute_authority(
        {"state": "active", "truth": "confirmed", "l_level": "L2",
         "truth_checked_at": _iso_days_ago(30)},
        staleness_days=7) == AU_RECALL


def test_confirmed_with_corrupt_checked_at_drops_to_recall_only():
    """When-plane fix: an unparseable truth_checked_at cannot certify freshness,
    so the confirmed fact loses behavior-shaping reach (drops to recall-only)
    until re-confirmed — rather than being silently treated as fresh forever."""
    assert _au(state="active", truth="confirmed", l_level="L2",
               truth_checked_at="not-a-timestamp") == AU_RECALL


def test_confirmed_with_absent_checked_at_stays_behavior_shaping():
    """Absent truth_checked_at remains the intentional 'fresh' case (a fact with
    no recorded recheck time is not demoted for staleness); only corrupt
    timestamps are demoted."""
    assert _au(state="active", truth="confirmed", l_level="L2") == AU_BEHAVIOR
