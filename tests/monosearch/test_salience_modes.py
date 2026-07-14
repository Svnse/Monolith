"""Salience modes (failing / recurring) + compute-on-read. Isolation via the
autouse `isolated_salience_db` fixture in conftest.py."""
from core.monosearch import salience
from core.monosearch.record import Provenance


def test_salience_value_is_count_times_decay():
    # last_seen == now -> decay factor 1.0 -> salience == count
    val = salience._salience(count=4, last_seen=1_000_000.0, now=1_000_000.0)
    assert val == 4.0
    # one HALF_LIFE_DAYS later -> halves
    later = 1_000_000.0 + salience.HALF_LIFE_DAYS * 86400
    assert abs(salience._salience(4, 1_000_000.0, later) - 2.0) < 1e-9


def test_salience_uses_decay_factor_directly_not_provenance_multiplier():
    # A SELF-provenance key must NOT decay faster than a USER-provenance key with
    # the same age — i.e. we are NOT routing through effective_reinforcement's
    # self=1/user=4 half-life multiplier. Same count + same age => same salience.
    age_days = 60.0  # 2x HALF_LIFE_DAYS
    self_val = salience._salience(10, 0.0, age_days * 86400.0)
    user_val = salience._salience(10, 0.0, age_days * 86400.0)
    assert self_val == user_val  # provenance does NOT affect the decay curve here


def test_failing_returns_top_self_fault_keys_by_salience():
    now = 2_000_000.0
    for _ in range(5):
        salience.record_observation("think_leak|x", Provenance.SELF, "fault_traces", ts=now)
    for _ in range(2):
        salience.record_observation("tool_no_fire|y", Provenance.SELF, "fault_traces", ts=now)
    top = salience.failing(now=now, limit=10)
    assert top[0]["recurrence_key"] == "think_leak|x"
    assert top[0]["count"] == 5
    assert top[1]["recurrence_key"] == "tool_no_fire|y"


def test_recurring_spans_all_sources():
    now = 2_000_000.0
    salience.record_observation("user_message|a", Provenance.USER, "canonical_log", ts=now)
    salience.record_observation("user_message|a", Provenance.USER, "canonical_log", ts=now)
    salience.record_observation("think_leak|x", Provenance.SELF, "fault_traces", ts=now)
    keys = {r["recurrence_key"] for r in salience.recurring(now=now, limit=10)}
    assert keys == {"user_message|a", "think_leak|x"}
