from core.monosearch import registry, bootstrap as ms_bootstrap


def test_init_registers_the_full_adapter_set():
    registry.clear()
    ms_bootstrap.init_monosearch()
    names = {a.name for a in registry.all_adapters()}
    assert {
        "fault_traces",
        "tools",
        "skills",
        "canonical_log",
        "turn_trace",
        "stage_traces",
        "acatalepsy-acus",
        "continuity",
        "bearing",
        "identity_signals",
        "identity",
        "plan_reminders",
        "investigations",
        "lag_watch",
        "mononote",
        "runtime_health",
        "outcome_traces",
        "acatalepsy-relations",
        "acatalepsy-warrants",
    }.issubset(names)
