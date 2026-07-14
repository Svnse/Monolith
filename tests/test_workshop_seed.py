from __future__ import annotations

from core.workshop_seed import seed_workshop_flows, BUNDLED_SEEDS_DIR


def test_seeds_into_empty_dir(tmp_path):
    n = seed_workshop_flows(worlds_dir=tmp_path)
    assert n >= 1
    copied = list(tmp_path.glob("*.monoline"))
    assert copied, "expected at least one seed copied"
    # bundled source actually exists (guards against an empty assets dir)
    assert any(BUNDLED_SEEDS_DIR.glob("*.monoline"))


def test_noop_when_dir_already_has_flows(tmp_path):
    (tmp_path / "mine.monoline").write_text("{}", encoding="utf-8")
    n = seed_workshop_flows(worlds_dir=tmp_path)
    assert n == 0  # never touch a non-empty worlds dir
    assert (tmp_path / "mine.monoline").read_text(encoding="utf-8") == "{}"  # untouched


def test_creates_missing_dir(tmp_path):
    target = tmp_path / "nested" / "worlds"
    n = seed_workshop_flows(worlds_dir=target)
    assert target.exists() and n >= 1


def test_seeder_does_not_import_monoline():
    # INV-#0: importing the seeder must not pull any Monoline module. Checked in a FRESH
    # interpreter -- the in-process sys.modules is shared across the session and is legitimately
    # populated with core.monoline_* once any bridge test runs load_monoline() (it keeps them
    # resident by design), so an ambient scan would be order-dependent.
    import subprocess
    import sys
    code = (
        "import sys;"
        "import core.workshop_seed;"
        "leaked=[m for m in sys.modules if m.startswith('monoline') "
        "or m.startswith('core.monoline_')];"
        "print('LEAK' if leaked else 'CLEAN', leaked)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.startswith("CLEAN"), out.stdout
