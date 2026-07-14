from __future__ import annotations

import subprocess
import sys


def test_chat_import_does_not_eagerly_load_bridge():
    # In a FRESH interpreter, importing the registry + world_state must not pull
    # engine.monoline_bridge or any monoline module. (INV-#0: lazy bridge.)
    code = (
        "import sys;"
        "import core.workflow_registry, core.world_state;"
        "leaked=[m for m in sys.modules if m=='engine.monoline_bridge' "
        "or m.startswith('monoline') or m.startswith('core.monoline_')];"
        "print('LEAK' if leaked else 'CLEAN', leaked)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.startswith("CLEAN"), out.stdout
