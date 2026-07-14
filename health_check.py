#!/usr/bin/env python
from __future__ import annotations

import sys

from core.health import get_runtime_health


def main() -> int:
    health = get_runtime_health(probe_endpoint_now="--probe-endpoint" in sys.argv)
    print("=" * 50)
    print("Monolith Runtime Health")
    print("=" * 50)
    for check in health.checks:
        marker = {
            "ok": "[OK]",
            "stale": "[STALE]",
            "warn": "[WARN]",
            "fail": "[FAIL]",
        }.get(check.status, "[?]")
        print(f"{marker} {check.name}: {check.message}")
    print("=" * 50)
    print(f"Status: {health.status.upper()}")
    return 0 if health.status in {"ok", "stale", "warn"} else 1


if __name__ == "__main__":
    sys.exit(main())
