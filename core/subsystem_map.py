"""Subsystem map — registry walker that dumps a structured snapshot of
load-bearing Monolith subsystems to disk at bootstrap.

Junior queries this via the scratchpad `introspect` op. Closes the
spatial-addressing gap noted in the 2026-05-20 audit: the substrate has
grown past what the system can enumerate from inside without grep.

Not a registry. Not a source of truth. A snapshot derived from real
registries. Regenerated at every bootstrap. If the registries diverge
from the code, that's a registry-validation problem (caught by
pipeline_registry.validate_against_filesystem); not a problem this
module needs to solve.

Scope caveat: the map reflects WHAT REGISTERED THIS RUN, not WHAT
COULD REGISTER. Subsystems that register conditionally (e.g.
lag_watch_interceptor inside a try/except in bootstrap.py) only show
up here if their registration actually fired. The map is descriptive
of the live process, not exhaustive of code-on-disk.
"""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.paths import CONFIG_DIR

_MAP_PATH = CONFIG_DIR / "subsystem_map.json"
_SCHEMA_VERSION = 1


def _git_sha() -> str:
    """Best-effort git HEAD sha; 'unknown' if not in a repo or git missing."""
    try:
        repo_root = Path(__file__).resolve().parent.parent
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return "unknown"


def _walk_policies() -> list[dict[str, Any]]:
    try:
        from core.pipeline_registry import iter_policies
        rows = []
        for p in iter_policies():
            tier = p.authority_tier
            tier_val = tier.value if hasattr(tier, "value") else str(tier)
            rows.append({
                "name": p.name,
                "module_path": p.module_path,
                "subscribes_to": list(p.subscribes_to),
                "depends_on": list(p.depends_on),
                "authority_tier": tier_val,
                "kill_switch_env_flag": p.kill_switch_env_flag,
                "retry_budget": p.retry_budget,
            })
        return rows
    except Exception:
        return []


def _walk_planes() -> list[dict[str, Any]]:
    rows = []
    for plane_name in ("effort", "conversation", "reasoning", "linguency"):
        try:
            mod = __import__(f"core.{plane_name}", fromlist=["_loader"])
            loader = getattr(mod, "_loader", None)
            if loader is None:
                continue
            cfg = loader.config
            rows.append({
                "name": cfg.plane_name,
                "valid_modes": sorted(cfg.valid_modes),
                "silent_modes": sorted(cfg.silent_modes),
                "default_mode": cfg.default_mode,
                "scaffolds_dir": str(cfg.scaffolds_dir),
                "flag_env": cfg.flag_env,
                "key_suffix": cfg.key_suffix,
                "classifier_attr": getattr(cfg, "classifier_attr", None),
                "module_path": f"core.{plane_name}",
            })
        except Exception:
            continue
    return rows


def _walk_skills() -> list[dict[str, Any]]:
    try:
        from core.skill_registry import list_tools
        rows = []
        for tool in list_tools():
            # ToolSpec is a dataclass — access via attributes, not .get().
            name = getattr(tool, "name", "") or ""
            desc = (getattr(tool, "description", "") or "").strip()
            if len(desc) > 200:
                desc = desc[:199].rstrip() + "…"
            params = getattr(tool, "params", ()) or ()
            param_names = [getattr(p, "name", str(p)) for p in params]
            rows.append({
                "name": name,
                "description": desc,
                "param_names": param_names,
            })
        return sorted(rows, key=lambda r: r["name"])
    except Exception:
        return []


def _walk_interceptors() -> list[dict[str, Any]]:
    try:
        from core.message_interceptors import iter_interceptors
        rows = []
        for i, fn in enumerate(iter_interceptors()):
            rows.append({
                "order": i,
                "name": getattr(fn, "__name__", "?"),
                "module": getattr(fn, "__module__", "?"),
            })
        return rows
    except Exception:
        return []


def build_subsystem_map() -> dict[str, Any]:
    """Build the snapshot dict from live registries. Pure function."""
    return {
        "schema_version": _SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "pipeline_policies": _walk_policies(),
        "planes": _walk_planes(),
        "skills": _walk_skills(),
        "interceptors": _walk_interceptors(),
    }


def dump_subsystem_map(path: Path | None = None) -> Path:
    """Build and atomically persist the subsystem map JSON."""
    target = path or _MAP_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    data = build_subsystem_map()
    tmp = target.with_name(target.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, target)
    return target


def read_subsystem_map(path: Path | None = None) -> dict[str, Any] | None:
    """Read the persisted map. Returns None if missing/corrupt."""
    target = path or _MAP_PATH
    if not target.exists():
        return None
    try:
        with target.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        return data
    except Exception:
        return None


def format_subsystem_map(
    data: dict[str, Any] | None,
    *,
    kind: str | None = None,
    name_filter: str | None = None,
) -> str:
    """Format the map for human/LLM consumption.

    kind: optional filter — "policies", "planes", "skills", "interceptors",
          or None for a summary across all kinds.
    name_filter: optional substring; matches against entry name (case-insensitive).
    """
    if not data:
        return (
            "[introspect: subsystem_map.json not present. "
            "Bootstrap must run dump_subsystem_map() first.]"
        )
    header = (
        f"[introspect: schema_v{data.get('schema_version', '?')} "
        f"generated_at={data.get('generated_at', '?')} "
        f"git_sha={data.get('git_sha', '?')}]"
    )
    sections: dict[str, list[dict[str, Any]]] = {
        "policies": list(data.get("pipeline_policies") or []),
        "planes": list(data.get("planes") or []),
        "skills": list(data.get("skills") or []),
        "interceptors": list(data.get("interceptors") or []),
    }

    def _matches_name(entry: dict[str, Any]) -> bool:
        if not name_filter:
            return True
        nf = name_filter.lower()
        return nf in str(entry.get("name", "")).lower()

    selected_kinds = [kind] if kind else list(sections.keys())
    lines = [header]
    for k in selected_kinds:
        rows = sections.get(k) or []
        rows = [r for r in rows if _matches_name(r)]
        lines.append(f"\n== {k} ({len(rows)}) ==")
        if not rows:
            lines.append("(none)")
            continue
        for r in rows:
            lines.append(_format_row(k, r))
    return "\n".join(lines)


def _format_row(kind: str, r: dict[str, Any]) -> str:
    if kind == "policies":
        return (
            f"- {r.get('name')} [{r.get('authority_tier')}] "
            f"subs={list(r.get('subscribes_to') or [])} "
            f"deps={list(r.get('depends_on') or [])} "
            f"flag={r.get('kill_switch_env_flag') or '-'}"
        )
    if kind == "planes":
        return (
            f"- {r.get('name')} default={r.get('default_mode')} "
            f"valid={list(r.get('valid_modes') or [])} "
            f"silent={list(r.get('silent_modes') or [])} "
            f"flag={r.get('flag_env')}"
        )
    if kind == "skills":
        params = r.get("param_names") or []
        return f"- {r.get('name')} ({', '.join(params)}): {r.get('description', '')}"
    if kind == "interceptors":
        return f"- [{r.get('order')}] {r.get('name')} ({r.get('module')})"
    return f"- {r}"
