from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any

from engine.loop.contracts import RunContext, ToolSpec
from engine.tools import get_boundary_config

POLICY_ACTION_ALLOW = "allow"
POLICY_ACTION_NEEDS_APPROVAL = "needs_approval"
POLICY_ACTION_DENY = "deny"

POLICY_REASON_UNKNOWN_TOOL = "unknown_tool"
POLICY_REASON_EMPTY_TOOL_NAME = "empty_tool_name"
POLICY_REASON_REQUIRES_APPROVAL_SCOPE = "requires_approval_scope"
POLICY_REASON_EXTERNAL_PATH_APPROVAL = "requires_approval_external_path"
POLICY_REASON_POLICY_SCOPE_CONFLICT = "policy_scope_conflict"
POLICY_REASON_AUTO_APPROVE_SCOPE = "auto_approve_scope"
POLICY_REASON_DEFAULT_ALLOW_SCOPE = "default_allow_scope"


@dataclass(frozen=True)
class PolicyDecision:
    action: str
    reason_code: str
    tool: str
    scope: str


class PolicyKernel:
    """
    Minimal tool-admission policy kernel for LoopRuntime.

    This centralizes the pre-execution admission decision so the runtime can
    emit stable policy events and later harden logic without scattering checks.
    """

    __slots__ = ()

    def decide(
        self,
        *,
        ctx: RunContext,
        tool_name: str,
        args: dict[str, Any],
        tool_index: dict[str, ToolSpec],
    ) -> PolicyDecision:
        normalized_name = str(tool_name or "").strip()
        if not normalized_name:
            return PolicyDecision(
                action=POLICY_ACTION_DENY,
                reason_code=POLICY_REASON_EMPTY_TOOL_NAME,
                tool="",
                scope="unknown",
            )

        spec = tool_index.get(normalized_name)
        if spec is None:
            return PolicyDecision(
                action=POLICY_ACTION_DENY,
                reason_code=POLICY_REASON_UNKNOWN_TOOL,
                tool=normalized_name,
                scope="unknown",
            )

        scope = str(getattr(spec, "scope", "") or "")
        in_require = scope in ctx.policy.require_approval
        in_auto = scope in ctx.policy.auto_approve

        if _requires_external_path_approval(args):
            return PolicyDecision(
                action=POLICY_ACTION_NEEDS_APPROVAL,
                reason_code=POLICY_REASON_EXTERNAL_PATH_APPROVAL,
                tool=normalized_name,
                scope=scope,
            )

        if in_require and in_auto:
            return PolicyDecision(
                action=POLICY_ACTION_NEEDS_APPROVAL,
                reason_code=POLICY_REASON_POLICY_SCOPE_CONFLICT,
                tool=normalized_name,
                scope=scope,
            )
        if in_require:
            return PolicyDecision(
                action=POLICY_ACTION_NEEDS_APPROVAL,
                reason_code=POLICY_REASON_REQUIRES_APPROVAL_SCOPE,
                tool=normalized_name,
                scope=scope,
            )
        if in_auto:
            return PolicyDecision(
                action=POLICY_ACTION_ALLOW,
                reason_code=POLICY_REASON_AUTO_APPROVE_SCOPE,
                tool=normalized_name,
                scope=scope,
            )

        return PolicyDecision(
            action=POLICY_ACTION_ALLOW,
            reason_code=POLICY_REASON_DEFAULT_ALLOW_SCOPE,
            tool=normalized_name,
            scope=scope,
        )


def _requires_external_path_approval(args: dict[str, Any]) -> bool:
    if not isinstance(args, dict):
        return False
    try:
        cfg = get_boundary_config()
        ws_root = Path(str(cfg.get("workspace_root") or "")).expanduser().resolve()
    except Exception:
        return False
    if not str(ws_root):
        return False

    for key in ("path", "src", "dst", "file"):
        value = args.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        raw = value.strip()
        try:
            p = Path(raw).expanduser()
            target = p.resolve() if p.is_absolute() else (ws_root / p).resolve()
            if not _is_within_root(target, ws_root):
                return True
        except Exception:
            continue
    return False


def _is_within_root(path: Path, root: Path) -> bool:
    p = str(path.resolve())
    r = str(root.resolve())
    return p == r or p.startswith(r + os.sep)
