# Capability Authorization Audit

This document summarizes capability request parsing, issuance, and tool authorization behavior in the current runtime implementation.

- Capability storage and authorization logic: `engine/capabilities.py` (`CapabilityManager`).
- Capability request parsing and issuance events: `engine/agent_runtime.py` (`_parse_capability_request`, `_handle_capability_request`).
- Tool-time authorization gate: `engine/agent_bridge.py` (`_validate`).
- UI approval payload emission: `ui/pages/code.py` (`_CapabilityApprovalWidget._approve`, `_deny`).

Notable behavior:

1. Capability scopes are enum-based (`CapabilityScope`) and comparisons are strict enum equality.
2. `_parse_capability_request` accepts JSON object payloads with canonical keys (`scope`, `path_pattern`, `ttl_seconds`, `constraints`, `reason`), or defaults to READ on non-JSON content.
3. UI approval currently emits `path_pattern: "**"` and omits `scope`, so approved decisions can broaden path constraints while preserving requested scope by fallback.
4. Authorization is bound to `branch_id`, `tool`, and extracted `path`; there is no task/engine key binding in this runtime code path.
