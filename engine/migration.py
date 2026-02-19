"""
Versioned Migration Policy — Phase 5 of Monolith Agent Contract V2.

Provides version constants, migration functions between contract/adapter
format versions, and a compatibility matrix for version interop checks.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Version constants
# ---------------------------------------------------------------------------

CONTRACT_FORMAT_VERSION = "3.0"
ADAPTER_FORMAT_VERSION = "2a.1"


# ---------------------------------------------------------------------------
# Version registries
# ---------------------------------------------------------------------------

CONTRACT_VERSIONS: dict[str, dict[str, Any]] = {
    "2.0": {
        "phase": 3,
        "fields": [
            "contract_id", "contract_hash", "parent_contract_hash",
            "tool_policy", "allowed_tools", "strict_mode",
            "max_inferences", "max_tokens_consumed", "max_format_retries",
            "step_timeout_ms", "total_timeout_ms",
            "context_budget", "tool_output_budget",
            "adapter_version", "model_profile_id", "token_gate",
            "source_page", "creation_timestamp",
        ],
    },
    "2.1": {
        "phase": 3,
        "fields": [
            "contract_id", "contract_hash", "parent_contract_hash",
            "tool_policy", "allowed_tools", "strict_mode",
            "max_inferences", "max_tokens_consumed", "max_format_retries",
            "step_timeout_ms", "total_timeout_ms",
            "context_budget", "tool_output_budget",
            "adapter_version", "model_profile_id", "token_gate",
            "cycle_forbid",
            "source_page", "creation_timestamp",
        ],
    },
    "3.0": {
        "phase": 5,
        "fields": [
            "contract_id", "contract_hash", "parent_contract_hash",
            "tool_policy", "allowed_tools", "strict_mode",
            "max_inferences", "max_tokens_consumed", "max_format_retries",
            "step_timeout_ms", "total_timeout_ms",
            "context_budget", "tool_output_budget",
            "adapter_version", "model_profile_id", "token_gate",
            "cycle_forbid",
            "contract_format_version", "model_fingerprint", "grammar_profile",
            "source_page", "creation_timestamp",
        ],
    },
}

ADAPTER_VERSIONS: dict[str, dict[str, Any]] = {
    "2a.0": {"phase": 2, "grammar_support": False},
    "2a.1": {"phase": 5, "grammar_support": True},
}

# Ordered migration path
_CONTRACT_VERSION_ORDER = ["2.0", "2.1", "3.0"]
_ADAPTER_VERSION_ORDER = ["2a.0", "2a.1"]


# ---------------------------------------------------------------------------
# Compatibility matrix
# ---------------------------------------------------------------------------

COMPATIBILITY_MATRIX: dict[tuple[str, str], bool] = {
    ("3.0", "2a.1"): True,
    ("3.0", "2a.0"): True,   # backward compat, no grammar
    ("2.1", "2a.1"): True,   # forward compat
    ("2.1", "2a.0"): True,
    ("2.0", "2a.0"): True,
    ("2.0", "2a.1"): True,   # forward compat
}


def check_compatibility(
    contract_version: str,
    adapter_version: str,
) -> tuple[bool, str]:
    """
    Check if a contract format version is compatible with an adapter version.

    Returns (compatible, reason).
    """
    key = (contract_version, adapter_version)
    if key in COMPATIBILITY_MATRIX:
        return (True, "")

    # Unknown versions — reject
    if contract_version not in CONTRACT_VERSIONS:
        return (False, f"unknown contract_format_version: {contract_version}")
    if adapter_version not in ADAPTER_VERSIONS:
        return (False, f"unknown adapter_version: {adapter_version}")

    return (False, f"incompatible versions: contract={contract_version} adapter={adapter_version}")


# ---------------------------------------------------------------------------
# Migration functions
# ---------------------------------------------------------------------------

def _migrate_2_0_to_2_1(d: dict[str, Any]) -> dict[str, Any]:
    """Add V2.1 hardening fields with safe defaults."""
    d.setdefault("cycle_forbid", [])
    if "tool_output_budget" not in d:
        d["tool_output_budget"] = {
            "max_bytes_per_call": 32768,
            "truncation_marker": "[TRUNCATED]",
        }
    d.setdefault("token_gate", False)
    return d


def _migrate_2_1_to_3_0(d: dict[str, Any]) -> dict[str, Any]:
    """Add Phase 5 fields with safe defaults."""
    d.setdefault("contract_format_version", "3.0")
    d.setdefault("model_fingerprint", "")
    d.setdefault("grammar_profile", None)
    return d


_CONTRACT_MIGRATION_STEPS: dict[tuple[str, str], Any] = {
    ("2.0", "2.1"): _migrate_2_0_to_2_1,
    ("2.1", "3.0"): _migrate_2_1_to_3_0,
}


def migrate_contract(
    contract_dict: dict[str, Any],
    from_version: str,
    to_version: str,
) -> dict[str, Any]:
    """
    Upgrade a serialized contract from one format version to another.

    Applies migration steps sequentially. Returns a new dict (does not
    mutate the input).
    """
    if from_version == to_version:
        return dict(contract_dict)

    if from_version not in CONTRACT_VERSIONS:
        raise ValueError(f"unknown from_version: {from_version}")
    if to_version not in CONTRACT_VERSIONS:
        raise ValueError(f"unknown to_version: {to_version}")

    from_idx = _CONTRACT_VERSION_ORDER.index(from_version)
    to_idx = _CONTRACT_VERSION_ORDER.index(to_version)

    if to_idx < from_idx:
        raise ValueError(f"downgrade not supported: {from_version} -> {to_version}")

    result = dict(contract_dict)
    for i in range(from_idx, to_idx):
        step_from = _CONTRACT_VERSION_ORDER[i]
        step_to = _CONTRACT_VERSION_ORDER[i + 1]
        migration_fn = _CONTRACT_MIGRATION_STEPS.get((step_from, step_to))
        if migration_fn is None:
            raise ValueError(f"no migration step for {step_from} -> {step_to}")
        result = migration_fn(result)

    return result


def migrate_adapter_config(
    config: dict[str, Any],
    from_version: str,
    to_version: str,
) -> dict[str, Any]:
    """
    Upgrade adapter configuration between versions.

    Currently minimal — adapter configs are largely backward-compatible.
    """
    if from_version == to_version:
        return dict(config)

    if from_version not in ADAPTER_VERSIONS:
        raise ValueError(f"unknown from_version: {from_version}")
    if to_version not in ADAPTER_VERSIONS:
        raise ValueError(f"unknown to_version: {to_version}")

    result = dict(config)

    if from_version == "2a.0" and to_version == "2a.1":
        result.setdefault("grammar_support", True)
        result["adapter_version"] = "2a.1"

    return result
