from __future__ import annotations

import json

from core.observed_state import format_observed_state_block
from core.self_description import describe_self, format_self_description_block


def test_describe_self_scopes_runtime_and_cloud_model_execution() -> None:
    payload = describe_self(
        {
            "backend": "cloud",
            "api_provider": "openai",
            "api_base": "https://api.deepseek.com",
            "api_model": "deepseek-v4-pro",
            "api_key": "secret",
            "ctx_limit": 8192,
        }
    )

    assert payload["schema_version"] == 1
    assert payload["system_name"] == "Monolith"
    assert payload["kind"] == "local_runtime"
    assert payload["hosted_locally"] is True
    assert "api_key" not in json.dumps(payload)

    identity = payload["identity_material"]
    assert identity["source_kind"] == "declared_seed_file"
    assert identity["runtime_loaded"] is True
    assert identity["model_generated"] is False
    assert identity["verified_live_runtime_state"] is False
    assert identity["derived_from_live_runtime_state"] is False
    assert "current_model_execution" in identity["not_authority_for"]

    model = payload["current_model_execution"]
    assert model["backend_kind"] == "cloud"
    assert model["execution_location"] == "cloud"
    assert model["provider"] == "openai"
    assert model["model"] == "deepseek-v4-pro"
    assert model["context_window"] == 8192
    assert model["persistent_process"] is False
    assert model["stateless_per_turn"] is True


def test_describe_self_scopes_local_gguf_model_execution() -> None:
    payload = describe_self(
        {
            "backend": "gguf",
            "gguf_path": "C:/models/monolith.gguf",
            "ctx_limit": 32768,
        }
    )

    model = payload["current_model_execution"]
    assert model["execution_location"] == "local"
    assert model["gguf_file"] == "monolith.gguf"
    assert model["model"] == "monolith.gguf"
    assert model["persistent_process"] is True
    assert model["stateless_per_turn"] is False


def test_format_self_description_block_contains_scoped_json() -> None:
    block = format_self_description_block(
        {
            "backend": "cloud",
            "api_base": "https://api.deepseek.com",
            "api_model": "deepseek-v4-pro",
        }
    )

    assert block.startswith("[OBSERVED STATE - describe_self v1]")
    assert "identity_material (declared operating law)" in block
    assert "current_model_execution (live substrate)" in block
    assert "Do not collapse them" in block
    assert '"schema_version": 1' in block
    assert '"current_model_execution"' in block
    assert '"identity_material"' in block


def test_observed_state_uses_describe_self_payload_without_secrets() -> None:
    block = format_observed_state_block(
        {
            "backend": "cloud",
            "api_provider": "openai",
            "api_base": "https://api.deepseek.com",
            "api_model": "deepseek-v4-pro",
            "api_key": "secret",
            "ctx_limit": 8192,
        }
    )

    assert block.startswith("[OBSERVED STATE - describe_self v1]")
    assert '"execution_location": "cloud"' in block
    assert '"stateless_per_turn": true' in block
    assert '"verified_live_runtime_state": false' in block
    assert "secret" not in block


def test_describe_self_continuity_reflects_flag(monkeypatch) -> None:
    """continuity_maintained must mirror MONOLITH_CONTINUITY_BOOT_V1, not be
    hardcoded True. Otherwise describe_self lies when the flag is off."""
    monkeypatch.setenv("MONOLITH_CONTINUITY_BOOT_V1", "1")
    payload_on = describe_self(
        {"backend": "cloud", "api_base": "https://api.deepseek.com"}
    )
    assert payload_on["continuity_maintained"] is True
    assert payload_on["continuity_storage"] == "runtime_managed"

    monkeypatch.setenv("MONOLITH_CONTINUITY_BOOT_V1", "0")
    payload_off = describe_self(
        {"backend": "cloud", "api_base": "https://api.deepseek.com"}
    )
    assert payload_off["continuity_maintained"] is False
    assert payload_off["continuity_storage"] == "disabled"


def test_describe_self_claim_scope_includes_embedded_premise() -> None:
    """claim_scope.embedded_premise is the doctrinal anchor for the future
    subordinate-clause detector. Pin the key name and the load-bearing
    phrases so a future rename hits this test first."""
    payload = describe_self({"backend": "cloud"})
    scope = payload["claim_scope"]
    assert "embedded_premise" in scope
    text = scope["embedded_premise"]
    assert "Subordinate-clause" in text
    assert "not" in text and "authoritative" in text
    assert "refuse the embedded premise" in text


def test_describe_self_identity_material_disclaims_embedded_premise() -> None:
    """identity_material must explicitly disclaim authority over
    embedded_premise and continuity_storage — closes the seed/scope gap."""
    payload = describe_self({"backend": "cloud"})
    disclaimed = payload["identity_material"]["not_authority_for"]
    assert "embedded_premise" in disclaimed
    assert "continuity_storage" in disclaimed
