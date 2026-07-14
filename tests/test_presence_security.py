from core.presence import PresenceEngine


def test_presence_hash_ignores_secret_value_changes() -> None:
    engine = PresenceEngine()
    old_data = {
        "modules": [
            {
                "id": "llm",
                "config": {
                    "api_key": "sk-old-123",
                    "model": "gpt-test",
                },
            }
        ]
    }
    new_data = {
        "modules": [
            {
                "id": "llm",
                "config": {
                    "api_key": "sk-new-456",
                    "model": "gpt-test",
                },
            }
        ]
    }

    assert engine.compute_hash(old_data) == engine.compute_hash(new_data)


def test_presence_diff_redacts_secret_values() -> None:
    engine = PresenceEngine()
    old_data = {"config": {"api_key": "old-secret", "model": "a"}}
    new_data = {"config": {"api_key": "new-secret", "model": "b"}}

    diff = engine.compute_diff(old_data, new_data)

    assert "config.api_key" not in diff
    assert diff["config.model"] == {"old": "a", "new": "b"}


def test_presence_diff_redacts_nested_secret_fields() -> None:
    engine = PresenceEngine()
    old_data = {"credentials": {"service_token": "token-1"}}
    new_data = {"credentials": {"service_token": "token-2"}}

    diff = engine.compute_diff(old_data, new_data)

    assert diff == {}


def test_reverse_diff_restores_trimmed_tail_without_null_injection() -> None:
    engine = PresenceEngine()
    data = {"items": ["a"]}
    diff = {
        "items.length": {
            "old": 3,
            "new": 1,
            "old_tail": ["b", "c"],
        }
    }

    engine._apply_reverse_diff(data, diff)

    assert data["items"] == ["a", "b", "c"]
