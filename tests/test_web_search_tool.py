from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from core.cmd_parser import process_response
from core.skill_registry import clear_skill_cache, get_tool


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _limit: int = -1) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_web_search_tool_is_discoverable() -> None:
    clear_skill_cache()
    spec = get_tool("web_search")
    alias = get_tool("search_web")

    assert spec is not None
    assert alias is spec
    assert spec.json_schema is not None
    assert spec.json_schema.get("required") == ["query"]
    param_names = {param.name for param in spec.params}
    assert {"query", "start_date", "end_date", "include_usage", "max_chars", "timeout"}.issubset(param_names)


def test_web_search_requires_tavily_key(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    from core import paths
    monkeypatch.setattr(paths, "CONFIG_DIR", tmp_path, raising=False)

    _clean, result, _artifacts = process_response(
        '<tool_call>{"tool":"web_search","query":"monolith"}</tool_call>',
        archive_dir=tmp_path,
    )

    assert result is not None
    assert "Tavily key not configured" in result


def test_web_search_calls_tavily_and_sorts_ranked_results(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    captured: dict[str, object] = {}

    def _fake_urlopen(req, timeout=0):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["authorization"] = req.get_header("Authorization")
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse(
            {
                "query": "ai audio",
                "answer": "Short answer.",
                "response_time": "0.42",
                "usage": {"credits": 1},
                "results": [
                    {
                        "title": "Lower",
                        "url": "https://example.com/lower",
                        "content": "less relevant",
                        "score": 0.2,
                    },
                    {
                        "title": "Higher",
                        "url": "https://example.com/higher",
                        "content": "more relevant",
                        "score": 0.9,
                    },
                ],
            }
        )

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    _clean, result, artifacts = process_response(
        (
            '<tool_call>{"tool":"web_search","query":"ai audio","max_results":2,'
            '"search_depth":"basic","include_answer":true,"include_usage":true}</tool_call>'
        ),
        archive_dir=tmp_path,
    )

    assert result is not None
    assert "[web_search: tavily" in result
    assert "answer: Short answer." in result
    assert result.index("Higher") < result.index("Lower")
    assert captured["url"] == "https://api.tavily.com/search"
    assert captured["authorization"] == "Bearer tvly-test"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["query"] == "ai audio"
    assert payload["max_results"] == 2
    assert payload["include_answer"] is True
    envelope = artifacts[1].get("envelope", {})
    data = envelope.get("data", {})
    assert data.get("url") == "https://example.com/higher"
    assert data.get("urls") == ["https://example.com/higher", "https://example.com/lower"]


def test_web_search_validation_rejects_missing_query(tmp_path: Path) -> None:
    _clean, result, artifacts = process_response(
        '<tool_call>{"tool":"web_search","max_results":3}</tool_call>',
        archive_dir=tmp_path,
    )

    assert result is not None
    assert "invalid arguments for 'web_search'" in result
    assert "missing required field 'query'" in result
    assert artifacts[1].get("envelope", {}).get("ok") is False
