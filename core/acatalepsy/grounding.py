"""Tavily grounding client for the Truth branch.

The external evidence source for fact-validation: the verifier reads retrieved
evidence; the local model is NOT the truth oracle. Key resolution order:
``TAVILY_API_KEY`` env var, then ``CONFIG_DIR/tavily.json`` (``{"api_key": ...}``).
No key / offline -> ``GroundingUnavailable`` (the verifier records 'unverifiable'
and queues the claim for re-check). Tests inject a ``search_fn``; this is the
production default.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

__all__ = ("Evidence", "GroundingUnavailable", "get_api_key", "tavily_search")

_TAVILY_URL = "https://api.tavily.com/search"


class GroundingUnavailable(RuntimeError):
    """No API key, or the grounding request failed (offline/timeout/HTTP error)."""


@dataclass(frozen=True)
class Evidence:
    url: str
    snippet: str
    score: float = 0.0


def get_api_key() -> str | None:
    """Resolve the Tavily key from env, then the Monolith config store."""
    key = (os.environ.get("TAVILY_API_KEY") or "").strip()
    if key:
        return key
    try:
        from core.paths import CONFIG_DIR
        path = CONFIG_DIR / "tavily.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            k = str(data.get("api_key") or "").strip()
            return k or None
    except Exception:
        pass
    return None


def tavily_search(query: str, *, max_results: int = 5, timeout: float = 10.0) -> list[Evidence]:
    """Query Tavily and return evidence. Raises ``GroundingUnavailable`` when no
    key is configured or the request fails."""
    key = get_api_key()
    if not key:
        raise GroundingUnavailable("no TAVILY_API_KEY (env or CONFIG_DIR/tavily.json)")
    body = json.dumps({
        "api_key": key,
        "query": query,
        "max_results": int(max_results),
        "search_depth": "basic",
    }).encode("utf-8")
    req = urllib.request.Request(
        _TAVILY_URL, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
        raise GroundingUnavailable(f"tavily request failed: {exc}") from exc
    out: list[Evidence] = []
    for r in data.get("results", []) or []:
        out.append(Evidence(
            url=str(r.get("url", "")),
            snippet=str(r.get("content", "")),
            score=float(r.get("score", 0) or 0),
        ))
    return out
