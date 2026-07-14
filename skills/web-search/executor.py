"""Tavily-backed live web search for Monolith.

Configuration follows the existing grounding convention:
``TAVILY_API_KEY`` env var first, then ``config/tavily.json`` with
``{"api_key": "..."}``.
"""
from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from typing import Any

from core.acatalepsy.grounding import get_api_key


_TAVILY_SEARCH_URL = "https://api.tavily.com/search"
_DEFAULT_TIMEOUT = 20
_VALID_DEPTHS = {"ultra-fast", "fast", "basic", "advanced"}
_VALID_TOPICS = {"general", "news", "finance"}
_VALID_TIME_RANGES = {"day", "week", "month", "year", "d", "w", "m", "y"}


def _coerce_int(value: object, default: int, lo: int, hi: int) -> int:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        parsed = default
    return max(lo, min(hi, parsed))


def _coerce_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _coerce_enum(value: object, valid: set[str], default: str) -> str:
    text = str(value or "").strip().lower()
    return text if text in valid else default


def _coerce_answer(value: object) -> bool | str:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"basic", "advanced"}:
        return text
    return _coerce_bool(value, False)


def _coerce_raw_content(value: object) -> bool | str:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"markdown", "text"}:
        return text
    return _coerce_bool(value, False)


def _coerce_string_list(value: object, *, cap: int) -> list[str] | None:
    if value is None:
        return None
    items: list[object]
    if isinstance(value, list):
        items = value
    else:
        items = str(value).split(",")
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip().lower()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= cap:
            break
    return out or None


def _clean_line(value: object, *, limit: int = 500) -> str:
    text = " ".join(str(value or "").split())
    if len(text) > limit:
        return text[: limit - 3].rstrip() + "..."
    return text


def _format_result(index: int, item: dict[str, Any], *, include_raw: bool) -> list[str]:
    title = _clean_line(item.get("title") or "(untitled)", limit=180)
    url = _clean_line(item.get("url") or "", limit=240)
    content = _clean_line(item.get("content") or "", limit=650)
    try:
        score = float(item.get("score", 0) or 0)
    except (TypeError, ValueError):
        score = 0.0
    lines = [f"{index}. score={score:.3f} | {title}"]
    if url:
        lines.append(f"   url: {url}")
    if content:
        lines.append(f"   snippet: {content}")
    if include_raw:
        raw = _clean_line(item.get("raw_content") or "", limit=1200)
        if raw:
            lines.append(f"   raw: {raw}")
    return lines


def _request_tavily(payload: dict[str, Any], *, api_key: str, timeout: int) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        _TAVILY_SEARCH_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "Monolith/0.1 (web_search tool)",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read(2_000_000)
    data = json.loads(raw.decode("utf-8", errors="replace"))
    return data if isinstance(data, dict) else {}


def run(cmd: dict, _ctx) -> str:
    query = str(cmd.get("query") or "").strip()
    if not query:
        return "[web_search: query required]"

    key = get_api_key()
    if not key:
        return "[web_search: Tavily key not configured - set TAVILY_API_KEY or config/tavily.json]"

    max_results = _coerce_int(cmd.get("max_results", cmd.get("limit", 5)), 5, 1, 20)
    timeout = _coerce_int(cmd.get("timeout", _DEFAULT_TIMEOUT), _DEFAULT_TIMEOUT, 1, 60)
    max_chars = _coerce_int(cmd.get("max_chars", 8000), 8000, 1000, 50000)
    depth = _coerce_enum(cmd.get("search_depth", "basic"), _VALID_DEPTHS, "basic")
    topic = _coerce_enum(cmd.get("topic", "general"), _VALID_TOPICS, "general")
    include_answer = _coerce_answer(cmd.get("include_answer", False))
    include_raw_content = _coerce_raw_content(cmd.get("include_raw_content", False))
    include_images = _coerce_bool(cmd.get("include_images", False))
    include_favicon = _coerce_bool(cmd.get("include_favicon", False))
    include_usage = _coerce_bool(cmd.get("include_usage", False))

    payload: dict[str, Any] = {
        "query": query,
        "search_depth": depth,
        "topic": topic,
        "max_results": max_results,
        "include_answer": include_answer,
        "include_raw_content": include_raw_content,
        "include_images": include_images,
        "include_favicon": include_favicon,
        "include_usage": include_usage,
    }
    include_domains = _coerce_string_list(cmd.get("include_domains"), cap=300)
    exclude_domains = _coerce_string_list(cmd.get("exclude_domains"), cap=150)
    if include_domains:
        payload["include_domains"] = include_domains
    if exclude_domains:
        payload["exclude_domains"] = exclude_domains
    time_range = str(cmd.get("time_range") or "").strip().lower()
    if time_range:
        if time_range not in _VALID_TIME_RANGES:
            return "[web_search: invalid time_range - use day/week/month/year or d/w/m/y]"
        payload["time_range"] = time_range
    for field in ("start_date", "end_date", "country"):
        value = str(cmd.get(field) or "").strip()
        if value:
            payload[field] = value
    if "safe_search" in cmd:
        payload["safe_search"] = _coerce_bool(cmd.get("safe_search"), False)
    if "auto_parameters" in cmd:
        payload["auto_parameters"] = _coerce_bool(cmd.get("auto_parameters"), False)
    if "exact_match" in cmd:
        payload["exact_match"] = _coerce_bool(cmd.get("exact_match"), False)

    try:
        data = _request_tavily(payload, api_key=key, timeout=timeout)
    except urllib.error.HTTPError as exc:
        return f"[web_search: HTTP {exc.code} {exc.reason}]"
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        return f"[web_search: connection error - {reason}]"
    except socket.timeout:
        return f"[web_search: timed out after {timeout}s]"
    except (OSError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
        return f"[web_search: error - {exc}]"

    results = data.get("results") or []
    if isinstance(results, list):
        results = [item for item in results if isinstance(item, dict)]
        results.sort(key=lambda item: float(item.get("score", 0) or 0), reverse=True)
    else:
        results = []

    response_time = data.get("response_time", "")
    header = (
        f"[web_search: tavily query={query!r} results={len(results)} "
        f"depth={depth} topic={topic}"
    )
    if response_time:
        header += f" response_time={response_time}"
    usage = data.get("usage")
    if isinstance(usage, dict) and usage.get("credits") is not None:
        header += f" credits={usage.get('credits')}"
    header += "]"

    lines = [header]
    answer = _clean_line(data.get("answer") or "", limit=1000)
    if answer:
        lines.append(f"answer: {answer}")
    for index, result in enumerate(results[:max_results], start=1):
        lines.extend(_format_result(index, result, include_raw=bool(include_raw_content)))
    if not results:
        lines.append("no ranked results")

    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + f"\n... [web_search truncated at {max_chars} chars]"
    return text
