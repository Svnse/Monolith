#!/usr/bin/env python3
"""Run the Acatalepsy auditor with the configured cloud LLM (DeepSeek).

Reads api_base / api_key / api_model from Monolith's LLMConfig snapshot,
calls the OpenAI-compatible chat completions endpoint, passes the response
to auditor.run_audit() for atomicity gating + candidate insertion.

Usage:
    python scripts/run_auditor_with_deepseek.py [--source auditor_claude]

Idempotent: re-running picks up from the last cursor advance. If no new
events have arrived since the last completed run, status is "empty_slice"
and no candidates are inserted.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.acatalepsy import auditor  # noqa: E402
from core.llm_config import load_config  # noqa: E402


DEFAULT_SOURCE = "auditor_claude"
HTTP_TIMEOUT_SECS = 300.0
# DeepSeek-v4-pro is a reasoning model — reasoning_tokens and content tokens
# share this budget. 24K leaves enough room for thorough analysis + the final
# JSON candidates payload. Empirically 4K was 100% consumed by reasoning,
# producing zero visible content (finish_reason='length').
MAX_OUTPUT_TOKENS = 24000


def deepseek_llm(*, system_prompt: str, user_content: str) -> str:
    """OpenAI-compatible chat-completions call via the configured cloud."""
    cfg = load_config()
    base = (cfg.get("api_base") or "").rstrip("/")
    key = cfg.get("api_key") or ""
    model = cfg.get("api_model") or ""
    if not (base and key and model):
        missing = [k for k, v in (("api_base", base), ("api_key", key), ("api_model", model)) if not v]
        raise RuntimeError(f"Missing API config fields: {missing}")

    url = f"{base}/v1/chat/completions"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.2,
        "max_tokens": MAX_OUTPUT_TOKENS,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SECS) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="ignore")[:500]
        raise RuntimeError(f"HTTPError {exc.code} from {url}: {body_text}") from exc

    # Dump full response for post-mortem — debugging empty/refused responses.
    art = _REPO / "artifacts"
    art.mkdir(exist_ok=True)
    (art / "auditor_raw_response.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError(f"No choices in response: {str(payload)[:300]}")
    content = choices[0].get("message", {}).get("content", "")
    finish = choices[0].get("finish_reason")
    print(f"  LLM finish_reason={finish!r}, content_len={len(content or '')}")
    return content or ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help=f"Auditor source label (default: {DEFAULT_SOURCE!r})",
    )
    args = parser.parse_args()

    print(f"Calling cloud LLM for audit run (source={args.source!r})...")
    result = auditor.run_audit(deepseek_llm, source=args.source)
    print()
    print("=== AUDIT RUN RESULT ===")
    print(f"  run_id:                {result.run_id}")
    print(f"  status:                {result.status}")
    print(f"  events_processed:      {result.events_processed}")
    print(f"  proposals_returned:    {result.proposals_returned}")
    print(f"  candidates_inserted:   {result.candidates_inserted}")
    print(f"  candidates_rejected:   {result.candidates_rejected}")
    if result.error:
        print(f"  error:                 {result.error}")
    if result.rejection_reasons:
        print("  rejection_reasons:")
        for r in result.rejection_reasons:
            print(f"    - {r}")
    return 0 if result.status in ("success", "empty_slice") else 1


if __name__ == "__main__":
    raise SystemExit(main())
