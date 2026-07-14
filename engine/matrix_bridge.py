"""engine/matrix_bridge.py — Matrix ↔ Monolith chat bridge (sidecar).

Run from the repo root:

    python engine/matrix_bridge.py

Required env vars:
    MATRIX_HOMESERVER       e.g. https://brainbrigade.xyz
    MATRIX_USER_ID          e.g. @monolith:brainbrigade.xyz
    MATRIX_PASSWORD         (first run only; cached afterward)

Optional env vars:
    MATRIX_DISPLAY_NAME       default "Monolith"
    MATRIX_MENTION_TRIGGER    default "@monolith" (case-insensitive)
    MATRIX_CREDENTIALS_PATH   default ~/.config/monolith/matrix_credentials.json
    MONOLITH_AGENT_URL        default http://localhost:7821
    MONOLITH_AGENT_TOKEN      reused if set (Bearer auth on HTTP calls)

Lifecycle:
    1. Wait for Monolith's /health to return 200 (CONNECT must be started).
    2. POST /join to register as a CONNECT peer named "Matrix".
    3. matrix-nio login (uses cached access_token if available).
    4. Sync forever; on @mention in a joined room, stream the reply back
       to the room via m.replace edits.
    5. SIGINT/SIGTERM → POST /leave → close client → exit cleanly.

Design spec: docs/superpowers/specs/2026-05-23-matrix-bridge-design.md
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Windows: Python's default console encoding (cp1252) can't print Unicode
# emoji/arrows. Reconfigure stdout/stderr so logging works on Windows
# without affecting UTF-8 platforms.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass

# Make repo-root imports work when running `python engine/matrix_bridge.py`
# from the repo root. Python normally only puts the script's own directory
# (engine/) on sys.path; we need core/* to be importable too.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import aiohttp
except ImportError:
    sys.stderr.write(
        "aiohttp is not installed. It is a transitive dependency of matrix-nio.\n"
        "Run: pip install -r docs/dependencies/requirements.txt\n"
    )
    sys.exit(2)

# `markdown` package is used to convert the model's markdown output into
# HTML for Matrix's `formatted_body` field. Without it, Element shows raw
# `*italic*` characters. The `nl2br` extension turns single newlines into
# `<br>` which matches how chat normally renders. `fenced_code` handles
# ```python style blocks; `tables` handles | pipe | tables.
try:
    import markdown as _markdown
except ImportError:
    sys.stderr.write(
        "python-markdown not installed. pip install markdown>=3.4\n"
    )
    sys.exit(2)

from html import escape as _html_escape

try:
    from nio import (
        AsyncClient,
        InviteMemberEvent,
        LoginResponse,
        MatrixRoom,
        RoomMessageText,
    )
except ImportError:
    sys.stderr.write(
        "matrix-nio is not installed.\n"
        "Run: pip install -r docs/dependencies/requirements.txt\n"
    )
    sys.exit(2)

from core.internal_tags import EXTERNAL_STRIP_TAGS, strip_tag_blocks


# ── Tunables ────────────────────────────────────────────────────────────────

EDIT_DEBOUNCE_MS = 3500          # matches brainbrigade.xyz's ~4s rate-limit window
EDIT_DEBOUNCE_CHARS = 800        # large chunks force a flush before the timer fires
HEALTH_POLL_INITIAL_SEC = 2.0
HEALTH_POLL_MAX_SEC = 60.0
CURSOR_CHAR = "▍"
PLACEHOLDER_BODY = "💭 thinking..."
PEER_NAME = "Matrix"

# Tags the model may emit; matches engine/agent_server.py:_THINK_BLOCK_RE.
# Used by the bridge as defense-in-depth: if the server's lane-splitter
# misses a tag, we extract the contents client-side so they render as a
# styled markdown block instead of raw "<think>" leaking into the room.
import re as _re
_THINK_LEAK_RE = _re.compile(
    r"<(think|analysis|reasoning)>(.*?)</\1>",
    _re.DOTALL | _re.IGNORECASE,
)


# ── Config ──────────────────────────────────────────────────────────────────

@dataclass
class Config:
    homeserver: str
    user_id: str
    password: str
    display_name: str
    mention_trigger: str
    credentials_path: Path
    monolith_url: str
    monolith_token: str

    @classmethod
    def from_env(cls) -> "Config":
        homeserver = os.environ.get("MATRIX_HOMESERVER", "").strip()
        user_id = os.environ.get("MATRIX_USER_ID", "").strip()

        missing = []
        if not homeserver:
            missing.append("MATRIX_HOMESERVER")
        if not user_id:
            missing.append("MATRIX_USER_ID")
        if missing:
            sys.stderr.write(
                f"Missing required env vars: {', '.join(missing)}\n"
            )
            sys.exit(2)

        password = os.environ.get("MATRIX_PASSWORD", "").strip()
        default_creds = Path.home() / ".config" / "monolith" / "matrix_credentials.json"
        creds_path = Path(
            os.environ.get("MATRIX_CREDENTIALS_PATH", str(default_creds))
        ).expanduser()

        if not password and not creds_path.exists():
            sys.stderr.write(
                "MATRIX_PASSWORD is required on first run "
                f"(no credentials cached at {creds_path})\n"
            )
            sys.exit(2)

        return cls(
            homeserver=homeserver,
            user_id=user_id,
            password=password,
            display_name=os.environ.get("MATRIX_DISPLAY_NAME", "Monolith"),
            mention_trigger=os.environ.get("MATRIX_MENTION_TRIGGER", "@monolith"),
            credentials_path=creds_path,
            monolith_url=os.environ.get(
                "MONOLITH_AGENT_URL", "http://localhost:7821"
            ).rstrip("/"),
            monolith_token=os.environ.get("MONOLITH_AGENT_TOKEN", "").strip(),
        )


# ── Credentials persistence ─────────────────────────────────────────────────

def load_credentials(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        sys.stderr.write(f"warning: could not load credentials at {path}: {exc}\n")
        return None


def save_credentials(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ── HTTP helpers (talk to engine/agent_server.py) ──────────────────────────

def _auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"} if token else {}


async def poll_monolith_health(cfg: Config) -> None:
    """Block until GET /health returns 200. Exponential backoff (1.5×), capped."""
    delay = HEALTH_POLL_INITIAL_SEC
    attempt = 0
    async with aiohttp.ClientSession() as session:
        while True:
            attempt += 1
            try:
                async with session.get(f"{cfg.monolith_url}/health", timeout=5) as resp:
                    if resp.status == 200:
                        print(f"monolith /health OK (attempt {attempt})", flush=True)
                        return
                    print(
                        f"monolith /health returned {resp.status} (attempt {attempt})",
                        flush=True,
                    )
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                # Only log every 5th retry to avoid spam during long waits.
                if attempt == 1 or attempt % 5 == 0:
                    print(
                        f"monolith /health unreachable (attempt {attempt}): {exc}",
                        flush=True,
                    )
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, HEALTH_POLL_MAX_SEC)


async def monolith_join(cfg: Config) -> None:
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                f"{cfg.monolith_url}/join",
                json={"name": PEER_NAME},
                headers=_auth_headers(cfg.monolith_token),
                timeout=5,
            ) as resp:
                body_text = await resp.text()
                print(f"join({PEER_NAME}): {resp.status} {body_text[:200]}", flush=True)
        except Exception as exc:
            print(f"warning: /join failed: {exc}", flush=True)


async def monolith_leave(cfg: Config) -> None:
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                f"{cfg.monolith_url}/leave",
                json={"name": PEER_NAME},
                headers=_auth_headers(cfg.monolith_token),
                timeout=5,
            ) as resp:
                await resp.read()
        except Exception:
            pass  # best-effort on shutdown


# ── Mention detection ──────────────────────────────────────────────────────

def _strip_one_token(body: str, token: str) -> str:
    """Remove the first case-insensitive occurrence of `token`, plus one
    trailing space/colon/comma if present. Returns the cleaned remainder."""
    lower_body = body.lower()
    lower_token = token.lower()
    idx = lower_body.find(lower_token)
    if idx < 0:
        return body
    end = idx + len(lower_token)
    if end < len(body) and body[end] in (":", ",", " "):
        end += 1
    return body[:idx] + body[end:]


def detect_mention(
    event_content: dict,
    body: str,
    *,
    bot_user_id: str,
    trigger: str,
    display_name: str,
) -> bool:
    """Check whether this message *explicitly* mentions the bot.

    INTENTIONAL: bare display name ("monolith" anywhere in body) does NOT
    trigger. The user said responses to casual mentions of the word
    "monolith" feel intrusive. We require:

      1. `m.mentions.user_ids` field — modern Element pill mention.
      2. `formatted_body` HTML containing the bot's matrix.to link — older
         clients still emit this.
      3. Substring of `<trigger>` where trigger starts with `@` (default
         `@monolith`) — covers typed `@monolith` without the autocomplete.

    `display_name` is accepted for signature stability but no longer used
    for detection (it IS used for body-stripping in extract_prompt).
    """
    # 1. Modern m.mentions
    mentions = event_content.get("m.mentions")
    if isinstance(mentions, dict):
        user_ids = mentions.get("user_ids") or []
        if bot_user_id in user_ids:
            return True

    # 2. formatted_body link
    formatted = event_content.get("formatted_body") or ""
    if isinstance(formatted, str) and bot_user_id in formatted:
        return True

    # 3. Substring of trigger (only triggers that start with @ are
    #    meaningful here; we defend against an empty/bare trigger).
    trigger_clean = trigger.strip()
    if trigger_clean.startswith("@") and trigger_clean.lower() in body.lower():
        return True

    return False


def extract_prompt(
    body: str, *, trigger: str, display_name: str
) -> str:
    """Strip the bot's name/trigger from `body`, returning the prompt.
    Tries trigger first, then display name. Both are case-insensitive."""
    cleaned = _strip_one_token(body, trigger)
    if cleaned == body and display_name:
        cleaned = _strip_one_token(body, display_name)
    return cleaned.strip()


# ── Debounce decision  ⚠️  USER IMPLEMENTS THIS  ⚠️  ─────────────────────────

def should_emit_edit(
    *,
    now_ms: int,
    last_edit_ms: int,
    accumulator: str,
    last_emitted: str,
) -> bool:
    """Decide whether to send an m.replace edit to Matrix right now.

    Implemented version. Tweak the EDIT_DEBOUNCE_* constants at the top of
    this file rather than the logic here.

    Returns True when EITHER:
      (a) ≥ EDIT_DEBOUNCE_MS have passed since last_edit_ms, OR
      (b) accumulator has grown ≥ EDIT_DEBOUNCE_CHARS beyond last_emitted.

    Special case: first edit (last_edit_ms == 0) fires immediately if
    accumulator has any content.
    """
    if not accumulator:
        return False
    if last_edit_ms == 0:
        return True
    if (now_ms - last_edit_ms) >= EDIT_DEBOUNCE_MS:
        return True
    growth = len(accumulator) - len(last_emitted)
    if growth >= EDIT_DEBOUNCE_CHARS:
        return True
    return False


def split_thinking(raw: str) -> tuple[str, str]:
    """Pull <think>/<analysis>/<reasoning> blocks out of `raw`.

    Returns (clean_answer, extracted_thinking). Used as defense-in-depth
    in case the server's lane-splitter (engine/agent_server.py) doesn't
    catch a tag — we still render the leak as markdown rather than raw.
    """
    extracted = "\n".join(
        m.group(2).strip() for m in _THINK_LEAK_RE.finditer(raw)
    ).strip()
    cleaned = _THINK_LEAK_RE.sub("", raw).strip()
    return cleaned, extracted


_PLAIN_SEPARATOR = "─" * 30


def _md_to_html(text: str) -> str:
    """Convert markdown to HTML for Matrix's formatted_body.

    Element only renders rich text when the message has
    `format: org.matrix.custom.html` + `formatted_body`. Without
    conversion, `*italic*` / `**bold**` / code blocks show as raw chars.
    """
    if not text:
        return ""
    return _markdown.markdown(
        text,
        extensions=["fenced_code", "tables", "nl2br"],
        output_format="html",
    )


def render_streaming_body(
    *,
    phase: str,
    thinking_text: str,
    answer_text: str,
    final: bool,
) -> str:
    """Build the PLAIN-TEXT body (fallback for non-HTML clients).

    Modern clients (Element, Cinny) use formatted_body instead, so this
    is mostly a safety net. Still: we keep it readable.

    Layout (both for streaming AND final — thinking is NEVER dropped):
        💭 Thought process:
        > {thinking text}

        ──────────────────────────────

        {answer text}
    """
    if phase == "thinking" and not answer_text:
        return PLACEHOLDER_BODY

    parts: list[str] = []
    if thinking_text:
        quoted = "\n".join(
            f"> {line}" if line.strip() else ">"
            for line in thinking_text.strip().splitlines()
        )
        parts.append(f"💭 Thought process:\n{quoted}")
    if answer_text:
        parts.append(answer_text.strip())
    if not parts:
        return PLACEHOLDER_BODY
    return f"\n\n{_PLAIN_SEPARATOR}\n\n".join(parts)


def render_streaming_body_html(
    *,
    phase: str,
    thinking_text: str,
    answer_text: str,
    final: bool,
) -> str:
    """Build the HTML formatted_body — this is what Element actually shows.

    Same layout as plain text but with real HTML elements:
      • `<blockquote>` for thinking
      • `<hr/>` separator between thinking and answer
      • Markdown in the answer is converted to HTML via _md_to_html
    """
    if phase == "thinking" and not answer_text:
        return f"<p><em>{_html_escape(PLACEHOLDER_BODY)}</em></p>"

    parts: list[str] = []
    if thinking_text:
        think_html = _md_to_html(thinking_text.strip())
        parts.append(
            "<p><strong>💭 Thought process:</strong></p>"
            f"<blockquote>{think_html}</blockquote>"
        )
    if answer_text:
        answer_html = _md_to_html(answer_text.strip())
        parts.append(answer_html)
    if not parts:
        return f"<p><em>{_html_escape(PLACEHOLDER_BODY)}</em></p>"
    return "<hr/>".join(parts)


# ── /chat/stream consumer + Matrix edit loop ───────────────────────────────

async def stream_reply(
    cfg: Config,
    client: AsyncClient,
    room_id: str,
    placeholder_event_id: str,
    prompt: str,
) -> None:
    """Consume /chat/stream SSE and emit ONE final m.replace on `done`.

    Rendering strategy (post user feedback 2026-05-23):
      • Placeholder "💭 thinking..." stays for the whole turn.
      • Matrix typing indicator (m.typing) fires throughout — Element
        renders "Monolith is typing..." in the room footer. This is an
        EPHEMERAL event, not a message edit, so it doesn't pile up on
        the rate-limit budget.
      • A background task refreshes typing every TYPING_REFRESH_SEC.
      • On `event: done`, typing stops and ONE final m.replace fires
        with the full content (thinking block + answer, rendered as
        markdown HTML).

    Tradeoff: no live token-streaming feel in Matrix. We accepted this
    because the LOCAL UI is the live view; Matrix is the broadcast.
    Eliminating mid-stream edits prevents the queue-catch-up lag that
    happened with the previous version (~15s drain after generation).
    """
    thinking_buf = ""
    answer_buf = ""
    last_emitted = ""
    last_edit_ms = 0
    phase = "thinking"
    edit_in_flight = False
    edits_skipped = 0  # diagnostic
    keep_typing = True

    async def typing_loop() -> None:
        """Refresh typing every ~25s. Synapse's max is 30s; we use 25."""
        try:
            while keep_typing:
                try:
                    await client.room_typing(room_id, typing_state=True, timeout=30000)
                except Exception as exc:
                    print(f"warning: room_typing failed: {exc}", flush=True)
                await asyncio.sleep(25)
        except asyncio.CancelledError:
            pass

    typing_task = asyncio.create_task(typing_loop())

    async def push_edit(
        body: str, body_html: str, *, final: bool = False
    ) -> None:
        """One m.replace edit. Sends BOTH plain `body` and HTML
        `formatted_body` so Element/Cinny render markdown properly."""
        is_placeholder = body == PLACEHOLDER_BODY
        if not final and body and not is_placeholder:
            rendered_plain = body + " " + CURSOR_CHAR
            rendered_html = body_html + f" <small>{CURSOR_CHAR}</small>"
        else:
            rendered_plain = body
            rendered_html = body_html

        new_content = {
            "msgtype": "m.text",
            "body": rendered_plain,
            "format": "org.matrix.custom.html",
            "formatted_body": rendered_html,
        }
        await client.room_send(
            room_id=room_id,
            message_type="m.room.message",
            content={
                "msgtype": "m.text",
                "body": f"* {rendered_plain}",
                "format": "org.matrix.custom.html",
                "formatted_body": f"* {rendered_html}",
                "m.new_content": new_content,
                "m.relates_to": {
                    "rel_type": "m.replace",
                    "event_id": placeholder_event_id,
                },
            },
        )

    async def maybe_emit() -> None:
        """Render current state and emit if debounce + in-flight allow.

        The in_flight guard is what saves us from the 429-cascade: nio
        is already auto-sleeping on 429s, but if we queued one edit per
        token while nio slept, we'd pile up dozens of edits. With the
        guard, only ONE edit is ever pending at a time.
        """
        nonlocal last_edit_ms, last_emitted, edit_in_flight, edits_skipped
        if edit_in_flight:
            edits_skipped += 1
            return
        # Strip any think tags that leaked into answer_buf as defense in depth.
        clean_answer, leaked_think = split_thinking(answer_buf)
        all_thinking = (
            thinking_buf + ("\n" + leaked_think if leaked_think else "")
        ).strip()
        body = render_streaming_body(
            phase=phase,
            thinking_text=all_thinking,
            answer_text=clean_answer,
            final=False,
        )
        body_html = render_streaming_body_html(
            phase=phase,
            thinking_text=all_thinking,
            answer_text=clean_answer,
            final=False,
        )
        if body == last_emitted:
            return
        now_ms = int(time.time() * 1000)
        if not should_emit_edit(
            now_ms=now_ms,
            last_edit_ms=last_edit_ms,
            accumulator=body,
            last_emitted=last_emitted,
        ):
            return
        edit_in_flight = True
        try:
            await push_edit(body, body_html)
            last_edit_ms = now_ms
            last_emitted = body
        finally:
            edit_in_flight = False

    payload = {"message": prompt, "agent": PEER_NAME}
    headers = _auth_headers(cfg.monolith_token)
    headers["Accept"] = "text/event-stream"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{cfg.monolith_url}/chat/stream",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=None, sock_read=300),
            ) as resp:
                if resp.status != 200:
                    err = await resp.text()
                    err_msg = f"⚠ /chat/stream returned {resp.status}: {err[:200]}"
                    await push_edit(err_msg, _html_escape(err_msg), final=True)
                    return

                event_name = ""
                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8", errors="replace").rstrip("\n")
                    if not line:
                        event_name = ""
                        continue
                    if line.startswith("event:"):
                        event_name = line[len("event:"):].strip()
                        continue
                    if not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if not data_str:
                        continue
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    if event_name == "thinking":
                        thinking_buf += str(data.get("text", ""))
                        # Don't emit during thinking phase — placeholder stays.
                    elif event_name == "token":
                        if phase == "thinking":
                            phase = "answering"
                            print(
                                f"[phase] thinking→answering (think_len={len(thinking_buf)})",
                                flush=True,
                            )
                        answer_buf += str(data.get("text", ""))
                        # NOTE: no mid-stream emit. Typing indicator
                        # (typing_task) tells Matrix "Monolith is typing..."
                        # The full content lands in ONE edit on `done`.
                    elif event_name == "tool_call":
                        # Tool-call surfacing is out-of-scope for v1 polish;
                        # just keep streaming the answer in the rendered output.
                        pass
                    elif event_name == "tool_result":
                        pass
                    elif event_name == "done":
                        full = str(data.get("full_response", answer_buf))
                        # Strip both standard tags (server-side guarantee) and
                        # any leaked think blocks (defense in depth). Any
                        # extracted thinking joins thinking_buf so it stays
                        # visible in the final rendering.
                        cleaned = strip_tag_blocks(full, EXTERNAL_STRIP_TAGS)
                        cleaned, leaked_final = split_thinking(cleaned)
                        all_thinking_final = (
                            thinking_buf
                            + ("\n" + leaked_final if leaked_final else "")
                        ).strip()
                        body = render_streaming_body(
                            phase="answering",
                            thinking_text=all_thinking_final,
                            answer_text=cleaned,
                            final=True,
                        )
                        body_html = render_streaming_body_html(
                            phase="answering",
                            thinking_text=all_thinking_final,
                            answer_text=cleaned,
                            final=True,
                        )
                        # Stop typing BEFORE the final edit so the
                        # indicator vanishes the moment the message lands.
                        keep_typing = False
                        try:
                            await client.room_typing(room_id, typing_state=False)
                        except Exception:
                            pass
                        await push_edit(body, body_html, final=True)
                        print(
                            f"[done] answer_len={len(cleaned)} "
                            f"think_len={len(all_thinking_final)} "
                            f"edits_skipped={edits_skipped}",
                            flush=True,
                        )
                        typing_task.cancel()
                        return
                    elif event_name == "error":
                        msg = str(data.get("text", "unknown error"))
                        err_text = f"⚠ {msg}"
                        keep_typing = False
                        try:
                            await client.room_typing(room_id, typing_state=False)
                        except Exception:
                            pass
                        await push_edit(
                            err_text, _html_escape(err_text), final=True
                        )
                        typing_task.cancel()
                        return
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        msg = f"⚠ stream interrupted: {exc}. Retry your message."
        keep_typing = False
        try:
            await client.room_typing(room_id, typing_state=False)
        except Exception:
            pass
        await push_edit(msg, _html_escape(msg), final=True)
        typing_task.cancel()
    except Exception as exc:
        msg = f"⚠ unexpected error: {exc}"
        keep_typing = False
        try:
            await client.room_typing(room_id, typing_state=False)
        except Exception:
            pass
        await push_edit(msg, _html_escape(msg), final=True)
        typing_task.cancel()


# ── Matrix event handlers ───────────────────────────────────────────────────

class Bridge:
    def __init__(self, cfg: Config, client: AsyncClient) -> None:
        self.cfg = cfg
        self.client = client

    async def on_message(self, room: MatrixRoom, event: RoomMessageText) -> None:
        if event.sender == self.cfg.user_id:
            return

        body = event.body or ""
        source = getattr(event, "source", {}) or {}
        content = source.get("content", {}) or {}

        # Every message: log a one-line summary. Cheap, and saves us from
        # guessing what the bridge saw next time a mention "fails".
        print(
            f"[msg] room={room.room_id[:20]}... sender={event.sender} body={body[:80]!r}",
            flush=True,
        )

        if not detect_mention(
            content,
            body,
            bot_user_id=self.cfg.user_id,
            trigger=self.cfg.mention_trigger,
            display_name=self.cfg.display_name,
        ):
            return

        prompt = extract_prompt(
            body,
            trigger=self.cfg.mention_trigger,
            display_name=self.cfg.display_name,
        )
        if not prompt:
            print(f"[mention] no prompt after strip; body={body!r}", flush=True)
            return

        print(f"[mention] from={event.sender} prompt={prompt[:120]!r}", flush=True)

        try:
            resp = await self.client.room_send(
                room_id=room.room_id,
                message_type="m.room.message",
                content={"msgtype": "m.text", "body": PLACEHOLDER_BODY},
            )
        except Exception as exc:
            print(f"warning: placeholder room_send raised: {exc}", flush=True)
            return

        placeholder_id = getattr(resp, "event_id", None)
        if not placeholder_id:
            print(f"warning: could not post placeholder: {resp}", flush=True)
            return

        await stream_reply(
            self.cfg, self.client, room.room_id, placeholder_id, prompt
        )

    async def on_invite(self, room: MatrixRoom, event: InviteMemberEvent) -> None:
        if event.membership != "invite":
            return
        if event.state_key != self.cfg.user_id:
            return
        print(f"invited to {room.room_id}; joining", flush=True)
        try:
            await self.client.join(room.room_id)
        except Exception as exc:
            print(f"warning: could not join {room.room_id}: {exc}", flush=True)


# ── Login + main loop ───────────────────────────────────────────────────────

async def login(cfg: Config) -> AsyncClient:
    creds = load_credentials(cfg.credentials_path)
    client = AsyncClient(cfg.homeserver, cfg.user_id)
    if creds and creds.get("access_token"):
        client.access_token = creds["access_token"]
        client.user_id = creds.get("user_id", cfg.user_id)
        client.device_id = creds.get("device_id")
        print(
            f"reusing cached credentials from {cfg.credentials_path}",
            flush=True,
        )
        return client

    if not cfg.password:
        sys.stderr.write(
            "no cached credentials and MATRIX_PASSWORD not set\n"
        )
        await client.close()
        sys.exit(2)

    resp = await client.login(cfg.password, device_name="monolith-bridge")
    if not isinstance(resp, LoginResponse):
        sys.stderr.write(f"matrix login failed: {resp}\n")
        await client.close()
        sys.exit(1)

    save_credentials(
        cfg.credentials_path,
        {
            "homeserver": cfg.homeserver,
            "user_id": resp.user_id,
            "device_id": resp.device_id,
            "access_token": resp.access_token,
        },
    )
    print(
        f"login OK; credentials cached at {cfg.credentials_path}",
        flush=True,
    )
    return client


async def maybe_set_display_name(client: AsyncClient, want: str) -> None:
    try:
        current = await client.get_displayname()
        cur_name = getattr(current, "displayname", None) or ""
        if cur_name != want:
            await client.set_displayname(want)
            print(f"display name set to {want!r}", flush=True)
    except Exception as exc:
        print(f"warning: could not set display name: {exc}", flush=True)


async def set_presence_online(client: AsyncClient, status_msg: str = "Online") -> None:
    """Tell Matrix the bot is online so Element doesn't show the grey dot.

    matrix-nio defaults presence to 'unavailable'. Without this call the
    bot sync_forever keeps the connection alive but the homeserver
    reports the user as offline to other clients.
    """
    try:
        await client.set_presence("online", status_msg)
        print(f"presence set to online ({status_msg!r})", flush=True)
    except Exception as exc:
        print(f"warning: could not set presence: {exc}", flush=True)


async def presence_keepalive(client: AsyncClient, interval_sec: float = 240.0) -> None:
    """Periodically re-set presence to online. Matrix downgrades presence
    to 'unavailable' after ~5 minutes of no activity-reporting; we ping
    every 4 minutes to stay green."""
    while True:
        try:
            await asyncio.sleep(interval_sec)
            await client.set_presence("online")
        except asyncio.CancelledError:
            return
        except Exception as exc:
            print(f"warning: presence keepalive failed: {exc}", flush=True)


async def main_async() -> int:
    cfg = Config.from_env()

    print(
        f"matrix bridge starting | user={cfg.user_id} | homeserver={cfg.homeserver}",
        flush=True,
    )
    print(
        f"target monolith={cfg.monolith_url} | mention trigger={cfg.mention_trigger!r}",
        flush=True,
    )

    await poll_monolith_health(cfg)
    await monolith_join(cfg)

    client = await login(cfg)
    await maybe_set_display_name(client, cfg.display_name)
    await set_presence_online(client, "Monolith — online via bridge")

    bridge = Bridge(cfg, client)
    client.add_event_callback(bridge.on_message, RoomMessageText)
    client.add_event_callback(bridge.on_invite, InviteMemberEvent)

    stop_event = asyncio.Event()

    def _request_stop(*_args) -> None:
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except (NotImplementedError, RuntimeError):
            # Windows doesn't support add_signal_handler for SIGTERM; fall back.
            try:
                signal.signal(sig, lambda *_a: _request_stop())
            except (ValueError, OSError):
                pass

    sync_task = asyncio.create_task(
        client.sync_forever(timeout=30000, full_state=True)
    )
    keepalive_task = asyncio.create_task(presence_keepalive(client))

    try:
        await stop_event.wait()
    finally:
        keepalive_task.cancel()
        sync_task.cancel()
        try:
            await client.set_presence("offline", "Bridge stopped")
        except Exception:
            pass
        for t in (keepalive_task, sync_task):
            try:
                await t
            except asyncio.CancelledError:
                pass
        await monolith_leave(cfg)
        await client.close()
        print("matrix bridge stopped cleanly", flush=True)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main_async()))
    except KeyboardInterrupt:
        sys.exit(0)
