from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Iterable

from core.slug import slugify


def _default_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ChatSessionManager:
    master_prompt: str
    now_iso: Callable[[], str] = _default_now_iso
    _session_counter: int = 0
    _current_session: dict | None = None
    _undo_snapshot: list[dict] | None = None
    _undo_tree_snapshot: dict | None = None
    _title_generated: bool = False
    _suppress_title_regen: bool = False

    def __post_init__(self) -> None:
        if self._current_session is None:
            self._current_session = self.create_session()

    @property
    def current(self) -> dict:
        assert self._current_session is not None
        return self._current_session

    def create_session(
        self,
        messages: list[dict] | None = None,
        created_at: str | None = None,
        updated_at: str | None = None,
        archive_path: str | None = None,
        summary: list[str] | None = None,
        title: str | None = None,
        assistant_tokens: int = 0,
    ) -> dict:
        self._session_counter += 1
        now = self.now_iso()
        return {
            "id": self._session_counter,
            "created_at": created_at or now,
            "updated_at": updated_at or now,
            "messages": messages or [],
            "archive_path": archive_path,
            "summary": summary or [],
            "title": title,
            "assistant_tokens": int(assistant_tokens),
        }

    def set_current(self, session: dict) -> None:
        self._current_session = session
        self._undo_snapshot = None
        self._undo_tree_snapshot = None
        self._title_generated = bool(session.get("title"))
        self._suppress_title_regen = False

    def build_engine_history(self) -> list[dict]:
        from core.channel_tag import build_channel_tag
        from core.session_tool_palette import render_session_tool_palette

        messages = self.current.get("messages", [])

        # Find last user message index for include_modes=True
        last_user_idx = -1
        for i, msg in enumerate(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                if msg.get("kind") != "command_block":
                    last_user_idx = i

        palette_block = None
        if last_user_idx > 0:
            palette_block = render_session_tool_palette(messages[:last_user_idx])
        palette_inserted = False

        history = [{"role": "system", "content": self.master_prompt}]
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            text = msg.get("text")
            if not isinstance(role, str) or not isinstance(text, str):
                continue
            content = text.strip()
            if not content:
                continue
            if role == "system":
                # Command receipts (/rating, /think, ...) are UI-only by
                # contract (_emit_command_block) — keep them out of the model
                # history; only genuine system events become [UI_EVENT].
                if msg.get("kind") == "command_block":
                    continue
                history.append({"role": "user", "content": f"[UI_EVENT]\n{content}"})
                continue
            if role == "tool_result":
                try:
                    payload = json.loads(content)
                except Exception:
                    payload = None
                if isinstance(payload, dict):
                    tool_name = str(payload.get("tool", "")).strip()
                    tool_body = str(payload.get("result", "")).strip() or content
                    if tool_name:
                        history.append({"role": "user", "content": f"[TOOL RESULT:{tool_name}]\n{tool_body}"})
                    else:
                        history.append({"role": "user", "content": f"[TOOL RESULT]\n{tool_body}"})
                else:
                    history.append({"role": "user", "content": f"[TOOL RESULT]\n{content}"})
                continue
            if role == "tool_call":
                continue
            if role == "agent":
                if not bool(msg.get("agent_approved", False)):
                    continue
                agent_name = str(msg.get("agent_name", "Agent") or "Agent").strip() or "Agent"
                body = content
                prefixed = f"[{agent_name}] "
                if body.startswith(prefixed):
                    body = body[len(prefixed):].lstrip()
                history.append({"role": "user", "content": f"[AGENT:{agent_name}]\n{body}"})
                continue
            if role not in {"user", "assistant"}:
                continue
            if role == "user" and i == last_user_idx and palette_block and not palette_inserted:
                history.append(
                    {
                        "role": "user",
                        "content": palette_block,
                        "ephemeral": True,
                        "source": "session_tool_palette",
                    }
                )
                palette_inserted = True
            if role == "assistant":
                content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL)
            # Inject [CHANNEL: ...] tag if not already present
            if not content.lstrip().startswith("[CHANNEL:"):
                role_token = {"user": "USER", "assistant": "ASSISTANT"}.get(role)
                if role_token:
                    # Replayed history shows the role token only (channel_tag.py
                    # contract): mode fields (monothink=/prompts=) belong ONLY on
                    # the generating turn, which LLMEngine.generate tags separately.
                    # Tagging this turn's last history message with modes made TWO
                    # messages carry the "turn to answer" marker, so right after a
                    # one-exchange history the model re-answered the prior message
                    # (2026-06-19 "E." duplicate-greeting bug).
                    tag = build_channel_tag(role_token, include_modes=False)
                    content = f"{tag}\n\n{content}"
            history.append({"role": role, "content": content})
        return history

    def snapshot(self) -> None:
        self._undo_snapshot = [dict(m) for m in self.current.get("messages", [])]
        from ui.pages import session_tree
        self._undo_tree_snapshot = session_tree.snapshot(self.current) if session_tree.active() else None

    def undo(self) -> bool:
        from ui.pages import session_tree
        if session_tree.active() and getattr(self, "_undo_tree_snapshot", None) is not None:
            ok = session_tree.restore(self.current, self._undo_tree_snapshot)
            self._undo_tree_snapshot = None
            self._undo_snapshot = None
            return ok
        if not self._undo_snapshot:
            return False
        self.current["messages"] = self._undo_snapshot
        self._undo_snapshot = None
        return True

    def add_message(self, role: str, text: str, extra: dict | None = None) -> int:
        now = self.now_iso()
        message = self.build_message(role, text, now=now, extra=extra)
        from ui.pages import session_tree
        if session_tree.active():
            index = session_tree.tree_append(self.current, message)
        else:
            self.current.setdefault("messages", []).append(message)
            index = len(self.current["messages"]) - 1
        self.current["updated_at"] = now
        return index

    def build_message(
        self,
        role: str,
        text: str,
        now: str | None = None,
        extra: dict | None = None,
    ) -> dict:
        stamp = now or self.now_iso()
        message = {
            "i": len(self.current.get("messages", [])) + 1,
            "time": stamp,
            "role": role,
            "text": text,
        }
        if isinstance(extra, dict):
            for key, value in extra.items():
                if key in {"i", "time", "role", "text"}:
                    continue
                message[key] = value
        return message

    def insert_message(self, index: int, role: str, text: str, extra: dict | None = None) -> int:
        now = self.now_iso()
        message = self.build_message(role, text, now=now, extra=extra)
        messages = self.current.setdefault("messages", [])
        if index < 0:
            index = 0
        if index > len(messages):
            index = len(messages)
        from ui.pages import session_tree
        if session_tree.active():
            index = session_tree.tree_insert(self.current, index, message)
        else:
            messages.insert(index, message)
            self._renumber_messages()
        self.current["updated_at"] = now
        return index

    def _renumber_messages(self) -> None:
        for idx, msg in enumerate(self.current.get("messages", []), start=1):
            if isinstance(msg, dict):
                msg["i"] = idx

    def append_assistant_token(self, token: str, target_index: int | None) -> None:
        if target_index is None:
            return
        msgs = self.current.get("messages", [])
        if not (0 <= target_index < len(msgs)):
            return
        msg = msgs[target_index]
        msg["text"] = (msg.get("text") or "") + token
        msg["time"] = self.now_iso()
        self.current["updated_at"] = msg["time"]
        self.current["assistant_tokens"] = int(self.current.get("assistant_tokens", 0)) + 1

    def cleanup_empty_assistant_if_needed(
        self,
        active_assistant_index: int | None,
        active_assistant_started: bool,
        active_assistant_token_count: int,
    ) -> bool:
        if active_assistant_index is None:
            return False
        if not active_assistant_started:
            return False
        if active_assistant_token_count > 0:
            return False
        msgs = self.current.get("messages", [])
        if 0 <= active_assistant_index < len(msgs):
            msg = msgs[active_assistant_index]
            if msg.get("role") == "assistant" and (msg.get("text") or "") == "":
                from ui.pages import session_tree
                if session_tree.active() and msg.get("node_id"):
                    session_tree.tree_prune_node(self.current, msg["node_id"])
                else:
                    del msgs[active_assistant_index]
                return True
        return False

    def set_suppress_title_regen(self, suppress: bool) -> None:
        self._suppress_title_regen = suppress

    def reset_title_flags(self) -> None:
        self._title_generated = False
        self._suppress_title_regen = False

    def maybe_generate_title(self) -> str | None:
        if self._suppress_title_regen:
            return None
        if self._title_generated:
            return None
        if self.current.get("title"):
            self._title_generated = True
            return None
        if not any(
            m.get("role") == "user" and m.get("text", "").strip()
            for m in self.current.get("messages", [])
        ):
            return None
        title = self.derive_title(self.current.get("messages", []))
        self.current["title"] = title
        self._title_generated = True
        return title

    def ensure_title(self, messages: Iterable[dict], title: str | None = None) -> str:
        if title:
            return title
        return self.derive_title(messages)

    def derive_title(self, messages: Iterable[dict]) -> str:
        stopwords = {
            "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
            "how", "i", "if", "in", "into", "is", "it", "me", "my", "of", "on", "or",
            "our", "please", "so", "that", "the", "their", "them", "then", "there", "these",
            "this", "to", "us", "we", "what", "when", "where", "which", "who", "why", "with",
            "you", "your",
        }
        user_texts: list[str] = []
        for msg in messages:
            if msg.get("role") != "user":
                continue
            text = " ".join((msg.get("text") or "").lower().split())
            if text:
                user_texts.append(text)
            if len(user_texts) == 3:
                break

        if not user_texts:
            return "chat"

        candidates: list[str] = []
        counts: dict[str, int] = {}
        for text in user_texts:
            for token in re.findall(r"[a-z0-9]+", text):
                if token in stopwords or len(token) < 3:
                    continue
                if token not in counts:
                    candidates.append(token)
                    counts[token] = 0
                counts[token] += 1

        ranked = sorted(candidates, key=lambda token: (-counts[token], candidates.index(token)))
        title_tokens = ranked[:6]
        title = " ".join(title_tokens)
        title = re.sub(r"\s+", " ", title).strip()
        title = re.sub(r"[^a-z0-9\- ]+", "", title)
        if len(title) > 40:
            title = title[:40].rstrip()
        return title or "chat"

    def build_summary(self, messages: Iterable[dict], title: str | None = None) -> list[str]:
        msgs = list(messages)
        summary: list[str] = []
        title = self.ensure_title(msgs, title)
        summary.append(f"Title: {title}")
        user_msgs = [m["text"] for m in msgs if m.get("role") == "user" and m.get("text")]
        assistant_msgs = [m["text"] for m in msgs if m.get("role") == "assistant" and m.get("text")]

        def _trim(text: str, limit: int = 120) -> str:
            return text if len(text) <= limit else f"{text[:limit]}\u2026"

        for msg in user_msgs[-3:]:
            summary.append(f"User: {_trim(msg)}")
        for msg in assistant_msgs[-3:]:
            summary.append(f"Assistant: {_trim(msg)}")
        if len(summary) < 3:
            summary.append(f"Messages: {len(msgs)}")
        if len(summary) < 3:
            summary.append("Summary: Not enough messages yet.")
        return summary[:6]

    def slugify(self, text: str) -> str:
        return slugify(text, "chat")

    def topic_dominant(self) -> bool:
        user_text = " ".join(
            [m.get("text", "") for m in self.current.get("messages", []) if m.get("role") == "user"]
        )
        words = [w.lower() for w in re.findall(r"[a-zA-Z]{4,}", user_text)]
        if not words:
            return False
        counts: dict[str, int] = {}
        for word in words:
            counts[word] = counts.get(word, 0) + 1
        return max(counts.values()) >= 3
