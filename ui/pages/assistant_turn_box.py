from __future__ import annotations

from dataclasses import dataclass, field

from ui.pages.chat_session import ChatSessionManager

_THINK_OPEN_TAGS = ("<think>", "<analysis>", "<reasoning>")
_THINK_CLOSE_TAGS = ("</think>", "</analysis>", "</reasoning>")
_ACU_OPEN_TAG = "<acatalepsy>"
_ACU_CLOSE_TAG = "</acatalepsy>"
_TOOL_OPEN_TAG = "<tool_call>"
_TOOL_CLOSE_TAG = "</tool_call>"
# End-of-response routing envelopes — emitted by the LLM at the very end
# of every turn, parsed by axis_orchestrator, suppressed from UI display.
# The model's self-classification routes the NEXT turn's gated injection.
_AXES_OPEN_TAG = "<axes>"
_AXES_CLOSE_TAG = "</axes>"
_INTENT_OPEN_TAG = "<intent>"
_INTENT_CLOSE_TAG = "</intent>"
# Tool-evidence envelope — emitted before [TOOL_LOOP_DONE] when tool
# results were used. One bullet per tool_result citing the load-bearing
# fact that drove the answer. Closes the verification gap on the
# PROVENANCE [observed] discipline: an answer that claims to be observed
# must cite the tool result that observed it.
_TOOL_EVIDENCE_OPEN_TAG = "<tool_evidence>"
_TOOL_EVIDENCE_CLOSE_TAG = "</tool_evidence>"
# Bearing-update envelope — emitted at turn end when the model updates its
# situational posture. Parsed and committed to bearing.json by chat_finalize's
# Bearing write hook; suppressed from UI display so the user doesn't see the
# raw JSON envelope. Same display discipline as <axes>/<intent>: lane exists
# so a future inspector can surface "Bearing updated" indicators, but nothing
# routes to the main answer area.
_BEARING_UPDATE_OPEN_TAG = "<bearing_update>"
_BEARING_UPDATE_CLOSE_TAG = "</bearing_update>"
_CURIOSITY_OPEN_TAG = "<curiosity>"
_CURIOSITY_CLOSE_TAG = "</curiosity>"
_ALL_OPEN_TAGS = _THINK_OPEN_TAGS + (
    _ACU_OPEN_TAG,
    _TOOL_OPEN_TAG,
    _AXES_OPEN_TAG,
    _INTENT_OPEN_TAG,
    _TOOL_EVIDENCE_OPEN_TAG,
    _BEARING_UPDATE_OPEN_TAG,
    _CURIOSITY_OPEN_TAG,
)
_ALL_CLOSE_TAGS = _THINK_CLOSE_TAGS + (
    _ACU_CLOSE_TAG,
    _TOOL_CLOSE_TAG,
    _AXES_CLOSE_TAG,
    _INTENT_CLOSE_TAG,
    _TOOL_EVIDENCE_CLOSE_TAG,
    _BEARING_UPDATE_CLOSE_TAG,
    _CURIOSITY_CLOSE_TAG,
)

# Internal engine sentinels that must NEVER reach the user's screen. Strip them
# at the stream entry, before the state machine sees them, so the post-stream
# rerender does not have to "clean up" what was briefly visible.
_INTERNAL_SENTINELS: tuple[str, ...] = ("[TOOL_LOOP_DONE]",)

# Cap on partial-tag accumulation. A malformed `<` followed by a long
# non-tag stream would otherwise stay in tag_open indefinitely; flushing
# at this size as plain text bounds the worst case.
_MAX_TAG_BUF = 64


def strip_internal_sentinels(text: str) -> str:
    """Remove engine-internal markers that should not display to the user."""
    if not text:
        return text
    for marker in _INTERNAL_SENTINELS:
        if marker in text:
            text = text.replace(marker, "")
    return text


@dataclass
class AssistantDisplayUpdate:
    answer_text: str = ""
    thinking_text: str = ""
    thinking_opened: bool = False
    thinking_closed: bool = False
    acu_text: str = ""
    acu_opened: bool = False
    acu_closed: bool = False
    tool_call_text: str = ""
    tool_call_opened: bool = False
    tool_call_closed: bool = False
    axes_text: str = ""
    axes_opened: bool = False
    axes_closed: bool = False
    intent_text: str = ""
    intent_opened: bool = False
    intent_closed: bool = False
    tool_evidence_text: str = ""
    tool_evidence_opened: bool = False
    tool_evidence_closed: bool = False
    bearing_update_text: str = ""
    bearing_update_opened: bool = False
    bearing_update_closed: bool = False
    curiosity_text: str = ""
    curiosity_opened: bool = False
    curiosity_closed: bool = False

    def has_changes(self) -> bool:
        return bool(
            self.answer_text
            or self.thinking_text
            or self.thinking_opened
            or self.thinking_closed
            or self.acu_text
            or self.acu_opened
            or self.acu_closed
            or self.tool_call_text
            or self.tool_call_opened
            or self.tool_call_closed
            or self.axes_text
            or self.axes_opened
            or self.axes_closed
            or self.intent_text
            or self.intent_opened
            or self.intent_closed
            or self.tool_evidence_text
            or self.tool_evidence_opened
            or self.tool_evidence_closed
            or self.bearing_update_text
            or self.bearing_update_opened
            or self.bearing_update_closed
            or self.curiosity_text
            or self.curiosity_opened
            or self.curiosity_closed
        )


@dataclass
class AssistantDisplaySnapshot:
    answer_text: str = ""
    thinking_text: str = ""
    thinking_active: bool = False
    thinking_seen: bool = False
    acu_text: str = ""
    acu_seen: bool = False
    tool_call_text: str = ""
    tool_call_seen: bool = False
    axes_text: str = ""
    axes_seen: bool = False
    intent_text: str = ""
    intent_seen: bool = False
    tool_evidence_text: str = ""
    tool_evidence_seen: bool = False
    bearing_update_text: str = ""
    bearing_update_seen: bool = False
    curiosity_text: str = ""
    curiosity_seen: bool = False


@dataclass
class AssistantStreamNormalizer:
    """Single source of truth for streamed-token tag handling.

    Routes incoming tokens into four lanes:
      - answer_text (the visible message body)
      - thinking_text (collapsible <think>/<analysis>/<reasoning> block)
      - acu_text (<acatalepsy> ACU memory block)
      - tool_call_text (<tool_call>...</tool_call> structured tool invocation)

    Engine-internal sentinels (e.g. [TOOL_LOOP_DONE]) are stripped at
    consume() entry so they never appear, even briefly, in any lane.
    """

    answer_text: str = ""
    thinking_text: str = ""
    thinking_active: bool = False
    thinking_seen: bool = False
    acu_text: str = ""
    acu_active: bool = False
    acu_seen: bool = False
    tool_call_text: str = ""
    tool_call_active: bool = False
    tool_call_seen: bool = False
    axes_text: str = ""
    axes_active: bool = False
    axes_seen: bool = False
    intent_text: str = ""
    intent_active: bool = False
    intent_seen: bool = False
    tool_evidence_text: str = ""
    tool_evidence_active: bool = False
    tool_evidence_seen: bool = False
    bearing_update_text: str = ""
    bearing_update_active: bool = False
    bearing_update_seen: bool = False
    curiosity_text: str = ""
    curiosity_active: bool = False
    curiosity_seen: bool = False
    _state: str = "normal"
    _tag_buf: str = ""
    # Last character pushed to answer_text. Used to detect inline-code
    # tag references — when the model writes `<bearing_update>` in prose
    # (backtick + tag + backtick), the state machine would otherwise
    # treat the `<` as the start of a real envelope. If the preceding
    # answer character was a backtick, the `<` is content not a tag-open.
    _last_answer_char: str = ""
    # Carry buffer for an internal sentinel ([TOOL_LOOP_DONE]) split across
    # streamed token boundaries. strip_internal_sentinels() only matches a
    # whole marker within one chunk; when the marker straddles a boundary the
    # pieces would slip through to the answer lane and flash on screen before
    # the post-stream rerender stripped them. We hold back any trailing run
    # that could still complete into a sentinel until the next chunk resolves.
    _pending_sentinel: str = ""

    def reset(self) -> None:
        self.answer_text = ""
        self.thinking_text = ""
        self.thinking_active = False
        self.thinking_seen = False
        self.acu_text = ""
        self.acu_active = False
        self.acu_seen = False
        self.tool_call_text = ""
        self.tool_call_active = False
        self.tool_call_seen = False
        self.axes_text = ""
        self.axes_active = False
        self.axes_seen = False
        self.intent_text = ""
        self.intent_active = False
        self.intent_seen = False
        self.tool_evidence_text = ""
        self.tool_evidence_active = False
        self.tool_evidence_seen = False
        self.bearing_update_text = ""
        self.bearing_update_active = False
        self.bearing_update_seen = False
        self.curiosity_text = ""
        self.curiosity_active = False
        self.curiosity_seen = False
        self._state = "normal"
        self._tag_buf = ""
        # Nesting depth of <think>/<analysis>/<reasoning>. The model sometimes
        # nests think blocks (or re-deliberates) — only the OUTERMOST close ends
        # thinking, so reasoning never leaks into the answer lane. "Catch the
        # true last think."
        self._think_depth = 0
        self._last_answer_char = ""
        self._pending_sentinel = ""

    @staticmethod
    def _held_sentinel_len(text: str) -> int:
        """Length of the trailing run of *text* that is a non-empty proper
        prefix of some internal sentinel (so it might complete on the next
        chunk). 0 when no trailing run could begin a sentinel."""
        best = 0
        for marker in _INTERNAL_SENTINELS:
            limit = min(len(text), len(marker) - 1)
            for k in range(limit, 0, -1):
                if marker.startswith(text[-k:]):
                    best = max(best, k)
                    break
        return best

    def consume(self, text: str) -> AssistantDisplayUpdate:
        update = AssistantDisplayUpdate()
        # Re-attach any partial sentinel held back from the previous chunk.
        text = self._pending_sentinel + (text or "")
        self._pending_sentinel = ""
        if not text:
            return update
        # Drop whole sentinels, then hold back a trailing partial that could
        # still complete into one on the next chunk (split-token safety).
        cleaned = strip_internal_sentinels(text)
        hold = self._held_sentinel_len(cleaned)
        if hold:
            self._pending_sentinel = cleaned[-hold:]
            cleaned = cleaned[:-hold]
        if not cleaned:
            return update
        for ch in cleaned:
            self._consume_char(ch, update)
        return update

    def _flush_buf_to(self, lane: str, update: AssistantDisplayUpdate) -> None:
        """Spill _tag_buf into a destination lane and clear it."""
        if not self._tag_buf:
            return
        buf = self._tag_buf
        self._tag_buf = ""
        if lane == "answer":
            self.answer_text += buf
            update.answer_text += buf
        elif lane == "thinking":
            self.thinking_text += buf
            update.thinking_text += buf
        elif lane == "acu":
            self.acu_text += buf
            update.acu_text += buf
        elif lane == "tool":
            self.tool_call_text += buf
            update.tool_call_text += buf
        elif lane == "axes":
            self.axes_text += buf
            update.axes_text += buf
        elif lane == "intent":
            self.intent_text += buf
            update.intent_text += buf
        elif lane == "tool_evidence":
            self.tool_evidence_text += buf
            update.tool_evidence_text += buf
        elif lane == "bearing_update":
            self.bearing_update_text += buf
            update.bearing_update_text += buf
        elif lane == "curiosity":
            self.curiosity_text += buf
            update.curiosity_text += buf

    def _consume_char(self, ch: str, update: AssistantDisplayUpdate) -> None:
        # Bound the partial-tag buffer. If we cross the cap without matching a
        # known prefix, treat it as plain content for whichever lane we were
        # collecting from. Prevents an unclosed `<` from stalling the parser.
        if len(self._tag_buf) > _MAX_TAG_BUF:
            if self._state == "tag_open":
                self._flush_buf_to("answer", update)
                self._state = "normal"
            elif self._state == "tag_close_think":
                self._flush_buf_to("thinking", update)
                self._state = "in_think"
            elif self._state == "tag_close_acu":
                self._flush_buf_to("acu", update)
                self._state = "in_acu"
            elif self._state == "tag_close_tool":
                self._flush_buf_to("tool", update)
                self._state = "in_tool"
            elif self._state == "tag_close_axes":
                self._flush_buf_to("axes", update)
                self._state = "in_axes"
            elif self._state == "tag_close_intent":
                self._flush_buf_to("intent", update)
                self._state = "in_intent"
            elif self._state == "tag_close_tool_evidence":
                self._flush_buf_to("tool_evidence", update)
                self._state = "in_tool_evidence"
            elif self._state == "tag_close_bearing_update":
                self._flush_buf_to("bearing_update", update)
                self._state = "in_bearing_update"
            elif self._state == "tag_close_curiosity":
                self._flush_buf_to("curiosity", update)
                self._state = "in_curiosity"

        if self._state == "normal":
            if ch == "<":
                # Backtick-quoted inline-code reference (`<tag>`) — treat
                # the `<` as content, not as a tag-open. Lets the model
                # write about its own envelope shape in prose without
                # the state machine treating the reference as a real tag.
                if self._last_answer_char == "`":
                    self.answer_text += ch
                    update.answer_text += ch
                    self._last_answer_char = ch
                    return
                self._state = "tag_open"
                self._tag_buf = "<"
            else:
                self.answer_text += ch
                update.answer_text += ch
                self._last_answer_char = ch
            return

        if self._state == "tag_open":
            self._tag_buf += ch
            if self._tag_buf in _THINK_OPEN_TAGS:
                self._state = "in_think"
                self._tag_buf = ""
                self.thinking_active = True
                self.thinking_seen = True
                self._think_depth = 1
                update.thinking_opened = True
                return
            if self._tag_buf == _ACU_OPEN_TAG:
                self._state = "in_acu"
                self._tag_buf = ""
                self.acu_active = True
                self.acu_seen = True
                update.acu_opened = True
                return
            if self._tag_buf == _TOOL_OPEN_TAG:
                self._state = "in_tool"
                self._tag_buf = ""
                self.tool_call_active = True
                self.tool_call_seen = True
                update.tool_call_opened = True
                return
            if self._tag_buf == _AXES_OPEN_TAG:
                self._state = "in_axes"
                self._tag_buf = ""
                self.axes_active = True
                self.axes_seen = True
                update.axes_opened = True
                return
            if self._tag_buf == _INTENT_OPEN_TAG:
                self._state = "in_intent"
                self._tag_buf = ""
                self.intent_active = True
                self.intent_seen = True
                update.intent_opened = True
                return
            if self._tag_buf == _TOOL_EVIDENCE_OPEN_TAG:
                self._state = "in_tool_evidence"
                self._tag_buf = ""
                self.tool_evidence_active = True
                self.tool_evidence_seen = True
                update.tool_evidence_opened = True
                return
            if self._tag_buf == _BEARING_UPDATE_OPEN_TAG:
                self._state = "in_bearing_update"
                self._tag_buf = ""
                self.bearing_update_active = True
                self.bearing_update_seen = True
                update.bearing_update_opened = True
                return
            if self._tag_buf == _CURIOSITY_OPEN_TAG:
                self._state = "in_curiosity"
                self._tag_buf = ""
                self.curiosity_active = True
                self.curiosity_seen = True
                update.curiosity_opened = True
                return
            if any(tag.startswith(self._tag_buf) for tag in _ALL_OPEN_TAGS):
                return
            self._flush_buf_to("answer", update)
            self._state = "normal"
            return

        if self._state == "in_think":
            if ch == "<":
                self._state = "tag_close_think"
                self._tag_buf = "<"
            else:
                self.thinking_text += ch
                update.thinking_text += ch
            return

        if self._state == "tag_close_think":
            self._tag_buf += ch
            # Nested <think> open while already thinking — go DEEPER, don't exit.
            if self._tag_buf in _THINK_OPEN_TAGS:
                self._think_depth += 1
                self._state = "in_think"
                self._tag_buf = ""
                return
            if self._tag_buf in _THINK_CLOSE_TAGS:
                self._tag_buf = ""
                self._think_depth -= 1
                if self._think_depth <= 0:
                    # True outermost close — only NOW does the answer lane begin.
                    self._think_depth = 0
                    self._state = "normal"
                    self.thinking_active = False
                    update.thinking_closed = True
                else:
                    self._state = "in_think"
                return
            # Still a prefix of a think OPEN or CLOSE tag — keep buffering so a
            # nested <think> isn't flushed as content before it completes.
            if any(
                tag.startswith(self._tag_buf)
                for tag in _THINK_OPEN_TAGS + _THINK_CLOSE_TAGS
            ):
                return
            self._flush_buf_to("thinking", update)
            self._state = "in_think"
            return

        if self._state == "in_acu":
            if ch == "<":
                self._state = "tag_close_acu"
                self._tag_buf = "<"
            else:
                self.acu_text += ch
                update.acu_text += ch
            return

        if self._state == "tag_close_acu":
            self._tag_buf += ch
            if self._tag_buf == _ACU_CLOSE_TAG:
                self._state = "normal"
                self._tag_buf = ""
                self.acu_active = False
                update.acu_closed = True
                return
            if _ACU_CLOSE_TAG.startswith(self._tag_buf):
                return
            self._flush_buf_to("acu", update)
            self._state = "in_acu"
            return

        if self._state == "in_tool":
            if ch == "<":
                self._state = "tag_close_tool"
                self._tag_buf = "<"
            else:
                self.tool_call_text += ch
                update.tool_call_text += ch
            return

        if self._state == "tag_close_tool":
            self._tag_buf += ch
            if self._tag_buf == _TOOL_CLOSE_TAG:
                self._state = "normal"
                self._tag_buf = ""
                self.tool_call_active = False
                update.tool_call_closed = True
                return
            if _TOOL_CLOSE_TAG.startswith(self._tag_buf):
                return
            self._flush_buf_to("tool", update)
            self._state = "in_tool"
            return

        if self._state == "in_axes":
            if ch == "<":
                self._state = "tag_close_axes"
                self._tag_buf = "<"
            else:
                self.axes_text += ch
                update.axes_text += ch
            return

        if self._state == "tag_close_axes":
            self._tag_buf += ch
            if self._tag_buf == _AXES_CLOSE_TAG:
                self._state = "normal"
                self._tag_buf = ""
                self.axes_active = False
                update.axes_closed = True
                return
            if _AXES_CLOSE_TAG.startswith(self._tag_buf):
                return
            self._flush_buf_to("axes", update)
            self._state = "in_axes"
            return

        if self._state == "in_intent":
            if ch == "<":
                self._state = "tag_close_intent"
                self._tag_buf = "<"
            else:
                self.intent_text += ch
                update.intent_text += ch
            return

        if self._state == "tag_close_intent":
            self._tag_buf += ch
            if self._tag_buf == _INTENT_CLOSE_TAG:
                self._state = "normal"
                self._tag_buf = ""
                self.intent_active = False
                update.intent_closed = True
                return
            if _INTENT_CLOSE_TAG.startswith(self._tag_buf):
                return
            self._flush_buf_to("intent", update)
            self._state = "in_intent"
            return

        if self._state == "in_tool_evidence":
            if ch == "<":
                self._state = "tag_close_tool_evidence"
                self._tag_buf = "<"
            else:
                self.tool_evidence_text += ch
                update.tool_evidence_text += ch
            return

        if self._state == "tag_close_tool_evidence":
            self._tag_buf += ch
            if self._tag_buf == _TOOL_EVIDENCE_CLOSE_TAG:
                self._state = "normal"
                self._tag_buf = ""
                self.tool_evidence_active = False
                update.tool_evidence_closed = True
                return
            if _TOOL_EVIDENCE_CLOSE_TAG.startswith(self._tag_buf):
                return
            self._flush_buf_to("tool_evidence", update)
            self._state = "in_tool_evidence"

        if self._state == "in_bearing_update":
            if ch == "<":
                self._state = "tag_close_bearing_update"
                self._tag_buf = "<"
            else:
                self.bearing_update_text += ch
                update.bearing_update_text += ch
            return

        if self._state == "tag_close_bearing_update":
            self._tag_buf += ch
            if self._tag_buf == _BEARING_UPDATE_CLOSE_TAG:
                self._state = "normal"
                self._tag_buf = ""
                self.bearing_update_active = False
                update.bearing_update_closed = True
                return
            if _BEARING_UPDATE_CLOSE_TAG.startswith(self._tag_buf):
                return
            self._flush_buf_to("bearing_update", update)
            self._state = "in_bearing_update"

        if self._state == "in_curiosity":
            if ch == "<":
                self._state = "tag_close_curiosity"
                self._tag_buf = "<"
            else:
                self.curiosity_text += ch
                update.curiosity_text += ch
            return

        if self._state == "tag_close_curiosity":
            self._tag_buf += ch
            if self._tag_buf == _CURIOSITY_CLOSE_TAG:
                self._state = "normal"
                self._tag_buf = ""
                self.curiosity_active = False
                update.curiosity_closed = True
                return
            if _CURIOSITY_CLOSE_TAG.startswith(self._tag_buf):
                return
            self._flush_buf_to("curiosity", update)
            self._state = "in_curiosity"

    def finalize(self, close_open: bool = True) -> AssistantDisplayUpdate:
        update = AssistantDisplayUpdate()
        # A held partial sentinel that never completed is genuine content (the
        # stream ended) — feed it through so it isn't silently dropped.
        if self._pending_sentinel:
            held = self._pending_sentinel
            self._pending_sentinel = ""
            for ch in held:
                self._consume_char(ch, update)
        # Spill any in-flight partial-tag buffer into the appropriate lane so
        # truncated streams don't lose characters.
        if self._tag_buf:
            if self._state == "tag_open":
                self._flush_buf_to("answer", update)
                self._state = "normal"
            elif self._state == "tag_close_think":
                self._flush_buf_to("thinking", update)
                self._state = "in_think"
            elif self._state == "tag_close_acu":
                self._flush_buf_to("acu", update)
                self._state = "in_acu"
            elif self._state == "tag_close_tool":
                self._flush_buf_to("tool", update)
                self._state = "in_tool"
            elif self._state == "tag_close_axes":
                self._flush_buf_to("axes", update)
                self._state = "in_axes"
            elif self._state == "tag_close_intent":
                self._flush_buf_to("intent", update)
                self._state = "in_intent"
            elif self._state == "tag_close_tool_evidence":
                self._flush_buf_to("tool_evidence", update)
                self._state = "in_tool_evidence"
            elif self._state == "tag_close_bearing_update":
                self._flush_buf_to("bearing_update", update)
                self._state = "in_bearing_update"
            elif self._state == "tag_close_curiosity":
                self._flush_buf_to("curiosity", update)
                self._state = "in_curiosity"
        if close_open and self.thinking_active:
            self.thinking_active = False
            self._state = "normal"
            update.thinking_closed = True
        if close_open and self.acu_active:
            self.acu_active = False
            self._state = "normal"
            update.acu_closed = True
        if close_open and self.tool_call_active:
            self.tool_call_active = False
            self._state = "normal"
            update.tool_call_closed = True
        if close_open and self.axes_active:
            self.axes_active = False
            self._state = "normal"
            update.axes_closed = True
        if close_open and self.intent_active:
            self.intent_active = False
            self._state = "normal"
            update.intent_closed = True
        if close_open and self.tool_evidence_active:
            self.tool_evidence_active = False
            self._state = "normal"
            update.tool_evidence_closed = True
        if close_open and self.bearing_update_active:
            self.bearing_update_active = False
            self._state = "normal"
            update.bearing_update_closed = True
        if close_open and self.curiosity_active:
            self.curiosity_active = False
            self._state = "normal"
            update.curiosity_closed = True
        return update

    def snapshot(self) -> AssistantDisplaySnapshot:
        return AssistantDisplaySnapshot(
            answer_text=self.answer_text,
            thinking_text=self.thinking_text,
            thinking_active=self.thinking_active,
            thinking_seen=self.thinking_seen,
            acu_text=self.acu_text,
            acu_seen=self.acu_seen,
            tool_call_text=self.tool_call_text,
            tool_call_seen=self.tool_call_seen,
            axes_text=self.axes_text,
            axes_seen=self.axes_seen,
            intent_text=self.intent_text,
            intent_seen=self.intent_seen,
            tool_evidence_text=self.tool_evidence_text,
            tool_evidence_seen=self.tool_evidence_seen,
            bearing_update_text=self.bearing_update_text,
            bearing_update_seen=self.bearing_update_seen,
            curiosity_text=self.curiosity_text,
            curiosity_seen=self.curiosity_seen,
        )

    @classmethod
    def from_text(cls, text: str, close_open: bool = True) -> "AssistantStreamNormalizer":
        parser = cls()
        parser.consume(text or "")
        parser.finalize(close_open=close_open)
        return parser


@dataclass
class AssistantTurnBox:
    sessions: ChatSessionManager
    active_assistant_index: int | None = None
    rewrite_assistant_index: int | None = None
    active_assistant_started: bool = False
    active_assistant_token_count: int = 0
    pending_tool_results: list[str] = field(default_factory=list)
    tool_followup_target_index: int | None = None
    last_task_id: str = ""
    pending_archive_save_task_id: str | None = None
    display_stream: AssistantStreamNormalizer = field(default_factory=AssistantStreamNormalizer)

    @property
    def current_session(self) -> dict:
        return self.sessions.current

    def bind_session(self, session: dict) -> None:
        self.sessions.set_current(session)
        self.reset_runtime_state()
        self.clear_tool_followup_state()
        self.pending_archive_save_task_id = None

    def reset_runtime_state(self) -> None:
        self.active_assistant_index = None
        self.rewrite_assistant_index = None
        self.active_assistant_started = False
        self.active_assistant_token_count = 0
        self.display_stream.reset()

    def clear_tool_followup_state(self) -> None:
        self.pending_tool_results.clear()
        self.tool_followup_target_index = None

    def _touch_session(self) -> str:
        stamp = self.sessions.now_iso()
        self.current_session["updated_at"] = stamp
        return stamp

    def _renumber_messages(self) -> None:
        self.sessions._renumber_messages()

    def stream_target_index(self) -> int | None:
        if self.rewrite_assistant_index is not None:
            return self.rewrite_assistant_index
        return self.active_assistant_index

    def start_new_stream(self) -> int:
        self.active_assistant_started = True
        self.active_assistant_token_count = 0
        self.rewrite_assistant_index = None
        self.display_stream.reset()
        self.active_assistant_index = self.sessions.add_message("assistant", "")
        return self.active_assistant_index

    def start_rewrite_stream(self, target_index: int | None) -> None:
        self.rewrite_assistant_index = target_index
        self.active_assistant_started = False
        self.active_assistant_token_count = 0
        self.display_stream.reset()
        messages = self.current_session.get("messages", [])
        if target_index is not None and 0 <= target_index < len(messages):
            existing_text = str(messages[target_index].get("text", ""))
            self.display_stream = AssistantStreamNormalizer.from_text(existing_text, close_open=False)

    def append_token(self, token: str) -> None:
        if not token:
            return
        self.active_assistant_token_count += 1
        self.sessions.append_assistant_token(token, self.stream_target_index())

    def consume_display_chunk(self, token: str) -> AssistantDisplayUpdate:
        return self.display_stream.consume(token)

    def finalize_display_stream(self) -> AssistantDisplayUpdate:
        return self.display_stream.finalize(close_open=True)

    def build_display_snapshot(self, text: str, close_open: bool = True) -> AssistantDisplaySnapshot:
        return AssistantStreamNormalizer.from_text(text or "", close_open=close_open).snapshot()

    def note_finished(self, task_id: str) -> None:
        self.pending_archive_save_task_id = str(task_id or self.last_task_id or "")

    def cleanup_empty_assistant_if_needed(self) -> bool:
        removed = self.sessions.cleanup_empty_assistant_if_needed(
            self.active_assistant_index,
            self.active_assistant_started,
            self.active_assistant_token_count,
        )
        if removed:
            self.reset_runtime_state()
        return removed

    def rewrite_assistant_text(self, index: int, text: str) -> None:
        from ui.pages import session_tree
        if session_tree.active():
            stamp = self._touch_session()
            session_tree.tree_fork_assistant_rewrite(self.current_session, index, text,
                                                     now=stamp)
            return
        messages = self.current_session.get("messages", [])
        if not (0 <= index < len(messages)):
            return
        message = messages[index]
        message["text"] = text
        stamp = self._touch_session()
        message["time"] = stamp

    def truncate_from(self, index: int) -> bool:
        from ui.pages import session_tree
        if session_tree.active():
            if not session_tree.tree_prune_from(self.current_session, index):
                return False
            self._touch_session()
            self.clear_tool_followup_state()
            self.pending_archive_save_task_id = None
            self.reset_runtime_state()
            return True
        messages = self.current_session.get("messages", [])
        if index < 0 or index >= len(messages):
            return False
        del messages[index:]
        self._renumber_messages()
        self._touch_session()
        self.clear_tool_followup_state()
        self.pending_archive_save_task_id = None
        self.reset_runtime_state()
        return True

    def delete_from_index(self, index: int) -> bool:
        return self.truncate_from(index)

    def edit_from_index(self, index: int) -> str | None:
        messages = self.current_session.get("messages", [])
        if index < 0 or index >= len(messages):
            return None
        message = messages[index]
        if message.get("role") != "user":
            return None
        return str(message.get("text", ""))

    def commit_edit_from_index(self, index: int, text: str) -> str | None:
        from ui.pages import session_tree
        if session_tree.active():
            stamp = self._touch_session()
            out = session_tree.tree_fork_user_edit(self.current_session, index, text,
                                                   now=stamp)
            if out is not None:
                self.clear_tool_followup_state()
                self.pending_archive_save_task_id = None
                self.reset_runtime_state()
            return out
        messages = self.current_session.get("messages", [])
        if index < 0 or index >= len(messages):
            return None
        message = messages[index]
        if message.get("role") != "user":
            return None
        updated_text = str(text or "").strip()
        if not updated_text:
            return None
        message["text"] = updated_text
        stamp = self._touch_session()
        message["time"] = stamp
        del messages[index + 1 :]
        self._renumber_messages()
        self.clear_tool_followup_state()
        self.pending_archive_save_task_id = None
        self.reset_runtime_state()
        return updated_text

    def regen_from_index(self, index: int) -> str | None:
        from ui.pages import session_tree
        if session_tree.active():
            prompt = session_tree.tree_regen_reset(self.current_session, index)
            if prompt is not None:
                self.clear_tool_followup_state()
                self.pending_archive_save_task_id = None
                self.reset_runtime_state()
            return prompt
        messages = self.current_session.get("messages", [])
        if index < 0 or index >= len(messages):
            return None
        message = messages[index]
        if message.get("role") != "assistant":
            return None
        prompt = None
        for offset in range(index - 1, -1, -1):
            candidate = messages[offset]
            if candidate.get("role") == "user":
                prompt = str(candidate.get("text", ""))
                break
        if not prompt:
            return None
        self.truncate_from(index)
        return prompt

    def regen_last_assistant(self) -> str | None:
        messages = self.current_session.get("messages", [])
        if not messages or messages[-1].get("role") != "assistant":
            return None
        return self.regen_from_index(len(messages) - 1)
