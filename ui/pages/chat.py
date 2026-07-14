import html
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
    QLineEdit, QPushButton, QLabel, QFileDialog,
    QListWidget, QListWidgetItem,
    QMessageBox, QSizePolicy, QFrame,
    QComboBox
)
from PySide6.QtCore import Signal, Qt, QTimer, QDateTime, QThread, QObject

import urllib.error
import urllib.request

from core.state import SystemStatus
import core.style as _s  # dynamic theme bridge
from ui.components.atoms import MonoGroupBox, MonoButton, MonoSlider
from ui.components.message_widget import MessageWidget
from ui.components.tool_bubbles import ToolCallBubble, ToolResultBubble
from core.llm_config import DEFAULT_CONFIG, MASTER_PROMPT, load_config, save_config
from core.llm_prompt import load_master_prompt
from core.paths import ARCHIVE_DIR
from core.cmd_parser import extract_commands, process_response
from core.incomplete_action import detect_incomplete_action, INCOMPLETE_ACTION_PATTERNS
from core.skill_runtime import (
    ToolResultCache,
    ToolExecutionContext,
    STREAMING_PREEXEC_TOOLS,
    execute_tool_call_enveloped,
    SpawnBudget,
    L1_PRINCIPAL_TOOLS,
    derive_child_context,
)
from core.history_search import search_archives
from core.skill_registry import canonical_tool_name
from ui.conversation_surface import ConversationSurface as ConversationSurfaceWidget
from ui.panels import (
    ActionReviewPanel,
    ArchiveBrowserPanel,
    AuditLogPanel,
    GenerationTracePanel,
    ModelConfigPanel,
    QuestionPanel,
)
from ui.pages.chat_archive import ChatArchiveManager
from ui.pages.assistant_turn_box import AssistantTurnBox, AssistantStreamNormalizer
from ui.pages.chat_session import ChatSessionManager
from core.acu_store import ACUStore
from core.chat_finalize import finalize_assistant_turn


def build_tool_followup_prompt(tool_results: list[str]) -> str:
    import re as _re
    blocks = []
    partial_hints: list[str] = []

    for idx, result in enumerate(tool_results, start=1):
        result = str(result).strip()
        if not result:
            continue
        blocks.append(f"[tool_result_{idx}]\n{result}")
        # Detect [PARTIAL — N chars remaining. To continue: tool(...)]
        m = _re.search(
            r"\[PARTIAL — ([\d,]+) chars remaining\. To continue: (\w+\([^)]+\))\]",
            result,
        )
        if m:
            remaining = m.group(1)
            continuation = m.group(2)
            partial_hints.append(
                f"  • tool_result_{idx} is PARTIAL ({remaining} chars not yet read)."
                f" If the missing content is needed to answer the question, call"
                f" {continuation} before responding. Only paginate if it actually matters."
            )

    joined = "\n\n".join(blocks).strip()
    partial_section = ""
    if partial_hints:
        partial_section = (
            "\n\nPartial results detected:\n"
            + "\n".join(partial_hints)
            + "\n"
        )

    return f"""
Tool results:
{joined}
{partial_section}
Continue from where you left off. Do not restate the user's original request. Do not re-analyze what was already decided in your prior thinking. The BEARING block (if present) carries your active goal and next move — use it as your anchor, not the raw user message.

If another tool call is needed, emit exactly one tool_call envelope and no other text.
If done, start with [TOOL_LOOP_DONE] then give the final answer.
"""


# ─── Message origin tags ───────────────────────────────────────────────
# Stored as msg["origin"] on session-row dicts. Lets downstream code
# (canonical_log, telemetry, render, routing) branch on *how* a message
# entered the session without inferring from role + side-channel state.
ORIGIN_UI_USER = "ui_user"                # user typed in the chat input
ORIGIN_CONNECT_AGENT = "connect_agent"    # arrived via the CONNECT bridge (:7821)
ORIGIN_AGENT_REPLY = "agent_reply"        # response from an @mentioned external agent
ORIGIN_SYSTEM_COMMAND = "system_command"  # locally-injected command receipt (e.g. /think)
ORIGIN_TOOL_INJECTION = "tool_injection"  # tool_call / tool_result rows inserted post-stream
ORIGIN_PIPELINE = "pipeline"  # assistant bubble produced by a Monoline flow run


# Dispatch sources the ENGINE issues on its own to keep the tool loop going —
# as opposed to a user-initiated send/edit/regen/agent/update. A pending user
# Stop must suppress these (Kernel Contract v2 §7: "STOP dominance"). Adding a
# new autonomous re-dispatch source? Add its `source=` string here too, or it
# will fire after Stop.
_AUTONOMOUS_CONTINUATION_SOURCES = frozenset({
    "tool_followup",
    "tool_followup_retry",
    "tool_parse_retry",
    "incomplete_action_nudge",
    "non_convergence_nudge",
})


def _is_autonomous_continuation(source: str) -> bool:
    """True when `source` is an engine-driven loop continuation (retry/nudge/
    followup) that a user Stop must suppress; False for user-initiated dispatches."""
    return (source or "") in _AUTONOMOUS_CONTINUATION_SOURCES


def _normalize_base_url(base_url: str) -> str:
    return (base_url or "").strip().rstrip("/")


def _models_url(base_url: str) -> str:
    base = _normalize_base_url(base_url)
    if base.endswith("/models"):
        return base
    if base.endswith("/v1"):
        return f"{base}/models"
    return f"{base}/v1/models"


def _extract_ctx_length_from_model_item(item: dict) -> int | None:
    """Return a conservative effective context length from a /models item."""
    if not isinstance(item, dict):
        return None
    meta = item.get("meta", {}) if isinstance(item.get("meta"), dict) else {}
    candidates: list[int] = []
    for source in (meta, item):
        for key in ("n_ctx", "context_length", "max_context_length", "n_ctx_train"):
            raw = source.get(key)
            if raw is None:
                continue
            try:
                value = int(raw)
            except (TypeError, ValueError):
                continue
            if value > 0:
                candidates.append(value)
    if not candidates:
        return None
    return min(candidates)


class ModelListLoader(QThread):
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, base_url: str, api_key: str | None):
        super().__init__()
        self.base_url = base_url
        self.api_key = api_key or ""

    def run(self):
        try:
            if not self.base_url:
                raise RuntimeError("Missing API base URL.")
            url = _models_url(self.base_url)
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            req = urllib.request.Request(url, headers=headers, method="GET")
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    body = resp.read()
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="ignore")
                raise RuntimeError(f"HTTP {exc.code}: {body or exc.reason}") from exc
            except urllib.error.URLError as exc:
                reason = getattr(exc, "reason", "") or str(exc)
                raise RuntimeError(f"URLError: {reason}") from exc
            data = json.loads(body)
            items = data.get("data", []) if isinstance(data, dict) else []
            models = []
            ctx_length = None
            for item in items:
                if isinstance(item, dict) and "id" in item:
                    models.append(str(item["id"]))
                    if ctx_length is None:
                        ctx_length = _extract_ctx_length_from_model_item(item)
                elif isinstance(item, str):
                    models.append(item)
            models = sorted({m for m in models if m})
            self.finished.emit({"models": models, "ctx_length": ctx_length})
        except Exception as e:
            msg = str(e).strip()
            if not msg:
                msg = repr(e)
            self.error.emit(msg)


# ── Slash-command catalog ───────────────────────────────────────────────
# Single source of truth for the command-picker popup. Add new commands
# here and the picker, hint text, and discoverability all stay in sync.
# Order is the order they appear in the picker.
SLASH_COMMANDS: tuple[tuple[str, str, str], ...] = (
    ("/think",        "on | off | toggle | status",           "toggle native <think> for this model"),
    ("/prompt",       "<name> [<name2> ...] | once <...> | clear", "inject prompt scaffolds (composable, opt-in)"),
    ("/monothink",    "on | off | once | status",             "self-evolving reasoning scaffold (separate from /prompt)"),
    ("/skill-creator", "[topic]",                             "create a skill ensemble → saves to /prompt"),
    ("/rating",       "<0-100> #<tag> [note]",                "rate the latest assistant turn (Layer D outcome)"),
    ("/trace",        "last | errors | <turn_id>",            "inspect recent turn traces"),
    ("/pipeline",     "last | faults | <turn_id>",            "inspect Turn Pipeline events (Layer E)"),
    ("/agent",        "on | off  approval:yes  thinking:no",  "agent mode + flags"),
    ("/approve",      "",                                     "approve the pending world action"),
    ("/reject",       "",                                     "reject the pending world action"),
    ("/act",          "<json>",                               "execute a world action by JSON literal"),
)

# Progressive arg hints: when N tokens are typed after the command,
# show the hint at index min(N, len-1). Omitted commands show full args always.
_ARG_STAGES: dict[str, list[str]] = {
    "/rating":       ["<0-100> #<tag> [note]", "#<tag> [#<tag> ...] [note]  — failure tag(s) drive monothink"],
    "/trace":        ["last | errors | <turn_id>"],
    "/pipeline":     ["last | faults | <turn_id>"],
    "/think":        ["on | off | toggle | status"],
    "/monothink":    ["on | off | once | status"],
    "/skill-creator": ["[topic]  — describe what kind of skill"],
    "/agent":        ["on | off", "approval:yes|no", "thinking:yes|no"],
    "/act":          ["<json>  — paste a world action literal"],
}


def _engine_is_busy(world_state) -> bool:
    """INV-C Arm 2: True if any engine reports a RUNNING/generating status.
    None world_state => assume free (unit-test / headless convention)."""
    if world_state is None:
        return False
    try:
        engines = (world_state.snapshot() or {}).get("engines", {}) or {}
    except Exception:
        return False
    busy = {"running", "generating", "streaming"}
    return any(str(e.get("status", "")).strip().lower() in busy
              for e in engines.values() if isinstance(e, dict))


class _SpawnWorker(QObject):
    """Runs the subagent atom OFF the Qt UI thread, emits sig_subagent_done with
    the fenced result so the L1 turn folds it back via the followup loop."""
    sig_subagent_done = Signal(str)  # fenced [SUBAGENT_RESULT ...] string

    def __init__(self, cmd: dict, child_level: int, parent_turn_id, allowed_tools,
                 should_cancel, spawn_budget, is_busy):
        super().__init__()
        self._cmd = dict(cmd)
        self._child_level = child_level
        self._parent_turn_id = parent_turn_id
        self._allowed_tools = allowed_tools
        self._should_cancel = should_cancel
        self._spawn_budget = spawn_budget
        self._is_busy = is_busy

    def run(self) -> None:
        from core.subagent import run_subagent
        from core.llm_config import load_config
        prompt = str(self._cmd.get("prompt", "")).strip()
        raw_messages = self._cmd.get("messages")
        messages = []
        if isinstance(raw_messages, list):
            for item in raw_messages:
                if isinstance(item, dict):
                    role = str(item.get("role", "")).strip().lower()
                    content = str(item.get("content", ""))
                    if role in {"system", "user", "assistant"} and content.strip():
                        messages.append({"role": role, "content": content})
        if not messages and prompt:
            messages.append({"role": "user", "content": prompt})
        frame = str(self._cmd.get("frame", f"L{self._child_level}")).strip() or f"L{self._child_level}"
        try:
            res = run_subagent(
                messages, load_config(), level=self._child_level, frame=frame,
                parent_turn_id=self._parent_turn_id, allowed_tools=self._allowed_tools,
                should_cancel=self._should_cancel, max_followups=0,
                spawn_budget=self._spawn_budget, is_busy=self._is_busy)
            self.sig_subagent_done.emit(res.fenced)
        except Exception as exc:
            self.sig_subagent_done.emit(
                f"[SUBAGENT_RESULT ok=false]\n[spawn error - {exc}]\n[/SUBAGENT_RESULT]")


class _PipelineWorker(QObject):
    """Runs a Monoline flow OFF the Qt UI thread (INV-OFFTHREAD). Marshals the normalized
    RunEvent stream + the final answer back via queued Qt signals (the _SpawnWorker pattern)."""
    sig_run_event = Signal(object)                      # RunStarted | BlockFinished | RunFinished
    sig_pipeline_done = Signal(str)                     # final OUTPUT-port answer
    sig_pipeline_error = Signal(str)
    sig_pipeline_stopped = Signal()                     # user STOP — clean halt, no message

    def __init__(self, workflow, payload, *, parent_turn_id, spawn_budget,
                 should_cancel, is_busy):
        super().__init__()
        self._workflow = workflow
        self._payload = payload
        self._parent_turn_id = parent_turn_id
        self._spawn_budget = spawn_budget
        self._should_cancel = should_cancel
        self._is_busy = is_busy

    def run(self) -> None:
        try:
            from engine.monoline_bridge import run_monoline_world, summarize_run_failure  # LAZY import (INV-#0)
        except Exception as exc:
            self.sig_pipeline_error.emit(f"[monoline plugin unavailable: {exc}]")
            return
        try:
            prompt = self._payload.get("prompt", "") if isinstance(self._payload, dict) else str(self._payload)
            run = run_monoline_world(
                self._workflow, user_input=str(prompt),
                parent_turn_id=self._parent_turn_id, spawn_budget=self._spawn_budget,
                should_cancel=self._should_cancel, is_busy=self._is_busy,
                on_step=None, should_stop=self._should_cancel,
                on_event=lambda ev: self.sig_run_event.emit(ev))
            # A user STOP is a clean halt, not a failure: take the silent stopped path (the
            # RunView already shows "stopped" via the RunFinished(stopped=True) event) instead of
            # surfacing the runtime's stop sentinel as a red error.
            try:
                _stopped = bool(self._should_cancel and self._should_cancel())
            except Exception:
                _stopped = False
            if _stopped:
                self.sig_pipeline_stopped.emit()
                return
            failure = summarize_run_failure(run)
            if failure:
                self.sig_pipeline_error.emit(failure)
            else:
                self.sig_pipeline_done.emit(str(run.result.output or ""))
        except Exception as exc:
            self.sig_pipeline_error.emit(f"[pipeline error - {exc}]")


class PageChat(QWidget):
    sig_generate = Signal(object)
    sig_load = Signal()
    sig_unload = Signal()
    sig_stop = Signal()
    sig_sync_history = Signal(list)
    sig_set_model_path = Signal(object)
    sig_set_ctx_limit = Signal(int)
    sig_operator_loaded = Signal(str)
    sig_debug = Signal(str)
    sig_agents_changed = Signal(object)  # list[AgentRecord] — active-agents spine -> UI strip

    # These are the PREFIXES (segment before the first ':') of the `source=` strings emitted at the
    # user-initiated _dispatch_generation call sites: f"send:{txt}", f"edit:{txt}", f"agent:{name}",
    # and the bare "regen". The guard (Step 5.9) matches `source.split(":",1)[0]` against this set --
    # NOT exact equality (the real sources carry a payload suffix). Internal machinery uses tokens
    # NOT in this set ("tool_followup", "tool_followup_retry", "update", "non_convergence_nudge",
    # "incomplete_action_nudge", "tool_parse_retry", "model") so it is NEVER diverted -> INV-#1 holds.
    # "at_mention:monolith" is DELIBERATELY excluded: typing @monolith always ran Genesis and still
    # does (a user with an active Monoline flow who @-mentions gets Genesis -- intended, conservative).
    # If you rename a `source=` literal at a call site, GREP these call sites and update this set --
    # the INV-#1 stub test supplies its own source literal and CANNOT catch source-vocabulary drift.
    _MONOLINE_ENTRY_SOURCES = frozenset({"send", "agent", "edit", "regen"})

    def __init__(self, state, ui_bridge, bridge=None, guard=None):
        super().__init__()
        self.state = state
        self.ui_bridge = ui_bridge
        self._bridge = bridge
        self._guard = guard
        # Optional VisionArtifactBridge; assigned by the addon factory from
        # AddonContext.services. None on the early-init path before the factory
        # runs; the generate_image executor falls back to text-only output then.
        self._vision_artifact_bridge = None
        self.config = load_config()
        self._sessions = ChatSessionManager(load_master_prompt())
        self._assistant_box = AssistantTurnBox(self._sessions)
        self._archive = ChatArchiveManager(ARCHIVE_DIR, self._sessions)
        self._archive.archive_dir.mkdir(parents=True, exist_ok=True)
        self._active_assistant_index = None
        self._rewrite_assistant_index = None
        self._active_widget: MessageWidget | None = None
        self._active_item = None
        self._last_status = None
        self._is_running = False
        self._is_model_loaded = False
        self._pending_update_text = None
        self._awaiting_update_restart = False
        self._pending_tool_results: list[str] = []
        self._tool_followup_target_index: int | None = None
        self._tool_followup_retries: int = 0
        self._MAX_TOOL_RETRIES: int = 3
        self._tool_parse_retries: int = 0
        self._MAX_TOOL_PARSE_RETRIES: int = 2
        self._tool_followup_depth: int = 0
        self._MAX_TOOL_FOLLOWUPS: int = 10
        self._MAX_TOOL_FOLLOWUP_CHARS: int = 24000
        # Non-convergence recovery: bound the "only reasoning, no answer/action"
        # re-prompt to one shot so a model that keeps spinning gives up cleanly.
        self._non_convergence_retries: int = 0
        self._MAX_NON_CONVERGENCE_RETRIES: int = 1
        self._tool_cancel_requested: bool = False
        self._tool_loop_active: bool = False
        self._editing_user_index: int | None = None
        self._update_trace_state = None
        self._update_token_count = 0
        self._update_progress_index = 0
        self._stream_debug_chunk_count = 0
        self._stream_debug_char_count = 0
        self._config_dirty = False
        self._last_task_id = ""
        self._spawn_workers: list = []      # live _SpawnWorker refs (anti-GC)
        self._active_agents: dict = {}      # id(worker) -> AgentRecord (the spine source)
        self._spawn_budget = None           # set per L1 turn before process_response
        from core.workflow_registry import WorkflowRegistry
        self._workflow_registry = WorkflowRegistry()
        _ws = getattr(self.state, "world_state", None)
        if _ws is not None:
            self._workflow_registry.bind_world_state(_ws)
        self._pipeline_workers: list = []  # live _PipelineWorker refs (anti-GC)
        self._pending_archive_save_task_id: str | None = None
        # When user clicks Edit/Regen/Delete while a generation is running, we STOP first,
        # then apply the mutation on the next READY.
        self._pending_mutation = None  # type: ignore[assignment]
        self._auto_scroll_on_height_change = False
        self._token_batch: list[str] = []
        self._token_batch_timer = QTimer(self)
        self._token_batch_timer.setSingleShot(True)
        self._token_batch_timer.setInterval(33)
        self._token_batch_timer.timeout.connect(self._flush_token_batch)

        # Tool result cache — warms mid-stream, hits at READY (features 4+5)
        self._tool_cache = ToolResultCache()
        # Raw stream accumulator for mid-stream tool pre-execution
        self._stream_raw: str = ""

        # ACU (Acatalepsy) memory store
        self._acu_store = ACUStore()

        # Agent mode state
        self._agent_enabled = False
        self._agent_approval = True       # ask before running tools
        self._agent_workspace = ""        # "" = use cwd
        self._agent_thinking = True       # model thinking on by default

        self._config_panel = ModelConfigPanel(state, ui_bridge, self.config, self)
        self._trace_panel = GenerationTracePanel(self)
        self._archive_panel = ArchiveBrowserPanel(ARCHIVE_DIR, self._archive, self)
        self._archive_panel.set_session_provider(self.current_session_data)
        self._audit_panel = AuditLogPanel(getattr(state, "world_state", None), self)
        self._action_review_panel = ActionReviewPanel(self)
        self._question_panel = QuestionPanel(self)
        from ui.panels.reasoning_tree import ReasoningTreePanel
        self._reasoning_tree_panel = ReasoningTreePanel()
        self._reasoning_tree_panel.bind_controller(self)
        # Pending ask_user state: only one question active at a time. Set when
        # the executor's on_ask_user callback fires; cleared when the user
        # answers, dismisses, or the session resets.
        self._pending_ask_user: dict | None = None

        self._config_panel.sig_model_payload_changed.connect(self.sig_set_model_path.emit)
        self._config_panel.sig_load_requested.connect(self.sig_load.emit)
        self._config_panel.sig_unload_requested.connect(self.sig_unload.emit)
        self._config_panel.sig_ctx_limit_changed.connect(self.sig_set_ctx_limit.emit)
        self._config_panel.sig_ctx_limit_changed.connect(self._update_state_ctx_limit)
        self._config_panel.sig_trace_requested.connect(self.trace_line)
        self._archive_panel.sig_new_requested.connect(self._start_new_session)
        self._archive_panel.sig_clear_requested.connect(self._on_archive_clear_request)
        self._archive_panel.sig_session_loaded.connect(self._load_archive_session)
        self._archive_panel.sig_current_archive_deleted.connect(self._reset_session_after_deleted_archive)
        self._archive_panel.sig_summarize_requested.connect(self._fill_input_for_summarize)
        self._action_review_panel.sig_approved.connect(
            lambda action: self._approve_pending(action, source="user/approve")
        )
        self._action_review_panel.sig_rejected.connect(
            lambda action: self._reject_pending(action, source="user/reject")
        )
        self._question_panel.sig_answered.connect(self._on_ask_user_answered)
        self._question_panel.sig_dismissed.connect(self._on_ask_user_dismissed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # === MODEL LOADER (lives in CONTROL tab) ===
        grp_load = MonoGroupBox("MODEL LOADER")
        grp_load.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        lbl_engine = QLabel("ENGINE")
        lbl_engine.setStyleSheet(
            f"color: {_s.FG_DIM}; font-size: 9px; font-weight: bold; background: transparent;"
        )
        self.engine_combo = QComboBox()
        self.engine_combo.addItem("GGUF (API)", "gguf_api")
        self.engine_combo.addItem("GGUF (llama.cpp)", "gguf")
        self.engine_combo.addItem("Model (API)", "openai")
        # vLLM / SGLang dropped from the menu (both speak OpenAI-compatible
        # protocol — "Model (API)" + custom API BASE covers them).
        self.engine_combo.setFixedHeight(24)
        self.engine_combo.setStyleSheet(
            f"""
            QComboBox {{
                background: {_s.BG_INPUT}; color: {_s.FG_TEXT};
                border: 1px solid {_s.BORDER_LIGHT}; padding: 2px 8px;
                font-size: 10px; font-weight: bold; border-radius: 2px;
            }}
            QComboBox:hover {{ border: 1px solid {_s.ACCENT_PRIMARY}; }}
            QComboBox::drop-down {{ border: none; width: 18px; }}
            QComboBox::down-arrow {{ image: none; border: none; }}
            QComboBox QAbstractItemView {{
                background: {_s.BG_INPUT}; color: {_s.FG_TEXT};
                border: 1px solid {_s.BORDER_LIGHT};
                selection-background-color: {_s.BG_BUTTON_HOVER};
                selection-color: {_s.ACCENT_PRIMARY};
            }}
        """
        )
        row_engine = QHBoxLayout()
        row_engine.addWidget(lbl_engine)
        row_engine.addWidget(self.engine_combo, 1)
        self.path_display = QLineEdit()
        self.path_display.setReadOnly(True)
        self.path_display.setPlaceholderText("No model selected")
        self.path_display.setObjectName("path_display")
        self.btn_browse = MonoButton("...")
        self.btn_browse.setFixedWidth(30)
        self.btn_browse.clicked.connect(self.pick_file)
        self.gguf_row = QWidget()
        row_file = QHBoxLayout(self.gguf_row)
        row_file.setContentsMargins(0, 0, 0, 0)
        row_file.addWidget(self.path_display)
        row_file.addWidget(self.btn_browse)
        self.remote_panel = QWidget()
        remote_layout = QVBoxLayout(self.remote_panel)
        remote_layout.setContentsMargins(0, 0, 0, 0)
        remote_layout.setSpacing(6)
        self.lbl_base = QLabel("API BASE")
        self.lbl_base.setStyleSheet(f"color: {_s.FG_DIM}; font-size: 9px; font-weight: bold; background: transparent;")
        self.api_base_input = QLineEdit()
        self.api_base_input.setPlaceholderText("http://localhost:8000/v1")
        self.lbl_model = QLabel("MODEL")
        self.lbl_model.setStyleSheet(f"color: {_s.FG_DIM}; font-size: 9px; font-weight: bold; background: transparent;")
        self.api_model_input = QLineEdit()
        self.api_model_input.setPlaceholderText("model-id")
        self.api_model_combo = QComboBox()
        self.api_model_combo.setVisible(False)
        self.api_model_combo.setFixedHeight(24)
        self.api_model_combo.setStyleSheet(
            f"""
            QComboBox {{
                background: {_s.BG_INPUT}; color: {_s.FG_TEXT};
                border: 1px solid {_s.BORDER_LIGHT}; padding: 2px 8px;
                font-size: 10px; font-weight: bold; border-radius: 2px;
            }}
            QComboBox:hover {{ border: 1px solid {_s.ACCENT_PRIMARY}; }}
            QComboBox::drop-down {{ border: none; width: 18px; }}
            QComboBox::down-arrow {{ image: none; border: none; }}
            QComboBox QAbstractItemView {{
                background: {_s.BG_INPUT}; color: {_s.FG_TEXT};
                border: 1px solid {_s.BORDER_LIGHT};
                selection-background-color: {_s.BG_BUTTON_HOVER};
                selection-color: {_s.ACCENT_PRIMARY};
            }}
        """
        )
        self.btn_fetch_models = MonoButton("FETCH MODELS")
        self.btn_fetch_models.setFixedHeight(24)
        self.lbl_key = QLabel("API KEY")
        self.lbl_key.setStyleSheet(f"color: {_s.FG_DIM}; font-size: 9px; font-weight: bold; background: transparent;")
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("(optional)")
        self.api_key_input.setEchoMode(QLineEdit.Password)
        remote_layout.addWidget(self.lbl_base)
        remote_layout.addWidget(self.api_base_input)
        remote_layout.addWidget(self.lbl_model)
        remote_layout.addWidget(self.api_model_input)
        remote_layout.addWidget(self.api_model_combo)
        remote_layout.addWidget(self.btn_fetch_models)
        remote_layout.addWidget(self.lbl_key)
        remote_layout.addWidget(self.api_key_input)

        self.btn_load = MonoButton("LOAD MODEL")
        self.btn_load.clicked.connect(self.toggle_load)
        grp_load.add_layout(row_engine)
        grp_load.add_widget(self.gguf_row)
        grp_load.add_widget(self.remote_panel)
        grp_load.add_widget(self.btn_load)

        # === AI CONFIGURATION (lives in SETTINGS tab) ===
        self.s_temp = MonoSlider("Temperature", 0.0, 2.0, self.config.get("temp", 1.0))
        self.s_temp.valueChanged.connect(lambda v: self._update_config_value("temp", v))
        self.s_top = MonoSlider("Top-P", 0.0, 1.0, self.config.get("top_p", 0.95))
        self.s_top.valueChanged.connect(lambda v: self._update_config_value("top_p", v))
        self.s_top_k = MonoSlider("Top-K", 0, 200, self.config.get("top_k", 20), is_int=True)
        self.s_top_k.valueChanged.connect(lambda v: self._update_config_value("top_k", int(v)))
        self.s_min_p = MonoSlider("Min-P", 0.0, 1.0, self.config.get("min_p", 0.0))
        self.s_min_p.valueChanged.connect(lambda v: self._update_config_value("min_p", v))
        self.s_presence = MonoSlider(
            "Presence Penalty", -2.0, 2.0, self.config.get("presence_penalty", 1.5)
        )
        self.s_presence.valueChanged.connect(
            lambda v: self._update_config_value("presence_penalty", v)
        )
        self.s_repetition = MonoSlider(
            "Repetition Penalty", 0.5, 2.0, self.config.get("repetition_penalty", 1.0)
        )
        self.s_repetition.valueChanged.connect(
            lambda v: self._update_config_value("repetition_penalty", v)
        )
        self.s_tok = MonoSlider(
            "Max Tokens", 512, 131072, self.config.get("max_tokens", 2048), is_int=True
        )
        self.s_tok.valueChanged.connect(
            lambda v: self._update_config_value("max_tokens", int(v))
        )
        # Initial slider value: prefer the resolved runtime window from
        # state.ctx_limit (set by the engine on model load); fall back to any
        # persisted user override; finally to the slider minimum so an unset
        # state is visually obvious.
        _state_ctx = int(getattr(getattr(self, "state", None), "ctx_limit", 0) or 0)
        _saved_ctx = int(self.config.get("ctx_limit", 0) or 0)
        _initial_ctx = _state_ctx or _saved_ctx or 1024
        self.s_ctx = MonoSlider(
            "Context Limit", 1024, 1_048_576, _initial_ctx, is_int=True
        )
        self.s_ctx.valueChanged.connect(self._on_ctx_limit_changed)

        save_row = QHBoxLayout()
        self.lbl_config_state = QLabel("SAVED")
        self.lbl_config_state.setObjectName("lbl_config_state")
        self.btn_save_config = MonoButton("SAVE SETTINGS")
        self.btn_save_config.setObjectName("btn_save_config")
        self.btn_save_config.clicked.connect(self._save_config)
        btn_reset_config = MonoButton("RESET")
        btn_reset_config.clicked.connect(self._reset_config)
        save_row.addWidget(self.lbl_config_state)
        save_row.addStretch()
        save_row.addWidget(btn_reset_config)
        save_row.addWidget(self.btn_save_config)

        # --- CONTROL tab: Model Loader (top-level, no collapsible) ---
        control_tab = QWidget()
        control_tab.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        control_layout = QVBoxLayout(control_tab)
        control_layout.setSpacing(12)
        control_layout.addWidget(grp_load)

        # --- Collapsible OPTIONS panel ---
        self._options_expanded = False
        self.btn_options_toggle = QPushButton("▸ OPTIONS")
        self.btn_options_toggle.setCursor(Qt.PointingHandCursor)
        self.btn_options_toggle.setObjectName("options_toggle_btn")
        self.btn_options_toggle.clicked.connect(self._toggle_options_panel)
        control_layout.addWidget(self.btn_options_toggle)

        self.options_panel = QWidget()
        self.options_panel.setVisible(False)
        options_layout = QVBoxLayout(self.options_panel)
        options_layout.setContentsMargins(0, 0, 0, 0)
        options_layout.setSpacing(8)


        control_layout.addWidget(self.options_panel)
        control_layout.addStretch()

        # --- ARCHIVE tab ---
        archive_tab = QWidget()
        archive_layout = QVBoxLayout(archive_tab)
        archive_layout.setSpacing(10)

        archive_controls = QHBoxLayout()
        self.btn_save_chat = MonoButton("SAVE")
        self.btn_save_chat.clicked.connect(self._save_chat_archive)
        self.btn_load_chat = MonoButton("LOAD")
        self.btn_load_chat.clicked.connect(self._load_chat_archive)
        self.btn_delete_chat = MonoButton("DELETE")
        self.btn_delete_chat.clicked.connect(self._delete_selected_archive)
        self.btn_clear_chat = MonoButton("CLEAR")
        self.btn_clear_chat.clicked.connect(lambda: self._clear_current_session(delete_archive=False))
        archive_controls.addWidget(self.btn_save_chat)
        archive_controls.addWidget(self.btn_load_chat)
        archive_controls.addWidget(self.btn_delete_chat)
        archive_controls.addWidget(self.btn_clear_chat)
        archive_controls.addStretch()
        archive_layout.addLayout(archive_controls)

        self.archive_search = QLineEdit()
        self.archive_search.setObjectName("archive_search")
        self.archive_search.setPlaceholderText("Search history…")
        self.archive_search.setClearButtonEnabled(True)
        self.archive_search.textChanged.connect(self._on_history_search_changed)
        archive_layout.addWidget(self.archive_search)

        self.archive_list = QListWidget()
        self.archive_list.setObjectName("archive_list")
        archive_layout.addWidget(self.archive_list)

        # --- SETTINGS tab: AI Configuration + Save/Reset ---
        settings_tab = QWidget()
        settings_layout = QVBoxLayout(settings_tab)
        settings_layout.setSpacing(10)
        settings_layout.addWidget(self.s_temp)
        settings_layout.addWidget(self.s_top)
        settings_layout.addWidget(self.s_top_k)
        settings_layout.addWidget(self.s_min_p)
        settings_layout.addWidget(self.s_presence)
        settings_layout.addWidget(self.s_repetition)
        settings_layout.addWidget(self.s_tok)
        settings_layout.addWidget(self.s_ctx)
        settings_layout.addLayout(save_row)
        settings_layout.addStretch()

        # --- AUDIT tab: Action audit log ---
        audit_tab = QWidget()
        audit_layout = QVBoxLayout(audit_tab)
        audit_layout.setSpacing(6)
        audit_layout.setContentsMargins(4, 6, 4, 6)

        audit_header = QHBoxLayout()
        _lbl_audit_title = QLabel("ACTION AUDIT LOG")
        _lbl_audit_title.setStyleSheet(
            f"color: {_s.FG_DIM}; font-size: 9px; font-family: Consolas; font-weight: bold;"
        )
        audit_header.addWidget(_lbl_audit_title)
        audit_header.addStretch()
        self.btn_clear_audit = MonoButton("CLEAR")
        self.btn_clear_audit.clicked.connect(self._clear_audit_log)
        audit_header.addWidget(self.btn_clear_audit)
        audit_layout.addLayout(audit_header)

        self.audit_list = QListWidget()
        self.audit_list.setObjectName("audit_list")
        self.audit_list.setStyleSheet(
            f"background: {_s.BG_MAIN}; color: {_s.FG_TEXT}; "
            f"border: 1px solid {_s.BORDER_DARK}; font-size: 9px; font-family: Consolas;"
        )
        audit_layout.addWidget(self.audit_list, 1)
        self._control_tab = control_tab
        self._archive_tab = archive_tab
        self._settings_tab = settings_tab
        self._audit_tab = audit_tab

        # === CHAT area ===
        self._surface = ConversationSurfaceWidget(self)
        self.message_list = self._surface.message_list
        self._agent_status_label = self._surface._agent_status_label
        self._agent_popup = self._surface._agent_popup
        self._agent_popup_label = self._surface._agent_popup_label
        self._action_review = self._action_review_panel
        self.input = self._surface.input
        self.btn_send = self._surface.btn_send
        self._surface.sig_send_requested.connect(self._submit_prompt)
        self._surface.sig_mutation_requested.connect(self._handle_surface_mutation)

        trace_group = MonoGroupBox("LOG")
        trace_group.setMaximumHeight(120)
        self.trace = QTextEdit()
        self.trace.setReadOnly(True)
        self.trace.setMinimumHeight(40)
        self.trace.setObjectName("trace_log")
        self.lbl_config_update = QLabel("")
        self.lbl_config_update.setObjectName("lbl_config_update")
        self.lbl_config_update.hide()
        self._config_update_fade = QTimer(self)
        self._config_update_fade.setSingleShot(True)
        self._config_update_fade.timeout.connect(self.lbl_config_update.hide)
        trace_group.add_widget(self.trace)
        trace_group.add_widget(self.lbl_config_update)

        layout.addWidget(self._surface, 1)
        # Active-agents strip — the little live UI under the chat. Fed by the
        # active-agents spine (sig_agents_changed); chips flip running… -> done in
        # place and zoom into each agent's own trace on click.
        try:
            from ui.components.agents_strip import AgentsStrip
            self._agents_strip = AgentsStrip(self)
            layout.addWidget(self._agents_strip, 0)
            self.sig_agents_changed.connect(self._agents_strip.update_agents)
            self._agents_strip.sig_zoom_agent.connect(self._on_zoom_agent)
        except Exception as _strip_exc:  # never break the chat over the strip
            self.sig_debug.emit(f"[agents_strip] init failed: {_strip_exc!r}")
        self._chat_outer = self._surface
        self._trace_group = trace_group
        self._active_assistant_started = False
        self._active_assistant_token_count = 0

        self.engine_combo.currentIndexChanged.connect(self._on_engine_changed)
        self.api_base_input.textChanged.connect(lambda v: self._update_config_value("api_base", v))
        self.api_base_input.editingFinished.connect(self._emit_model_payload)
        self.api_model_input.textChanged.connect(lambda v: self._update_config_value("api_model", v))
        self.api_model_input.editingFinished.connect(self._emit_model_payload)
        self.api_model_combo.currentIndexChanged.connect(self._on_model_combo_changed)
        self.btn_fetch_models.clicked.connect(self._fetch_models)
        self.api_key_input.textChanged.connect(lambda v: self._update_config_value("api_key", v))
        self.api_key_input.editingFinished.connect(self._emit_model_payload)
        self._apply_model_config()
        self._update_load_button_text()
        self._refresh_archive_list()
        self.destroyed.connect(lambda *_args: self._stop_local_server())
        self._set_config_dirty(False)
        if not self._is_model_loaded:
            self._apply_default_limits()
        self._config_panel.auto_probe_server()
        self.ui_bridge.sig_theme_changed.connect(self._on_theme_changed)
        self.ui_bridge.sig_config_changed.connect(self._on_external_config_changed)
        if hasattr(self.ui_bridge, "sig_world_action_pending"):
            self.ui_bridge.sig_world_action_pending.connect(self._on_world_action_pending)
        self.engine_combo = self._config_panel.engine_combo
        self.path_display = self._config_panel.path_display
        self.btn_browse = self._config_panel.btn_browse
        self.gguf_row = self._config_panel.gguf_row
        self.remote_panel = self._config_panel.remote_panel
        self.lbl_base = self._config_panel.lbl_base
        self.api_base_input = self._config_panel.api_base_input
        self.lbl_model = self._config_panel.lbl_model
        self.api_model_input = self._config_panel.api_model_input
        self.api_model_combo = self._config_panel.api_model_combo
        self.btn_fetch_models = self._config_panel.btn_fetch_models
        self.lbl_key = self._config_panel.lbl_key
        self.api_key_input = self._config_panel.api_key_input
        self.btn_load = self._config_panel.btn_load
        self.trace = self._trace_panel.trace
        self.archive_search = self._archive_panel.archive_search
        self.archive_list = self._archive_panel.archive_list
        self.audit_list = self._audit_panel.audit_list

        # Node registrations deferred — called again when engine probe is attached

    def _rebuild_send_template(self):
        self._surface._rebuild_send_template()

    def _on_theme_changed(self, _theme_name: str = "") -> None:
        self._surface._on_theme_changed(_theme_name)

    def build_session_messages_snapshot(self, n: int) -> list[dict]:
        """Return the last N messages from the active session.

        Safe to call from any thread for diagnostic snapshots — the
        read is atomic at the list level and the dicts are copied.
        Used by `ConnectionsPage._build_session_messages_snapshot`
        which feeds AgentServer's /session/messages endpoint.
        """
        try:
            msgs = self._current_session.get("messages", []) or []
        except Exception:
            return []
        if not msgs:
            return []
        slice_n = max(1, min(int(n), len(msgs)))
        return [dict(m) for m in msgs[-slice_n:] if isinstance(m, dict)]

    def build_interceptor_state_snapshot(self) -> dict:
        """Return a snapshot of interceptor-visible state.

        Reads effort tier (world_state), continuity pin counts, recent
        rating summary, and a few chat-page scalars. Best-effort; any
        sub-read failure is swallowed and that field omitted.
        """
        out: dict[str, object] = {}
        try:
            ws = getattr(self.state, "world_state", None)
            if ws is not None and hasattr(ws, "get_effort_state"):
                out["effort"] = ws.get_effort_state()
        except Exception:
            pass
        try:
            from core import continuity as _cont
            snap = _cont.read(include_retired=False)
            out["continuity"] = {
                "active_count": snap.get("counts", {}).get("active", 0),
                "retired_total": snap.get("counts", {}).get("retired_total", 0),
                "active": snap.get("active", []),
            }
        except Exception:
            pass
        try:
            from core import turn_trace as _tt
            out["recent_ratings"] = _tt.recent_ratings_summary()
        except Exception:
            pass
        try:
            out["last_task_id"] = str(self._last_task_id or "")
            out["is_running"] = bool(self._is_running)
            out["is_model_loaded"] = bool(self._is_model_loaded)
            out["tool_loop_active"] = bool(self._tool_loop_active)
        except Exception:
            pass
        return out

    def companion_panels(self) -> dict[str, QWidget]:
        return {
            "CONFIG": self._config_panel,
            "GENERATING": self._trace_panel,
            "ACTION_REVIEW": self._action_review_panel,
            "ASK_USER": self._question_panel,
            "ARCHIVE": self._archive_panel,
            "AUDIT": self._audit_panel,
            "REASONING_TREE": self._reasoning_tree_panel,
        }

    def set_vision_artifact_bridge(self, bridge) -> None:
        """Called by the PageChat addon factory after the singleton bridge is
        resolved from AddonContext.services. Subscribes us to artifact arrivals
        so we can swap the tool-result bubble's placeholder state for the real
        thumb strip when the vision engine finishes a skill-triggered
        generation. Safe to call multiple times; reconnects idempotently."""
        self._vision_artifact_bridge = bridge
        if bridge is None:
            return
        try:
            bridge.sig_artifact_ready.connect(self._on_artifact_ready)
        except Exception:
            pass

    def _on_artifact_ready(self, artifact_id: str, batch_index: int, png_path: str, sidecar: dict) -> None:
        """Find the tool_result message whose payload carries this artifact_id
        and mutate its JSON text to append the new png_path. The chat surface
        then rerenders that row so ToolResultBubble picks up image_paths and
        renders the thumb strip. Per-image (not per-batch) so the strip can
        grow as additional batch members arrive."""
        session = self._current_session
        if not session:
            return
        messages = session.get("messages", [])
        target_idx: int | None = None
        target_payload: dict | None = None
        for i, msg in enumerate(messages):
            if msg.get("role") != "tool_result":
                continue
            text = msg.get("text", "") or ""
            if f"artifact_id={artifact_id}" not in text:
                continue
            try:
                target_payload = json.loads(text)
            except Exception:
                target_payload = None
            if not isinstance(target_payload, dict):
                continue
            target_idx = i
            break
        if target_idx is None or target_payload is None:
            return

        image_paths = list(target_payload.get("image_paths") or [])
        if str(png_path) not in image_paths:
            image_paths.append(str(png_path))
        target_payload["image_paths"] = image_paths
        target_payload["status"] = "ready"
        target_payload["artifact_id"] = artifact_id
        if sidecar:
            target_payload["sidecar"] = sidecar
        messages[target_idx]["text"] = json.dumps(target_payload, ensure_ascii=False)

        surface = getattr(self, "_surface", None)
        if surface is not None and hasattr(surface, "rerender_row"):
            try:
                surface.rerender_row(target_idx)
            except Exception:
                pass

    def current_session_data(self) -> dict:
        return self._current_session

    def switch_reasoning_path(self, node_id: str) -> bool:
        """Pane click → switch the active branch. Refused mid-generation (the
        active path's last node being an empty assistant = a streaming
        placeholder)."""
        from ui.pages import session_tree
        if not session_tree.active():
            return False
        session = self._current_session
        msgs = session.get("messages", [])
        if msgs and msgs[-1].get("role") == "assistant" and not (msgs[-1].get("text") or "").strip():
            return False                       # streaming in flight — no mid-generation switches
        if not session_tree.tree_switch(session, node_id):
            return False
        self._active_widget = None
        self._render_session()
        self.sig_sync_history.emit(self._build_engine_history_from_session())
        return True

    def _switch_take_from_index(self, index: int, direction: int) -> bool:
        """In-chat ‹k/n› control → step the message at ``index`` to its
        prev/next sibling take. Mirrors switch_reasoning_path: refused
        mid-generation, then the same post-mutation rebuild trio."""
        from ui.pages import session_tree
        if not session_tree.active():
            return False
        session = self._current_session
        msgs = session.get("messages", [])
        if msgs and msgs[-1].get("role") == "assistant" and not (msgs[-1].get("text") or "").strip():
            return False                       # streaming in flight — no mid-generation switches
        if not session_tree.switch_take(session, index, direction):
            return False
        self._active_widget = None
        self._render_session()
        self.sig_sync_history.emit(self._build_engine_history_from_session())
        return True

    def open_reasoning_trace(self, task_id: str) -> None:
        """Pane double-click → surface the trace for ``task_id``.

        No single-call trace-open path exists in PageChat/main_window (the
        chat surface has no MonoBase/open_trace affordance — confirmed by grep),
        so per the plan's sanctioned fallback we transition the companion pane
        to the GENERATING (generation-trace) panel."""
        shell = self.window()
        companion = getattr(shell, "companion", None)
        if companion is None:
            return
        try:
            from ui.companion_pane import CompanionState
        except Exception:
            return
        if hasattr(companion, "show_state"):
            companion.show_state(CompanionState.GENERATING)
        elif hasattr(companion, "pin_state"):
            companion.pin_state(CompanionState.GENERATING)

    @property
    def _current_session(self):
        return self._sessions.current

    @_current_session.setter
    def _current_session(self, session):
        self._assistant_box.bind_session(session)

    @property
    def _active_assistant_index(self):
        return self._assistant_box.active_assistant_index

    @_active_assistant_index.setter
    def _active_assistant_index(self, value):
        self._assistant_box.active_assistant_index = value

    @property
    def _rewrite_assistant_index(self):
        return self._assistant_box.rewrite_assistant_index

    @_rewrite_assistant_index.setter
    def _rewrite_assistant_index(self, value):
        self._assistant_box.rewrite_assistant_index = value

    @property
    def _active_assistant_started(self):
        return self._assistant_box.active_assistant_started

    @_active_assistant_started.setter
    def _active_assistant_started(self, value):
        self._assistant_box.active_assistant_started = bool(value)

    @property
    def _active_assistant_token_count(self):
        return self._assistant_box.active_assistant_token_count

    @_active_assistant_token_count.setter
    def _active_assistant_token_count(self, value):
        self._assistant_box.active_assistant_token_count = int(value or 0)

    @property
    def _pending_tool_results(self):
        return self._assistant_box.pending_tool_results

    @_pending_tool_results.setter
    def _pending_tool_results(self, value):
        self._assistant_box.pending_tool_results = list(value or [])

    @property
    def _tool_followup_target_index(self):
        return self._assistant_box.tool_followup_target_index

    @_tool_followup_target_index.setter
    def _tool_followup_target_index(self, value):
        self._assistant_box.tool_followup_target_index = value

    @property
    def _last_task_id(self):
        return self._assistant_box.last_task_id

    @_last_task_id.setter
    def _last_task_id(self, value):
        self._assistant_box.last_task_id = str(value or "")

    @property
    def _pending_archive_save_task_id(self):
        return self._assistant_box.pending_archive_save_task_id

    @_pending_archive_save_task_id.setter
    def _pending_archive_save_task_id(self, value):
        self._assistant_box.pending_archive_save_task_id = str(value) if value is not None else None

    def eventFilter(self, obj, event):
        return self._surface.eventFilter(obj, event)

    def _resize_all_message_items(self):
        self._surface._resize_all_message_items()

    def send(self):
        self._surface.send()

    def handle_send_click(self):
        self._surface.handle_send_click()

    def accept_external_text(
        self,
        text: str,
        *,
        label: str = "external text",
        force_attachment: bool = False,
    ) -> str:
        return self._surface.accept_external_text(
            text,
            label=label,
            force_attachment=force_attachment,
        )

    def request_stop_generation(self) -> None:
        self._tool_cancel_requested = True
        self.sig_stop.emit()

    def _set_send_button_state(self, is_running: bool, stopping: bool = False):
        self._is_running = is_running
        if is_running:
            has_input = bool(self.input.text().strip())
            if has_input:
                self.btn_send.setText("UPDATE")
                color = _s.ACCENT_PRIMARY
            else:
                self.btn_send.setText("■")
                color = _s.FG_ERROR
            self.btn_send.setStyleSheet(
                self._btn_style_template.format(
                    bg=_s.BG_INPUT,
                    color=color,
                )
            )
            self.btn_send.setEnabled(not stopping)
        else:
            self.btn_send.setText("SEND")
            self.btn_send.setStyleSheet(
                self._btn_style_template.format(
                    bg=_s.BG_INPUT,
                    color=_s.ACCENT_PRIMARY,
                )
            )
            self.btn_send.setEnabled(True)

    def _on_input_changed(self, text):
        self._update_agent_popup(text)
        if not self._is_running:
            return
        self._set_send_button_state(is_running=True)

    def _model_supports_thinking(self) -> tuple[bool | None, str]:
        """Look up the active model's native-thinking capability.

        Returns (supports, family_label). `supports` is None when the model
        registry hasn't classified the model yet (loader still warming up,
        unrecognized GGUF, etc.) -- callers should treat None as 'unknown'
        and surface that to the user instead of silently picking a default.
        """
        ws = getattr(self.state, "world_state", None)
        if ws is None:
            return None, ""
        engine_key = getattr(self, "_engine_key", "llm")
        engines = (ws.state.get("engines") or {}) if hasattr(ws, "state") else {}
        engine_meta = engines.get(engine_key) or {}
        preset = engine_meta.get("model_preset") or {}
        caps = preset.get("capabilities") or {}
        family = str(preset.get("family_name") or preset.get("family_id") or "model")
        if "supports_thinking" not in caps:
            return None, family
        return bool(caps["supports_thinking"]), family

    def _handle_think_command(self, cmd: str) -> bool:
        """`/think [on|off|toggle|status]` — control native-thinking gating.

        The flag itself only matters when the model has supports_thinking=True;
        gating happens in ui/addons/builtin.py at request build time. This
        command's job is to (a) toggle the flag and (b) tell the user whether
        it will actually do anything for the active model.
        """
        parts = cmd.split()
        arg = parts[1].lower() if len(parts) > 1 else "toggle"
        supports, family = self._model_supports_thinking()

        if arg == "status":
            cur = "ON" if self._agent_thinking else "OFF"
            if supports is None:
                self._emit_command_block(
                    f"/think  →  {cur}",
                    f"capability for '{family}' unknown",
                )
            elif supports:
                self._emit_command_block(
                    f"/think  →  {cur}",
                    f"'{family}' supports native <think>",
                )
            else:
                note = (
                    "no effect — model has no native think"
                    if self._agent_thinking
                    else "matches model (no native think)"
                )
                self._emit_command_block(f"/think  →  {cur}", note)
            return True

        if arg in ("on", "1", "true", "yes"):
            target = True
        elif arg in ("off", "0", "false", "no"):
            target = False
        elif arg in ("toggle", ""):
            target = not self._agent_thinking
        else:
            self._emit_command_block(
                f"/think  →  ?",
                f"unknown arg '{arg}' (use on|off|toggle|status)",
            )
            return True

        self._agent_thinking = bool(target)
        self._refresh_agent_status()

        cur = "ON" if target else "OFF"
        if supports is True:
            self._emit_command_block(
                f"/think  →  {cur}",
                f"'{family}' will respect this",
            )
        elif supports is False:
            if target:
                self._emit_command_block(
                    f"/think  →  ON",
                    f"'{family}' has no native think; flag stored but ignored "
                    "until you switch to a thinking-capable model",
                )
            else:
                self._emit_command_block(
                    f"/think  →  OFF",
                    f"'{family}' has no native think anyway",
                )
        else:
            self._emit_command_block(
                f"/think  →  {cur}",
                f"capability for '{family}' unknown until model loads",
            )
        return True

    def _handle_trace_command(self, cmd: str) -> bool:
        """`/trace` | `/trace last` | `/trace errors` | `/trace <turn_id>`.

        Read-only inspector for the turn-trace store. Renders summaries
        to the chat trace pane.
        """
        from core import turn_trace as _tt
        parts = cmd.split()
        if len(parts) == 1:
            # /trace alone — show last 5 summaries.
            rows = _tt.list_recent_turns(limit=5)
            if not rows:
                self._trace_html("no turns recorded yet", "TRACE")
                return True
            for r in rows:
                tag = " [errored]" if r.errored_stage_count else ""
                self._trace_html(
                    f"{r.captured_at} · {r.turn_id[:8]} · {r.backend} · "
                    f"{r.stage_count} stages{tag} · {r.total_chars} chars",
                    "TRACE",
                )
            return True
        verb_or_id = parts[1].lower()
        if verb_or_id == "last":
            rows = _tt.list_recent_turns(limit=1)
            if not rows:
                self._trace_html("no turns recorded yet", "TRACE")
                return True
            joined = _tt.get_turn_trace(rows[0].turn_id)
            self._render_trace_joined(joined)
            return True
        if verb_or_id == "errors":
            rows = _tt.search_turns(has_errored_stage=True, limit=10)
            if not rows:
                self._trace_html("no errored turns", "TRACE")
                return True
            for r in rows:
                self._trace_html(
                    f"{r.captured_at} · {r.turn_id[:8]} · "
                    f"{r.errored_stage_count}/{r.stage_count} errored",
                    "TRACE",
                    error=True,
                )
            return True
        # Treat anything else as a turn_id (full or prefix).
        joined = _tt.get_turn_trace(parts[1])
        if joined is None:
            # Try as a prefix lookup against recent turns.
            recent = _tt.list_recent_turns(limit=50)
            match = next(
                (r for r in recent if r.turn_id.startswith(parts[1])), None
            )
            if match is None:
                self._trace_html(f"turn '{parts[1]}' not found", "TRACE", error=True)
                return True
            joined = _tt.get_turn_trace(match.turn_id)
        self._render_trace_joined(joined)
        return True

    def _handle_pipeline_command(self, cmd: str) -> bool:
        """`/pipeline` | `/pipeline last` | `/pipeline faults` | `/pipeline <turn_id>`.

        Read-only inspector for the Turn Pipeline Layer E event store
        (fault_traces). Renders rows to the chat trace pane — same surface
        the existing /trace command writes to.
        """
        from core import turn_trace as _tt
        parts = cmd.split()

        def _render_events(rows) -> None:
            if not rows:
                self._trace_html("no pipeline events for this turn", "PIPELINE")
                return
            for r in rows:
                tag = ""
                if r.fault_kind:
                    tag = f" [{r.severity}: {r.fault_kind}]"
                self._trace_html(
                    f"{r.emitted_at} · seq={r.seq:>3} · {r.event_kind} · "
                    f"{r.source_kind}/{r.source_name}{tag}",
                    "PIPELINE",
                    error=bool(r.severity == "hard"),
                )

        if len(parts) == 1 or parts[1].lower() == "last":
            latest = _tt.most_recent_pipeline_turn_id()
            if latest is None:
                self._trace_html("no turns with pipeline events yet", "PIPELINE")
                return True
            evs = _tt.list_pipeline_events(latest)
            self._trace_html(
                f"turn={latest[:8]} events={len(evs)}",
                "PIPELINE",
            )
            _render_events(evs)
            return True

        verb_or_id = parts[1].lower()

        if verb_or_id == "faults":
            from datetime import datetime, timedelta, timezone
            since_iso = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
            rows = _tt.list_faults_since(since_iso, limit=50)
            if not rows:
                self._trace_html("no faults in last 24h", "PIPELINE")
                return True
            for r in rows:
                self._trace_html(
                    f"{r.emitted_at} · turn={r.turn_id[:8]} · "
                    f"{r.fault_kind} [{r.severity}] · "
                    f"from {r.source_name}",
                    "PIPELINE",
                    error=(r.severity == "hard"),
                )
            return True

        # Treat anything else as a turn_id (full or prefix).
        evs = _tt.list_pipeline_events(parts[1])
        if not evs:
            recent = _tt.list_recent_turns(limit=50)
            match = next(
                (r for r in recent if r.turn_id.startswith(parts[1])), None
            )
            if match is None:
                self._trace_html(
                    f"turn '{parts[1]}' has no pipeline events", "PIPELINE", error=True,
                )
                return True
            evs = _tt.list_pipeline_events(match.turn_id)
        self._trace_html(
            f"turn={evs[0].turn_id[:8]} events={len(evs)}",
            "PIPELINE",
        )
        _render_events(evs)
        return True

    def _render_trace_joined(self, joined) -> None:
        if joined is None:
            self._trace_html("turn not found", "TRACE", error=True)
            return
        self._trace_html(
            f"turn={joined.turn_id[:8]} parent={(joined.parent_turn_id or '-')[:8]} "
            f"stages={joined.summary['stage_count']} "
            f"errored={joined.summary['errored_stage_count']} "
            f"chars={joined.summary['total_chars']}",
            "TRACE",
        )
        for s in joined.stages:
            tag = "" if s.outcome == "ran" else f" [{s.outcome}: {s.outcome_reason}]"
            added_n = len(s.items_added)
            self._trace_html(
                f"  {s.seq:02d} {s.stage_name} ({s.messages_in}->{s.messages_out}, +{added_n} items){tag}",
                "TRACE",
                error=(s.outcome == "errored"),
            )
        if joined.frame is not None:
            f = joined.frame
            self._trace_html(
                f"  frame: {f.backend} · {len(f.final_messages)} msgs · "
                f"system={f.system_prompt_chars} user={f.user_prompt_chars}",
                "TRACE",
            )

    def _handle_rating_command(self, cmd: str) -> bool:
        """`/rating` (help) | `/rating <0-100> #<tag> [#<tag> ...] [note]`.

        Records a Layer D outcome (kind=rating) on the most recent assistant
        turn. A rating now steers monothink evolution via a CLOSED `failure_tags`
        enum (core.failure_tags), not free text. Tokens are parsed as:
          * the first token is the integer score (0-100);
          * any ``#<tag>`` token is a candidate failure tag (validated against
            the closed vocabulary; unknown tags are reported, never coerced);
          * everything else is trailing prose → a holistic ``surface_note``.

        Only a rating carrying at least one VALID tag drives monothink (the
        triviality gate). Examples:
            /rating 40 #premature_convergence
            /rating 30 #missing_branch_pressure #premise_unchecked felt rushed
            /rating 95 "exact, no padding"   ← records, but does NOT train
        """
        from core.failure_tags import (
            FAILURE_TAGS,
            compose_reasoning_why,
            is_valid_tag,
            normalize_tags,
        )

        parts = cmd.split(maxsplit=1)
        if len(parts) < 2 or not parts[1].strip():
            self._emit_command_block(
                "/rating  →  ?",
                "usage: /rating <0-100> #<tag> [#<tag> ...] [note] "
                "— applies to the latest assistant turn",
            )
            return True

        body = parts[1].strip()
        score_token, _, rest = body.partition(" ")
        score_token = score_token.strip().rstrip("%")
        try:
            score = int(score_token)
        except ValueError:
            self._emit_command_block(
                "/rating  →  ?",
                f"score must be an integer 0–100 (got {score_token!r})",
            )
            return True
        if score < 0 or score > 100:
            self._emit_command_block(
                "/rating  →  ?",
                f"score out of range: {score} (must be 0–100)",
            )
            return True

        # Split the remainder into #tag candidates vs trailing prose. A token
        # is a tag candidate iff it matches `#<name>` (leading '#'); everything
        # else accumulates into the surface_note.
        tag_candidates: list[str] = []      # raw names (before validation)
        unknown_tags: list[str] = []        # candidates that failed validation
        prose_tokens: list[str] = []
        for tok in rest.split():
            if tok.startswith("#") and len(tok) > 1:
                name = tok[1:]
                tag_candidates.append(name)
                if not is_valid_tag(name):
                    unknown_tags.append(name)
            else:
                prose_tokens.append(tok)

        failure_tags = normalize_tags(tag_candidates)
        surface_note = " ".join(prose_tokens).strip().strip('"').strip("'") or None

        # Find the most recent assistant message to attach the rating to.
        msgs = self._current_session.get("messages", [])
        last_assistant_idx = -1
        for i in range(len(msgs) - 1, -1, -1):
            msg = msgs[i]
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                last_assistant_idx = i
                break
        if last_assistant_idx < 0:
            self._emit_command_block(
                "/rating  →  noop", "no assistant turn yet to rate",
            )
            return True

        ok = self._record_turn_outcome(
            "rating",
            last_assistant_idx,
            rating_value=score,
            failure_tags=failure_tags,
            surface_note=surface_note,
        )
        if not ok:
            self._emit_command_block(
                "/rating  →  failed",
                "could not resolve turn_id for the latest assistant message",
            )
            return True

        # D3 outcome reporting.
        if failure_tags:
            # Branch 1: at least one valid tag → recorded + drives evolution.
            why = compose_reasoning_why(failure_tags)
            note_tail = f"\nnote: {surface_note}" if surface_note else ""
            unknown_tail = (
                f"\nignored unknown tag(s): {', '.join(unknown_tags)}"
                if unknown_tags else ""
            )
            self._emit_command_block(
                f"/rating  →  {score}  ·  {', '.join(failure_tags)}",
                f"recorded on latest turn — drives monothink.\n{why}"
                f"{note_tail}{unknown_tail}",
            )
        elif tag_candidates:
            # Branch 2: #tokens present but NONE valid → recorded, no evolution.
            vocab = ", ".join(FAILURE_TAGS.keys())
            note_tail = f"\nnote: {surface_note}" if surface_note else ""
            self._emit_command_block(
                f"/rating  →  {score}  (no valid tag — not trained)",
                f"recorded on latest turn (telemetry only).\n"
                f"unknown tag(s): {', '.join(unknown_tags)}{note_tail}\n"
                f"valid tags: {vocab}",
            )
        else:
            # Branch 3: no #tokens at all → recorded, no evolution.
            vocab = ", ".join(FAILURE_TAGS.keys())
            note_tail = f"\nnote: {surface_note}" if surface_note else ""
            self._emit_command_block(
                f"/rating  →  {score}  (no #tag — not trained)",
                f"recorded on latest turn (telemetry only).{note_tail}\n"
                f"add a #tag to train monothink. valid tags: {vocab}",
            )
        return True

    # ── /prompt and /monothink handlers ─────────────────────────────────

    def _handle_prompt_command(self, cmd: str) -> bool:
        """`/prompt` (status) | `/prompt <n1> [n2 ...]` | `/prompt once <...>` | `/prompt clear`."""
        from core.prompt_library import valid_prompts
        world_state = getattr(self.state, "world_state", None)
        if world_state is None:
            self._emit_command_block("/prompt  →  ?", "world state unavailable")
            return True
        parts = cmd.split()
        valid = valid_prompts()
        valid_label = " | ".join(sorted(valid)) if valid else "(none found)"

        # Status: /prompt with no args — show active + available
        if len(parts) == 1:
            snap = world_state.get_prompt_state()
            active = snap.get("prompts") or []
            once = snap.get("once")
            lines = []
            if active:
                lines.append(f"active: {' + '.join(active)}")
            else:
                lines.append("active: none")
            if once:
                lines.append(f"next turn: {' + '.join(once)}")
            lines.append(f"available: {valid_label}")
            self._emit_command_block("/prompt  →  status", "\n".join(lines))
            return True

        # Clear: /prompt clear
        if parts[1].lower() == "clear":
            world_state.clear_prompts()
            self._emit_command_block("/prompt  →  cleared", "no scaffold will inject")
            return True

        # Once: /prompt once <name1> [name2 ...]
        if parts[1].lower() == "once":
            if len(parts) < 3:
                self._emit_command_block("/prompt once  →  missing names", f"available: {valid_label}")
                return True
            if parts[2].lower() == "clear":
                world_state.set_prompts_once(None)
                self._emit_command_block("/prompt once  →  cleared", "once-override removed")
                return True
            names = [p.lower() for p in parts[2:]]
            bad = [n for n in names if n not in valid]
            if bad:
                self._emit_command_block(
                    f"/prompt once  →  unknown: {', '.join(bad)}",
                    f"available: {valid_label}",
                )
                return True
            world_state.set_prompts_once(names)
            self._emit_command_block(
                f"/prompt once  →  {' + '.join(names)}",
                "applies to next turn only, then reverts",
            )
            return True

        # Set baseline: /prompt <name1> [name2 ...]
        names = [p.lower() for p in parts[1:]]
        bad = [n for n in names if n not in valid]
        if bad:
            self._emit_command_block(
                f"/prompt  →  unknown: {', '.join(bad)}",
                f"available: {valid_label}",
            )
            return True
        world_state.set_active_prompts(names)
        self._emit_command_block(
            f"/prompt  →  {' + '.join(names)}",
            "persistent until next /prompt or /prompt clear",
        )
        return True

    def _handle_monothink_command(self, cmd: str) -> bool:
        """`/monothink` (status) | `/monothink on|off` | `/monothink once`."""
        world_state = getattr(self.state, "world_state", None)
        if world_state is None:
            self._emit_command_block("/monothink  →  ?", "world state unavailable")
            return True
        parts = cmd.split()

        # Status
        if len(parts) == 1:
            snap = world_state.get_monothink_state()
            enabled = snap.get("enabled", False)
            once = snap.get("once")
            detail = f"{'on' if enabled else 'off'}"
            if once is not None:
                detail += f"  ·  next turn: {'on' if once else 'off'}"
            self._emit_command_block("/monothink  →  status", detail)
            return True

        arg = parts[1].lower()
        if arg == "on":
            world_state.set_monothink(True)
            self._emit_command_block("/monothink  →  on", "persistent until /monothink off")
            return True
        if arg == "off":
            world_state.clear_monothink()
            self._emit_command_block("/monothink  →  off", "monothink disabled")
            return True
        if arg == "once":
            world_state.set_monothink_once(True)
            self._emit_command_block("/monothink  →  once", "fires next turn only")
            return True
        if arg == "clear":
            world_state.clear_monothink()
            self._emit_command_block("/monothink  →  cleared", "monothink disabled")
            return True

        self._emit_command_block(
            f"/monothink  →  unknown: {arg}",
            "usage: /monothink on | off | once | clear",
        )
        return True

    # ── tool command handlers (/skill-creator, future tools) ────────────

    def _handle_tool_command(self, cmd: str, tool_name: str) -> bool:
        """Generic handler for tool-prompt slash commands.

        Injects the tool prompt for the next turn via world_state, then
        forwards any remaining text as the user message so the model sees
        both the scaffold and the user's intent in one turn.
        """
        from core.prompt_library import load_tool_prompt
        world_state = getattr(self.state, "world_state", None)
        if world_state is None:
            self._emit_command_block(f"/{tool_name}  →  ?", "world state unavailable")
            return True
        content = load_tool_prompt(tool_name)
        if not content:
            self._emit_command_block(
                f"/{tool_name}  →  error",
                f"prompt file not found: prompts/tools/{tool_name}.md",
            )
            return True
        # Set the tool prompt for next-turn injection
        world_state.state["tool_prompt_once"] = tool_name
        world_state.mark_dirty()
        # Extract any trailing text as the user's topic/request
        prefix = f"/{tool_name}"
        trailing = cmd[len(prefix):].strip()
        if trailing:
            self._emit_command_block(
                f"/{tool_name}  →  active",
                f"scaffold loaded · sending: \"{trailing[:80]}{'…' if len(trailing) > 80 else ''}\"",
            )
            # Forward the trailing text as a normal message so the model
            # gets both the tool scaffold AND the user's request
            self._submit_prompt(trailing)
        else:
            self._emit_command_block(
                f"/{tool_name}  →  active",
                "scaffold loaded for next turn — type your request",
            )
        return True

    def _handle_frame_command(self, cmd: str) -> bool:
        """/frame <better frame> — a HUMAN correction of the LAST turn's frame
        (MonoFrame v2). This is the surface that TRAINS: it assembles a
        CorrectionCard (stateless control -> synthesis -> advisor gate -> store)
        off-thread. bad_frame is the current bearing frame; recent_asks come from
        this session. Thin glue over the tested process_correction_async."""
        better = cmd[len("/frame"):].strip()
        if not better:
            self._emit_command_block("/frame  →  usage", "/frame <a better frame for the last turn>")
            return True
        try:
            from addons.system.bearing import store as _bstore
            from addons.system.bearing.drift import recent_asks as _ra
            from addons.system.bearing.correction_runner import process_correction_async
            from addons.system.bearing import correction_card as _cc
            bad_frame = _bstore.get_bearing().current_frame or ""
            msgs = getattr(self, "_current_session", {}).get("messages", []) or []
            process_correction_async(
                str(self.config.get("_turn_id") or ""),
                bad_frame=bad_frame,
                better_frame=better,
                recent_asks=_ra(msgs),
                base_config=self.config,
                source=_cc.Source.HUMAN,
            )
            self._emit_command_block(
                "/frame  →  training",
                f"correcting last frame · advisor gate running · "
                f"\"{better[:60]}{'…' if len(better) > 60 else ''}\"",
            )
        except Exception as exc:
            self._emit_command_block("/frame  →  error", str(exc))
        return True

    def _handle_world_commands(self, text: str) -> bool:
        cmd = text.strip()
        if not cmd.startswith("/"):
            return False
        # /think handled before world_state lookup so it works pre-load.
        if cmd == "/think" or cmd.startswith("/think "):
            return self._handle_think_command(cmd)
        if cmd == "/prompt" or cmd.startswith("/prompt "):
            return self._handle_prompt_command(cmd)
        if cmd == "/monothink" or cmd.startswith("/monothink "):
            return self._handle_monothink_command(cmd)
        if cmd == "/skill-creator" or cmd.startswith("/skill-creator "):
            return self._handle_tool_command(cmd, "skill-creator")
        if cmd == "/trace" or cmd.startswith("/trace "):
            return self._handle_trace_command(cmd)
        if cmd == "/pipeline" or cmd.startswith("/pipeline "):
            return self._handle_pipeline_command(cmd)
        if cmd == "/rating" or cmd.startswith("/rating "):
            return self._handle_rating_command(cmd)
        if cmd == "/frame" or cmd.startswith("/frame "):
            return self._handle_frame_command(cmd)
        if cmd == "/workshop" or cmd.startswith("/workshop "):
            return self._handle_workshop_command(cmd)
        world_state = getattr(self.state, "world_state", None)
        if world_state is None:
            return False
        if cmd == "/approve":
            pending = world_state.get_pending_action()
            if pending:
                self._approve_pending(pending, source="user/approve")
                self._emit_command_block(
                    "/approve  →  ok",
                    f"action: {pending.get('type', 'unknown')}",
                )
            else:
                self._emit_command_block(
                    "/approve  →  noop", "no pending action to approve"
                )
            return True
        if cmd == "/reject":
            pending = world_state.get_pending_action()
            self._reject_pending(pending, source="user/reject")
            self._emit_command_block(
                "/reject  →  ok",
                "pending action discarded" if pending else "nothing to reject",
            )
            return True
        if cmd.startswith("/act " ):
            raw = cmd[len("/act " ):].strip()
            try:
                action = json.loads(raw)
            except Exception as exc:
                self._emit_command_block(
                    "/act  →  invalid",
                    f"json parse error: {exc}",
                )
                return True
            self._execute_world_action(action, source="user/act")
            self._emit_command_block(
                "/act  →  dispatched",
                f"type: {action.get('type', 'unknown')}",
            )
            return True
        return False

    def _handle_workshop_command(self, cmd: str) -> bool:
        """`/workshop` lists workflows + the active one; `/workshop <name|id>` sets the active chat
        flow (so the next message runs through it); `/workshop genesis|off|none|reset` returns to
        Genesis. Reuses the registry's active-flow flag -- the same flag the chat guard reads."""
        reg = getattr(self, "_workflow_registry", None)
        if reg is None:
            self._emit_command_block("/workshop  →  unavailable", "workflow registry not ready")
            return True
        try:
            flows = reg.list_workflows()
        except Exception as exc:
            self._emit_command_block("/workshop  →  error", f"could not list workflows: {exc}")
            return True
        try:
            active = reg.active_id() or ""
        except Exception:
            active = ""
        arg = cmd[len("/workshop"):].strip()

        if not arg:  # list
            lines = []
            for w in flows:
                is_active = (w.id == active) or (active == "" and w.id == "genesis")
                lines.append(f"{w.name}  [{w.id}]" + ("   ● active" if is_active else ""))
            lines.append("")
            lines.append("usage: /workshop <name|id>   (genesis|off resets to Genesis)")
            self._emit_command_block(f"/workshop  →  {len(flows)} workflow(s)", "\n".join(lines))
            return True

        low = arg.lower()
        if low in ("genesis", "off", "none", "reset"):
            reg.set_active("")
            self._emit_command_block("/workshop  →  Genesis", "active flow reset to Genesis (native).")
            return True

        target = next((w for w in flows if w.id == arg), None)
        if target is None:
            target = next((w for w in flows if w.name.lower() == low), None)
        if target is None or target.kind != "monoline":
            avail = ", ".join(w.name for w in flows if w.kind == "monoline") or "(none saved)"
            self._emit_command_block("/workshop  →  not found",
                                     f"no monoline workflow '{arg}'. available: {avail}")
            return True
        reg.set_active(target.id)
        self._emit_command_block(f"/workshop  →  {target.name}",
                                 f"active flow set — your next message runs through '{target.name}'.")
        return True

    def _execute_world_action(self, action: dict, source: str, approved: bool = False) -> None:
        world_state = getattr(self.state, "world_state", None)
        if world_state is not None:
            world_state.set_last_action(f"{source}: {action.get('type')}")
        if approved and hasattr(self.ui_bridge, "sig_world_action_approved"):
            self.ui_bridge.sig_world_action_approved.emit(action)
        else:
            self.ui_bridge.sig_world_action.emit(action)

    def _extract_action_from_text(self, text: str) -> dict | None:
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("ACTION " ):
                raw = line[len("ACTION " ):].strip()
                try:
                    return json.loads(raw)
                except Exception:
                    return None
        return None

    def _maybe_handle_action_proposal(self) -> None:
        world_state = getattr(self.state, "world_state", None)
        if world_state is None:
            return
        msgs = self._current_session.get("messages", [])
        if not msgs or msgs[-1].get("role") != "assistant":
            return
        action = self._extract_action_from_text(msgs[-1].get("text", ""))
        if not action:
            return

        from core.world_actions import check_policy, PolicyDecision
        decision = check_policy(action)

        if decision == PolicyDecision.BLOCKED:
            self._trace_html("ACTION BLOCKED by policy", "WORLD", error=True)
            world_state.append_action_log({
                "type": action.get("type"), "command": action.get("command"),
                "engine": action.get("engine"), "source": "policy",
                "outcome": "blocked",
            })
            return

        if decision == PolicyDecision.AUTO_APPROVE:
            self._execute_world_action(action, source="policy/auto")
            world_state.append_action_log({
                "type": action.get("type"), "command": action.get("command"),
                "engine": action.get("engine"), "source": "policy/auto",
                "outcome": "auto_approved",
            })
            self._trace_html(
                f"ACTION AUTO-APPROVED ({action.get('command', action.get('type'))})", "WORLD"
            )
            return

        # REQUIRE_APPROVAL — show review panel
        world_state.set_pending_action(action)
        self._action_review_show(action)

    # ------------------------------------------------------------------
    # Action review panel helpers
    # ------------------------------------------------------------------

    def _build_action_review_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("action_review_panel")
        panel.setFrameShape(QFrame.StyledPanel)
        panel.setStyleSheet(f"""
            QFrame#action_review_panel {{
                background: {_s.BG_PANEL};
                border: 1px solid {_s.ACCENT_PRIMARY};
                border-radius: 4px;
            }}
        """)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)

        # Header row
        header = QHBoxLayout()
        lbl_title = QLabel("⚡ ACTION PROPOSED")
        lbl_title.setStyleSheet(
            f"color: {_s.ACCENT_PRIMARY}; font-size: 10px; font-family: Consolas; font-weight: bold;"
        )
        header.addWidget(lbl_title)
        header.addStretch()
        btn_close = QPushButton("✕")
        btn_close.setFixedSize(18, 18)
        btn_close.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {_s.FG_DIM}; border: none; font-size: 10px; }}"
            f"QPushButton:hover {{ color: {_s.FG_TEXT}; }}"
        )
        btn_close.clicked.connect(lambda: self._reject_pending(
            getattr(self.state, "world_state", None) and
            self.state.world_state.get_pending_action(), source="user/dismiss"
        ))
        header.addWidget(btn_close)
        layout.addLayout(header)

        # Detail labels
        self._ar_lbl_type = QLabel("")
        self._ar_lbl_type.setStyleSheet(
            f"color: {_s.FG_TEXT}; font-size: 9px; font-family: Consolas;"
        )
        layout.addWidget(self._ar_lbl_type)

        self._ar_lbl_engine = QLabel("")
        self._ar_lbl_engine.setStyleSheet(
            f"color: {_s.FG_TEXT}; font-size: 9px; font-family: Consolas;"
        )
        layout.addWidget(self._ar_lbl_engine)

        self._ar_lbl_payload = QLabel("")
        self._ar_lbl_payload.setStyleSheet(
            f"color: {_s.FG_DIM}; font-size: 9px; font-family: Consolas;"
        )
        self._ar_lbl_payload.setWordWrap(True)
        layout.addWidget(self._ar_lbl_payload)

        # Action buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_reject = QPushButton("REJECT")
        btn_reject.setFixedHeight(24)
        btn_reject.setStyleSheet(
            f"QPushButton {{ background: transparent; color: #e05555; border: 1px solid #e05555; "
            f"border-radius: 2px; font-size: 9px; font-family: Consolas; padding: 0 10px; }}"
            f"QPushButton:hover {{ background: #e0555522; }}"
        )
        btn_reject.clicked.connect(lambda: self._reject_pending(
            getattr(getattr(self.state, "world_state", None), "get_pending_action", lambda: None)(),
            source="user/reject",
        ))
        btn_row.addWidget(btn_reject)

        btn_approve = QPushButton("APPROVE ✓")
        btn_approve.setFixedHeight(24)
        btn_approve.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {_s.ACCENT_PRIMARY}; "
            f"border: 1px solid {_s.ACCENT_PRIMARY}; border-radius: 2px; "
            f"font-size: 9px; font-family: Consolas; padding: 0 10px; }}"
            f"QPushButton:hover {{ background: {_s.ACCENT_PRIMARY}22; }}"
        )
        btn_approve.clicked.connect(lambda: self._approve_pending(
            getattr(getattr(self.state, "world_state", None), "get_pending_action", lambda: None)(),
            source="user/approve",
        ))
        btn_row.addWidget(btn_approve)
        layout.addLayout(btn_row)

        return panel

    def _action_review_show(self, action: dict) -> None:
        kind = action.get("type", "—")
        cmd = action.get("command", "")
        self._action_review_panel.show_action(action)
        self._trace_html(
            f"ACTION PROPOSED ({cmd or kind}) — click APPROVE / REJECT above, or type /approve / /reject",
            "WORLD",
        )

    def _on_world_action_pending(self, action: dict) -> None:
        if not isinstance(action, dict):
            return
        self._action_review_show(action)

    def _approve_pending(self, action: dict | None, source: str = "user/approve") -> None:
        world_state = getattr(self.state, "world_state", None)
        if action:
            self._execute_world_action(action, source=source, approved=True)
            if world_state:
                world_state.set_pending_action(None)
                world_state.append_action_log({
                    "type": action.get("type"), "command": action.get("command"),
                    "engine": action.get("engine"), "source": source,
                    "outcome": "approved",
                })
            self._trace_html("WORLD ACTION APPROVED", "WORLD")
        else:
            self._trace_html("NO PENDING ACTION", "WORLD")
        self._action_review_panel.clear()

    # ── ask_user question lifecycle ────────────────────────────────

    def _on_ask_user(self, payload: dict) -> bool | str:
        """Callback invoked by the ask_user executor via ToolExecutionContext.

        Returns:
          * True  — accepted, panel shown + world_state pending_question set
          * False — busy (another question is already pending)
          * "error: ..."  — other failure

        Pending state lives in BOTH self._pending_ask_user (chat-side; lets us
        guard against busy/double-fire without going through world_state) AND
        world_state.session["pending_question"] (lets CompanionPane.
        evaluate_state auto-route to ASK_USER on the next refresh).
        """
        if not isinstance(payload, dict):
            return "error: payload must be a dict"
        if self._pending_ask_user is not None:
            return False  # busy
        try:
            self._pending_ask_user = dict(payload)
            ws = getattr(self.state, "world_state", None)
            if ws is not None:
                ws.set_pending_question(self._pending_ask_user)
            self._question_panel.show_question(payload)
        except Exception as exc:
            # Roll back chat-side + world_state state on render failure so a
            # later ask_user call doesn't get a stale "busy" rejection.
            self._pending_ask_user = None
            ws = getattr(self.state, "world_state", None)
            if ws is not None:
                try:
                    ws.set_pending_question(None)
                except Exception:
                    pass
            return f"error: failed to render question panel: {type(exc).__name__}: {exc}"
        return True

    def _clear_pending_ask_user(self) -> None:
        """Single point for clearing both chat-side and world_state pending
        question state. Called from answered, dismissed, and session-reset
        paths so the panel can't outlive its trigger."""
        self._pending_ask_user = None
        ws = getattr(self.state, "world_state", None)
        if ws is not None:
            try:
                ws.set_pending_question(None)
            except Exception:
                pass
        try:
            self._question_panel.clear()
        except Exception:
            pass

    def _on_ask_user_answered(self, question_id: str, answers: list) -> None:
        """User picked one or more option(s). Inject the answer as the next user message.

        Format documented in skills/ask-user/SKILL.md — keep this in sync if
        the contract changes. The model on the next turn sees the
        [ASK_USER_ANSWER] block and continues from there.
        """
        self._clear_pending_ask_user()
        if not answers:
            payload_lines = "(no option selected)"
        elif len(answers) == 1:
            payload_lines = str(answers[0])
        else:
            payload_lines = "\n".join(f"- {str(a)}" for a in answers)
        msg = (
            f"[ASK_USER_ANSWER question_id={question_id}]\n"
            f"{payload_lines}\n"
            f"[/ASK_USER_ANSWER]"
        )
        self._send_message(msg)

    def _on_ask_user_dismissed(self, question_id: str) -> None:
        """User closed the panel without answering. Tell the model so it can
        proceed without an answer (or re-ask with adjusted framing)."""
        self._clear_pending_ask_user()
        msg = (
            f"[ASK_USER_DISMISSED question_id={question_id}]\n"
            f"(user dismissed the question without answering)\n"
            f"[/ASK_USER_DISMISSED]"
        )
        self._send_message(msg)

    def _reject_pending(self, action: dict | None, source: str = "user/reject") -> None:
        world_state = getattr(self.state, "world_state", None)
        if world_state:
            world_state.set_pending_action(None)
            if action:
                world_state.append_action_log({
                    "type": action.get("type"), "command": action.get("command"),
                    "engine": action.get("engine"), "source": source,
                    "outcome": "rejected",
                })
        if hasattr(self.ui_bridge, "sig_world_action_rejected"):
            self.ui_bridge.sig_world_action_rejected.emit(action)
        self._trace_html("PENDING ACTION REJECTED", "WORLD")
        self._action_review_panel.clear()

    # ------------------------------------------------------------------
    # Audit log helpers
    # ------------------------------------------------------------------

    _AUDIT_OUTCOME_COLOR = {
        "auto_approved": "#88cc88",
        "approved":      "#4a9eff",
        "rejected":      "#e05555",
        "blocked":       "#e8a838",
    }

    def _refresh_audit_list(self) -> None:
        world_state = getattr(self.state, "world_state", None)
        if world_state is None:
            return
        self.audit_list.clear()
        log = world_state.get_action_log()
        for entry in reversed(log):
            ts = entry.get("ts", "")[:19].replace("T", " ")
            outcome = entry.get("outcome", "?")
            cmd = entry.get("command") or entry.get("type", "?")
            engine = entry.get("engine") or "—"
            source = entry.get("source", "")
            text = f"{ts}  {outcome:<14}  {cmd:<18}  eng:{engine}  src:{source}"
            item = QListWidgetItem(text)
            color = self._AUDIT_OUTCOME_COLOR.get(outcome, _s.FG_DIM)
            item.setForeground(__import__("PySide6.QtGui", fromlist=["QColor"]).QColor(color))
            self.audit_list.addItem(item)

    def _clear_audit_log(self) -> None:
        world_state = getattr(self.state, "world_state", None)
        if world_state:
            world_state.clear_action_log()
        self.audit_list.clear()

    # ------------------------------------------------------------------
    # Agent command helpers
    # ------------------------------------------------------------------

    def _update_agent_popup(self, text: str) -> None:
        """Show/update autocomplete popup when user is typing /agent or @mention."""
        _surface = getattr(self, "_surface", None)
        _picker = _surface._command_picker if _surface else None

        # ── @mention picker ───────────────────────────────────────────────
        at_pos = text.rfind("@")
        if at_pos != -1:
            if _picker:
                _picker.hide()
            fragment = text[at_pos + 1:].lower()
            from engine.external_agents import peer_names as _ext_names
            all_names = ["monolith"] + _ext_names()
            matches = [n for n in all_names if n.startswith(fragment)]
            if matches:
                items = "  ".join(
                    f'<span style="color:{_s.ACCENT_PRIMARY};">@{n}</span>'
                    for n in matches
                )
                self._agent_popup_label.setText(items)
                self._agent_popup.adjustSize()
                popup_h = self._agent_popup.sizeHint().height()
                self._agent_popup.setGeometry(
                    self.input.x(),
                    self.input.y() - popup_h - 4,
                    self.input.width() + self.btn_send.width() + 4,
                    popup_h,
                )
                self._agent_popup.show()
                self._agent_popup.raise_()
                return

        # ── slash command picker (new CommandPicker widget) ──────────────
        stripped = text.strip()
        if stripped.startswith("/") and not stripped.startswith("/agent") and _picker:
            self._agent_popup.hide()
            partial = stripped.split(None, 1)[0]

            # /prompt special case: show filterable prompt names after the command
            if partial == "/prompt" and " " in stripped:
                from core.prompt_library import valid_prompts as _vp
                available = sorted(_vp())
                # Get the text after "/prompt " — could be partial name(s)
                after_cmd = stripped.split(None, 1)[1] if len(stripped.split(None, 1)) > 1 else ""
                # The last token is what's being typed now
                tokens = after_cmd.split()
                current_partial = tokens[-1].lower() if tokens else ""
                # Filter available prompts by the current partial
                already_selected = set(t.lower() for t in tokens[:-1]) if len(tokens) > 1 else set()
                candidates = [
                    p for p in available
                    if p.startswith(current_partial) and p not in already_selected
                ] if current_partial else [p for p in available if p not in already_selected]
                if candidates:
                    _picker.update_matches([
                        (p, "", "") for p in candidates
                    ])
                    _surface._position_command_picker()
                    _picker.show()
                    _picker.raise_()
                    # Override accept behavior: Tab fills just the prompt name, not the whole command
                    _picker._prompt_completion_prefix = stripped.rsplit(current_partial, 1)[0] if current_partial else stripped + " "
                else:
                    _picker.hide()
                return

            # Exact match → progressive arg hint mode
            exact = None
            for (name, args, desc) in SLASH_COMMANDS:
                if name == partial and " " in stripped:
                    exact = (name, args, desc)
                    break

            if exact is not None:
                after_cmd = stripped[len(exact[0]):].strip()
                filled_count = len(after_cmd.split()) if after_cmd else 0
                stages = _ARG_STAGES.get(exact[0])
                if stages:
                    stage_idx = min(filled_count, len(stages) - 1)
                    hint_text = stages[stage_idx]
                else:
                    hint_text = exact[1]
                _picker.show_arg_hint(exact[0], hint_text, exact[2])
                _surface._position_command_picker()
                _picker.show()
                _picker.raise_()
                return

            # Partial match → filtered list
            matches = [
                (name, args, desc)
                for (name, args, desc) in SLASH_COMMANDS
                if name.startswith(partial) or partial == "/"
            ]
            if not matches:
                _picker.hide()
                return

            _picker.update_matches(matches)
            _surface._position_command_picker()
            _picker.show()
            _picker.raise_()
            return

        # ── /agent picker ─────────────────────────────────────────────────
        if _picker:
            _picker.hide()
        if not text.startswith("/agent"):
            self._agent_popup.hide()
            return

        # Build hint lines showing current parsed state
        args = self._parse_agent_args(text)
        on_off   = "on ✓" if args.get("on") else "off"
        approval = "yes" if args.get("approval", True) else "no"
        thinking = "yes" if args.get("thinking", True) else "no"
        workspace = args.get("workspace") or "(cwd)"

        hint = (
            f'<span style="color:{_s.ACCENT_PRIMARY};">/agent</span> '
            f'<span style="color:{_s.FG_TEXT};">on|off</span>  '
            f'<span style="color:{_s.FG_DIM};">approval:yes|no  thinking:yes|no  workspace:"path"</span><br>'
            f'<span style="color:{_s.FG_DIM};">  state →</span> '
            f'<span style="color:{_s.FG_TEXT};">{on_off}</span>  '
            f'approval:<span style="color:{_s.FG_ACCENT};">{approval}</span>  '
            f'thinking:<span style="color:{_s.FG_ACCENT};">{thinking}</span>  '
            f'workspace:<span style="color:{_s.FG_ACCENT};">{workspace}</span>'
        )
        self._agent_popup_label.setText(hint)

        # Position popup above the input field
        self._agent_popup.adjustSize()
        input_pos = self.input.mapTo(self, self.input.pos())
        popup_h = self._agent_popup.sizeHint().height()
        self._agent_popup.setGeometry(
            self.input.x(),
            self.input.y() - popup_h - 4,
            self.input.width() + self.btn_send.width() + 4,
            popup_h,
        )
        self._agent_popup.show()
        self._agent_popup.raise_()

    def _parse_agent_args(self, text: str) -> dict:
        """Parse /agent command text into a dict of args."""
        import re
        args: dict = {}
        lower = text.lower()

        if " on" in lower:
            args["on"] = True
        elif " off" in lower:
            args["on"] = False

        m = re.search(r"approval:(yes|no)", lower)
        if m:
            args["approval"] = m.group(1) == "yes"

        m = re.search(r"thinking:(yes|no)", lower)
        if m:
            args["thinking"] = m.group(1) == "yes"

        m = re.search(r'workspace:"([^"]*)"', text)
        if m:
            args["workspace"] = m.group(1)

        return args

    def _resolve_at_mentions(self, text: str) -> list[dict]:
        """Return participant dicts for @names found in text that are connected with a URL.
        Returns empty list if CONNECT is not active.
        Each dict: {name, url, label}
        """
        import re as _re
        from engine.agent_server import get_server as _get_server
        srv = _get_server()
        if srv is None or not srv.is_running:
            return []
        at_words = {m.lower() for m in _re.findall(r"@(\w+)", text)}
        if not at_words:
            return []
        with srv._participants_lock:
            participants = dict(srv._participants)
        results = []
        for word in at_words:
            if word == "monolith":
                continue
            p = participants.get(word)
            if p and p.get("url"):
                results.append(p)
            elif p:
                # Connected but no URL — will show a note
                results.append(p)
        return results

    def _get_chat_history(self, limit: int = 20) -> list[dict]:
        """Return recent messages as a list of {role, text} dicts for context."""
        msgs = getattr(self, "_messages", [])
        out = []
        for m in msgs[-limit:]:
            role = m.get("role", "user")
            text = str(m.get("content") or m.get("text") or "")
            if role in ("user", "assistant", "agent"):
                out.append({"role": "user" if role in ("user", "agent") else "assistant", "text": text[:500]})
        return out

    def _dispatch_at_mentions(self, text: str, participants: list[dict]) -> None:
        """Route message + history to connected agents. Monolith stays silent unless @monolith."""
        import re as _re
        from engine.external_agents import dispatch as _dispatch, strip_mention as _strip

        monolith_mentioned = bool(_re.search(r"@monolith\b", text, _re.IGNORECASE))
        history = self._get_chat_history(limit=20)

        for p in participants:
            target = p["name"].lower()
            label = p.get("label") or p["name"]
            url = p.get("url") or ""

            # Strip all @mentions from what the agent sees
            clean = _re.sub(r"@\w+", "", text, flags=_re.IGNORECASE).strip() or text
            self.sig_debug.emit(f"[CHAT] @mention → {target}: {clean[:60]}")

            if not url:
                note = f"[{label} is connected but has no /chat URL — message not delivered]"
                idx = self._add_message("agent", note, extra={"agent_name": label})
                self._append_message_widget(idx)
                continue

            def _on_reply(lbl, reply, _lbl=label):
                idx = self._add_message("agent", reply, extra={"agent_name": _lbl})
                self._append_message_widget(idx)

            def _on_err(lbl, err, _lbl=label):
                idx = self._add_message("agent", err, extra={"agent_name": _lbl})
                self._append_message_widget(idx)

            _dispatch(
                target,
                clean,
                on_reply=lambda lbl, reply, _fn=_on_reply: self._qt_invoke(_fn, lbl, reply),
                on_error=lambda lbl, err, _fn=_on_err: self._qt_invoke(_fn, lbl, err),
                history=history,
                url=url,
            )

        if monolith_mentioned:
            clean = _re.sub(r"@\w+", "", text, flags=_re.IGNORECASE).strip()
            self._dispatch_generation(clean or text, source="at_mention:monolith")

    def _qt_invoke(self, fn, *args) -> None:
        """Call fn(*args) safely on the Qt main thread via a single-shot timer."""
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, lambda: fn(*args))

    def _apply_agent_command(self, text: str) -> bool:
        """
        If text is an /agent command, apply it and return True.
        Return False if it's a normal message.
        """
        if not text.lower().startswith("/agent"):
            return False

        self._agent_popup.hide()
        args = self._parse_agent_args(text)

        enabling = args.get("on", None)
        if enabling is None:
            # bare /agent toggles
            enabling = not self._agent_enabled

        self._agent_enabled  = enabling
        self._agent_approval = args.get("approval", self._agent_approval)
        self._agent_thinking = args.get("thinking", self._agent_thinking)
        self._agent_workspace = args.get("workspace", self._agent_workspace)

        self._refresh_agent_status()
        return True

    def _refresh_agent_status(self) -> None:
        if self._agent_enabled:
            approval_str  = "approval:on" if self._agent_approval else "approval:off"
            thinking_str  = "thinking:on" if self._agent_thinking else "thinking:off"
            workspace_str = f'ws:"{self._agent_workspace}"' if self._agent_workspace else "ws:cwd"
            self._agent_status_label.setText(
                f"⬡ AGENT  {approval_str}  {thinking_str}  {workspace_str}"
            )
            self._agent_status_label.show()
        else:
            self._agent_status_label.hide()

    def _send_message(self, text):
        self.input.setText(text)
        self.send()

    def _is_editing_message(self) -> bool:
        return self._editing_user_index is not None

    def _clear_editing_message(self, *, clear_input: bool = False) -> None:
        self._editing_user_index = None
        self._surface.set_editing_message(None)
        self.input.setPlaceholderText(self._surface.DEFAULT_INPUT_PLACEHOLDER)
        if clear_input:
            self.input.clear()
        self._set_send_button_state(is_running=self._is_running)

    def _begin_editing_message(self, index: int, text: str) -> None:
        if self._editing_user_index == index:
            self._clear_editing_message(clear_input=True)
            return
        self._editing_user_index = index
        self._surface.set_editing_message(index)
        self.input.setPlaceholderText(self._surface.EDIT_INPUT_PLACEHOLDER)
        self.input.setText(text)
        self.input.setFocus(Qt.OtherFocusReason)
        self.input.selectAll()
        self._set_send_button_state(is_running=self._is_running)

    def _commit_edit_prompt(self, text: str) -> None:
        index = self._editing_user_index
        if index is None:
            return
        self._snapshot_session()
        self._sessions.set_suppress_title_regen(True)
        updated_text = self._assistant_box.commit_edit_from_index(index, text)
        if updated_text is None:
            self._clear_editing_message()
            return
        self._active_widget = None
        self._render_session()
        self.sig_sync_history.emit(
            self._build_engine_history_from_session()
        )
        self._clear_editing_message()
        self._dispatch_generation(updated_text, source=f"edit:{updated_text[:20]}")

    def _dispatch_generation(self, payload: object, *, rewrite_index: int | None = None, source: str = "chat") -> None:
        # STOP dominance (Kernel Contract v2 §7): once the user presses Stop, no
        # autonomous continuation (parse-retry / nudge / followup-retry) may
        # re-dispatch a generation. A Stop-interrupted reply often leaves a
        # malformed <tool_call> that parse-errors and would otherwise trigger a
        # retry the user never asked for. User-initiated dispatches (send / edit /
        # regen / agent / update) instead CLEAR the stale Stop so the new turn
        # starts clean — mirroring the monoline lane's reset in _dispatch_monoline_run.
        if _is_autonomous_continuation(source):
            # getattr default: _dispatch_generation is exercised in isolation by
            # flow-swap tests on a minimal stub; a real PageChat always sets this
            # flag in __init__, so the default only applies there (no Stop pending).
            if getattr(self, "_tool_cancel_requested", False):
                self._tool_cancel_requested = False
                self._clear_tool_followup_state()
                return
        elif hasattr(self, "_tool_cancel_requested"):
            self._tool_cancel_requested = False
        # --- ACTIVE-FLOW BRANCH (additive; read BEFORE any Genesis stream setup) ---
        ws = getattr(self.state, "world_state", None)
        active_id = ws.get_active_workflow() if ws is not None else ""
        wf = self._workflow_registry.get(active_id) if active_id else None
        if (wf is not None and wf.kind == "monoline"
                and source.split(":", 1)[0] in self._MONOLINE_ENTRY_SOURCES):
            # prefix match: real sources are "send:<txt>"/"edit:<txt>"/"agent:<name>"/"regen";
            # internal sources (tool_followup, update, *_nudge, *_retry, model) never match -> Genesis.
            self._dispatch_monoline_run(wf, payload, rewrite_index=rewrite_index, source=source)
            return
        # --- GENESIS PATH BELOW: byte-for-byte the existing code, untouched ---
        self._set_send_button_state(is_running=True)
        if rewrite_index is not None:
            self._assistant_box.start_rewrite_stream(rewrite_index)
            self._active_widget = None
        else:
            self._start_assistant_stream()
        self.message_list.scrollToBottom()
        self.sig_debug.emit(f"[CHAT] about to emit sig_generate ({source})")
        self.sig_generate.emit(payload)
        self.sig_debug.emit(f"[CHAT] sig_generate emitted ({source})")

    def _dispatch_monoline_run(self, workflow, payload, *, rewrite_index=None, source="chat"):
        """The additive lane: one conversation row holding the unified RunView, fed by a
        UI-thread RunModelBuilder off the worker's RunEvent stream. Never enters
        _start_assistant_stream / sig_generate / the READY handler."""
        import threading as _threading
        from core.skill_runtime import SpawnBudget
        from core.run_model import RunModelBuilder
        from ui.components.run_view import RunView
        # Build + launch INSIDE a guard, and set the running-state flags ONLY AFTER the worker
        # thread actually starts. A setup failure (preflight, surface append, the load_monoline
        # sys.modules swap, thread spawn) therefore can NEVER leave the send button stuck
        # 'running' or the workshop flag stuck 'RUNNING' -- which would corrupt the next Genesis
        # turn (INV-#1).
        try:
            self._tool_cancel_requested = False
            # Preflight on the MAIN thread BEFORE building the row: WARM the plugin (the one-time
            # sys.modules swap must not run cold on the worker thread) + validate. A failure leaves
            # NO empty run row and NO stuck running flag.
            from engine.monoline_bridge import load_monoline, validate_chat_workflow  # LAZY (INV-#0)
            load_monoline()
            problem = validate_chat_workflow(workflow)
            if problem:
                self._trace_html(f"Pipeline run failed: {problem}", "TOOLS", error=True)
                return
            # One conversation row: the unified RunView. The builder folds the worker's RunEvents
            # on the UI thread; the view subscribes to the resulting model (see _on_run_event).
            builder = RunModelBuilder()
            # one collapsible "tool block" in the chat: small by default, click to reveal the
            # workflow's inner blocks. The ORIGIN_PIPELINE bubble carries the answer (show_final=False).
            view = RunView(show_final=False, collapsible=True)
            if hasattr(self._surface, "_append_card_widget"):
                self._surface._append_card_widget(view)  # one conversation row
            worker = _PipelineWorker(
                workflow, payload,
                parent_turn_id=str(self._last_task_id or ""),
                spawn_budget=SpawnBudget(),
                should_cancel=lambda: bool(self._tool_cancel_requested),
                is_busy=self._monoline_is_busy)  # filters out the 'workshop' flag (see _monoline_is_busy)
            worker.builder = builder
            worker.inline_view = view
            worker.sig_run_event.connect(lambda ev, w=worker: self._apply_run_event(w, ev))
            worker.sig_pipeline_done.connect(lambda ans, w=worker: self._on_pipeline_done(ans, w))
            worker.sig_pipeline_error.connect(self._on_pipeline_error)
            worker.sig_pipeline_stopped.connect(self._on_pipeline_stopped)
            self._pipeline_workers.append(worker)  # anti-GC
            _threading.Thread(target=worker.run, daemon=True).start()
        except Exception as exc:
            self._set_send_button_state(is_running=False)
            self._set_workshop_active(False)
            self._trace_html(f"Monoline run could not start: {exc}", "TOOLS", error=True)
            return
        # The run is genuinely in flight. Mark running NOW (Qt-queued done/error slots run later on
        # the event loop, after this method returns -- so there is no reset-then-set race).
        self._set_workshop_active(True)
        self._set_send_button_state(is_running=True)

    def _monoline_is_busy(self) -> bool:
        """is_busy predicate for a Monoline run. CRITICAL: _dispatch_monoline_run calls
        _set_workshop_active(True), which sets engines['workshop'].status='RUNNING'.
        _engine_is_busy counts 'running' as busy -> WITHOUT filtering the 'workshop' key,
        every 'monolith'-provider block would see itself busy (Arm 2 refuse), exhaust the
        bounded retry, and the pipeline would FAIL on its own activity flag. Filter it out:
        a Monoline run is busy only vs a GENUINE external Genesis/expedition generation."""
        ws = getattr(self.state, "world_state", None)
        if ws is None:
            return False
        try:
            engines = dict((ws.snapshot() or {}).get("engines", {}) or {})
        except Exception:
            return False
        engines.pop("workshop", None)  # do not count our own run-activity flag as busy
        busy = {"running", "generating", "streaming"}
        return any(str(e.get("status", "")).strip().lower() in busy
                   for e in engines.values() if isinstance(e, dict))

    def _apply_run_event(self, worker, ev) -> None:
        """Fold a worker's RunEvent into its RunModel on the UI thread. On RunStarted, register
        the live model (so the companion run browser can list/bind it) and bind the inline view
        (equipped run only; a dry-run has inline_view=None and lives only in the browser)."""
        builder = getattr(worker, "builder", None)
        if builder is None:
            return
        builder.apply(ev)
        from core.run_model import RunStarted, live_runs
        if isinstance(ev, RunStarted) and builder.model is not None:
            live_runs.register(builder.model)
            view = getattr(worker, "inline_view", None)
            if view is not None:
                view.bind(builder.model)

    def _on_pipeline_done(self, answer: str, worker=None) -> None:
        # The OUTPUT-port value renders as an ordinary assistant bubble, origin=ORIGIN_PIPELINE.
        # It does NOT flow through _process_last_response_commands / verifier / ACU / bearing.
        # We also fold a compact per-block trace into the STORED content as an [ATTACHED] block:
        # core.attached_blocks strips it from the bubble, but build_engine_history keeps it, so the
        # next turn's context carries both the final answer and what each workshop step produced.
        stored = answer
        model = getattr(getattr(worker, "builder", None), "model", None)
        if model is not None:
            from core.run_model import build_workshop_trace_attachment
            trace = build_workshop_trace_attachment(model)
            if trace:
                stored = f"{answer}\n\n{trace}" if answer else trace
        idx = self._sessions.insert_message(
            len(self._current_session["messages"]), "assistant", stored,
            extra={"origin": ORIGIN_PIPELINE})
        self._append_message_widget(idx)
        self._set_send_button_state(is_running=False)
        self._set_workshop_active(False)
        # Sync the new turn into the engine NOW so the workshop output is in context for the very
        # next message. Previously this handler never synced, so the answer reached the model only
        # after the following turn's generation (a turn late).
        self.sig_sync_history.emit(self._build_engine_history_from_session())

    def _on_pipeline_error(self, message: str) -> None:
        # A completed-with-error run already flipped its RunView via the RunFinished event; a hard
        # worker exception (no RunFinished) is surfaced here in the trace. Always reset run state.
        self._trace_html(f"Pipeline run failed: {message}", "TOOLS", error=True)
        self._set_send_button_state(is_running=False)
        self._set_workshop_active(False)

    def _on_pipeline_stopped(self) -> None:
        # User STOP: a clean halt, not a failure. Say NOTHING in chat — the RunView already shows
        # "stopped" in place (RunFinished(stopped=True)). Just reset run state. No message, no insert.
        self._set_send_button_state(is_running=False)
        self._set_workshop_active(False)

    def _run_monoline_dry(self, workflow_id: str) -> None:
        """decision 5: in-panel Test dry-run -- runs the flow off-thread WITHOUT Set-Active. It has
        no chat row, so the companion run browser is its only live surface: _on_run_event registers
        the live RunModel in run_model.live_runs for the browser to bind. Never touches active-flow."""
        import threading as _threading
        from core.skill_runtime import SpawnBudget
        from core.run_model import RunModelBuilder
        wf = self._workflow_registry.get(workflow_id)
        if wf is None or wf.kind != "monoline":
            return
        # Same guard discipline as _dispatch_monoline_run: set the workshop flag only after the
        # worker launches, so a setup failure leaves no stuck 'RUNNING' flag. (No send button here:
        # a dry-run streams into the browser and does not lock the chat composer.)
        try:
            # WARM on the MAIN thread before the worker starts (see _dispatch_monoline_run rationale).
            from engine.monoline_bridge import load_monoline  # LAZY (INV-#0)
            load_monoline()
            worker = _PipelineWorker(
                wf, {"prompt": ""},  # dry-run uses empty input; the flow's own seeds drive it
                parent_turn_id=str(self._last_task_id or ""),
                spawn_budget=SpawnBudget(),
                should_cancel=lambda: bool(getattr(self, "_tool_cancel_requested", False)),
                is_busy=self._monoline_is_busy)
            worker.builder = RunModelBuilder()
            worker.inline_view = None   # no chat row; the browser binds the live model
            worker.sig_run_event.connect(lambda ev, w=worker: self._apply_run_event(w, ev))
            worker.sig_pipeline_error.connect(self._on_pipeline_error)
            worker.sig_pipeline_stopped.connect(self._on_pipeline_stopped)
            self._pipeline_workers.append(worker)
            _threading.Thread(target=worker.run, daemon=True).start()
        except Exception as exc:
            self._set_workshop_active(False)
            self._trace_html(f"Monoline dry-run could not start: {exc}", "TOOLS", error=True)
            return
        self._set_workshop_active(True)

    def _open_monoline_canvas(self, world_id) -> None:
        """Open the Monoline Create/Edit canvas as a separate top-level window (same
        QApplication). LAZY bridge import (INV-#0). world_id=None => Create; else Edit."""
        try:
            from engine.monoline_bridge import open_create_canvas  # LAZY
        except Exception as exc:
            self._trace_html(f"Monoline canvas unavailable: {exc}", "TOOLS", error=True)
            return
        open_create_canvas(self, world_id=world_id)

    def wire_workshop_library(self, pane) -> None:
        """Connect the WorkshopLibraryPane's signals to their chat-host consumers, so
        the Test / Edit / Create / Run buttons are live (not dead signals)."""
        pane.sig_test_run.connect(self._run_monoline_dry)
        pane.sig_open_monoline.connect(self._open_monoline_canvas)
        pane.sig_focus_chat.connect(lambda: self.setFocus())

    def _submit_prompt(self, prompt: object) -> None:
        txt = str(prompt or "").strip()
        if not txt:
            return
        self._tool_cancel_requested = False
        self._clear_tool_followup_state()
        world_state = getattr(self.state, "world_state", None)
        if world_state is not None:
            world_state.set_last_prompt(txt)
        if self._is_editing_message():
            self.sig_debug.emit(
                f"[CHAT] submit_edit: idx={self._editing_user_index}, text={repr(txt[:60])}, msgs={len(self._current_session['messages'])}"
            )
            self._commit_edit_prompt(txt)
            return
        self.sig_debug.emit(f"[CHAT] send: text={repr(txt[:60])}, msgs={len(self._current_session['messages'])}")
        # Acatalepsy v1.1.5: arm the assistant-capture flag for THIS UI turn.
        # CONNECT-routed messages don't go through _submit_prompt, so they
        # never arm the flag — that prevents the CONNECT response from being
        # double-logged via the UI mirror path. The flag is one-shot: the
        # first terminal READY consumes it.
        self._chat_canonical_assistant_pending_capture = True
        user_idx = self._add_message("user", txt, extra={"origin": ORIGIN_UI_USER})
        # Acatalepsy v1.1: capture UI-side user message in canonical_log.
        self._log_chat_to_canonical("user", txt)
        self._append_message_widget(user_idx)

        # @mention routing is only meaningful for UI-typed messages — this
        # entire path is reached only from local user input. CONNECT-routed
        # messages bypass _submit_prompt entirely and enter via
        # _submit_agent_message, so they never hit _resolve_at_mentions.
        mentions = self._resolve_at_mentions(txt)
        if mentions:
            self._dispatch_at_mentions(txt, mentions)
            return

        self._dispatch_generation(txt, source=f"send:{txt[:20]}")

    def _submit_agent_message(
        self,
        agent_name: object,
        prompt: object,
        *,
        auto_dispatch: bool = False,
        add_to_timeline: bool = True,
        approved: bool = False,
        message_index: int | None = None,
    ) -> int | None:
        name = str(agent_name or "Agent").strip() or "Agent"
        txt = str(prompt or "").strip()
        if not txt:
            return None

        idx: int | None = message_index if isinstance(message_index, int) else None
        if add_to_timeline:
            display_text = f"[{name}] {txt}"
            idx = self._add_message(
                "agent",
                display_text,
                extra={
                    "agent_name": name,
                    "agent_approved": bool(approved),
                    # Tag origin so downstream consumers (canonical_log,
                    # telemetry, render) can distinguish a CONNECT-routed
                    # inbound from a locally-typed message without
                    # inferring from role.
                    "origin": ORIGIN_CONNECT_AGENT,
                },
            )
            self._append_message_widget(idx)

        if not auto_dispatch:
            return idx

        self._clear_tool_followup_state()
        if idx is not None:
            msgs = self._current_session.get("messages", [])
            if 0 <= idx < len(msgs):
                msg = msgs[idx]
                if isinstance(msg, dict):
                    msg["agent_approved"] = True
                    msg["updated_at"] = self._sessions.now_iso()
                    self._current_session["updated_at"] = msg["updated_at"]
        injected = f"[AGENT:{name}]\n{txt}"
        self._dispatch_generation(injected, source=f"agent:{name}")
        return idx

    def _clear_tool_followup_state(self) -> None:
        self._assistant_box.clear_tool_followup_state()
        self._tool_followup_retries = 0
        self._tool_parse_retries = 0
        self._non_convergence_retries = 0
        self._tool_followup_depth = 0
        self._tool_loop_active = False
        self._tool_cache.clear()
        self._stream_raw = ""
        self._pending_tool_results.clear()
        self._workshop_turn_count = 0  # per-turn run_workshop launch cap resets at the turn boundary
        self._card_author_turn_count = 0  # per-turn author_workshop_card cap resets at the turn boundary

    def _build_tool_followup_prompt(self) -> str:
        return build_tool_followup_prompt(self._pending_tool_results)

    def _maybe_nudge_non_convergence(self) -> bool:
        """Recover a turn that produced ONLY reasoning — no public answer and no
        tool action ("didn't say anything"). Re-prompt once, bounded, anchored
        on what the user actually asked (the bearing goal may be stale/None).
        Returns True iff a nudge was dispatched. Detector is pure:
        core.command_feedback.is_non_convergent. Soft-fail — never breaks the
        READY handler.
        """
        try:
            if self._non_convergence_retries >= self._MAX_NON_CONVERGENCE_RETRIES:
                return False
            messages = self._current_session.get("messages", [])
            if not messages:
                return False
            last = messages[-1]
            if last.get("role") != "assistant":
                return False
            raw = str(last.get("text", ""))
            # Genuinely-empty output is the empty-retry path's job; a model that
            # signalled done has converged. Only act on reasoning-without-answer.
            if not raw.strip() or "[TOOL_LOOP_DONE]" in raw:
                return False
            try:
                public = AssistantStreamNormalizer.from_text(raw).answer_text
            except Exception:
                public = raw
            from core.command_feedback import (
                answer_trapped_in_think,
                build_non_convergence_nudge,
                is_non_convergent,
                recover_trapped_answer,
                should_recover_trapped,
            )
            if not is_non_convergent(public, had_tool_call=False, done_signal=False):
                return False
            # Recover-and-suppress (2026-06-24): when the answer is CONFIDENTLY
            # trapped in an unbalanced <think> (a completed block, then a re-opened
            # one holding the answer), pull it out and surface it INSTEAD of firing
            # the re-emit. The re-emit injects an ephemeral role:user nudge the model
            # reads as the USER correcting it -> it hallucinates "E corrected my think
            # tag", re-reasons, and doubles the output, contaminating the rated turn.
            # The peer/training path's _clean_agent_response already orphan-salvages
            # this same text, so returning False rides the normal idle-completion and
            # the peer gets a clean single-turn answer. should_recover_trapped is
            # biased to regen; pure-truncation / unrecoverable cases fall through to
            # the nudge below (the blessed worst-case).
            if should_recover_trapped(raw):
                recovered = recover_trapped_answer(raw)
                # capture-on-fire: a rare, under-modelled bug — log the raw so its
                # real shape/frequency becomes observable (observability contract).
                self.sig_debug.emit(
                    "[CONVERGENCE] answer trapped in unbalanced <think>; recovered "
                    f"{len(recovered)} chars, suppressing re-emit. "
                    f"raw[:400]={raw[:400]!r}"
                )
                try:
                    last["text"] = recovered
                    surface = getattr(self, "_surface", None)
                    if surface is not None and hasattr(surface, "rerender_row"):
                        surface.rerender_row(len(messages) - 1)
                except Exception:
                    pass
                return False
            ask = ""
            for m in reversed(messages):
                if m.get("role") == "user" and not m.get("ephemeral"):
                    ask = str(m.get("text", "") or m.get("content", "") or "")
                    break
            self._non_convergence_retries += 1
            # Cause-aware recovery: when the answer is trapped in an unbalanced
            # <think> (the common "behind the reasoning curtain" failure), the
            # single allowed retry gets a surgical "re-emit outside think" nudge
            # instead of the generic one that took two re-emits to recover.
            trapped = answer_trapped_in_think(raw)
            nudge = build_non_convergence_nudge(ask, trapped=trapped)
            self.sig_debug.emit(
                f"[CONVERGENCE] non-convergent turn "
                f"({'answer trapped in unbalanced <think>' if trapped else 'reasoning, no answer/action'}); "
                f"nudging {self._non_convergence_retries}/"
                f"{self._MAX_NON_CONVERGENCE_RETRIES}"
            )
            self._dispatch_generation(
                {"prompt": nudge, "ephemeral": True},
                rewrite_index=None,
                source="non_convergence_nudge",
            )
            return True
        except Exception as exc:  # noqa: BLE001
            self.sig_debug.emit(
                f"[CONVERGENCE] nudge failed: {type(exc).__name__}: {exc}"
            )
            return False

    def _build_parse_retry_prompt(
        self,
        raw: str,
        *,
        attempted_backslash_repair: bool = False,
        attempted_tag_repair: bool = False,
    ) -> str:
        hints: list[str] = [
            "Re-emit exactly one corrected <tool_call>...</tool_call> block and nothing else.",
            'Use strict valid JSON. Single calls: {"name":"<tool>","arguments":{...}}. Batch/chain: {"calls":[...],"mode":"parallel|chain"}.',
            "For Windows paths, either use forward slashes (C:/Users/...) or escaped backslashes (C:\\\\Users\\\\...).",
        ]
        if attempted_tag_repair:
            hints.append("Previous attempt looked like a missing closing </tool_call> tag.")
        if attempted_backslash_repair:
            hints.append("Previous attempt looked like invalid JSON backslashes in a string.")
        raw_snippet = str(raw or "").strip()[:300]
        return (
            "The previous tool command could not be parsed.\n\n"
            + "\n".join(hints)
            + f"\n\nMalformed block content:\n{raw_snippet}"
        )

    def _trim_pending_tool_results(self) -> None:
        total_len = sum(len(item) for item in self._pending_tool_results)
        while self._pending_tool_results and total_len > self._MAX_TOOL_FOLLOWUP_CHARS:
            removed = self._pending_tool_results.pop(0)
            total_len -= len(removed)

    _INCOMPLETE_ACTION_PATTERNS = INCOMPLETE_ACTION_PATTERNS

    def _detect_incomplete_action(self, text: str, tool_ran: bool = False) -> str | None:
        """Return a nudge prompt if the response narrates an unexecuted action.

        Delegates to core.incomplete_action; `tool_ran` suppresses the nudge
        when a tool already executed this turn (the narrated action was
        fulfilled), and <think> reasoning is excluded from the scan.
        """
        return detect_incomplete_action(
            text, tool_ran=tool_ran, patterns=self._INCOMPLETE_ACTION_PATTERNS
        )

    @staticmethod
    def _strip_tool_loop_done_tag(text: str) -> tuple[str, bool]:
        marker = "[TOOL_LOOP_DONE]"
        if marker not in text:
            return text, False
        cleaned_lines: list[str] = []
        removed = False
        for raw_line in str(text or "").splitlines():
            line = raw_line.replace(marker, "").strip()
            if raw_line.strip() == marker:
                removed = True
                continue
            if marker in raw_line:
                removed = True
            cleaned_lines.append(line if marker in raw_line else raw_line)
        cleaned = "\n".join(cleaned_lines).strip()
        return cleaned, removed

    # ---- Capability verb handlers (AddonRegistry verification) ----
    def generate_text(self, prompt: str) -> None:
        self._send_message(str(prompt))

    def chat(self, prompt: str) -> None:
        self._send_message(str(prompt))

    def load_model(self) -> None:
        self.sig_load.emit()

    def unload_model(self) -> None:
        self.sig_unload.emit()

    def stream_tokens(self, prompt: str) -> None:
        self._send_message(str(prompt))

    def set_task_id(self, task_id: str) -> None:
        self._last_task_id = str(task_id or "")

    def thinking_enabled(self) -> bool:
        return bool(self._agent_thinking)

    def get_session_messages(self) -> list[dict[str, str]]:
        """Return current chat history in engine/runtime message format.

        Each retained message is prefixed with a [CHANNEL: ...] header via
        core.channel_tag. Past turns get role-only tags (USER / ASSISTANT /
        AGENT); the current generating turn (last user-role message) gets
        a full tag with live plane modes from world_state.

        Already-tagged messages (peer turns whose tag was authored by
        agent_server at intake time) are passed through unchanged — no
        double-tagging.
        """
        from core.channel_tag import build_channel_tag

        messages = self._current_session.get("messages", [])
        if not isinstance(messages, list):
            return []
        # Pre-pass: find the index of the last user-role message (= current
        # generating turn). That message gets include_modes=True.
        last_user_idx = -1
        for i, item in enumerate(messages):
            if not isinstance(item, dict):
                continue
            if str(item.get("role", "")).strip().lower() == "user":
                if item.get("kind") == "command_block":
                    continue
                last_user_idx = i

        out: list[dict[str, str]] = []
        for i, item in enumerate(messages):
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).strip().lower()
            content = str(item.get("text", "")).strip()
            # "agent" role is admitted here so peer-routed turns reach the
            # model's history. Their tag was authored by agent_server at
            # intake (post-_apply_pending_all, via peek) — we don't re-tag.
            if role not in ("user", "assistant", "agent", "system"):
                continue
            if not content:
                continue
            if item.get("kind") == "command_block":
                continue
            # Skip injection when the content already carries a [CHANNEL: ...]
            # header (peer turns; tool envelopes that authored their own).
            if not content.lstrip().startswith("[CHANNEL:"):
                role_token = {
                    "user": "USER",
                    "assistant": "ASSISTANT",
                    "agent": "AGENT",
                }.get(role)
                if role_token is not None:
                    include_modes = (role == "user" and i == last_user_idx)
                    tag = build_channel_tag(role_token, include_modes=include_modes)
                    content = f"{tag}\n\n{content}"
            # Engine expects "user" / "assistant" / "system" only; the
            # "agent" role is collapsed to "user" for prompt-shape purposes
            # (its tag tells the model who actually spoke).
            out_role = "user" if role == "agent" else role
            out.append({"role": out_role, "content": content})
        return out

    def _emit_command_block(self, title: str, detail: str = "") -> None:
        """Render a slash-command receipt as a system-bubble in the chat.

        The trace panel kept these invisible to anyone whose eyes were on
        the conversation; the receipt belongs in the conversation itself.
        Marked `kind=command_block` so it's filtered out of the history
        sent to the model -- the LLM doesn't need to see "user typed /think".
        """
        text = title if not detail else f"{title}\n{detail}"
        idx = self._sessions.add_message(
            "system", text, extra={"kind": "command_block", "origin": ORIGIN_SYSTEM_COMMAND}
        )
        self._append_message_widget(idx)

    def _submit_update(self, update_text):
        partial = "(no output yet)"
        if self._active_assistant_index is not None:
            txt = self._current_session["messages"][self._active_assistant_index]["text"]
            if txt:
                partial = txt

        original = ""
        for msg in reversed(self._current_session["messages"]):
            if msg["role"] == "user":
                original = msg["text"]
                break

        injected = f"""
You were interrupted mid-generation.

Original user request:
{original}

Partial assistant output so far:
{partial}

User update:
{update_text}

Continue from the interruption point. Do not repeat earlier content. Prioritize the user update.
"""

        self.input.clear()
        self._start_update_streaming()
        self._dispatch_generation(injected, rewrite_index=self._active_assistant_index, source="update")

    def _submit_tool_followup(self, tool_result: str) -> None:
        if not tool_result.strip():
            return
        if self._tool_cancel_requested:
            self._tool_cancel_requested = False
            self._clear_tool_followup_state()
            return
        if not self._pending_tool_results:
            self._pending_tool_results.append(tool_result.strip())
        self._trim_pending_tool_results()
        total_len = sum(len(item) for item in self._pending_tool_results)
        self.sig_debug.emit(
            f"[CHAT] _submit_tool_followup: results={len(self._pending_tool_results)}, total_len={total_len}"
        )
        target_index = self._tool_followup_target_index
        if target_index is not None and not self._widget_for_index(target_index):
            target_index = None
        injected = self._build_tool_followup_prompt()
        self._dispatch_generation(
            {"prompt": injected, "ephemeral": True},
            rewrite_index=target_index,
            source="tool_followup",
        )

    def _on_spawn_subagent(self, cmd: dict) -> str:
        """L1 host hook (ctx.on_spawn_subagent). Runs the atom OFF the Qt thread and
        returns immediately so the UI never blocks (R3); folds back via sig_subagent_done."""
        import threading as _threading
        child_level = int(cmd.get("_validated_child_level", 2))
        l1 = ToolExecutionContext(
            archive_dir=ARCHIVE_DIR, level=1, allowed_tools=L1_PRINCIPAL_TOOLS,
            parent_turn_id=str(self._last_task_id or ""),
            spawn_budget=self._spawn_budget,
            should_cancel=lambda: bool(self._tool_cancel_requested))
        child = derive_child_context(l1, child_level,
                                     label=str(cmd.get("frame", f"L{child_level}")))
        ws = getattr(self.state, "world_state", None)
        worker = _SpawnWorker(
            cmd, child_level, child.parent_turn_id, child.allowed_tools,
            child.should_cancel, child.spawn_budget,
            is_busy=lambda ws=ws: _engine_is_busy(ws))
        worker.sig_subagent_done.connect(lambda fenced, w=worker: self._on_subagent_done(fenced, w))
        self._spawn_workers.append(worker)  # keep a ref so it isn't GC'd mid-run
        # active-agents spine: this worker is the source. Register it running and
        # project the snapshot to the model lane (+ UI) — both main-thread here.
        from core.active_agents import AgentRecord, RUNNING
        self._active_agents[id(worker)] = AgentRecord(
            agent_id=str(id(worker)),
            frame=str(cmd.get("frame", f"L{child_level}")),
            level=child_level,
            status=RUNNING,
        )
        self._publish_active_agents()
        self._set_workshop_active(True)
        _threading.Thread(target=worker.run, daemon=True).start()
        return "[spawn_subagent: PENDING]"

    def _publish_active_agents(self) -> None:
        """Project the live source (self._active_agents) two ways: the model-context
        lane (set_active) and the UI strip (sig_agents_changed)."""
        records = list(self._active_agents.values())
        try:
            from core.active_agents import set_active
            set_active(records)
        except Exception:
            pass
        try:
            self.sig_agents_changed.emit(records)
        except Exception:
            pass

    def _on_zoom_agent(self, child_turn_id: str, frame: str) -> None:
        """Zoom into a sub-agent's own trace (the child_turn_id). First cut:
        surface it in the LOG + load the child turn's stage trace into a popup so
        the click DOES something. The richer 'zoomed chat' render is the next
        slice. Never breaks the chat."""
        cid = str(child_turn_id or "").strip()
        if not cid:
            return
        try:
            self._trace_html(f"Zoom into agent '{frame}' — trace {cid}", "AGENT")
        except Exception:
            pass
        try:
            from core import turn_trace
            rec = turn_trace.inspect_one(cid) if hasattr(turn_trace, "inspect_one") else None
            if rec:
                self.sig_debug.emit(f"[agent zoom] {frame}: {rec}")
        except Exception as exc:
            self.sig_debug.emit(f"[agent zoom] trace load failed: {exc!r}")

    def _on_subagent_done(self, fenced: str, worker=None) -> None:
        """Main-thread slot (queued from the worker thread). Flips the agent to
        DONE in the live source (keeping it, with its child_turn_id, so the UI
        can show 'done' and zoom into its trace), then folds the result back into
        the turn via the followup loop (the model's work product — untouched)."""
        if worker is not None:
            from core.active_agents import AgentRecord, DONE, parse_child_turn_id
            prev = self._active_agents.get(id(worker))
            if prev is not None:
                self._active_agents[id(worker)] = AgentRecord(
                    agent_id=prev.agent_id, frame=prev.frame, level=prev.level,
                    status=DONE, child_turn_id=parse_child_turn_id(fenced),
                )
            self._publish_active_agents()
        ws = getattr(self.state, "world_state", None)
        if not _engine_is_busy(ws):
            self._set_workshop_active(False)
        self._queue_tool_followup(fenced, rewrite_index=None)

    _MAX_WORKSHOP_RUNS_PER_TURN = 4  # the model can fan out at most this many flow runs per turn

    def _on_run_workshop(self, cmd: dict) -> str:
        """L1 host hook (ctx.on_run_workshop): the model asked to run a named Workshop workflow.
        Runs it OFF the Qt thread (UI never blocks) via the Phase-B pipeline worker and folds the
        OUTPUT back into the turn as a tool result through the SAME followup loop spawn_subagent
        uses. L1-ONLY by construction (the gate denies it below L1, so a flow's L3 blocks cannot
        recurse into it). Hard-capped at _MAX_WORKSHOP_RUNS_PER_TURN launches per turn (the counter
        resets at the turn boundary in _clear_tool_followup_state) so the model cannot fan out
        unbounded concurrent flows -- the per-block spawn_budget does NOT bound this (a flow's
        blocks run with empty allowed_tools and never charge it), so the launch cap is the real
        ceiling."""
        import threading as _threading
        cap = getattr(self, "_MAX_WORKSHOP_RUNS_PER_TURN", 4)
        if getattr(self, "_workshop_turn_count", 0) >= cap:
            return f"[run_workshop: per-turn limit of {cap} workflow runs reached]"
        reg = getattr(self, "_workflow_registry", None)
        name = str(cmd.get("name") or cmd.get("workflow") or cmd.get("input") or "").strip()
        wf = reg.get(name) if (reg is not None and name) else None
        if (wf is None or wf.kind != "monoline") and reg is not None and name:
            wf = next((w for w in reg.list_workflows()
                       if w.kind == "monoline" and w.name.lower() == name.lower()), None)
        if wf is None or wf.kind != "monoline":
            return f"[run_workshop: no workflow named '{name}']"
        prompt = str(cmd.get("input") or cmd.get("prompt") or "")
        # warm the plugin on the MAIN thread before the worker (the bridge's one-time sys.modules swap)
        try:
            from engine.monoline_bridge import load_monoline
            load_monoline()
        except Exception as exc:
            return f"[run_workshop: monoline plugin unavailable - {exc}]"
        worker = _PipelineWorker(
            wf, {"prompt": prompt},
            parent_turn_id=str(self._last_task_id or ""),
            spawn_budget=getattr(self, "_spawn_budget", None),
            should_cancel=lambda: bool(self._tool_cancel_requested),
            is_busy=self._monoline_is_busy)   # drops our own 'workshop' flag -> no self-deadlock
        worker.sig_pipeline_done.connect(
            lambda out, n=wf.name, w=worker: self._on_workshop_tool_done(n, out, w))
        worker.sig_pipeline_error.connect(
            lambda err, n=wf.name, w=worker: self._on_workshop_tool_error(n, err, w))
        # User STOP on a model-invoked workshop: just clean up (decrement in-flight, clear the
        # flag) — no error, and no tool-followup (the user stopped; don't re-prompt the model).
        worker.sig_pipeline_stopped.connect(lambda w=worker: self._workshop_run_finished(w))
        self._pipeline_workers.append(worker)
        self._workshop_turn_count = getattr(self, "_workshop_turn_count", 0) + 1
        self._workshop_inflight = getattr(self, "_workshop_inflight", 0) + 1
        self._set_workshop_active(True)
        _threading.Thread(target=worker.run, daemon=True).start()
        return f"[run_workshop: '{wf.name}' PENDING]"

    def _workshop_run_finished(self, worker) -> None:
        """Decrement the in-flight count and clear the 'workshop' activity flag when the LAST run
        finishes -- UNCONDITIONALLY. (Do NOT gate the clear on _engine_is_busy: it counts our own
        'workshop=RUNNING' flag as busy, so the clear would never fire, leaving the flag stuck and
        poisoning every later spawn_subagent with a false 'busy'.)"""
        self._workshop_inflight = max(0, getattr(self, "_workshop_inflight", 1) - 1)
        if self._workshop_inflight == 0:
            self._set_workshop_active(False)
        try:
            self._pipeline_workers.remove(worker)
        except (ValueError, AttributeError):
            pass

    def _on_workshop_tool_done(self, name: str, output: str, worker=None) -> None:
        self._workshop_run_finished(worker)
        self._queue_tool_followup(f"[run_workshop: '{name}' completed]\n{output}", rewrite_index=None)

    def _on_workshop_tool_error(self, name: str, error: str, worker=None) -> None:
        # fail-soft: the model sees the failure as a tool result and can react (never hangs).
        self._workshop_run_finished(worker)
        self._queue_tool_followup(f"[run_workshop: '{name}' failed - {error}]", rewrite_index=None)

    _MAX_CARD_AUTHORS_PER_TURN = 3  # model can author at most this many new cards per turn

    def _on_author_workshop_card(self, cmd: dict) -> str:
        """L1 host hook (ctx.on_author_workshop_card): the model wants to author a new Workshop card.
        SYNCHRONOUS: build + validate + write-file is fast and model-free, so the result is
        returned in-line, closing the fix loop within the SAME turn — the model sees validation
        errors immediately and can re-author. L1-ONLY by construction (the gate denies it below
        L1). Hard-capped at _MAX_CARD_AUTHORS_PER_TURN per turn (counter resets at the turn
        boundary in _clear_tool_followup_state). Never writes outside WORKFLOWS_DIR; refuses
        if the id already exists (no clobber)."""
        import re as _re
        import json as _json

        cap = getattr(self, "_MAX_CARD_AUTHORS_PER_TURN", 3)
        if getattr(self, "_card_author_turn_count", 0) >= cap:
            return f"[author_workshop_card: per-turn limit of {cap} card authors reached]"

        # --- warm the bridge on the MAIN thread (same as _on_run_workshop) ---
        try:
            from engine.monoline_bridge import load_monoline
            m = load_monoline()
        except Exception as exc:
            return f"[author_workshop_card: monoline plugin unavailable - {exc}]"

        # --- parse cmd: name + blueprint payload ---
        raw_name = str(cmd.get("name") or "").strip()
        payload = cmd.get("blueprint") or cmd.get("world") or cmd.get("preset")
        # Tolerate the model dumping the blueprint at the top level (no 'blueprint' key)
        if payload is None:
            # Check if cmd itself looks like a blueprint (has 'blocks')
            if "blocks" in cmd:
                payload = dict(cmd)
            else:
                return "[author_workshop_card: missing 'blueprint' key with blocks/connections]"
        if not isinstance(payload, dict):
            return "[author_workshop_card: 'blueprint' must be a JSON object]"

        # --- id sanitizer ---
        if not raw_name:
            return "[author_workshop_card: 'name' is required]"
        # Slug: lowercase, replace runs of non-alphanumeric with '-', strip leading/trailing '-'
        slug = _re.sub(r"[^a-z0-9]+", "-", raw_name.lower()).strip("-")
        if not slug:
            return "[author_workshop_card: name produces an empty id after sanitization]"
        # Reject reserved sentinel
        from core.workflow_registry import GENESIS_ID
        if slug == GENESIS_ID:
            return f"[author_workshop_card: id 'genesis' is reserved; choose another name]"
        # Reject any path-separator or traversal pattern in the RAW name
        if any(c in raw_name for c in ("/", "\\")) or ".." in raw_name:
            return f"[author_workshop_card: name must not contain path separators or '..']"

        # --- determine WORKFLOWS_DIR from the registry (so tests can inject a tmp dir) ---
        reg = getattr(self, "_workflow_registry", None)
        if reg is not None:
            workflows_dir = getattr(reg, "workflows_dir", None)
        else:
            from core.workflow_registry import WORKFLOWS_DIR
            workflows_dir = WORKFLOWS_DIR
        if workflows_dir is None:
            from core.workflow_registry import WORKFLOWS_DIR
            workflows_dir = WORKFLOWS_DIR

        # --- collision check BEFORE building ---
        dest = workflows_dir / f"{slug}.monoline"
        if dest.exists():
            return f"[author_workshop_card: card '{slug}' already exists; choose another name]"

        # --- build preset ---
        try:
            preset = m["blueprint"].build_preset_from_blueprint(payload, strict=False)
        except Exception as exc:
            return f"[author_workshop_card: blueprint error - {exc}]"

        # --- validate (Phase-1 gate) ---
        try:
            result = m["blueprint"].validate_preset(preset)
        except Exception as exc:
            return f"[author_workshop_card: validation error - {exc}]"

        if result.errors:
            return "[author_workshop_card: validation failed - " + "; ".join(result.errors) + "]"

        # --- set id / name / description on preset, then write ---
        preset.id = slug
        preset.name = str(payload.get("name") or raw_name or slug)
        if payload.get("description"):
            preset.description = str(payload["description"])

        try:
            from pathlib import Path as _Path
            workflows_dir = _Path(workflows_dir)
            workflows_dir.mkdir(parents=True, exist_ok=True)
            dest = workflows_dir / f"{slug}.monoline"
            data = preset.to_dict()
            data["id"] = slug  # ensure the top-level JSON key is the slug
            dest.write_text(_json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            return f"[author_workshop_card: write error - {exc}]"

        # increment cap counter
        self._card_author_turn_count = getattr(self, "_card_author_turn_count", 0) + 1

        n_blocks = len(getattr(preset, "blocks", []))
        warnings_part = ""
        if result.warnings:
            warnings_part = "; warnings: " + "; ".join(result.warnings[:3])
        return (
            f"[author_workshop_card: saved '{slug}' ({n_blocks} blocks){warnings_part}. "
            f"Equip it or run_workshop '{slug}'.]"
        )

    def _set_workshop_active(self, active: bool) -> None:
        """Set the world_state 'workshop' flag that the companion's evaluate_state reads
        (best-effort; the WorkshopPane polls turn_trace independently of this flag)."""
        ws = getattr(self.state, "world_state", None)
        if ws is None:
            return
        setter = getattr(ws, "set_engine_status", None)
        if callable(setter):
            try:
                setter("workshop", "RUNNING" if active else "idle")
            except Exception:
                pass

    def _queue_tool_followup(self, tool_result: str, rewrite_index: int | None = None) -> None:
        if not tool_result.strip():
            return
        if self._tool_cancel_requested:
            self._tool_cancel_requested = False
            self._clear_tool_followup_state()
            return
        if self._tool_followup_depth >= self._MAX_TOOL_FOLLOWUPS:
            self.sig_debug.emit(
                f"[CHAT] tool followup depth limit reached ({self._MAX_TOOL_FOLLOWUPS}); stopping loop"
            )
            self._trace_html(
                f"Tool follow-up limit reached ({self._MAX_TOOL_FOLLOWUPS}). Stopping autonomous loop.",
                "TOOLS",
                error=True,
            )
            self._clear_tool_followup_state()
            return
        self.sig_debug.emit(f"[CHAT] _queue_tool_followup: tool_len={len(tool_result)}")
        self._pending_tool_results.append(tool_result.strip())
        self._tool_followup_depth += 1
        self._trim_pending_tool_results()
        self._tool_followup_target_index = rewrite_index
        self._tool_followup_retries = 0  # reset on successful progression
        self._tool_parse_retries = 0
        # Phase 4 migration: emit a TurnStreamEndedEvent indicating the
        # happy-path continuation (had_tool_call=True, had_continuation=True).
        # This gives the tool_loop_continuation policy a baseline of clean
        # closes; the no-fire detector relies on the matching emission from
        # paths that *don't* end up here (Phase 4b — not yet wired).
        self._emit_pipeline_stream_ended(had_tool_call=True, had_continuation=True)
        QTimer.singleShot(0, lambda tr=tool_result: self._submit_tool_followup(tr))

    def _emit_pipeline_stream_ended(
        self,
        *,
        had_tool_call: bool,
        had_continuation: bool,
        closed_lanes: tuple = ("answer",),
    ) -> None:
        """Best-effort: publish TurnStreamEndedEvent for the current turn."""
        try:
            from monokernel.turn_pipeline import TurnContext, get_pipeline
            from core.turn_pipeline_events import TurnStreamEndedEvent
            turn_id = self._turn_id_for_message_index(
                len(self._current_session.get("messages", [])) - 1
            )
            if not turn_id:
                return
            ctx = TurnContext(turn_id=turn_id, started_at=0.0)
            get_pipeline().publish(
                TurnStreamEndedEvent(
                    closed_lanes=tuple(closed_lanes),
                    had_tool_call=had_tool_call,
                    had_continuation=had_continuation,
                ),
                ctx,
                source_kind="kernel",
                source_name="chat_page",
            )
        except Exception as exc:  # noqa: BLE001
            self.sig_debug.emit(
                f"[PIPELINE] TurnStreamEndedEvent emit failed: "
                f"{type(exc).__name__}: {exc}"
            )

    def _start_assistant_stream(self):
        self.sig_debug.emit(f"[CHAT] _start_assistant_stream: msgs_before={len(self._current_session['messages'])}")
        self._stream_debug_chunk_count = 0
        self._stream_debug_char_count = 0
        self._stream_raw = ""
        self._active_assistant_index = self._assistant_box.start_new_stream()
        self._active_widget = self._append_message_widget(self._active_assistant_index)
        self._active_item = None

    def on_engine_event(self, engine_key: str, event: str, payload: object) -> None:
        if engine_key != getattr(self, "_engine_key", "llm"):
            return
        if event == "token":
            self.append_token(str(payload or ""))
            return
        if event == "trace":
            self.append_trace(str(payload or ""))
            return
        if event == "status":
            status = self._coerce_status(payload)
            if status is not None:
                self._handle_status(status)
            else:
                self._trace_html(f"Unrecognized status payload: {html.escape(repr(payload))}", "ERROR", error=True)
            return
        if event == "finished":
            self._handle_finished(str(payload or ""))
            return
        if event == "model_capabilities" and isinstance(payload, dict):
            self._handle_model_capabilities(payload)

    def _coerce_status(self, payload: object) -> SystemStatus | None:
        if isinstance(payload, SystemStatus):
            return payload
        if hasattr(payload, "value"):
            raw = getattr(payload, "value", None)
            if isinstance(raw, str):
                try:
                    return SystemStatus(raw)
                except ValueError:
                    return None
        if isinstance(payload, str):
            try:
                return SystemStatus(payload)
            except ValueError:
                upper = payload.upper()
                try:
                    return SystemStatus[upper]
                except KeyError:
                    return None
        return None

    def append_token(self, t):
        if not t:
            return
        self._stream_debug_chunk_count += 1
        self._stream_debug_char_count += len(t)
        self._stream_raw += t
        self._try_stream_preexec()
        self._append_assistant_token(t)
        self._update_progress_markers()
        if self._active_widget is None:
            target_index = self._rewrite_assistant_index
            if target_index is None:
                target_index = self._active_assistant_index
            if target_index is not None:
                self._active_widget = self._widget_for_index(target_index)
                self._active_item = None
        if self._active_widget is None:
            return
        # Reveal the widget on first token (hidden for tool-followup streams)
        if not self._active_widget.isVisible():
            self._active_widget.setVisible(True)
            self._active_widget.sig_height_changed.emit()

        sb = self.message_list.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - 40
        self._auto_scroll_on_height_change = at_bottom
        render_start = time.perf_counter()
        display_update = self._assistant_box.consume_display_chunk(t)
        self._apply_display_update(self._active_widget, display_update)
        render_ms = (time.perf_counter() - render_start) * 1000.0
        if (
            self._stream_debug_chunk_count <= 4
            or self._stream_debug_chunk_count % 64 == 0
            or render_ms >= 8.0
        ):
            self.sig_debug.emit(
                "[STREAM] "
                f"chunk={self._stream_debug_chunk_count} "
                f"chars={self._stream_debug_char_count} "
                f"len={len(t)} "
                f"render_ms={render_ms:.2f} "
                f"widget_h={self._active_widget.height() if self._active_widget is not None else -1} "
                f"scroll_max={sb.maximum()}"
            )

    def _flush_token_batch(self):
        if self._token_batch_timer.isActive():
            self._token_batch_timer.stop()
        self._token_batch.clear()

    def _try_stream_preexec(self) -> None:
        """Pre-execute safe, fast tools as soon as their closing tag lands mid-stream.

        Warms ``self._tool_cache`` so ``process_response()`` gets a cache hit
        instead of re-running the same tool after generation finishes.
        Only runs tools in :data:`STREAMING_PREEXEC_TOOLS` (read-only, no side
        effects).

        Routes parsing through :func:`core.cmd_parser.extract_commands` so
        double-nested envelopes — the case where a model echoes the literal
        ``<tool_call>...</tool_call>`` markup from the retry-prompt around its
        actual JSON envelope — are extracted correctly. The previous inline
        ``re.finditer`` + ``json.loads`` approach silently failed on nested
        envelopes because the non-greedy regex captured outer-open through
        inner-close, leaving the inner ``<tool_call>`` prefix on the JSON
        payload. ``extract_commands`` peels that prefix; tools fire mid-stream.
        See ``tests/test_skills.py::test_extract_commands_handles_double_nested_envelope_with_name_args``.

        Trim semantics: after each pass that found at least one complete
        envelope, advance past the last ``</tool_call>`` so subsequent tokens
        don't re-scan the same buffer prefix. Safe because ``_stream_raw`` is
        only consumed here — the canonical assistant message text accumulates
        through a separate path in :meth:`append_token`.
        """
        raw = self._stream_raw
        # Quick bail-out — no closing tag yet.
        if "</tool_call>" not in raw:
            return

        from core.cmd_parser import extract_commands
        from core.skill_registry import canonical_tool_name as _ctn

        def _get_ctx() -> ToolExecutionContext:
            return ToolExecutionContext(
                archive_dir=ARCHIVE_DIR,
                on_open_addon=None,
                bridge=self._bridge,
                guard=self._guard,
                world_state=getattr(self.state, "world_state", None),
                on_generate_audio=None,
                on_set_session_meta=None,
                on_ask_user=self._on_ask_user,
                should_cancel=None,
                result_cache=self._tool_cache,
                vision_artifact_bridge=self._vision_artifact_bridge,
            )

        cmds = extract_commands(raw)
        if not cmds:
            return

        for cmd in cmds:
            if cmd.get("_parse_error"):
                continue
            # Skip batch/chain — preexec is single-call read-only only.
            if isinstance(cmd.get("calls"), list):
                continue
            tool = _ctn(str(cmd.get("tool") or cmd.get("skill") or cmd.get("op") or ""))
            if tool not in STREAMING_PREEXEC_TOOLS:
                continue
            try:
                execute_tool_call_enveloped(cmd, _get_ctx())
            except Exception:
                pass

        # extract_commands has already examined everything up to the last close
        # tag; future tokens land beyond it. Trim past the last </tool_call> so
        # we don't re-scan on every subsequent token batch.
        last_close = raw.rfind("</tool_call>")
        if last_close >= 0:
            self._stream_raw = raw[last_close + len("</tool_call>"):]

    def _apply_display_update(self, widget, update) -> None:
        if widget is None or update is None or not getattr(update, "has_changes", lambda: False)():
            return
        if hasattr(widget, "apply_stream_update"):
            widget.apply_stream_update(update)

    def _apply_assistant_snapshot(self, widget, text: str, *, close_open: bool = True) -> None:
        if widget is None:
            return
        snapshot = self._assistant_box.build_display_snapshot(text or "", close_open=close_open)
        has_visible_text = bool((snapshot.answer_text or "").strip() or (snapshot.thinking_text or "").strip())
        widget.setVisible(has_visible_text)
        if hasattr(widget, "apply_assistant_display"):
            widget.apply_assistant_display(
                snapshot.answer_text,
                snapshot.thinking_text,
                thinking_done=not snapshot.thinking_active,
            )
        if hasattr(self, "_surface") and hasattr(self._surface, "sync_widget_geometry"):
            self._surface.sync_widget_geometry(widget)

    def on_guard_finished(self, engine_key, task_id):
        if engine_key != getattr(self, "_engine_key", "llm"):
            return
        self._handle_finished(str(task_id or ""))

    def _handle_finished(self, task_id: str) -> None:
        if not self._current_session.get("messages"):
            return
        self._assistant_box.note_finished(task_id)
        # Stamp the task_id on the most recent assistant message so later
        # outcome recording (thumbs/copy/delete/regen, /rating) can map a
        # message index back to the turn that produced it. Required for
        # Layer D outcome capture.
        try:
            tid = str(task_id or "").strip()
            if tid:
                msgs = self._current_session.get("messages", [])
                for msg in reversed(msgs):
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        msg["task_id"] = tid
                        break
        except Exception:
            pass
        # Force a single, deterministic markdown re-render at stream end so
        # any plain-text streamed content is reformatted in one place. This
        # eliminates the late-finalize flicker where post-hoc sentinel strip
        # would change the document after the user had already seen it.
        widget = getattr(self, "_active_widget", None)
        if widget is not None and hasattr(widget, "finalize"):
            try:
                widget.finalize()
            except Exception:
                pass

    # ── ACU (Acatalepsy) memory ──────────────────────────────────────────

    def _extract_and_store_acus(self) -> None:
        """Extract ACU content from the last assistant message and store it."""
        messages = self._current_session.get("messages", [])
        if not messages:
            return
        last = messages[-1]
        if last.get("role") != "assistant":
            return
        text = str(last.get("text", ""))
        if "<acatalepsy>" not in text:
            return
        norm = AssistantStreamNormalizer.from_text(text)
        acu_raw = norm.acu_text.strip()
        if not acu_raw:
            return
        # Split on newlines — each line is a separate claim
        claims = [line.strip() for line in acu_raw.splitlines() if line.strip()]
        if claims:
            ids = self._acu_store.ingest_many(claims, source="model")
            self.sig_debug.emit(f"[ACU] stored {len(ids)} claims")

    # ── Response verifier (READY-time deterministic checks) ──────────────

    def _run_response_verifier(self) -> None:
        """READY-time pipeline emit + optional deterministic verifier.

        Pipeline emit (TurnReadyEvent) fires regardless of
        MONOLITH_VERIFIER_V1 so output_sanitizer + verifier_bridge
        receive the event and gate on their own kill switches. The
        synchronous verify_response call is gated by the verifier flag.

        Observation-only: any verdict is stashed under
        ``_current_session["last_verification"]`` and emitted to the
        debug trace. Does NOT mutate the assistant's text and does NOT
        inject anything into the model's context — per the architectural
        choice (advanced Monolith ``self_aware_loop.py:5-9``) that
        self-judgment text causes spiraling.

        Body delegates to ``core.chat_finalize.finalize_assistant_turn``
        for testability — chat.py is Qt-coupled at module load; the
        helper is not.
        """
        messages = self._current_session.get("messages", [])
        if not messages:
            return
        last = messages[-1]
        if last.get("role") != "assistant":
            return
        raw = str(last.get("text", ""))
        if not raw.strip():
            return
        try:
            norm = AssistantStreamNormalizer.from_text(raw)
            public = norm.answer_text
        except Exception:
            public = raw

        def _record(payload: dict) -> None:
            self._current_session["last_verification"] = payload

        # MonoFrame v2: thread the live ask into config so chat_finalize can
        # digest it (input_digest) for the standing frame-selection recorder.
        try:
            from addons.system.bearing.stateless_reframe import session_asks as _mf_sa
            _mf_asks = _mf_sa(messages)
            self.config["_user_input"] = _mf_asks[-1] if _mf_asks else ""
        except Exception:
            pass

        finalize_assistant_turn(
            raw=raw,
            public=public,
            config=self.config,
            emit_pipeline_ready=lambda r, p, t: self._emit_pipeline_turn_ready(r, p, tools_used=t),
            record_verdict=_record,
            on_debug=self.sig_debug.emit,
        )

        # MonoFrame v2 (CorrectionCards) is correction-triggered (/frame), not
        # per-turn — the v1 auto per-turn second-opinion loop was superseded and
        # removed. Read-side (nearest human card injected into the frame) lives in
        # bearing/compiler._apply_correction_example; write-side is the /frame
        # command. Nothing fires here on a normal turn.

    def _emit_pipeline_turn_ready(
        self, raw_answer: str, public_answer: str, *, tools_used: tuple = (),
    ) -> None:
        """Best-effort: publish TurnReadyEvent for the current assistant turn."""
        try:
            from monokernel.turn_pipeline import TurnContext, get_pipeline
            from core.turn_pipeline_events import TurnReadyEvent
            msg_idx = len(self._current_session.get("messages", [])) - 1
            turn_id = self._turn_id_for_message_index(msg_idx)
            if not turn_id:
                return
            ctx = TurnContext(turn_id=turn_id, started_at=0.0)
            get_pipeline().publish(
                TurnReadyEvent(
                    raw_answer=raw_answer,
                    public_answer=public_answer,
                    tools_used=tools_used,
                ),
                ctx,
                source_kind="kernel",
                source_name="chat_page",
            )
            # ── Consumption seam (M1) ─────────────────────────────────────
            # output_sanitizer wrote any correction onto ctx.sanitized_text
            # during the synchronous publish() above. The apply step is
            # delegated to the Qt-free, unit-tested
            # core.chat_finalize.apply_terminal_correction. The bubble is
            # re-fetched by message index because self._active_widget is already
            # nulled by the READY handler (chat.py:3328) before this runs. The
            # raw stored message is left untouched (think/acu content lives
            # there); only the rendered answer is corrected. Non-performative.
            from core.chat_finalize import apply_terminal_correction
            apply_terminal_correction(
                corrected=getattr(ctx, "sanitized_text", None),
                public=public_answer,
                get_widget=lambda: self._widget_for_index(msg_idx),
                on_debug=self.sig_debug.emit,
            )
        except Exception as exc:  # noqa: BLE001
            self.sig_debug.emit(
                f"[PIPELINE] TurnReadyEvent emit failed: "
                f"{type(exc).__name__}: {exc}"
            )

    def append_trace(self, trace_msg):
        lowered = trace_msg.lower()

        # --- Filter: only show LLM-relevant trace info ---
        # Skip guard internals, status transitions, and noise
        skip_patterns = [
            "guard", "dispatch", "route", "bridge", "dock",
            "addon", "registry", "host", "mount",
        ]
        for pat in skip_patterns:
            if pat in lowered and "error" not in lowered:
                return

        # Categorize what we show
        if "error" in lowered:
            state = "ERROR"
        elif "token" in lowered:
            state = "TOKENIZING"
        elif "inference started" in lowered:
            state = "INFERENCE"
        elif "inference" in lowered and ("complete" in lowered or "aborted" in lowered):
            state = "COMPLETE"
        elif "init backend" in lowered or "system online" in lowered:
            state = "MODEL"
        elif "unload" in lowered or "cancel" in lowered:
            state = "MODEL"
        elif "ctx" in lowered or "context" in lowered:
            state = "CTX"
        else:
            state = "INFO"

        display_msg = trace_msg
        if "→" in display_msg:
            display_msg = display_msg[display_msg.index("→") + 1:].strip()
        elif display_msg.startswith("ERROR"):
            display_msg = display_msg.replace("ERROR:", "").strip()

        self._trace_html(display_msg, state, error=(state == "ERROR"))

    def clear_chat(self):
        self._set_current_session(self._create_session(), show_reset=True, sync_history=True)

    def _sync_path_display(self):
        self._config_panel._sync_path_display()

    def build_model_payload(self) -> dict:
        backend = self.config.get("backend", "gguf_api")
        payload_backend = backend
        if backend == "gguf_api":
            if self.config.get("api_base") and self.config.get("api_model"):
                payload_backend = "openai"
        return {
            "backend": payload_backend,
            "api_provider": self.config.get("api_provider", "openai"),
            "api_base": self.config.get("api_base", ""),
            "api_model": self.config.get("api_model", ""),
            "api_key": self.config.get("api_key", ""),
            "gguf_path": self.config.get("gguf_path"),
            "path": self.config.get("gguf_path"),
        }

    def describe_active_model(self) -> str:
        backend = self.config.get("backend", "gguf_api")
        if backend in ("gguf", "gguf_api"):
            api_base = self.config.get("api_base") or ""
            api_model = self.config.get("api_model") or ""
            if backend == "gguf_api" and api_base and api_model:
                return f"local:{api_model}@{api_base}"
            path = self.config.get("gguf_path")
            if not path:
                return "none"
            return Path(path).name
        provider = self.config.get("api_provider", "openai")
        model = self.config.get("api_model") or "unknown"
        base = self.config.get("api_base") or "no base"
        return f"{provider}:{model}@{base}"

    def _emit_model_payload(self):
        self.sig_set_model_path.emit(self.build_model_payload())

    def _apply_backend_visibility(self):
        self._config_panel.apply_backend_visibility()

    def _apply_model_config(self):
        self._config_panel.apply_model_config()

    def _pick_free_port(self) -> int:
        return self._config_panel._pick_free_port()

    def _coerce_local_base(self, raw: str) -> tuple[str, str, int] | None:
        return self._config_panel._coerce_local_base(raw)

    def _ensure_local_api_base(self) -> tuple[str, str, int] | None:
        return self._config_panel._ensure_local_api_base()

    def _start_local_server(self, gguf_path: str, base: str, host: str, port: int) -> bool:
        if self._local_server_proc and self._local_server_proc.poll() is None:
            if self._local_server_model == gguf_path and self._local_server_base == base:
                return True
            self._stop_local_server()

        try:
            self._trace_html(f"Starting local server on {host}:{port} ...", "MODEL")
            cmd = self._build_server_cmd(gguf_path, host, port)
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            if os.name == "nt":
                creationflags |= subprocess.BELOW_NORMAL_PRIORITY_CLASS
            # Write stderr to a temp log file so the subprocess never blocks
            # on a full pipe buffer (Windows default is ~4KB).
            from core.paths import LOG_DIR, ensure_safe_local_path
            self._local_server_log = ensure_safe_local_path(LOG_DIR / "llama_server.log")
            log_fh = open(self._local_server_log, "w", encoding="utf-8")
            self._local_server_log_fh = log_fh
            self._local_server_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=log_fh,
                creationflags=creationflags,
            )
        except Exception as exc:
            self._trace_html(f"Failed to start local server: {exc}", "MODEL", error=True)
            return False

        self._local_server_model = gguf_path
        self._local_server_base = base
        return True

    def _stop_local_server(self) -> None:
        self._config_panel.stop_local_server()

    def _local_server_error_output(self, proc: subprocess.Popen | None) -> str:
        if proc is None:
            return ""
        log_path = getattr(self, "_local_server_log", None)
        if log_path and log_path.exists():
            try:
                text = log_path.read_text(encoding="utf-8", errors="ignore")
                # Return last 500 chars — enough for the error, not overwhelming
                return text[-500:].strip()
            except Exception:
                return ""
        return ""

    def _resolve_native_llama_server(self) -> str | None:
        """Find llama-server binary. Checks env var, common paths, then PATH."""
        env = os.getenv("MONOLITH_LLAMA_SERVER", "").strip()
        if env and Path(env).is_file():
            return env
        candidates = [
            Path.home() / "llama.cpp" / "build" / "bin" / "Release" / "llama-server.exe",
            Path.home() / "llama.cpp" / "build" / "bin" / "llama-server.exe",
        ]
        for p in candidates:
            if p.is_file():
                return str(p)
        # Check PATH
        import shutil
        found = shutil.which("llama-server")
        return found

    def _resolve_local_server_python(self) -> str:
        env_py = os.getenv("MONOLITH_LLAMA_PY", "").strip()
        if env_py and Path(env_py).exists():
            return env_py
        return sys.executable

    def _build_server_cmd(self, gguf_path: str, host: str, port: int) -> list[str]:
        """Build the command to launch a local GGUF server.

        Prefers the native llama-server binary (faster, latest arch support).
        Falls back to python -m llama_cpp.server if no binary found.
        """
        n_gpu_layers = str(self.config.get("n_gpu_layers", -1))
        n_threads = os.cpu_count() or 4
        n_threads = max(1, n_threads // 2)
        native = self._resolve_native_llama_server()
        if native:
            self._trace_html(f"Using native llama-server: {native}", "MODEL")
            return [
                native,
                "--model", gguf_path,
                "--host", host,
                "--port", str(port),
                "--n-gpu-layers", n_gpu_layers,
                "--threads", str(n_threads),
            ]
        python_exe = self._resolve_local_server_python()
        self._trace_html(f"Using python llama_cpp.server: {python_exe}", "MODEL")
        return [
            python_exe,
            "-m", "llama_cpp.server",
            "--model", gguf_path,
            "--host", host,
            "--port", str(port),
            "--n_gpu_layers", n_gpu_layers,
            "--n_threads", str(n_threads),
        ]

    def _start_local_server_and_load(self) -> None:
        gguf_path = self.config.get("gguf_path")
        if not gguf_path:
            self._trace_html("No GGUF selected.", "MODEL", error=True)
            self.btn_load.setEnabled(True)
            self._update_load_button_text()
            return

        base_info = self._ensure_local_api_base()
        if not base_info:
            self._trace_html("Invalid local API base.", "MODEL", error=True)
            self.btn_load.setEnabled(True)
            self._update_load_button_text()
            return

        base, host, port = base_info
        if not self._start_local_server(gguf_path, base, host, port):
            self.btn_load.setEnabled(True)
            self._update_load_button_text()
            return

        self._pending_local_load = True
        self._local_fetch_attempts = 0

        if self._local_fetch_timer is None:
            self._local_fetch_timer = QTimer(self)
            self._local_fetch_timer.setInterval(500)
            self._local_fetch_timer.timeout.connect(self._attempt_local_model_fetch)
        self._local_fetch_timer.start()
        self._attempt_local_model_fetch()

    def _attempt_local_model_fetch(self) -> None:
        if not self._pending_local_load:
            if self._local_fetch_timer:
                self._local_fetch_timer.stop()
            return
        if self._local_server_proc and self._local_server_proc.poll() is not None:
            self._pending_local_load = False
            if self._local_fetch_timer:
                self._local_fetch_timer.stop()
            err = self._local_server_error_output(self._local_server_proc)
            msg = "Local server exited before responding."
            if err:
                msg = f"{msg} {err}"
            self._trace_html(msg, "MODEL", error=True)
            self.btn_load.setEnabled(True)
            self._update_load_button_text()
            return
        if self._model_fetcher and self._model_fetcher.isRunning():
            return
        self._local_fetch_attempts += 1
        if self._local_fetch_attempts > 180:
            self._pending_local_load = False
            if self._local_fetch_timer:
                self._local_fetch_timer.stop()
            self._trace_html(
                "Local server did not respond in time. Check model path and server deps.",
                "MODEL",
                error=True,
            )
            self.btn_load.setEnabled(True)
            self._update_load_button_text()
            return
        if self._local_fetch_attempts == 1:
            self._trace_html("Waiting for local server to respond...", "MODEL")
        elif self._local_fetch_attempts % 20 == 0:
            pid = self._local_server_proc.pid if self._local_server_proc else "?"
            self._trace_html(f"Still waiting for local server (pid={pid})...", "MODEL")
        self._model_fetcher = ModelListLoader(self.config.get("api_base", ""), None)
        self._model_fetcher.finished.connect(self._on_models_loaded)
        self._model_fetcher.error.connect(self._on_models_error)
        self._model_fetcher.start()

    def _apply_config_to_controls(self) -> None:
        self._config_panel.apply_config_to_controls()

    def _on_engine_changed(self, _index: int):
        self._config_panel._on_engine_changed(_index)

    def _fetch_models(self):
        self._config_panel.fetch_models()

    def _on_models_loaded(self, payload: object):
        self._config_panel._on_models_loaded(payload)

    def _on_models_error(self, message: str):
        self._config_panel._on_models_error(message)

    def _on_model_combo_changed(self, _index: int):
        self._config_panel._on_model_combo_changed(_index)

    def _set_config_dirty(self, dirty=True):
        self._config_dirty = bool(dirty)
        self._config_panel.set_config_dirty(bool(dirty))

    def _on_external_config_changed(self, payload: dict) -> None:
        self._config_panel.on_external_config_changed(payload)
        self._config_dirty = self._config_panel._config_dirty

    def _save_config(self):
        self._config_panel.save_current_config()
        self._last_config_update = QDateTime.currentDateTime()
        stamp = self._last_config_update.toString("HH:mm:ss")
        self._trace_panel.show_config_saved(stamp)
        self._set_config_dirty(False)

    def _update_config_value(self, key, value):
        self._config_panel._update_config_value(key, value)
        self._config_dirty = self._config_panel._config_dirty

    def _set_slider_limits(self, slider, max_value, value):
        self._config_panel._set_slider_limits(slider, max_value, value)

    def _apply_default_limits(self):
        self._config_panel.apply_default_limits()

    def _trace_html(self, msg, tag="INFO", error=False):
        self._trace_panel.append_html(str(msg), str(tag), bool(error))

    def _trace_plain(self, msg):
        self._trace_panel.append_plain(str(msg))

    def trace_line(self, msg: str, tag: str = "INFO", error: bool = False) -> None:
        self._trace_html(msg, tag, error)

    def _on_model_capabilities(self, payload):
        self._handle_model_capabilities(payload)

    def _handle_model_capabilities(self, payload: dict):
        self._is_model_loaded = True
        self._config_panel.handle_model_capabilities(payload)
        self._set_config_dirty(self._config_panel._config_dirty)

    def _on_ctx_limit_changed(self, value):
        self._config_panel._on_ctx_limit_changed(value)

    def _update_state_ctx_limit(self, value: int) -> None:
        if hasattr(self, "state") and self.state is not None:
            self.state.ctx_limit = int(value)

    def _toggle_options_panel(self):
        self._config_panel.toggle_endpoint_edit()

    def _reset_config(self):
        self._config_panel.reset_config()
        self._set_config_dirty(True)

    def pick_file(self):
        self._config_panel.pick_file()
        self._set_config_dirty(self._config_panel._config_dirty)

    def toggle_load(self):
        self._config_panel.toggle_load()

    def _update_load_button_text(self):
        self._config_panel.update_load_button_text()

    def _request_mutation(self, fn):
        """Run a session mutation safely.

        If a generation is currently running, STOP first, then run `fn` on the next READY.
        This prevents stale indices / widgets during streaming.
        """
        if self._is_running:
            # Cancel any queued UPDATE-restart; mutation wins.
            self._awaiting_update_restart = False
            self._pending_update_text = None
            self._pending_mutation = fn
            self._set_send_button_state(is_running=True, stopping=True)
            self.request_stop_generation()
            return
        fn()

    def update_status(self, engine_key: str, status: SystemStatus):
        if engine_key != getattr(self, "_engine_key", "llm"):
            return
        self._handle_status(status)

    def _handle_status(self, status: SystemStatus) -> None:
        engine_key = getattr(self, "_engine_key", "llm")
        ek_short = engine_key[-8:] if engine_key else "?"
        prev = getattr(self, '_last_status', None)
        transition = f"{prev.name if prev else '?'}→{status.name}" if hasattr(status, 'name') else str(status)
        self.sig_debug.emit(f"[CHAT:{ek_short}] status {transition}, running={self._is_running}, mutation={'yes' if self._pending_mutation else 'no'}")
        is_processing = status in (SystemStatus.LOADING, SystemStatus.RUNNING, SystemStatus.UNLOADING)
        self._config_panel.set_engine_status(status, self._is_model_loaded)
        if status == SystemStatus.READY and self._pending_mutation is not None:
            # Finish transition to READY (stop state) first, then mutate.
            # Keep this path above UPDATE-restart.
            self._auto_scroll_on_height_change = False
            self._flush_token_batch()
            self._set_send_button_state(is_running=False)
            self._rewrite_assistant_index = None
            if self._active_widget is not None:
                self._apply_display_update(self._active_widget, self._assistant_box.finalize_display_stream())
                self._active_widget.finalize()
                # Acatalepsy v1.1: capture finalized assistant text.
                self._capture_finalized_assistant_to_canonical()
            self._active_widget = None
            if self._update_trace_state == "streaming":
                self._finalize_update_progress()
            # If generation ended before any tokens arrived, remove the empty assistant bubble.
            self._cleanup_empty_assistant_if_needed()
            # Reset assistant stream trackers after end-of-generation.
            self._active_assistant_started = False
            self._active_assistant_token_count = 0
            self._pending_archive_save_task_id = None
            self.sig_debug.emit(
                f"[STREAM] ready-before-mutation chunks={self._stream_debug_chunk_count} chars={self._stream_debug_char_count}"
            )
            self._sessions.set_suppress_title_regen(False)
            pending = self._pending_mutation
            self._pending_mutation = None
            try:
                pending()
            finally:
                self._last_status = status
            return

        if status == SystemStatus.READY and self._awaiting_update_restart:
            self._auto_scroll_on_height_change = False
            self._awaiting_update_restart = False
            self.btn_send.setEnabled(True)

            update_text = self._pending_update_text
            self._pending_update_text = None
            self._pending_archive_save_task_id = None
            self._submit_update(update_text)
            return
        if status == SystemStatus.RUNNING:
            self._set_send_button_state(is_running=True)
        elif status == SystemStatus.READY:
            self._auto_scroll_on_height_change = False
            self._flush_token_batch()
            if self._last_status == SystemStatus.UNLOADING:
                self._is_model_loaded = False
                self._config_panel.handle_ready_after_unload()
            self._set_send_button_state(is_running=False)
            self._rewrite_assistant_index = None
            if self._active_widget is not None:
                self._apply_display_update(self._active_widget, self._assistant_box.finalize_display_stream())
                self._active_widget.finalize()
                # Acatalepsy v1.1: capture finalized assistant text.
                self._capture_finalized_assistant_to_canonical()
            self._active_widget = None
            if self._update_trace_state == "streaming":
                self._finalize_update_progress()
            # If generation ended before any tokens arrived, remove the empty assistant bubble.
            self._cleanup_empty_assistant_if_needed()
            # Reset assistant stream trackers after end-of-generation.
            self._active_assistant_started = False
            self._active_assistant_token_count = 0
            self.sig_debug.emit(
                f"[STREAM] ready chunks={self._stream_debug_chunk_count} chars={self._stream_debug_char_count}"
            )
            # Title generation is finalized ONLY on READY.

            # Retry: if we were in a tool followup and the LLM produced nothing, nudge it.
            if (
                self._pending_tool_results
                and self._stream_debug_char_count == 0
                and self._tool_followup_retries < self._MAX_TOOL_RETRIES
            ):
                self._tool_followup_retries += 1
                self.sig_debug.emit(
                    f"[CHAT] tool followup retry {self._tool_followup_retries}/{self._MAX_TOOL_RETRIES}"
                )
                base = self._build_tool_followup_prompt()
                nudge = (
                    f"{base}\n\n"
                    "You must respond now. If tools are complete, start with [TOOL_LOOP_DONE] "
                    "and provide the final answer."
                )
                self._dispatch_generation(
                    {"prompt": nudge, "ephemeral": True},
                    rewrite_index=None,
                    source="tool_followup_retry",
                )
                return

            # READY is emitted after _on_gen_finish completes and assistant text is final.
            # STOP also transitions to READY; _maybe_generate_title self-guards.
            # Do NOT call this method from token flush, send paths, or mutation handlers.
            if self._pending_mutation is None:
                handled_tools = self._process_last_response_commands()
                if not handled_tools:
                    # Non-convergence recovery: if the turn produced ONLY
                    # reasoning — no public answer and no tool action — re-prompt
                    # once instead of letting it silently die ("didn't say
                    # anything"). Bounded; anchored on the user's actual ask.
                    if self._maybe_nudge_non_convergence():
                        return
                    # Source-Tier Gate: capture whether THIS exchange used tools
                    # before the terminal cleanup resets _tool_loop_active, so the
                    # classifier (in finalize, below) sees real exchange tool usage
                    # rather than the terminal turn's tool-free raw text.
                    self.config["_source_tier_tools"] = (
                        ("tool",) if self._tool_loop_active else ()
                    )
                    self._clear_tool_followup_state()
                    if self._tool_cancel_requested:
                        self._tool_cancel_requested = False
                    # Extract ACUs from the final assistant response
                    try:
                        self._extract_and_store_acus()
                    except Exception:
                        pass
                    # Deterministic READY-time verifier — observation-only
                    try:
                        self._run_response_verifier()
                    except Exception:
                        pass
                self._maybe_handle_action_proposal()
                self._maybe_generate_title()
                self.sig_sync_history.emit(self._build_engine_history_from_session())
                if not self._pending_tool_results and self._pending_archive_save_task_id is not None:
                    try:
                        self._save_chat_archive()
                    except Exception:
                        pass
                    self._pending_archive_save_task_id = None
            self._sessions.set_suppress_title_regen(False)
        elif status == SystemStatus.LOADING:
            self._set_send_button_state(is_running=False)
            self.btn_send.setEnabled(False)
        elif status in (SystemStatus.UNLOADING, SystemStatus.ERROR):
            self._is_model_loaded = False
            self._config_panel.set_engine_status(status, self._is_model_loaded)

        if status == SystemStatus.READY and not self._is_model_loaded:
            # Don't reset to defaults if the model's context cap was already
            # auto-configured (e.g. during local server startup).
            if not getattr(self._config_panel, "_model_ctx_cap", None):
                self._apply_default_limits()
        self._last_status = status

    def apply_operator(self, operator_data: dict):
        if not isinstance(operator_data, dict):
            return
        config = operator_data.get("config")
        if not isinstance(config, dict):
            return

        slider_values = {
            "temp": float(config.get("temp", self.config.get("temp", 1.0))),
            "top_p": float(config.get("top_p", self.config.get("top_p", 0.95))),
            "top_k": int(config.get("top_k", self.config.get("top_k", 20))),
            "min_p": float(config.get("min_p", self.config.get("min_p", 0.0))),
            "presence_penalty": float(
                config.get("presence_penalty", self.config.get("presence_penalty", 1.5))
            ),
            "repetition_penalty": float(
                config.get("repetition_penalty", self.config.get("repetition_penalty", 1.0))
            ),
            "max_tokens": int(config.get("max_tokens", self.config.get("max_tokens", 2048))),
            "ctx_limit": int(config.get("ctx_limit", self.config.get("ctx_limit", int(getattr(getattr(self, "state", None), "ctx_limit", 0) or 0)))),
        }

        self.config.update(config)

        self.s_temp.slider.blockSignals(True)
        self.s_top.slider.blockSignals(True)
        self.s_top_k.slider.blockSignals(True)
        self.s_min_p.slider.blockSignals(True)
        self.s_presence.slider.blockSignals(True)
        self.s_repetition.slider.blockSignals(True)
        self.s_tok.slider.blockSignals(True)
        self.s_ctx.slider.blockSignals(True)
        self.s_temp.slider.setValue(int(slider_values["temp"] * 100))
        self.s_temp.val_lbl.setText(f"{slider_values['temp']:.2f}")
        self.s_top.slider.setValue(int(slider_values["top_p"] * 100))
        self.s_top.val_lbl.setText(f"{slider_values['top_p']:.2f}")
        self.s_top_k.slider.setValue(int(slider_values["top_k"]))
        self.s_top_k.val_lbl.setText(str(int(slider_values["top_k"])))
        self.s_min_p.slider.setValue(int(slider_values["min_p"] * 100))
        self.s_min_p.val_lbl.setText(f"{slider_values['min_p']:.2f}")
        self.s_presence.slider.setValue(int(slider_values["presence_penalty"] * 100))
        self.s_presence.val_lbl.setText(f"{slider_values['presence_penalty']:.2f}")
        self.s_repetition.slider.setValue(int(slider_values["repetition_penalty"] * 100))
        self.s_repetition.val_lbl.setText(f"{slider_values['repetition_penalty']:.2f}")
        self.s_tok.slider.setValue(int(slider_values["max_tokens"]))
        self.s_tok.val_lbl.setText(str(int(slider_values["max_tokens"])))
        self.s_ctx.slider.setValue(int(slider_values["ctx_limit"]))
        self.s_ctx.val_lbl.setText(str(int(slider_values["ctx_limit"])))
        self.s_temp.slider.blockSignals(False)
        self.s_top.slider.blockSignals(False)
        self.s_top_k.slider.blockSignals(False)
        self.s_min_p.slider.blockSignals(False)
        self.s_presence.slider.blockSignals(False)
        self.s_repetition.slider.blockSignals(False)
        self.s_tok.slider.blockSignals(False)
        self.s_ctx.slider.blockSignals(False)

        self.sig_set_ctx_limit.emit(int(slider_values["ctx_limit"]))

        self._apply_model_config()
        self._emit_model_payload()

        # Restore chat messages if snapshot included them, otherwise fresh session
        messages = operator_data.get("messages")
        if isinstance(messages, list) and messages:
            session = self._create_session(
                messages=messages,
                title=operator_data.get("session_title"),
                assistant_tokens=operator_data.get("assistant_tokens", 0),
            )
            self._set_current_session(session, show_reset=True, sync_history=True)
        else:
            self._start_new_session()

        self._set_config_dirty(True)
        self.sig_operator_loaded.emit(str(operator_data.get("name", "")))

    def _start_new_session(self):
        self._sessions.reset_title_flags()
        self._trace_panel.clear()
        self._set_current_session(self._create_session(), show_reset=True, sync_history=True)
        self._trace_plain("--- TRACE RESET ---")

    def _on_archive_clear_request(self, scope: str) -> None:
        """Dispatcher for the archive panel's /clear command. The panel
        emits the scope as a string ("chat"|"all"|"logs") so we can route
        to the right surface. /clear all is handled inside the archive
        panel itself (it owns the file deletion); we only handle chat
        and logs here. chat keeps the original confirmation dialog so an
        accidental Enter doesn't nuke the session."""
        scope = (scope or "chat").lower().strip()
        if scope == "chat":
            self._prompt_clear_session()
        elif scope == "logs":
            trace = getattr(self, "_trace_panel", None)
            if trace is not None and hasattr(trace, "clear"):
                try:
                    trace.clear()
                except Exception:
                    pass
        # scope == "all" is owned by ArchiveBrowserPanel; nothing to do here.

    def _fill_input_for_summarize(self, prompt_text: str) -> None:
        """Drop a summarize prompt into the conversation input box so the
        user can press Send when ready. Avoids needing a separate LLM
        wiring — reuses the existing per-turn submit path."""
        if not isinstance(prompt_text, str) or not prompt_text.strip():
            return
        surface = getattr(self, "_surface", None)
        if surface is None:
            return
        input_widget = getattr(surface, "input", None)
        if input_widget is None or not hasattr(input_widget, "setText"):
            return
        input_widget.setText(prompt_text)
        if hasattr(input_widget, "setFocus"):
            input_widget.setFocus()

    def _prompt_clear_session(self):
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Clear Session")
        dialog.setText("Choose how to clear the current session.")
        dialog.setStyleSheet(f"""
            QMessageBox {{
                background: {_s.BG_INPUT};
                color: {_s.FG_TEXT};
            }}
            QLabel {{
                color: {_s.FG_TEXT};
            }}
            QPushButton {{
                color: {_s.FG_TEXT};
                background: transparent;
                border: 1px solid {_s.BORDER_LIGHT};
                padding: 6px 12px;
                font-size: 10px;
                font-weight: bold;
                border-radius: 2px;
            }}
            QPushButton:hover {{
                border: 1px solid {_s.ACCENT_PRIMARY};
                color: {_s.ACCENT_PRIMARY};
            }}
            QPushButton:checked {{
                border: 1px solid {_s.ACCENT_PRIMARY};
                color: {_s.ACCENT_PRIMARY};
            }}
        """)
        btn_clear = dialog.addButton("Clear Logs", QMessageBox.AcceptRole)
        btn_delete = dialog.addButton("Delete Chat", QMessageBox.DestructiveRole)
        dialog.addButton("Cancel", QMessageBox.RejectRole)
        dialog.exec()
        clicked = dialog.clickedButton()
        if clicked == btn_clear:
            self._clear_current_session(delete_archive=False)
        elif clicked == btn_delete:
            self._clear_current_session(delete_archive=True)

    def _clear_current_session(self, delete_archive):
        archive_path = self._current_session.get("archive_path")
        if delete_archive and archive_path:
            try:
                Path(archive_path).unlink()
            except OSError:
                pass
        self._trace_panel.clear()
        self._set_current_session(self._create_session(), show_reset=True, sync_history=True)
        self._refresh_archive_list()

    def _delete_selected_archive(self):
        item = self.archive_list.currentItem()
        if not item:
            return
        archive_path = Path(item.data(Qt.UserRole))
        try:
            archive_path.unlink()
        except OSError:
            return
        if self._current_session.get("archive_path") == str(archive_path):
            self._set_current_session(self._create_session(), show_reset=True, sync_history=True)
        self._refresh_archive_list()

    def _save_chat_archive(self):
        self._archive.save_session(self._current_session)
        self._archive_panel.set_current_archive_path(self._current_session.get("archive_path"))
        self._archive_panel.refresh()

    def _load_chat_archive(self):
        return

    def _load_archive_session(self, session) -> None:
        self._set_current_session(session, show_reset=False, sync_history=True)
        self._notify_header_update()
        self._refresh_archive_list()

    def _reset_session_after_deleted_archive(self) -> None:
        self._set_current_session(self._create_session(), show_reset=True, sync_history=True)
        self._refresh_archive_list()

    def _refresh_archive_list(self):
        self._archive_panel.set_current_archive_path(self._current_session.get("archive_path"))
        self._archive_panel.refresh()

    def _on_history_search_changed(self, _text: str) -> None:
        self._refresh_archive_list()

    def _on_set_session_meta(self, changes: dict) -> str | None:
        if not isinstance(changes, dict):
            return "no valid metadata changes"
        updated: list[str] = []
        title = changes.get("title")
        if isinstance(title, str) and title.strip():
            self._current_session["title"] = title.strip()
            updated.append("title")
        summary = changes.get("summary")
        if isinstance(summary, list):
            lines = [str(item).strip() for item in summary if str(item).strip()]
            self._current_session["summary"] = lines
            updated.append("summary")
        if updated:
            self._current_session["updated_at"] = self._sessions.now_iso()
            self._notify_header_update()
            self._refresh_archive_list()
        return ", ".join(updated) if updated else "no valid metadata changes"

    def _process_last_response_commands(self) -> bool:
        """Parse tool_call envelopes from the last assistant message and execute them."""
        msgs = self._current_session.get("messages", [])
        last_index = next(
            (idx for idx in range(len(msgs) - 1, -1, -1) if msgs[idx].get("role") == "assistant"),
            None,
        )
        if last_index is None:
            return False
        last_assistant = msgs[last_index]
        text = last_assistant.get("text", "")
        stripped_text, saw_done_tag = self._strip_tool_loop_done_tag(text)
        if saw_done_tag:
            if stripped_text != text:
                self._assistant_box.rewrite_assistant_text(last_index, stripped_text)
                widget = self._widget_for_index(last_index)
                self._apply_assistant_snapshot(widget, stripped_text, close_open=True)
            self._clear_tool_followup_state()
            text = stripped_text
        if not extract_commands(text):
            nudge = self._detect_incomplete_action(text, tool_ran=self._tool_loop_active)
            if nudge and self._tool_followup_depth < self._MAX_TOOL_FOLLOWUPS:
                self._tool_followup_depth += 1
                self._dispatch_generation(
                    {"prompt": nudge, "ephemeral": True},
                    rewrite_index=None,
                    source="incomplete_action_nudge",
                )
                return True
            return False

        self._tool_loop_active = True

        def _open_addon(addon_id: str) -> None:
            if hasattr(self, "ui_bridge"):
                self.ui_bridge.sig_launch_addon.emit(addon_id)

        self._spawn_budget = SpawnBudget()
        clean_text, tool_result, artifacts = process_response(
            text,
            archive_dir=ARCHIVE_DIR,
            on_open_addon=_open_addon,
            bridge=self._bridge,
            guard=self._guard,
            world_state=getattr(self.state, "world_state", None),
            on_generate_audio=getattr(self, "_on_generate_audio", None),
            on_soundtrap=getattr(self, "_on_soundtrap", None),
            on_set_session_meta=self._on_set_session_meta,
            on_ask_user=self._on_ask_user,
            should_cancel=lambda: bool(self._tool_cancel_requested),
            result_cache=self._tool_cache,
            vision_artifact_bridge=self._vision_artifact_bridge,
            level=1,
            allowed_tools=L1_PRINCIPAL_TOOLS,
            parent_turn_id=str(self._last_task_id or ""),
            spawn_budget=self._spawn_budget,
            on_spawn_subagent=self._on_spawn_subagent,
            on_run_workshop=self._on_run_workshop,
            on_author_workshop_card=self._on_author_workshop_card,
        )

        rerender = False
        insertion_index = last_index + 1
        parse_error_artifacts: list[dict] = []

        if clean_text != text:
            self._assistant_box.rewrite_assistant_text(last_index, clean_text)
            snapshot = self._assistant_box.build_display_snapshot(clean_text, close_open=True)
            if not (snapshot.answer_text or "").strip():
                self._assistant_box.truncate_from(last_index)
                self._active_widget = None
                insertion_index = last_index
                rerender = True
            else:
                widget = self._widget_for_index(last_index)
                self._apply_assistant_snapshot(widget, clean_text, close_open=True)

        for artifact in artifacts:
            kind = str(artifact.get("kind", "")).strip().lower()
            if kind == "tool_call":
                payload = json.dumps(artifact.get("command", {}), ensure_ascii=False)
                self._sessions.insert_message(
                    insertion_index, "tool_call", payload,
                    extra={"origin": ORIGIN_TOOL_INJECTION},
                )
                insertion_index += 1
                rerender = True
                continue
            if kind == "tool_result":
                if artifact.get("parse_error"):
                    parse_error_artifacts.append(artifact)
                payload = json.dumps(
                    {
                        "tool": artifact.get("tool", ""),
                        "call": artifact.get("call", {}),
                        "result": artifact.get("result", ""),
                    },
                    ensure_ascii=False,
                )
                self._sessions.insert_message(
                    insertion_index, "tool_result", payload,
                    extra={"origin": ORIGIN_TOOL_INJECTION},
                )
                insertion_index += 1
                rerender = True

        if rerender:
            self._render_session()

        if parse_error_artifacts:
            if self._tool_parse_retries >= self._MAX_TOOL_PARSE_RETRIES:
                self._trace_html(
                    (
                        "Tool command parse retry limit reached; stopping automatic retries. "
                        "Please issue a corrected <tool_call> block."
                    ),
                    "TOOLS",
                    error=True,
                )
                self._clear_tool_followup_state()
                return True

            self._tool_parse_retries += 1
            latest = parse_error_artifacts[-1]
            retry_prompt = self._build_parse_retry_prompt(
                str(latest.get("raw", "")),
                attempted_backslash_repair=bool(latest.get("repair_json")),
                attempted_tag_repair=bool(latest.get("repair_close_tag")),
            )
            self._dispatch_generation(
                {"prompt": retry_prompt, "ephemeral": True},
                rewrite_index=None,
                source="tool_parse_retry",
            )
            return True

        if self._tool_cancel_requested:
            self._tool_cancel_requested = False
            self._clear_tool_followup_state()
            return True
        if tool_result:
            self._queue_tool_followup(tool_result, rewrite_index=None)
        return True

    def _create_session(self, messages=None, created_at=None, updated_at=None, archive_path=None, summary=None, title=None, assistant_tokens=0):
        return self._sessions.create_session(
            messages=messages,
            created_at=created_at,
            updated_at=updated_at,
            archive_path=archive_path,
            summary=summary,
            title=title,
            assistant_tokens=assistant_tokens,
        )

    def _set_current_session(self, session, show_reset=False, sync_history=False):
        self._current_session = session
        self._active_widget = None
        self._active_item = None
        self._clear_editing_message(clear_input=True)
        # Pending ask_user questions don't survive a session boundary — the
        # question was for the prior session's context and would arrive into
        # the new session as orphan ask_user state if we didn't clear here.
        self._clear_pending_ask_user()
        self._render_session(session, show_reset=show_reset)
        if sync_history:
            history = self._build_engine_history_from_session()
            self.sig_sync_history.emit(history)
        self._notify_header_update()
        # Session switched — refresh the reasoning-tree pane so it never shows a
        # stale tree (its fingerprint short-circuit makes this cheap).
        panel = getattr(self, "_reasoning_tree_panel", None)
        if panel is not None:
            try:
                panel.invalidate()
            except Exception:
                pass

    def _build_engine_history_from_session(self):
        return self._sessions.build_engine_history()

    def _snapshot_session(self):
        self._sessions.snapshot()

    def _undo_last_mutation(self):
        if not self._sessions.undo():
            return
        self._render_session()
        self.sig_sync_history.emit(
            self._build_engine_history_from_session()
        )

    def _delete_from_index(self, idx: int):
        self.sig_debug.emit(f"[CHAT] _delete_from_index: idx={idx}, msgs={len(self._current_session['messages'])}, is_running={self._is_running}")
        def _do_delete():
            self._snapshot_session()
            if not self._assistant_box.delete_from_index(idx):
                return
            self._clear_editing_message(clear_input=True)
            self._active_widget = None
            self._render_session()
            self.sig_sync_history.emit(
                self._build_engine_history_from_session()
            )

        self._request_mutation(_do_delete)

    def _edit_from_index(self, idx: int):
        self.sig_debug.emit(f"[CHAT] _edit_from_index: idx={idx}, msgs={len(self._current_session['messages'])}, is_running={self._is_running}")
        def _do_edit():
            text = self._assistant_box.edit_from_index(idx)
            if text is None:
                return
            self._begin_editing_message(idx, text)

        self._request_mutation(_do_edit)

    def _regen_from_index(self, idx: int):
        self.sig_debug.emit(f"[CHAT] _regen_from_index: idx={idx}, msgs={len(self._current_session['messages'])}, is_running={self._is_running}")
        def _do_regen():
            self._snapshot_session()
            self._sessions.set_suppress_title_regen(True)
            prompt = self._assistant_box.regen_from_index(idx)
            if prompt is None:
                return
            self._clear_editing_message(clear_input=True)
            self._active_widget = None
            self._render_session()
            self.sig_sync_history.emit(
                self._build_engine_history_from_session()
            )

            self._dispatch_generation(prompt, source="regen")

        self._request_mutation(_do_regen)

    def _handle_surface_mutation(self, action: str, payload: object) -> None:
        data = payload if isinstance(payload, dict) else {}
        idx = int(data.get("index", -1))
        # Layer D outcome capture — record before any state mutation so a
        # delete/regen still leaves a trace of the outcome on the turn that
        # produced the message. Read-only actions (thumbs / copy) only record.
        if action in ("thumbs_up", "thumbs_down", "copy", "delete", "regen"):
            self._record_turn_outcome(action, idx)
        if action == "delete":
            self._delete_from_index(idx)
            return
        if action == "edit":
            self._edit_from_index(idx)
            return
        if action == "regen":
            self._regen_from_index(idx)
            return
        if action == "switch_take":
            self._switch_take_from_index(idx, int(data.get("direction", 0)))
            return
        if action in ("thumbs_up", "thumbs_down", "copy"):
            return  # outcome already recorded; no further mutation needed

    def _turn_id_for_message_index(self, idx: int) -> str | None:
        """Return the turn_id stamped on the assistant message at *idx*.

        Falls back to the engine's last_task_id when the message is the
        most recent assistant message but doesn't yet carry a stamp
        (race window between finalize and finished signal).
        """
        msgs = self._current_session.get("messages", [])
        if 0 <= idx < len(msgs):
            entry = msgs[idx]
            if isinstance(entry, dict) and entry.get("role") == "assistant":
                tid = str(entry.get("task_id", "") or "").strip()
                if tid:
                    return tid
                last_assistant_idx = -1
                for i in range(len(msgs) - 1, -1, -1):
                    msg = msgs[i]
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        last_assistant_idx = i
                        break
                if idx == last_assistant_idx:
                    fallback = str(self._last_task_id or "").strip()
                    return fallback or None
        return None

    def _extract_think_block(self, message_index: int) -> str | None:
        """Return the assistant's <think>...</think> trace at *message_index*.

        Concatenates all <think> blocks (most turns have one; tool-call turns
        can have several) with blank-line separators, then tail-caps to fit
        inside ``core.turn_trace._METADATA_CAP_BYTES`` (4096) once embedded
        in the metadata JSON. A 6k-char real-world trace exceeded the cap on
        the first live /rating and lost the whole metadata blob to the
        truncation marker; 3500 chars leaves room for the JSON wrappers.
        Returns None when there is no usable trace. Best-effort — any error
        returns None.
        """
        try:
            msgs = self._current_session.get("messages", [])
            if not (0 <= message_index < len(msgs)):
                return None
            msg = msgs[message_index]
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                return None
            text = str(msg.get("text") or "")
            if not text:
                return None
            import re
            blocks = re.findall(
                r"<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE,
            )
            joined = "\n\n".join(b.strip() for b in blocks if b.strip())
            if not joined:
                return None
            # Tail-cap (mirrors core/monothink._clip_think_block's posture —
            # the conclusion of a reasoning trace lives at the end).
            _CAP = 3500
            if len(joined) > _CAP:
                joined = "…(trace truncated, latest tail shown)…\n" + joined[-_CAP:]
            return joined
        except Exception:
            return None

    def _extract_replay_input(self, message_index: int) -> str | None:
        """Return the nearest prior non-ephemeral user text for contrast replay."""
        try:
            msgs = self._current_session.get("messages", [])
            if not (0 <= message_index < len(msgs)):
                return None
            for i in range(message_index - 1, -1, -1):
                msg = msgs[i]
                if not isinstance(msg, dict) or msg.get("role") != "user":
                    continue
                if msg.get("ephemeral"):
                    continue
                text = str(msg.get("text") or msg.get("content") or "").strip()
                if not text:
                    continue
                _CAP = 2000
                if len(text) > _CAP:
                    text = "(input truncated, latest tail shown)\n" + text[-_CAP:]
                return text
        except Exception:
            return None
        return None

    def _record_turn_outcome(
        self,
        kind: str,
        message_index: int,
        *,
        rating_value: int | None = None,
        reason: str | None = None,
        failure_tags: list[str] | None = None,
        surface_note: str | None = None,
    ) -> bool:
        """Persist a Layer D outcome for the turn at *message_index*.

        Returns True on success, False if turn_id can't be resolved or the
        record fails. Best-effort — never raises.

        ``failure_tags`` (the closed-enum directional signal) and
        ``surface_note`` (holistic prose) come from /rating. When tags are
        present the outcome ``reason`` is AUTO-COMPOSED from them
        (core.failure_tags.compose_reasoning_why) — never rater free text —
        and both ride in ``metadata`` so the turn_trace evolution hook can
        read ``metadata["failure_tags"]``. Other callers (thumbs/copy/…) pass
        neither and are unaffected.
        """
        turn_id = self._turn_id_for_message_index(message_index)
        if not turn_id:
            self.sig_debug.emit(
                f"[outcome] could not resolve turn_id for index={message_index} kind={kind}"
            )
            return False
        try:
            from datetime import datetime, timezone

            from core import turn_trace as _tt

            # Carry the rated turn's reasoning trace forward in metadata so
            # monothink.maybe_evolve_after_rating can show the model how it
            # thought alongside the rating signal. (Thumbs no longer drive
            # evolution, but the trace is cheap to keep and harmless.)
            metadata: dict = {}
            if kind in ("rating", "thumbs_up", "thumbs_down"):
                think_block = self._extract_think_block(message_index)
                if think_block:
                    metadata["think_block"] = think_block
            if kind == "rating":
                replay_input = self._extract_replay_input(message_index)
                if replay_input:
                    metadata["replay_input"] = replay_input

            # SP1 rating contract: stash the closed-enum failure_tags + the
            # holistic surface_note in metadata, and derive `reason` (the human
            # echo in the outcome row) from the tags when present.
            if failure_tags is not None:
                metadata["failure_tags"] = failure_tags
            if surface_note is not None:
                metadata["surface_note"] = surface_note
            if failure_tags:
                from core.failure_tags import compose_reasoning_why
                reason = compose_reasoning_why(failure_tags)
            elif reason is None:
                reason = surface_note

            record = _tt.OutcomeTraceRecord(
                turn_id=turn_id,
                recorded_at=datetime.now(timezone.utc).isoformat(),
                kind=kind,
                rating_value=rating_value,
                reason=reason,
                metadata=metadata,
            )
            _tt.record_outcome(record)
            tail = f" rating={rating_value}" if rating_value is not None else ""
            tail += f" reason={reason!r}" if reason else ""
            self.sig_debug.emit(
                f"[outcome] turn={turn_id[:8]} kind={kind}{tail}"
            )
            return True
        except Exception as exc:
            self.sig_debug.emit(f"[outcome] record failed: {exc}")
            return False

    def _expand_tool_artifact(self, tool_name: str, payload: object) -> None:
        tool = canonical_tool_name(tool_name) or str(tool_name or "")
        shell = self.window()
        companion = getattr(shell, "companion", None)
        if companion is None:
            return

        try:
            from ui.companion_pane import CompanionState
        except Exception:
            return

        data = payload if isinstance(payload, dict) else {}
        call = data.get("call") if isinstance(data.get("call"), dict) else {}

        if tool == "search_history":
            if hasattr(companion, "show_state"):
                companion.show_state(CompanionState.ARCHIVE)
            else:
                companion.pin_state(CompanionState.ARCHIVE)
            panel = companion.get_panel(CompanionState.ARCHIVE)
            query = str(call.get("query", "")).strip()
            if panel is not None and hasattr(panel, "set_query"):
                panel.set_query(query)
            return

        if tool in {"read_file", "list_files", "grep"}:
            if hasattr(companion, "show_state"):
                companion.show_state(CompanionState.DATABANK)
            else:
                companion.pin_state(CompanionState.DATABANK)
            panel = companion.get_panel(CompanionState.DATABANK)
            path = str(call.get("path", "")).strip()
            if panel is not None and hasattr(panel, "reveal_path"):
                panel.reveal_path(path)
            return

        if tool == "open_addon":
            addon_id = str(call.get("addon", "")).strip()
            if addon_id and hasattr(self.ui_bridge, "sig_launch_addon"):
                self.ui_bridge.sig_launch_addon.emit(addon_id)

    # ------------------------------------------------------------------
    # ConversationSurface delegates
    # ------------------------------------------------------------------

    def focus_input(self) -> None:
        self._surface.focus_input()

    def _set_send_button_state(self, is_running: bool, stopping: bool = False):
        self._is_running = is_running
        self._surface._set_send_button_state(is_running, stopping)

    def _on_input_changed(self, text):
        self._surface._on_input_changed(text)

    def _render_session(self, session=None, show_reset=False):
        self._surface._render_session(session=session, show_reset=show_reset)

    def _append_message_widget(self, idx: int, role=None, text=None, timestamp=None):
        return self._surface._append_message_widget(idx, role=role, text=text, timestamp=timestamp)

    def _widget_for_index(self, idx: int):
        return self._surface._widget_for_index(idx)

    def _begin_update_trace(self, update_text):
        self._update_trace_state = "requested"
        self._update_token_count = 0
        self._update_progress_index = 0
        self._trace_html("UPDATE REQUESTED", "UPDATE")
        self._trace_html(f'USER PATCH: "{update_text}"', "UPDATE")

    def _start_update_streaming(self):
        self._update_trace_state = "streaming"
        self._update_token_count = 0
        self._update_progress_index = 0

    def _update_progress_markers(self):
        if self._update_trace_state != "streaming":
            return
        self._update_token_count += 1
        thresholds = [25, 50, 100]
        pct = min(
            100,
            int((self._update_token_count / self.config["max_tokens"]) * 100),
        )
        while self._update_progress_index < len(thresholds):
            if pct >= thresholds[self._update_progress_index]:
                percent = thresholds[self._update_progress_index]
                self._trace_html(f"UPDATE PROGRESS {percent}%", "UPDATE")
                self._update_progress_index += 1
                continue
            break

    def _finalize_update_progress(self):
        if self._update_trace_state != "streaming":
            return
        thresholds = [25, 50, 100]
        while self._update_progress_index < len(thresholds):
            percent = thresholds[self._update_progress_index]
            self._trace_html(f"UPDATE PROGRESS {percent}%", "UPDATE")
            self._update_progress_index += 1
        self._update_trace_state = None

    def _add_message(self, role, text, extra: dict | None = None):
        return self._sessions.add_message(role, text, extra=extra)

    def _capture_finalized_assistant_to_canonical(self) -> None:
        """Read the finalized assistant text from the current session and
        write it to canonical_log. Called right after the assistant widget
        finalizes on a READY transition. Best-effort; swallowed errors.

        v1.1.5 gates:
          - Only fire when this UI turn was initiated via _submit_prompt
            (CONNECT-routed turns skip — agent_server already logged the
            assistant response on its side; the UI is just rendering).
          - One-shot per UI turn (the first terminal READY consumes the
            flag; intermediate tool-followup READYs are skipped via
            _tool_loop_active / _pending_tool_results checks).
        """
        # Gate 1: only UI-initiated turns
        if not getattr(self, "_chat_canonical_assistant_pending_capture", False):
            return
        # Gate 2: skip intermediate READYs during tool followups
        if getattr(self, "_tool_loop_active", False):
            return
        if getattr(self, "_pending_tool_results", None):
            return
        try:
            idx = getattr(self, "_active_assistant_index", None)
            if idx is None:
                return
            messages = self._current_session.get("messages", [])
            if not isinstance(messages, list) or not (0 <= idx < len(messages)):
                return
            msg = messages[idx]
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                return
            text = msg.get("text") or ""
            if text.strip():
                self._log_chat_to_canonical("assistant", text)
                # One-shot — consume the flag so subsequent READYs in the
                # same turn (rare but possible) don't double-write.
                self._chat_canonical_assistant_pending_capture = False
        except Exception:
            pass

    def _log_chat_to_canonical(self, role: str, text: str) -> None:
        """Best-effort canonical_log write for UI-side chat turns.

        Acatalepsy v1.1 — extends substrate coverage from CONNECT-only
        to the primary UI chat surface. The auditor consumes canonical_log
        regardless of source, so capturing UI chats here gives it the
        primary working surface (most chats happen here, not via CONNECT).

        Best-effort: substrate write failures are swallowed via sig_debug
        so substrate issues never break the chat path. The user never
        sees an error from this hook.
        """
        if not text or not text.strip():
            return
        if role not in ("user", "assistant"):
            return  # 'agent', 'system', 'command_block' etc. are not chat turns
        try:
            from core.acatalepsy import canonical_log as _cl
            engine_key = getattr(self, "_engine_key", "default")
            session_id = f"ui:{engine_key}"
            kind = "user_message" if role == "user" else "assistant_message"
            payload = {"text": text, "agent": role}
            if role == "assistant":
                try:
                    from core.irp_chat_annotator import annotate_assistant_payload
                    payload = annotate_assistant_payload(text, payload)
                except Exception:
                    pass
            _cl.append(
                kind,
                payload=payload,
                session_id=session_id,
            )
        except Exception as exc:
            try:
                self.sig_debug.emit(f"[CHAT:acatalepsy] {kind} write failed: {exc!r}")
            except Exception:
                pass

    def _append_assistant_token(self, token):
        self._assistant_box.append_token(token)

    def _cleanup_empty_assistant_if_needed(self):
        """Remove a placeholder assistant message if generation ended before any tokens arrived."""
        removed = self._assistant_box.cleanup_empty_assistant_if_needed()
        if not removed:
            return
        # After deletion, any stored indices are stale.
        self._active_widget = None
        self._render_session()
        self.sig_sync_history.emit(self._build_engine_history_from_session())

    def _maybe_generate_title(self):
        if self._sessions.maybe_generate_title():
            self._notify_header_update()

    def _topic_dominant(self):
        return self._sessions.topic_dominant()

    def _notify_header_update(self):
        dt = QDateTime.currentDateTime().toString("ddd \u2022 HH:mm")
        title = self._sessions.ensure_title(
            self._current_session.get("messages", []),
            self._current_session.get("title"),
        )
        self.ui_bridge.sig_terminal_header.emit(getattr(self, "_mod_id", ""), title, dt)

    def _derive_title(self, messages):
        return self._sessions.derive_title(messages)

    def _build_summary(self, messages):
        return self._sessions.build_summary(messages, self._current_session.get("title"))
    def _slugify(self, text):
        return self._sessions.slugify(text)

    def _now_iso(self):
        return self._sessions.now_iso()

