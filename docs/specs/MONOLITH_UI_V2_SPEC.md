# MONOLITH UI V2 — UPGRADE SPECIFICATION

**Target**: Codex implementation tasks  
**Scope**: UI layer only — kernel contract, engine layer, and world state are frozen  
**Constraint**: PySide6/Qt — no framework migration  
**Philosophy**: Conversation-centered OS, not workspace IDE  

---

## 0. FROZEN CONTRACTS — DO NOT MODIFY

These files define the execution hierarchy and are architecturally frozen.  
Any v2 UI work must consume their signals and data, never restructure them.

| File | Role | Why frozen |
|------|------|-----------|
| `monokernel/guard.py` | Sole execution authority | Kernel Contract v2 §1 |
| `monokernel/dock.py` | Queue management | Kernel Contract v2 §2.2 |
| `monokernel/bridge.py` | Task translator | Kernel Contract v2 §2.1 |
| `engine/llm.py` | LLM engine implementation | Engine port interface |
| `engine/bridge.py` | Engine signal bridge | SIGNAL PASSTHROUGH LAW |
| `core/world_state.py` | WorldStateStore | Single source of truth for state |
| `core/world_actions.py` | Action validation + policy | PolicyDecision enum |
| `core/task.py` | Task dataclass | Shared contract |
| `ui/bridge.py` | UIBridge signal hub | All UI ↔ kernel signals |

The signal interface on `UIBridge` is the v2 UI's only contract surface:

```
sig_terminal_header(str, str, str)
sig_apply_operator(dict)
sig_open_overseer()
sig_theme_changed(str)
sig_config_changed(dict)
sig_launch_addon(str)
sig_world_action(dict)
sig_world_action_pending(dict)
sig_world_action_approved(dict)
sig_world_action_rejected(object)
sig_monitor_log(str, str)
sig_reload_modules()
```

---

## 1. CORE PHILOSOPHY

### 1.1 Conversation-centered, not workspace-centered

v1 uses a `QStackedLayout` page model: you're in Chat OR Hub OR Databank.  
v2 eliminates page switching. The conversation surface is always visible, always primary.  
Everything else is a **companion pane** that reacts to conversation and kernel state.

### 1.2 Borderless elevation

v1 uses `MonoGroupBox` (bordered rectangles with painted title text) to separate every UI group.  
v2 eliminates 80% of visible borders. Hierarchy is communicated through:

- **Background lightness shifts** between nested surfaces (bg_main → bg_panel → bg_group)
- **Typographic weight and size** (14px/500 for section headers, 12px/400 for content)
- **Spacing** (16px between groups, 8px between items within groups)

`MonoGroupBox` remains available but becomes a rarely-used explicit container, not the default grouping mechanism.

### 1.3 State-driven UI, not navigation-driven

The companion pane content is selected by kernel state, not user navigation:

| WorldState condition | Companion shows |
|---------------------|-----------------|
| Engine status == GENERATING | Generation trace + progress |
| Engine status == IDLE, no pending | Model config (collapsed) |
| `pending_action` is not None | Action review (full) |
| Vision engine active | SD preview + parameters |
| Audio engine active | Waveform + audio controls |
| User triggered history | Archive browser |
| User triggered search | Search results |
| Default / nothing active | Collapsed (conversation full-width) |

The companion pane reads `WorldStateStore.snapshot()` on every `sig_status` emission from MonoGuard and transitions accordingly.

---

## 2. LAYOUT ARCHITECTURE

### 2.1 Top-level structure

Replace the current `MonolithUI` layout in `ui/main_window.py`.

**Current** (v1):
```
root_layout (QVBoxLayout)
├── gradient_line
├── top_bar
├── content_layout (QHBoxLayout)
│   ├── sidebar (70px, QFrame)
│   │   ├── module_strip (scrollable icons)
│   │   ├── btn_hub
│   │   └── btn_addons
│   └── stack (QStackedLayout)
│       ├── empty_page
│       ├── hub_page
│       ├── chat_page (2629 lines)
│       └── [addon pages]
└── bottom_bar (status label)
```

**Target** (v2):
```
root_layout (QVBoxLayout)
├── gradient_line (existing, keep)
├── omni_bar (new, replaces top_bar)
├── content_layout (QHBoxLayout)
│   ├── icon_rail (40px, static, new)
│   ├── conversation_surface (flex, always visible, new)
│   └── companion_pane (0–360px, collapsible, new)
└── vitals_footer (new, replaces floating VitalsWindow)
```

### 2.2 Removed components

| Component | File | Replacement |
|-----------|------|-------------|
| `QStackedLayout` page stack | `main_window.py` | Eliminated — conversation is permanent |
| Hub page | `ui/pages/hub.py` | Omni-bar commands for profiles/themes |
| `ModuleStrip` | `ui/components/module_strip.py` | Icon rail (static, non-scrollable) |
| `VitalsWindow` (floating dialog) | `ui/components/complex.py` | Vitals footer (inline) |
| Top bar with title/time/buttons | `main_window.py:_build_top_bar` | Omni-bar |
| `SplitControlBlock` | `ui/components/complex.py` | Removed — no tabbed sidebar panels |

### 2.3 New file structure

```
ui/
├── main_window.py          # REWRITE — new root layout
├── bridge.py               # FROZEN
├── conversation_surface.py  # NEW — extracted from chat.py
├── companion_pane.py        # NEW — state-driven right panel
├── omni_bar.py              # NEW — replaces command_palette + top_bar
├── vitals_footer.py         # NEW — inline system metrics
├── icon_rail.py             # NEW — replaces module_strip
├── components/
│   ├── atoms.py             # MODIFY — keep MonoButton, MonoSlider; deprecate MonoGroupBox usage
│   ├── message_widget.py    # MODIFY — add ToolCallBubble, ToolResultBubble
│   ├── command_palette.py   # DELETE — absorbed into omni_bar.py
│   ├── complex.py           # MODIFY — remove VitalsWindow, keep GradientLine, FlameLabel
│   ├── module_strip.py      # DELETE — replaced by icon_rail.py
│   └── drop_zone.py         # KEEP
├── panels/                  # NEW directory
│   ├── model_config.py      # NEW — extracted from chat.py model/config controls
│   ├── generation_trace.py  # NEW — extracted from chat.py trace display
│   ├── action_review.py     # NEW — extracted from chat.py action review
│   ├── archive_browser.py   # NEW — extracted from chat.py archive panel
│   └── audit_log.py         # NEW — extracted from chat.py audit panel
├── pages/
│   ├── chat.py              # DELETE after extraction complete
│   ├── chat_session.py      # KEEP — session management (standalone)
│   ├── chat_archive.py      # KEEP — archive I/O (standalone)
│   ├── hub.py               # DELETE — replaced by omni-bar
│   └── databank.py          # MODIFY — becomes a companion pane panel
├── modules/                 # KEEP — addon modules unchanged
├── addons/                  # KEEP — addon framework unchanged
├── tools/                   # KEEP
└── overseer.py              # KEEP
```

---

## 3. TASK BREAKDOWN

### TASK 1: Extract ConversationSurface from chat.py

**Priority**: CRITICAL — unlocks all other tasks  
**New file**: `ui/conversation_surface.py`  
**Estimated size**: ~500 lines

#### What it owns

- The `QListWidget` message list (currently `self.lst_chat` in PageChat)
- The input row: text input, send/stop button, file attach button
- Message rendering: `_append_message_widget`, `_render_session`, `_add_message`
- Token streaming: `_append_assistant_token`
- Send flow: `send()`, `_send_message()`, `handle_send_click()`
- Session display: `_set_current_session`, `_build_engine_history_from_session`
- Message mutation: `_undo_last_mutation`, `_delete_from_index`, `_edit_from_index`, `_regen_last_assistant`
- Input handling: `eventFilter` for Enter key, `_on_input_changed`
- Agent autocomplete: `_update_agent_popup`, `_parse_agent_args`, `_apply_agent_command`

#### What it does NOT own

- Model config UI (→ `panels/model_config.py`)
- Archive list UI (→ `panels/archive_browser.py`)
- Trace display UI (→ `panels/generation_trace.py`)
- Action review UI (→ `panels/action_review.py`)
- Audit log UI (→ `panels/audit_log.py`)
- API/model loading logic (→ `panels/model_config.py`)
- Local server management (→ `panels/model_config.py`)

#### Class interface

```python
class ConversationSurface(QWidget):
    # Signals emitted to parent
    sig_send_requested = Signal(str)           # user pressed Enter/Send
    sig_file_dropped = Signal(str)             # file path dropped into input
    sig_mutation_requested = Signal(str, dict)  # "delete"|"edit"|"regen", payload
    sig_agent_command = Signal(str, dict)       # agent subcommand, parsed args
    
    def __init__(self, state: AppState, ui_bridge: UIBridge, parent=None):
        ...
    
    # Called by parent when engine emits tokens
    def append_token(self, token: str) -> None: ...
    
    # Called by parent to load a session
    def load_session(self, session: ChatSessionManager) -> None: ...
    
    # Called by parent when generation finishes
    def finalize_response(self) -> None: ...
    
    # Called by parent to add a complete message
    def add_message(self, role: str, text: str) -> None: ...
    
    # Returns the current message history for engine submission
    def get_history(self) -> list[dict]: ...
```

#### Extraction method

1. Copy `PageChat.__init__` lines 498–637 (chat area construction) into `ConversationSurface.__init__`
2. Move all methods listed under "What it owns" above
3. Replace direct `self.state` / `self.ui_bridge` references with constructor-injected references
4. Remove the `QSplitter` wrapping — ConversationSurface is a flat `QVBoxLayout` widget
5. Verify: ConversationSurface should have zero references to model combo boxes, sliders, archive lists, or trace displays

#### Acceptance criteria

- ConversationSurface can render a session, stream tokens, and emit send requests
- It has no knowledge of model config, archives, or traces
- chat.py line count drops by ~1200 lines

---

### TASK 2: Extract companion pane panels

**Priority**: CRITICAL — must happen alongside Task 1  
**New files**: `ui/panels/model_config.py`, `ui/panels/generation_trace.py`, `ui/panels/action_review.py`, `ui/panels/archive_browser.py`, `ui/panels/audit_log.py`

#### model_config.py (~300 lines)

Extract from `PageChat.__init__` lines 195–342 and associated methods:

- Engine selector combo (`self.combo_engine`)
- API base URL input (`self.inp_base_url`)
- API key input (`self.inp_api_key`)
- Model combo + fetch button
- Context limit slider
- Temperature, top_p, top_k, min_p, repeat_penalty sliders
- System prompt text area
- Load/unload button
- Save/reset config buttons
- `ModelListLoader` thread and `_fetch_models`, `_on_models_loaded`, `_on_models_error`
- Local server management: `_start_local_server`, `_stop_local_server`, `_resolve_native_llama_server`, `_build_server_cmd`, etc.
- `_apply_config_to_controls`, `_save_config`, `_update_config_value`

```python
class ModelConfigPanel(QWidget):
    sig_config_changed = Signal(dict)    # emitted when user changes any config value
    sig_load_requested = Signal()         # user clicked Load
    sig_unload_requested = Signal()       # user clicked Unload
    
    def __init__(self, state: AppState, ui_bridge: UIBridge, parent=None): ...
    def apply_config(self, config: dict) -> None: ...
    def set_engine_status(self, status: SystemStatus) -> None: ...
```

#### generation_trace.py (~100 lines)

Extract the trace display `QTextEdit` and methods `_trace_html`, `_trace_plain`.

```python
class GenerationTracePanel(QWidget):
    def __init__(self, parent=None): ...
    def append_trace(self, engine_key: str, message: str) -> None: ...
    def clear(self) -> None: ...
```

#### action_review.py (~150 lines)

Extract `_build_action_review_panel` and handlers `_action_review_show`, `_approve_pending`, `_reject_pending`, `_on_world_action_pending`.

```python
class ActionReviewPanel(QWidget):
    sig_approved = Signal(dict)    # approved action
    sig_rejected = Signal(object)  # rejected action
    
    def __init__(self, parent=None): ...
    def show_action(self, action: dict) -> None: ...
    def clear(self) -> None: ...
```

#### archive_browser.py (~150 lines)

Extract the archive list widget and methods `_refresh_archive_list`, `_load_chat_archive`, `_delete_selected_archive`, `_save_chat_archive`, history search.

```python
class ArchiveBrowserPanel(QWidget):
    sig_archive_selected = Signal(str)   # archive path
    sig_archive_deleted = Signal(str)
    sig_new_session = Signal()
    
    def __init__(self, archive_dir: Path, parent=None): ...
    def refresh(self) -> None: ...
```

#### audit_log.py (~80 lines)

Extract `_refresh_audit_list`, `_clear_audit_log`, and the audit list widget.

```python
class AuditLogPanel(QWidget):
    def __init__(self, world_state: WorldStateStore, parent=None): ...
    def refresh(self) -> None: ...
    def clear(self) -> None: ...
```

---

### TASK 3: Create CompanionPane

**Priority**: HIGH — depends on Task 2  
**New file**: `ui/companion_pane.py`  
**Estimated size**: ~250 lines

The companion pane is a `QWidget` with a `QStackedLayout` that transitions between panel contents based on kernel state. It differs from v1's tabbed sidebar in that the **transitions are state-driven, not user-navigated**.

#### State machine

```python
from enum import Enum, auto

class CompanionState(Enum):
    COLLAPSED = auto()      # pane hidden, conversation takes full width
    CONFIG = auto()         # model config panel visible
    GENERATING = auto()     # generation trace + live progress
    ACTION_REVIEW = auto()  # pending action approval
    VISION = auto()         # SD preview + parameters (if vision engine active)
    AUDIO = auto()          # audio controls (if audio engine active)
    ARCHIVE = auto()        # archive browser
    AUDIT = auto()          # action audit log
    DATABANK = auto()       # databank file browser
    ADDON = auto()          # running addon module view
```

#### Transition rules

```python
class CompanionPane(QWidget):
    sig_state_changed = Signal(str)  # emits new CompanionState.name
    
    PRIORITY_ORDER = [
        # Higher priority states override lower ones
        CompanionState.ACTION_REVIEW,   # always wins — safety critical
        CompanionState.GENERATING,       # active work visible
        CompanionState.VISION,
        CompanionState.AUDIO,
        CompanionState.CONFIG,
        CompanionState.ARCHIVE,
        CompanionState.AUDIT,
        CompanionState.DATABANK,
        CompanionState.ADDON,
        CompanionState.COLLAPSED,
    ]
    
    def evaluate_state(self, world_snapshot: dict) -> CompanionState:
        """Determine companion state from world state snapshot.
        
        Called on every sig_status emission from MonoGuard.
        Called on every sig_world_action_pending emission.
        Called on user icon_rail clicks (override to specific state).
        """
        # Priority 1: pending action always takes over
        if world_snapshot.get("session", {}).get("pending_action"):
            return CompanionState.ACTION_REVIEW
        
        engines = world_snapshot.get("engines", {})
        
        # Priority 2: any engine actively generating
        for key, engine_state in engines.items():
            status = engine_state.get("status", "")
            if status in ("GENERATING", "LOADING"):
                if key == "vision":
                    return CompanionState.VISION
                if key == "audio":
                    return CompanionState.AUDIO
                return CompanionState.GENERATING
        
        # Priority 3: user-pinned state (set by icon rail click)
        if self._pinned_state is not None:
            return self._pinned_state
        
        # Default: collapsed
        return CompanionState.COLLAPSED
    
    def transition_to(self, new_state: CompanionState) -> None:
        """Animate transition to new companion state."""
        if new_state == self._current_state:
            return
        
        if new_state == CompanionState.COLLAPSED:
            self._animate_width(target=0, duration_ms=200)
        else:
            self._animate_width(target=360, duration_ms=200)
            self._stack.setCurrentWidget(self._panels[new_state])
        
        self._current_state = new_state
        self.sig_state_changed.emit(new_state.name)
```

#### Width animation

Use `QPropertyAnimation` on `maximumWidth`:

```python
def _animate_width(self, target: int, duration_ms: int = 200) -> None:
    self._anim = QPropertyAnimation(self, b"maximumWidth")
    self._anim.setDuration(duration_ms)
    self._anim.setStartValue(self.maximumWidth())
    self._anim.setEndValue(target)
    self._anim.setEasingCurve(QEasingCurve.OutCubic)
    self._anim.start()
```

#### Panel registration

```python
def __init__(self, state, ui_bridge, parent=None):
    super().__init__(parent)
    self._stack = QStackedLayout(self)
    self._panels = {}
    self._current_state = CompanionState.COLLAPSED
    self._pinned_state = None  # user-override from icon rail
    
    # Register panels
    self._panels[CompanionState.CONFIG] = ModelConfigPanel(state, ui_bridge)
    self._panels[CompanionState.GENERATING] = GenerationTracePanel()
    self._panels[CompanionState.ACTION_REVIEW] = ActionReviewPanel()
    self._panels[CompanionState.ARCHIVE] = ArchiveBrowserPanel(ARCHIVE_DIR)
    self._panels[CompanionState.AUDIT] = AuditLogPanel(state.world_state)
    
    for panel in self._panels.values():
        self._stack.addWidget(panel)
    
    self.setMaximumWidth(0)  # start collapsed

def pin_state(self, state: CompanionState) -> None:
    """Called by icon rail when user clicks a panel icon."""
    if self._pinned_state == state:
        self._pinned_state = None  # toggle off
    else:
        self._pinned_state = state
    self.transition_to(self.evaluate_state(self._get_world_snapshot()))
```

---

### TASK 4: Create IconRail

**Priority**: MEDIUM — depends on Task 3  
**New file**: `ui/icon_rail.py`  
**Estimated size**: ~120 lines

Replaces `ModuleStrip`. Static vertical strip of icons, 40px wide. No scrolling, no drag reorder, no pulse animations.

```python
class IconRail(QWidget):
    sig_panel_requested = Signal(str)  # CompanionState name
    
    ICONS = [
        ("C", "Config",  CompanionState.CONFIG),
        ("H", "History", CompanionState.ARCHIVE),
        ("A", "Audit",   CompanionState.AUDIT),
        ("D", "Databank",CompanionState.DATABANK),
    ]
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(40)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 12, 0, 12)
        layout.setSpacing(6)
        
        for icon_char, tooltip, state in self.ICONS:
            btn = _RailButton(icon_char, tooltip)
            btn.clicked.connect(lambda checked, s=state: self.sig_panel_requested.emit(s.name))
            layout.addWidget(btn, alignment=Qt.AlignCenter)
        
        layout.addStretch()
```

#### Behavior

- Click toggles the companion pane to that state (or collapses if already showing it)
- Active state button gets a subtle left-edge accent line (2px, accent_primary color)
- No text labels — tooltip on hover only
- When addons are running, dynamically append addon icons below the static set
- When an addon icon is clicked, companion pane shows that addon's widget

#### Dynamic addon icons

```python
def add_addon_icon(self, addon_id: str, icon_char: str, label: str) -> None:
    """Called by AddonHost when a module is launched."""
    btn = _RailButton(icon_char, label)
    btn.clicked.connect(lambda: self.sig_panel_requested.emit(f"addon:{addon_id}"))
    self._addon_buttons[addon_id] = btn
    self._layout.insertWidget(self._layout.count() - 1, btn, alignment=Qt.AlignCenter)

def remove_addon_icon(self, addon_id: str) -> None:
    """Called by AddonHost when a module is closed."""
    btn = self._addon_buttons.pop(addon_id, None)
    if btn:
        self._layout.removeWidget(btn)
        btn.deleteLater()
```

---

### TASK 5: Create OmniBar

**Priority**: MEDIUM  
**New file**: `ui/omni_bar.py`  
**Estimated size**: ~300 lines  
**Replaces**: `ui/components/command_palette.py` + `main_window.py:_build_top_bar`

The omni-bar is a persistent top-center input widget that provides unified search/command access. It absorbs the current `CommandPalette` functionality and extends it.

#### Layout

- Centered horizontally in the top bar area
- Height: 32px, border-radius: 16px (pill shape)
- Background: bg_panel, border: 1px border_subtle
- Left edge: small magnifying glass icon or `/` hint
- Right edge: current model name badge + keyboard shortcut hint (Ctrl+K)
- Left of omni-bar: drag area for window movement (preserve frameless drag behavior)
- Right of omni-bar: window control buttons (minimize, maximize, close)

#### Command grammar

The omni-bar input is parsed with a prefix system:

| Prefix | Action | Source |
|--------|--------|--------|
| (no prefix) | Fuzzy search addon names | `AddonRegistry` |
| `>model ` | Switch model | Current config `models` list |
| `>theme ` | Switch theme | `list_themes()` |
| `>profile ` | Switch operator profile | `OperatorManager.list()` |
| `/history ` | Search chat archives | `search_archives()` |
| `/agent ` | Agent commands | Existing agent subcommand parser |
| `>load` | Load current model | Direct action |
| `>unload` | Unload current model | Direct action |
| `>new` | New chat session | Direct action |
| `>vitals` | Toggle vitals footer expanded view | Direct action |

#### Result rendering

```python
class _OmniResult(QFrame):
    """Single result row in the dropdown."""
    clicked = Signal(str)  # action identifier
    
    def __init__(self, icon: str, title: str, subtitle: str, action_id: str):
        ...
        # Layout: [icon 18px] [title expanding] [subtitle dim right-aligned]
        # No borders — separated by 1px spacing
        # Hover: bg_button_hover background
        # Selected (keyboard): bg_button_pressed background + accent_primary left edge
```

#### Keyboard interaction

- `Ctrl+K` or clicking the bar: focus the input
- `Escape`: clear and unfocus
- `Up/Down`: navigate results
- `Enter`: execute selected result
- Typing immediately filters — no debounce needed for local search
- `/history` prefix triggers `search_archives()` which may be slower — debounce 300ms

#### Integration with hub functionality

The following hub page features move into omni-bar commands:

| Hub feature | Omni-bar equivalent |
|-------------|-------------------|
| Operator profile cards + selection | `>profile research` → applies operator |
| Theme switcher dropdown | `>theme midnight` → calls `apply_theme()` |
| Operator creation dialog | `>profile new` → opens `_NameDialog` inline or as small modal |
| Module launch grid | (no prefix) → shows addon search results, Enter launches |

The hub page (`ui/pages/hub.py`) is deleted after this task.

---

### TASK 6: Create VitalsFooter

**Priority**: LOW  
**New file**: `ui/vitals_footer.py`  
**Estimated size**: ~150 lines  
**Replaces**: `VitalsWindow` in `ui/components/complex.py`

#### Default state (collapsed, 28px height)

A single horizontal bar at the bottom of the window:

```
[ CPU 12% | VRAM 4.2G/8G | CTX 2048/8192 | ··· qwen2.5:14b | READY ]
```

- Left section: resource metrics from `WorldStateStore.state["resources"]`
- Right section: active model name + engine status from `WorldStateStore.state["engines"]`
- Separator: thin 1px line at top edge (border_subtle)
- Background: bg_sidebar
- Text: fg_dim, 11px monospace (Consolas)
- Status keyword colored: READY=fg_accent, GENERATING=accent_primary, ERROR=fg_error

#### Expanded state (hover or click, animated to ~120px height)

When user hovers over any metric, the footer smoothly expands to show:

- Per-engine breakdown (LLM engine status, Vision engine status, Audio engine status)
- VRAM detail: which model is consuming how much
- One-click "Unload" button next to each loaded model
- Context window visual bar (filled portion = used tokens / max tokens)
- Queue depth if any tasks are queued

Use `QPropertyAnimation` on `maximumHeight` with `OutCubic` easing, 150ms duration.

#### Data source

Subscribe to `MonoGuard.sig_status` for engine state changes.  
Poll `WorldStateStore.state["resources"]` on a `QTimer` (interval: 2000ms, matching current `VitalsWindow` behavior).

---

### TASK 7: Inline tool widgets in message stream

**Priority**: HIGH  
**Modified file**: `ui/components/message_widget.py`  
**Estimated additions**: ~200 lines

#### New message types

Currently `MessageWidget` renders all content as styled text (markdown-ish HTML). v2 adds two new inline widget types that render inside the chat stream as first-class message elements.

##### ToolCallBubble

Rendered when the assistant emits a `monolith_cmd` envelope. Instead of showing raw JSON in the chat, render a compact inline card:

```
┌─────────────────────────────────────┐
│ ⚡ generate_image                    │
│ prompt: "a castle on a hill"        │
│ steps: 20, cfg: 7.5                 │
│                                     │
│ [Approve]  [Reject]     ← only if REQUIRE_APPROVAL
└─────────────────────────────────────┘
```

- Background: bg_group
- Left accent: 3px line, accent_primary color
- Icon: ⚡ for engine commands, 🔧 for skill calls
- Tool name: 13px, bold
- Parameters: 11px monospace, fg_dim, max 3 lines then "show more" toggle
- If `check_policy()` returns `REQUIRE_APPROVAL`: show Approve/Reject buttons inline
- If `AUTO_APPROVE`: show a subtle "Auto-approved ✓" badge instead of buttons
- On approve: the bubble morphs — buttons fade out, a spinner appears, then the ToolResultBubble appears below

##### ToolResultBubble

Rendered when a tool returns its result. Content varies by tool type:

| Tool | Inline rendering |
|------|-----------------|
| `generate_image` | Thumbnail (max 200px wide) + "View full" link that pins companion to VISION |
| `calculate` | Result value in a styled code block |
| `search-history` | Compact result list (max 3 items) + "Show all in companion" link |
| `read-file` | First 5 lines of file content in a code block + "Full file" link |
| `save-note` | Confirmation badge: "Saved to {path}" |
| Other / unknown | Raw text result in a subtle card, same style as current `[tool_result]` but with card styling |

```python
class ToolCallBubble(QFrame):
    sig_approved = Signal(dict)
    sig_rejected = Signal(dict)
    
    def __init__(self, action: dict, policy: PolicyDecision, parent=None):
        ...

class ToolResultBubble(QFrame):
    sig_expand_in_companion = Signal(str, object)  # tool_name, full_result
    
    def __init__(self, tool_name: str, result: object, parent=None):
        ...
```

#### Integration point

In `ConversationSurface._append_message_widget`, detect tool call/result messages by role or content pattern and instantiate the appropriate bubble widget instead of a standard `MessageWidget`.

The detection logic (simplified):

```python
def _widget_for_message(self, role: str, text: str, idx: int) -> QWidget:
    if role == "tool_call":
        action = json.loads(text)
        policy = check_policy(action)
        bubble = ToolCallBubble(action, policy)
        bubble.sig_approved.connect(lambda a: self.ui_bridge.sig_world_action_approved.emit(a))
        bubble.sig_rejected.connect(lambda a: self.ui_bridge.sig_world_action_rejected.emit(a))
        return bubble
    elif role == "tool_result":
        payload = json.loads(text)
        return ToolResultBubble(payload.get("tool"), payload.get("result"))
    else:
        return MessageWidget(...)  # existing path
```

---

### TASK 8: Borderless elevation pass

**Priority**: MEDIUM — can be done incrementally  
**Modified files**: `ui/components/atoms.py`, `core/themes.py`, `core/theme_engine.py`

#### Theme additions

Add two new tokens to the `Theme` dataclass in `core/themes.py`:

```python
@dataclass(frozen=True)
class Theme:
    ...
    # NEW: elevation surfaces (v2)
    bg_surface_1: str    # slightly lighter than bg_main — primary content cards
    bg_surface_2: str    # slightly lighter than surface_1 — nested groups
```

For the Midnight theme:
```python
bg_surface_1 = "#131620"   # between bg_main (#0e1117) and bg_panel (#161922)
bg_surface_2 = "#181c27"   # between bg_panel and bg_group
```

#### Elevation classes

Add a utility function for applying elevation without borders:

```python
# In atoms.py or a new ui/elevation.py

def apply_elevation(widget: QWidget, level: int = 1) -> None:
    """Apply borderless elevation styling to a widget.
    
    level 0: bg_main (base canvas)
    level 1: bg_surface_1 (primary content area)
    level 2: bg_surface_2 (nested group within content)
    """
    import core.style as s
    colors = [s.BG_MAIN, getattr(s, 'BG_SURFACE_1', s.BG_PANEL), getattr(s, 'BG_SURFACE_2', s.BG_GROUP)]
    bg = colors[min(level, len(colors) - 1)]
    widget.setStyleSheet(f"""
        background: {bg};
        border: none;
        border-radius: 6px;
        padding: 12px;
    """)
```

#### Migration pattern

Replace `MonoGroupBox` instances incrementally:

**Before** (v1):
```python
group = MonoGroupBox("GENERATION")
group.add_widget(temperature_slider)
group.add_widget(top_p_slider)
```

**After** (v2):
```python
group = QFrame()
group.setObjectName("elevation_2")
layout = QVBoxLayout(group)
layout.setContentsMargins(12, 8, 12, 8)
layout.setSpacing(6)

header = QLabel("GENERATION")
header.setStyleSheet(f"color: {_s.FG_DIM}; font-size: 11px; font-weight: 500; letter-spacing: 1.5px;")
layout.addWidget(header)
layout.addWidget(temperature_slider)
layout.addWidget(top_p_slider)

apply_elevation(group, level=2)
```

The section header replaces the painted border-title. No box, no border — just a typographic label above the controls.

---

### TASK 9: Wire new layout in main_window.py

**Priority**: CRITICAL — final integration task, depends on Tasks 1–6  
**Modified file**: `ui/main_window.py`  
**Estimated rewrite**: ~200 lines (down from ~400)

#### New __init__ structure

```python
class MonolithUI(QMainWindow):
    def __init__(self, state: AppState, ui_bridge: UIBridge):
        super().__init__()
        self.state = state
        self.ui_bridge = ui_bridge
        
        # Window setup (keep existing frameless + translucent)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setMouseTracking(True)
        self.resize(1100, 700)
        self.setMinimumSize(600, 400)
        
        # Keep existing resize grip logic unchanged
        self._resize_edge = None
        self._resize_origin = None
        self._resize_geo = None
        self._GRIP = 4
        
        # Root layout
        main_widget = QWidget()
        main_widget.setObjectName("MainFrame")
        self.setCentralWidget(main_widget)
        root = QVBoxLayout(main_widget)
        root.setContentsMargins(1, 1, 1, 1)
        root.setSpacing(0)
        
        # Gradient line (keep existing)
        self.gradient_line = GradientLine()
        root.addWidget(self.gradient_line)
        
        # Omni-bar (replaces top_bar)
        self.omni_bar = OmniBar(state, ui_bridge)
        root.addWidget(self.omni_bar)
        
        # Content area
        content = QHBoxLayout()
        content.setSpacing(0)
        
        # Icon rail (replaces sidebar + module_strip)
        self.icon_rail = IconRail()
        content.addWidget(self.icon_rail)
        
        # Conversation surface (always visible, flex width)
        self.conversation = ConversationSurface(state, ui_bridge)
        content.addWidget(self.conversation, stretch=1)
        
        # Companion pane (state-driven, 0-360px)
        self.companion = CompanionPane(state, ui_bridge)
        content.addWidget(self.companion)
        
        root.addLayout(content)
        
        # Vitals footer (replaces floating VitalsWindow)
        self.vitals = VitalsFooter(state)
        root.addWidget(self.vitals)
        
        # --- Signal wiring ---
        self._wire_signals()
    
    def _wire_signals(self):
        # Icon rail → companion pane
        self.icon_rail.sig_panel_requested.connect(
            lambda name: self.companion.pin_state(CompanionState[name])
        )
        
        # Conversation send → engine submission
        self.conversation.sig_send_requested.connect(self._on_send)
        
        # MonoGuard status → companion pane state evaluation
        # (connected via bootstrap.py when guard is created)
        
        # UIBridge signals
        self.ui_bridge.sig_world_action_pending.connect(
            lambda action: self.companion.transition_to(CompanionState.ACTION_REVIEW)
        )
        self.ui_bridge.sig_theme_changed.connect(self._on_theme_changed)
    
    def attach_guard_signals(self, guard: MonoGuard):
        """Called by bootstrap after guard creation."""
        guard.sig_status.connect(self._on_engine_status)
        guard.sig_token.connect(
            lambda key, token: self.conversation.append_token(token)
        )
        guard.sig_trace.connect(
            lambda key, msg: self.companion.get_panel(CompanionState.GENERATING).append_trace(key, msg)
        )
        guard.sig_finished.connect(
            lambda key, result: self.conversation.finalize_response()
        )
    
    def _on_engine_status(self, engine_key: str, status: SystemStatus):
        """Re-evaluate companion pane state on any engine status change."""
        snapshot = self.state.world_state.snapshot()
        new_state = self.companion.evaluate_state(snapshot)
        self.companion.transition_to(new_state)
        self.vitals.update_engine(engine_key, status)
```

#### What's removed from main_window.py

- `_build_top_bar()` — replaced by OmniBar
- `set_page()` / `switch_to_module()` / `close_module()` — no page stack
- `QStackedLayout` and `self.pages` dict — eliminated
- `ModuleStrip` instantiation and signal connections
- `btn_hub`, `btn_addons` sidebar buttons
- `VitalsWindow` creation and toggle logic
- `_terminal_titles` tracking — absorbed by companion pane

---

## 4. STYLE GUIDE — V2 VISUAL LANGUAGE

### 4.1 Spacing scale

| Token | Value | Use |
|-------|-------|-----|
| `space-xs` | 4px | Between icon and label |
| `space-sm` | 8px | Between items in a group |
| `space-md` | 12px | Padding inside cards/panels |
| `space-lg` | 16px | Between groups |
| `space-xl` | 24px | Between major sections |

### 4.2 Typography

| Role | Font | Size | Weight | Color |
|------|------|------|--------|-------|
| Section header | Consolas | 11px | 500 | fg_dim |
| Control label | Inter/system sans | 12px | 400 | fg_secondary |
| Control value | Consolas | 12px | 400 | fg_text |
| Message body | system sans | 14px | 400 | fg_text |
| Message role badge | Consolas | 10px | 700 | fg_dim |
| Omni-bar input | system sans | 13px | 400 | fg_text |
| Vitals metric | Consolas | 11px | 400 | fg_dim |

### 4.3 Borders — when to use them

| Context | Border? | Instead |
|---------|---------|---------|
| Group of sliders | No | Elevation background + header label |
| Input field | Yes, 1px border_subtle | Focus: 1px accent_primary |
| Companion pane edge | Yes, 1px border_subtle on left edge only | — |
| Message bubble | No | Elevation level 1 background |
| Tool call card | Yes, 3px left accent line only | — |
| Icon rail edge | Yes, 1px border_subtle on right edge only | — |
| Vitals footer edge | Yes, 1px border_subtle on top edge only | — |

### 4.4 Animation budget

Only two animated elements in v2:

1. **Companion pane slide**: `QPropertyAnimation` on `maximumWidth`, 200ms, `OutCubic`
2. **Gradient line breathing**: existing `GradientLine` animation during generation (keep as-is)

Everything else is instant state transitions. No spring physics, no scale transitions, no pulse effects.

### 4.5 Scrollbar styling

Keep the existing `_build_scrollbar_style()` from `core/style.py` but reduce the border on the handle from 1px accent_primary to 1px border_subtle. The v1 scrollbar handle border is too prominent. The handle hover state keeps `scrollbar_handle_hover` background — no border change on hover.

---

## 5. MIGRATION ORDER

Execute tasks in this order to maintain a working application at every step:

```
Phase 1: Extraction (app stays on v1 layout, but code is decomposed)
  1. Task 1: Extract ConversationSurface → verify chat still works
  2. Task 2: Extract all companion panels → verify chat still works

Phase 2: New components (built but not yet wired into main layout)
  3. Task 4: Build IconRail
  4. Task 3: Build CompanionPane (consumes panels from Task 2)
  5. Task 5: Build OmniBar
  6. Task 6: Build VitalsFooter

Phase 3: Integration (swap layout)
  7. Task 9: Rewrite main_window.py with new layout
  8. Task 7: Add inline tool widgets to message_widget.py
  9. Task 8: Borderless elevation pass (incremental, panel by panel)

Phase 4: Cleanup
  10. Delete ui/pages/chat.py (replaced by conversation_surface.py)
  11. Delete ui/pages/hub.py (replaced by omni_bar.py)
  12. Delete ui/components/command_palette.py (absorbed into omni_bar.py)
  13. Delete ui/components/module_strip.py (replaced by icon_rail.py)
  14. Remove VitalsWindow from ui/components/complex.py
```

---

## 6. TESTING STRATEGY

### 6.1 Smoke tests per phase

**Phase 1 gate**: Application boots, chat sends messages, tokens stream, session saves/loads, archives browse, model config applies. All via the existing v1 layout — but the internal code is the new decomposed modules.

**Phase 2 gate**: Each new component can be instantiated standalone in a test harness:
```python
# Quick verification script
app = QApplication(sys.argv)
pane = CompanionPane(mock_state, mock_bridge)
pane.transition_to(CompanionState.CONFIG)
pane.show()
app.exec()
```

**Phase 3 gate**: Full v2 layout boots. Conversation surface renders. Companion pane transitions on engine state changes. Omni-bar searches addons and switches models. Vitals footer shows metrics.

### 6.2 Signal integrity checks

After each extraction, verify no signal connections were dropped:

```python
# In test or debug mode, validate signal graph
def verify_signal_wiring(main_window):
    assert main_window.conversation.sig_send_requested.receivers() > 0
    assert main_window.companion._panels[CompanionState.ACTION_REVIEW].sig_approved.receivers() > 0
    assert main_window.companion._panels[CompanionState.CONFIG].sig_config_changed.receivers() > 0
    # ... etc
```

### 6.3 Kernel contract compliance

No v2 UI code should:
- Import from `monokernel/` directly (use UIBridge signals)
- Call engine methods directly (route through MonoBridge → MonoDock → MonoGuard)
- Mutate `WorldStateStore` directly (read via `.snapshot()`, write via world actions)
- Bypass `check_policy()` for any action execution

---

## 7. OPEN QUESTIONS

These are design decisions deferred to implementation time:

1. **Addon rendering in companion pane**: Do addon modules (Vision SD, AudioGen, ExternalProcess) render as companion pane panels, or do they still get their own full-width view? Current recommendation: companion pane panels, but allow a "pop out to full" button that overlays the conversation temporarily.

2. **Multi-engine trace**: When both LLM and Vision engines are active simultaneously, which trace does the companion pane show? Current recommendation: show LLM trace (it's the conversational one), put a tab toggle at the top of GenerationTracePanel.

3. **Databank integration**: The current Databank page (`ui/pages/databank.py`) is a full file browser. Does it become a companion panel (constrained width) or stay accessible via omni-bar as a modal? Current recommendation: companion panel with simplified list view, "Open in system file manager" button for complex operations.

4. **Overseer integration**: The Overseer (`ui/overseer.py`) is currently a separate window. Keep it separate or absorb into companion pane? Current recommendation: keep separate — it's a debugging/monitoring tool that benefits from its own window.

5. **Keyboard shortcuts**: Define the full shortcut map for v2 (Ctrl+K for omni-bar is settled, but what about companion pane toggle, vitals expand, etc.).

---

*End of spec. Implementation begins with Phase 1, Task 1: Extract ConversationSurface.*
