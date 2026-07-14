# MONOLITH UI V2 — DELTA SPEC (March 17 2026)

**Context**: This document updates the original V2 spec based on an audit of the current codebase. It identifies what shipped, what shipped as a shell, and what remains. Every task below is **no-code design/spec work** except where explicitly noted — Codex implements, this doc directs.

**Rule**: No frozen contract files are touched. See original spec §0.

---

## 1. AUDIT SUMMARY — WHAT SHIPPED VS WHAT DIDN'T

### 1.1 Shipped and working

| Component | File | Lines | Status |
|-----------|------|-------|--------|
| Main window rewrite | `ui/main_window.py` | 682 | Topology correct: OmniBar + IconRail + CompanionPane + VitalsFooter + conversation_stack |
| Companion pane state machine | `ui/companion_pane.py` | 359 | Full: CompanionState enum, evaluate_state(), animate_width(), addon reg, auto-suppression logic |
| Omni-bar | `ui/omni_bar.py` | 329 | Working: `>model`, `>theme`, `>profile`, `/history` commands, result list, keyboard nav |
| Vitals footer | `ui/vitals_footer.py` | 293 | Working: resource polling, engine status, model display |
| Tool bubbles | `ui/components/tool_bubbles.py` | 365 | ToolCallBubble + ToolResultBubble rendering inline in message list |
| Assistant turn box | `ui/pages/assistant_turn_box.py` | 313 | Think block streaming state machine extracted from chat.py |
| Vision panel | `ui/panels/vision_panel.py` | 300 | Real companion panel with preview, progress, controls |
| Audio panel | `ui/panels/audio_panel.py` | 205 | Real companion panel with waveform, playback controls |
| Conversation surface | `ui/conversation_surface.py` | 322 | Extracted as class, used by PageChat — but see §1.2 |

### 1.2 Shipped as proxy shells (structure exists, code still lives in chat.py)

These panels have correct class signatures and signal interfaces, but their `bind_controller()` method just reparents widgets that are still constructed inside `PageChat.__init__`. They are **not standalone**.

| Panel | File | Lines | What it proxies |
|-------|------|-------|-----------------|
| `ModelConfigPanel` | `ui/panels/model_config.py` | 78 | Reparents `controller._control_tab` and `controller._settings_tab` |
| `GenerationTracePanel` | `ui/panels/generation_trace.py` | 49 | Reparents `controller._trace_group` |
| `ActionReviewPanel` | `ui/panels/action_review.py` | 52 | Reparents `controller._action_review` |
| `ArchiveBrowserPanel` | `ui/panels/archive_browser.py` | 59 | Reparents `controller._archive_tab` |
| `AuditLogPanel` | `ui/panels/audit_log.py` | 48 | Reparents `controller._audit_tab` |

**The consequence**: `chat.py` is still 2,625 lines. It still constructs all the widgets these panels claim to own. The extraction from the original spec (Phase 1, Tasks 1–2) has not happened — only the *interfaces* shipped, not the *internals*.

### 1.3 Visually untouched from v1

| Element | Current state | Target state |
|---------|--------------|-------------|
| Icon rail buttons | Single letters (C, G, H, A, D) via `_RailButton` QPushButton with text | QPainter-drawn 16x16 stroke icons |
| Message actions | EDIT/DELETE/REGEN always visible in header row | Hover-reveal floating bar above message |
| Message thought blocks | Full collapsible `_ThinkBlock` section between header and body | Inline pill badge next to timestamp |
| Config panel content | `MonoGroupBox` borders, 8 unsplit sliders, raw 262144 values | Borderless elevation, 3 semantic groups, model status card |
| Conversation surface wrapper | `MonoGroupBox("CHAT")` with painted border | Borderless — no wrapping group box |
| Hub page | `ui/pages/hub.py` still exists (22,411 bytes) | Delete — functionality in omni-bar |
| `command_palette.py` | Deleted | Confirmed gone — absorbed into omni_bar |
| `module_strip.py` | Deleted | Confirmed gone — replaced by icon_rail |

---

## 2. REMAINING TASKS — ORDERED BY PRIORITY

### TASK A: Icon rail — QPainter icons replacing letters

**File**: `ui/icon_rail.py` (99 lines → ~200 lines)  
**Priority**: HIGH — most visible v1 holdover  
**Depends on**: Nothing

Replace `_RailButton(QPushButton)` with `_RailIcon(QWidget)` that uses `paintEvent` to draw stroke icons.

#### Icon set

| Position | Tooltip | Icon description | Paint guide |
|----------|---------|-----------------|-------------|
| 1 | Config | Settings gear | Circle r=3 at center + 8 radial ticks (short lines from r=5.5 outward, stroke-linecap round) |
| 2 | Trace | Activity graph | Polyline path: M3,12 L6,6 L9,9 L13,3 (stroke-linecap round, stroke-linejoin round) |
| 3 | History | Clock | Circle r=5 at (8, 8.5) + two-segment hand: M8,5.5 V8 L10,9.5 |
| 4 | Audit | Log lines | Three horizontal lines: M4,4.5 H12 / M4,8 H9.5 / M4,11.5 H11 (stroke-linecap round) |
| — | *divider* | 18px wide, 1px line | `QPen(border_subtle, 1)`, horizontal centered in rail width |
| 5 | Databank | Grid | Four rounded rects: (3,3,4.2,4.2), (8.8,3,4.2,4.2), (3,8.8,4.2,4.2), (8.8,8.8,4.2,4.2), rx=1 |
| bottom | Overseer | Eye | Outer: ellipse path M2,8 C2,5 5.5,2.5 8,2.5 S14,5 14,8 S10.5,13.5 8,13.5 S2,11 2,8 / Inner: circle r=2.5 at center |

#### Paint implementation pattern

```python
class _RailIcon(QWidget):
    clicked = Signal()
    
    def __init__(self, paint_fn, tooltip, parent=None):
        super().__init__(parent)
        self.setFixedSize(34, 34)
        self.setToolTip(tooltip)
        self.setCursor(Qt.PointingHandCursor)
        self._paint_fn = paint_fn  # Callable[[QPainter, QRectF, QColor], None]
        self._active = False
        self._badge = False
    
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        
        # Background
        if self._active:
            p.fillRect(self.rect(), QColor(_s.BG_PANEL))
        elif self.underMouse():
            p.fillRect(self.rect(), QColor(_s.BG_BUTTON_HOVER))
        
        # Active left accent
        if self._active:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(_s.ACCENT_PRIMARY))
            p.drawRoundedRect(QRectF(0, 7, 2.5, self.height() - 14), 1, 1)
        
        # Icon
        icon_color = QColor(_s.ACCENT_PRIMARY if self._active else _s.FG_DIM)
        if self.underMouse() and not self._active:
            icon_color = QColor(_s.FG_SECONDARY)
        
        # Center the 16x16 icon area in the 34x34 widget
        p.translate(9, 9)
        self._paint_fn(p, QRectF(0, 0, 16, 16), icon_color)
        p.resetTransform()
        
        # Badge dot
        if self._badge:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(_s.FG_WARN))
            p.drawEllipse(self.width() - 10, 3, 6, 6)
        
        p.end()
    
    def set_badge(self, visible: bool) -> None:
        self._badge = visible
        self.update()
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
    
    def enterEvent(self, event):
        self.update()
    
    def leaveEvent(self, event):
        self.update()
```

#### Example paint function (config gear)

```python
def _paint_config(p: QPainter, rect: QRectF, color: QColor):
    pen = QPen(color, 1.2)
    pen.setCapStyle(Qt.RoundCap)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    cx, cy = 8.0, 8.0
    p.drawEllipse(QPointF(cx, cy), 3, 3)
    import math
    for i in range(8):
        angle = math.radians(i * 45)
        x1 = cx + 4.8 * math.cos(angle)
        y1 = cy + 4.8 * math.sin(angle)
        x2 = cx + 6.2 * math.cos(angle)
        y2 = cy + 6.2 * math.sin(angle)
        p.drawLine(QPointF(x1, y1), QPointF(x2, y2))
```

#### Divider

Insert a `QFrame` with `setFixedSize(18, 1)` and background `border_subtle` between the Audit button and the Databank button. Use `layout.addWidget(divider, alignment=Qt.AlignHCenter)`.

#### Dynamic addon icons

Keep the existing `add_addon_icon` / `remove_addon_icon` interface, but icons added dynamically go below the divider (before the stretch). If an addon provides no icon, use a generic diamond or asterisk paint function.

---

### TASK B: Message widget hover-reveal actions

**File**: `ui/components/message_widget.py` (1,067 lines)  
**Priority**: HIGH — directly addresses the crowded header problem  
**Depends on**: Nothing

#### What changes

1. The `self.actions` widget (containing EDIT, REGEN, DELETE buttons) becomes **hidden by default**
2. On `enterEvent`, a floating action bar appears **above the top-right corner** of the message
3. On `leaveEvent`, it hides
4. The action bar is a child of the `MessageWidget` itself, positioned with negative y offset

#### Implementation

The `enterEvent` and `leaveEvent` stubs already exist at lines 801–805 — they just need bodies.

```python
def __init__(self, ...):
    ...
    # Replace the inline actions widget with a floating bar
    self._hover_bar = QFrame(self)
    self._hover_bar.setObjectName("msg_hover_bar")
    hover_layout = QHBoxLayout(self._hover_bar)
    hover_layout.setContentsMargins(3, 3, 3, 3)
    hover_layout.setSpacing(2)
    
    if role == "user":
        btn_edit = _HoverAction("Edit")
        btn_edit.clicked.connect(lambda: self._emit_action("edit"))
        hover_layout.addWidget(btn_edit)
    
    if role == "assistant":
        btn_regen = _HoverAction("Regen")
        btn_regen.clicked.connect(lambda: self._emit_action("regen"))
        hover_layout.addWidget(btn_regen)
        btn_copy = _HoverAction("Copy")
        btn_copy.clicked.connect(lambda: self._copy_to_clipboard())
        hover_layout.addWidget(btn_copy)
    
    if role != "system":
        btn_delete = _HoverAction("Delete")
        btn_delete.clicked.connect(lambda: self._emit_action("delete"))
        hover_layout.addWidget(btn_delete)
    
    self._hover_bar.adjustSize()
    self._hover_bar.hide()

def enterEvent(self, event):
    super().enterEvent(event)
    bar = self._hover_bar
    bar.move(self.width() - bar.width() - 8, -bar.height() + 4)
    bar.show()
    bar.raise_()

def leaveEvent(self, event):
    super().leaveEvent(event)
    self._hover_bar.hide()
```

#### Remove the old always-visible actions

Delete the `self.actions` widget construction (lines ~672–691) and the `head.addWidget(self.actions)` call. The header row becomes: role label + timestamp + stretch. Nothing else.

#### _HoverAction button style

```python
class _HoverAction(QPushButton):
    def __init__(self, label):
        super().__init__(label)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(22)
        self.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {_s.FG_DIM};
                border: none;
                border-radius: 3px;
                padding: 0 8px;
                font-size: 10px;
                font-family: Consolas;
            }}
            QPushButton:hover {{
                background: {_s.BG_BUTTON_HOVER};
                color: {_s.FG_TEXT};
            }}
        """)
```

#### Hover bar container style

```python
self._hover_bar.setStyleSheet(f"""
    QFrame#msg_hover_bar {{
        background: {_s.BG_PANEL};
        border: 1px solid {_s.BORDER_LIGHT};
        border-radius: 5px;
    }}
""")
```

#### Copy action

Add a `_copy_to_clipboard` method that copies the message content:

```python
def _copy_to_clipboard(self):
    from PySide6.QtWidgets import QApplication
    clipboard = QApplication.clipboard()
    if clipboard is not None:
        clipboard.setText(self._content)
```

---

### TASK C: Inline thought badge replacing collapsible blocks

**File**: `ui/components/message_widget.py`  
**Priority**: MEDIUM — reduces vertical noise  
**Depends on**: Nothing (can be done with or without Task B)

#### Current behavior

Think blocks render as `_ThinkBlock` widgets inside `_think_container`, positioned between the header row and the body text. Each block has a clickable header that expands/collapses the content. This takes a full line minimum.

#### Target behavior

The thought indicator moves into the header row as a tiny inline badge:

```
ASSISTANT  23:01  ● 3.5s
```

The `●` is a 4px circle in `accent_primary` at 50% opacity. The `3.5s` is the total think time in `fg_dim`, 9px monospace.

Clicking the badge opens the think content **below the response body** (not between header and body), or alternatively expands it in the companion pane's generation trace.

#### Implementation

In the `MessageWidget.__init__` header layout, after the timestamp label:

```python
self._think_badge = QWidget()
self._think_badge.setFixedHeight(self._HEADER_H)
self._think_badge.hide()
badge_layout = QHBoxLayout(self._think_badge)
badge_layout.setContentsMargins(4, 0, 0, 0)
badge_layout.setSpacing(3)

self._think_dot = QLabel("●")
self._think_dot.setStyleSheet(
    f"color: {_s.ACCENT_PRIMARY}; font-size: 6px; background: transparent; opacity: 0.5;"
)
self._think_dot.setFixedWidth(8)
badge_layout.addWidget(self._think_dot)

self._think_duration = QLabel("")
self._think_duration.setStyleSheet(
    f"color: {_s.FG_DIM}; font-size: 9px; font-family: Consolas; background: transparent;"
)
badge_layout.addWidget(self._think_duration)

self._think_badge.setCursor(Qt.PointingHandCursor)
self._think_badge.mousePressEvent = lambda e: self._toggle_think_expand()
head.addWidget(self._think_badge)
```

When `apply_assistant_display` is called with thinking text:

```python
def _update_think_badge(self, thinking_text: str, thinking_done: bool):
    if not thinking_text:
        self._think_badge.hide()
        return
    
    # Calculate approximate duration from text length or stored timing
    duration = getattr(self, '_think_duration_sec', None)
    if duration is not None:
        label = f"{duration:.1f}s"
    else:
        # Estimate from character count (rough: ~50 chars/sec for display)
        label = "thinking" if not thinking_done else f"~{len(thinking_text) // 50}s"
    
    self._think_duration.setText(label)
    self._think_badge.show()
```

The `_think_container` remains in the widget tree but is **hidden by default** and only shown when the badge is clicked. Move it to **after** the `text_view` in the layout order (currently it's between header and text_view).

---

### TASK D: Config panel — borderless elevation redesign

**File**: `ui/panels/model_config.py` (78 lines → ~250 lines)  
**Priority**: MEDIUM — the proxy pattern must be replaced for this to work  
**Depends on**: Task E (chat.py extraction) OR can be done as new standalone code that replaces the proxy

This is the only task that requires real widget construction code to move out of `chat.py`. The current `ModelConfigPanel.bind_controller()` reparents widgets built elsewhere. The v2 config panel builds its own widgets.

#### Model section — status card

```python
def _build_model_section(self):
    card = QFrame()
    card.setStyleSheet(f"background: {_s.BG_SURFACE_1}; border: none; border-radius: 6px; padding: 10px 12px;")
    
    layout = QVBoxLayout(card)
    layout.setSpacing(2)
    
    name_row = QHBoxLayout()
    self._model_name = QLabel("No model")
    self._model_name.setStyleSheet(f"color: {_s.FG_TEXT}; font-size: 12px; font-weight: 500;")
    name_row.addWidget(self._model_name)
    
    self._model_status = QLabel("")  # "● Loaded" or empty
    self._model_status.setStyleSheet(f"color: {_s.FG_ACCENT}; font-size: 9px; font-family: Consolas;")
    name_row.addWidget(self._model_status)
    name_row.addStretch()
    layout.addLayout(name_row)
    
    self._model_meta = QLabel("")  # "GGUF (API) · 262k ctx"
    self._model_meta.setStyleSheet(f"color: {_s.FG_DIM}; font-size: 10px; font-family: Consolas;")
    layout.addWidget(self._model_meta)
    
    return card
```

#### Endpoint — collapsed single line

```python
def _build_endpoint_row(self):
    row = QFrame()
    row.setStyleSheet(f"background: {_s.BG_SURFACE_1}; border: none; border-radius: 6px; padding: 7px 10px;")
    
    h = QHBoxLayout(row)
    h.setContentsMargins(0, 0, 0, 0)
    
    self._endpoint_label = QLabel("127.0.0.1:59129/v1")
    self._endpoint_label.setStyleSheet(f"color: {_s.FG_SECONDARY}; font-size: 10px; font-family: Consolas;")
    h.addWidget(self._endpoint_label)
    h.addStretch()
    
    edit_btn = QPushButton("Edit")
    edit_btn.setStyleSheet(f"color: {_s.ACCENT_PRIMARY}; font-size: 9px; border: none; background: transparent;")
    edit_btn.setCursor(Qt.PointingHandCursor)
    edit_btn.clicked.connect(self._toggle_endpoint_edit)
    h.addWidget(edit_btn)
    
    return row
```

When "Edit" is clicked, the row expands to show the full input fields (engine dropdown, API base, API key, model combo, fetch button). When collapsed again, it returns to the single-line display.

#### Slider groups — three semantic sections

Replace the flat list of 8 sliders with three groups, each preceded by a 9px section header:

```
SAMPLING
  Temperature  ─────●──── 1.05
  Top-P        ──────────● 0.89
  Top-K        ──●──────── 20
  Min-P        ────────●── 0.83

PENALTIES
  Presence     ──────●──── 1.50
  Repetition   ─────●───── 1.00

LIMITS
  Max tokens   ──────────● 262k
  Context      ──────────● 262k
```

Section headers: 9px, `FG_DIM`, Consolas, letter-spacing 1px, weight 500.  
Slider labels: 10px, `FG_DIM`.  
Slider values: 10px, `FG_SECONDARY`, Consolas. For values >= 10000, abbreviate with k suffix.

Slider track: `BG_SURFACE_1` background, `ACCENT_PRIMARY` fill, height 3px.  
Slider thumb: 10px circle, `FG_TEXT` fill, 2px `BG_MAIN` border.

#### Save row — text links

```
Unsaved              Reset  Save
```

- "Unsaved": 9px, `FG_WARN`, visible only when dirty
- "Reset": 10px, `FG_DIM`, clickable text
- "Save": 10px, `ACCENT_PRIMARY`, clickable text
- Separated by a 1px `border_subtle` top line

#### New theme tokens required

Add to `Theme` dataclass in `core/themes.py`:

```python
bg_surface_1: str    # #131620 for Midnight
bg_surface_2: str    # #181c27 for Midnight
```

Add to `core/style.py` `refresh_styles()`:

```python
g["BG_SURFACE_1"] = t.bg_surface_1
g["BG_SURFACE_2"] = t.bg_surface_2
```

---

### TASK E: Complete chat.py extraction

**File**: `ui/pages/chat.py` (2,625 lines → target ~800 lines)  
**Priority**: HIGH but highest effort  
**Depends on**: Nothing — but Task D benefits from it

The proxy panels (§1.2) need to become standalone. This means moving widget *construction* out of `PageChat.__init__` and into each panel class.

#### Extraction map

| Lines in chat.py | Destination | What moves |
|-----------------|-------------|-----------|
| 195–297 (model loader UI) | `panels/model_config.py` | Engine combo, API base input, API key input, model combo, fetch button, load/unload button, `ModelListLoader` thread |
| 298–342 (AI configuration UI) | `panels/model_config.py` | System prompt textarea, all sliders, save/reset buttons |
| 343–412 (panel group tabs) | DELETE | Tab buttons and ops_stack QStackedWidget — companion pane replaces this |
| 413–443 (archive tab UI) | `panels/archive_browser.py` | Archive list widget, search input, new/clear/delete buttons |
| 460–497 (audit tab UI) | `panels/audit_log.py` | Audit list widget, clear button |
| 498–517 (trace display) | `panels/generation_trace.py` | Trace QTextEdit |
| 549–575 (action review UI) | `panels/action_review.py` | Action review frame with JSON display, approve/reject buttons |
| 1432–1709 (model/config logic) | `panels/model_config.py` | `_emit_model_payload`, `_apply_backend_visibility`, `_apply_model_config`, `_pick_free_port`, `_coerce_local_base`, `_ensure_local_api_base`, `_start_local_server`, `_stop_local_server`, all server management, `_apply_config_to_controls`, `_fetch_models`, `_on_models_loaded`, `_on_models_error` |
| 1784–1935 (model list/config) | `panels/model_config.py` | Continued config management |
| 2229–2340 (archive logic) | `panels/archive_browser.py` | `_start_new_session`, `_prompt_clear_session`, `_clear_current_session`, `_delete_selected_archive`, `_save_chat_archive`, `_load_chat_archive`, `_refresh_archive_list` |
| 1009–1027 (audit logic) | `panels/audit_log.py` | `_refresh_audit_list`, `_clear_audit_log` |

#### What stays in chat.py (~800 lines)

- `PageChat.__init__` — creates `ConversationSurface` and wires signals, but no longer builds panel widgets
- Send flow: `send()`, `_send_message()`, token handling, generation lifecycle
- Session management: `_create_session`, `_set_current_session`, `_snapshot_session`
- Tool execution flow: `_process_last_response_commands`, `_execute_world_action`, `_maybe_handle_action_proposal`
- Agent command handling
- Message mutation: `_undo_last_mutation`, `_delete_from_index`, `_edit_from_index`, `_regen_last_assistant`

#### Acceptance criteria

After extraction:
- `chat.py` is under 900 lines
- Each panel can be instantiated with `(state, ui_bridge)` constructor — no `bind_controller()` reparenting
- Panels communicate with chat via signals, not by reaching into `controller._private_method()`
- `_control_tab`, `_settings_tab`, `_archive_tab`, `_audit_tab`, `_trace_group` attributes no longer exist on `PageChat`

---

### TASK F: Remove MonoGroupBox("CHAT") wrapper from ConversationSurface

**File**: `ui/conversation_surface.py` (line 42)  
**Priority**: LOW — quick win  
**Depends on**: Nothing

Currently:

```python
self.chat_group = MonoGroupBox("CHAT")
chat_layout = QVBoxLayout()
...
self.chat_group.add_layout(chat_layout)
root.addWidget(self.chat_group)
```

Replace with:

```python
chat_layout = QVBoxLayout()
chat_layout.setContentsMargins(0, 0, 0, 0)
chat_layout.setSpacing(0)
...
root.addLayout(chat_layout)
```

No border, no title. The conversation surface is the primary content area — it doesn't need a box around it. The `CHAT` label in the top-left corner of the conversation area (visible in the screenshots) disappears. The message list fills the space directly.

Additionally: the existing `MonoGroupBox` import can be removed from `conversation_surface.py`.

---

### TASK G: Delete hub.py

**File**: `ui/pages/hub.py` (22,411 bytes)  
**Priority**: LOW  
**Depends on**: Verify all hub functionality is in omni-bar

The omni-bar already handles `>model`, `>theme`, `>profile`. Verify these cover the hub's feature set:

| Hub feature | Omni-bar coverage | Status |
|------------|-------------------|--------|
| Operator profile cards | `>profile {name}` switches operator | Check `_query` handles listing |
| Operator creation | `>profile new` or direct dialog | May need a small modal for name input |
| Theme switcher | `>theme {name}` switches theme | Covered |
| Module launch grid | No-prefix search shows addon results | Covered via AddonRegistry |
| Lineage dialog | Not in omni-bar | Consider: audit log panel or omni-bar `>lineage {name}` |

If lineage dialog is important, add `>lineage` as an omni-bar command that opens a popup. Otherwise, the feature can live in the audit panel.

Once verified: delete `ui/pages/hub.py` and remove any imports/references in `main_window.py` and `bootstrap.py`.

---

## 3. TASK EXECUTION ORDER

```
Phase 1: Visual pass (independent tasks, can be parallelized)
  A. Icon rail QPainter icons
  B. Message hover-reveal actions
  C. Inline thought badges
  F. Remove MonoGroupBox("CHAT") wrapper

Phase 2: Structural extraction
  E. Extract panel code from chat.py (largest task)

Phase 3: Design implementation (depends on Phase 2 for full effect)
  D. Config panel borderless elevation redesign

Phase 4: Cleanup
  G. Delete hub.py
```

Phase 1 tasks are all independent and touch different files — they can be done in any order or simultaneously. Phase 2 is the bottleneck. Phase 3 can be started during Phase 2 if the config panel is built as new standalone code rather than refactoring the proxy.

---

## 4. WHAT IS EXPLICITLY NOT IN SCOPE

These items from the original spec are deferred — they either shipped already or aren't worth the complexity right now:

- **Companion pane state machine** — Already shipped and working. No changes needed.
- **Omni-bar behavior** — Already shipped. Minor polish (result styling) is cosmetic and low priority.
- **Vitals footer** — Already shipped. Hover-expand is nice-to-have but not critical.
- **Spring physics / motion design** — Explicitly rejected. The two-animation budget (companion slide + gradient breathe) is correct and already implemented.
- **Docking workspace** — Rejected in original spec. Still rejected. Conversation-centered layout shipped.
- **Tool call approve/reject inline in chat** — `ToolCallBubble` and `ToolResultBubble` already render. The `check_policy()` → `REQUIRE_APPROVAL` flow exists in `core/world_actions.py`. Wiring them for inline approve/reject inside the chat stream is a separate task that depends on confirming the skill runtime actually produces these flows end-to-end. Defer until confirmed working.

---

## 5. REFERENCE: V2 MOCKUP TARGET

The full-application mockup rendered earlier in this conversation is the target for all visual tasks. Key properties:

- Dark glass canvas, background `#0e1117`
- No `MonoGroupBox` borders anywhere
- Icon rail: 42px wide, QPainter stroke icons, divider line, badge dots
- Message actions: hidden until hover, floating above message top-right
- Think blocks: inline `● 3.5s` badge in header row
- Config panel: model status card, endpoint collapsed to one line, three slider groups, text-link save/reset
- Vitals footer: 26px tall, monospace metrics, model name + status right-aligned
- Gradient accent line at top (existing, unchanged)
- Omni-bar: centered pill with search, status badge, Ctrl+K hint (existing, unchanged)

---

*End of delta spec. Execute Phase 1 first — the visual tasks have the highest UX impact per line of code changed.*
