# Changelog

## v0.2.2a — 2025-02-16

### Theme Engine Overhaul

- **Dynamic theme system** — 4 built-in themes: Midnight (blue), Obsidian (dark blue), Monolithic (gold), Slate (ChatGPT green). Default: Midnight.
- **Slate theme** — New ChatGPT-inspired dark mode with `#10a37f` green accent, `#343541` background.
- **Theme persistence** — Selected theme saved to `%APPDATA%/Monolith/config/theme.json` and restored on launch.
- **Live theme switching** — All UI components update instantly when theme changes via Appearance dropdown in Hub.

### Style Migration

- **Eliminated stale color imports** — Replaced all `from core.style import X` with dynamic `import core.style as s` pattern across 18 files. Prevents Python binding bugs where colors freeze at import time.
- **Widget refresh system** — All atom widgets (`MonoButton`, `MonoGroupBox`, `MonoTriangleButton`, `MonoSlider`, `CollapsibleSection`) now have `refresh_style()` methods for live theme updates.
- **Bootstrap theme handler** — `_on_theme_changed` now walks all mounted pages and open modules, calling `apply_theme_refresh()` on each.
- **Sidebar + window controls** — Sidebar buttons and `SplitControlBlock` (min/max/close) refresh on theme change.

### New Files

- `core/themes.py` — Theme registry with 4 theme dataclasses and `apply_theme()` function.
- `core/theme_config.py` — JSON-based theme persistence (`load_theme_config` / `save_theme_config`).

### Bug Fixes

- Fixed group box title clipping ("MODUL..." / "SYSTE...") — `adjustSize()` now runs after stylesheet is applied.
- Fixed `BG_INPUT` NameError crash in chat send button — bare reference not caught by f-string migration.
- Fixed `_s._s.FG_ERROR` double-prefix in chat trace HTML.
- Fixed OverseerDB shutdown crash — all DB methods now gracefully return when connection is closed instead of raising `RuntimeError`.
- Fixed Modules page retaining wrong theme colors after theme switch.

---

## v0.2a — 2025-02-09

### New Systems

- **Operator System** — Save and restore full workspace snapshots (all open modules, terminal configs, chat history). Load via LOAD button or double-click. Legacy single-config format backward compatible.
- **Overseer** — Real-time trace viewer and debug dashboard. Tracks kernel tasks, engine events, and LLM generation with searchable log. VizTracer integration for profiling.
- **UI Bridge** — Dedicated signal bus (`ui/bridge.py`) decoupling UI events from kernel internals. Handles operator apply, terminal headers, overseer toggling.
- **Operator Manager** (`core/operators.py`) — JSON-based operator storage with save/load/delete/list. Supports both legacy (single config) and new (multi-module snapshot) formats.
- **Overseer Database** (`core/overseer_db.py`) — Persistent trace storage for session history.

### New Files

- `ui/pages/hub.py` — Operator hub page with card grid, LOAD/NEW/DELETE actions, double-click to open
- `ui/components/message_widget.py` — Dedicated message bubble widget with per-message edit/delete/regenerate actions, proper word-wrap sizing
- `ui/bridge.py` — UI signal bridge
- `ui/overseer.py` — Overseer trace window
- `core/operators.py` — Operator persistence layer
- `core/overseer_db.py` — Trace database
- `core/paths.py` — Centralized config/data path resolution
- `combiner.py` — Source bundler utility
- `requirements.txt` — Dependency manifest

### Chat Overhaul

- Replaced monolithic QTextEdit chat with individual `MessageWidget` bubbles in a QListWidget
- Per-message actions on hover: edit, delete, regenerate
- Thinking mode support (Off / Standard / Extended) via collapsible OPTIONS panel in CONTROL tab
- File attachment support via OPTIONS panel
- Race condition prevention (`_pending_mutation` system) — safe edit/delete/regen during active generation
- Ghost bubble cleanup (`_cleanup_empty_assistant_if_needed`)
- `sig_debug` signal for structured trace output to Overseer
- Streaming state cleanup on interruption (UNLOADING handling)

### UI Changes

- **Top bar**: MONOLITH label is now static muted gold, left-aligned (removed fire animation)
- **Status**: Moved from top bar to small label in bottom-right corner
- **Chat title/time**: Only visible when viewing terminal modules (hidden for SD, audio, runtime)
- **Live clock**: 60-second timer keeps chat timestamp current
- **Gradient line**: Kept at top of window (unchanged)
- **Input toolbar**: Removed + and THINK buttons from chat input area
- **OPTIONS panel**: Collapsible panel in CONTROL tab with attach file + think mode toggles

### Addon System

- `AddonHost.launch_module()` now stamps `widget._addon_id` on every module instance for type identification
- `hub_factory` added to builtin registry — wires operator hub with snapshot/load/save
- Terminal factory enhanced with named exception-wrapped handlers, short engine ID traces (`[LLM:xxxx]`), descriptive generation messages
- Operator trace messages routed to Overseer via `ctx.guard.sig_trace`

### Architecture

- Removed direct `LLMEngine` instantiation from bootstrap — engine creation delegated to addon system
- Added `UIBridge` to bootstrap wiring
- Added `OverseerWindow` to bootstrap with guard connection
- `aboutToQuit` now cleans up viztracer and overseer database

### Bug Fixes

- Fixed operator loading doing nothing (missing view switch + module creation)
- Fixed assistant bubble showing large empty gap during streaming (sizeHint override for word-wrap)
- Fixed `btn_thinking` AttributeError when applying operators (removed stale reference)
- Fixed race conditions during edit/delete/regenerate while LLM is generating
- Fixed ghost empty assistant bubbles appearing after interrupted generation
