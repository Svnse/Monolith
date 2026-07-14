# Monolith UI Contract

**Status:** ACTIVE — first written 2026-07-08, the day the UI was consolidated to
chat-only. This is the document every UI change is measured against. It exists
because the UI was restyled several times (ATMOS toolkit, ui_v2 cockpit — both
since removed) without a written law, so each pass invented its own chrome and
the next pass had nothing to honor.

**The rule about this rule:** a new surface, panel, or restyle must cite this
contract in its spec. If a change genuinely needs to break a law here, amend
this document *first* (design-gate discipline), then build.

---

## 1. Region census

The window is exactly these regions, and no change may add a region without
amending this section:

| Region | Widget / objectName | Role |
|---|---|---|
| Gradient line | `GradientLine` | top accent strip, focus glow |
| Omni bar | `omni_bar_frame` | THE command entry: search, `>` commands, panels |
| Conversation stack | `conversation_stack` | **chat, and only chat** — the sole central workspace |
| Icon rail | `icon_rail` | launcher only: 4 pinned panels + addon icons + Overseer |
| Companion pane | `companion_pane` | transient right-side panels (resizable 360–1200px) |
| Vitals footer | `vitals_footer` | resource readout |

Standing decisions this census encodes (E, 2026-07-08):

- **Chat-only center.** The conversation stack hosts terminal (chat) widgets
  only. No workspace switcher, no central tabs, no module that takes over the
  center. (The soundtrap/monoline/mononote central workspaces and the
  ui_v2 cockpit were removed to establish this; see
  `backup/full-snapshot-2026-07-08` for the pre-consolidation state.)
- **Rail pins are capped.** CONFIG, GENERATING, ARCHIVE, STATS (+ OVERSEER at
  bottom). Everything else opens from the omni bar (plain search or a `>`
  command). Do not add a pin without removing one and amending this table.
- **Boot opens on chat.** Nothing may launch focused at startup except the
  terminal. Background modules (e.g. CONNECT under `MONOLITH_AGENT_AUTOSTART`)
  mount with `launch_module(id, focus=False)`.

## 2. Chrome budget — one frame per region

**The law:** each region draws exactly one frame and one header. Content inside
a region gets ZERO additional frames, title bars, or boxed borders.

For the companion pane specifically:

- The pane owns the frame (`QFrame#companion_pane`) and the single header
  (`_title`) — and `transition_to` **already sets the header to the panel's
  name** ("EXPEDITION", "SELF MAINT", the addon's real title, …).
- Therefore a panel widget must NOT:
  - draw its own title label repeating its name (that is the
    "panel within a panel" effect);
  - wrap itself in a `QFrame`/`MonoGroupBox` covering its whole body;
  - re-implement a header row for identity purposes.
- A panel MAY have a top row for *live status* (badges, chips, counts) — status
  is content, identity is chrome. If a panel needs internal grouping, use a
  **dim section label + hairline divider**, not a box:

  ```python
  lbl = QLabel("ROUTING")           # section label: FG_DIM, 9px, mono, bold
  line = QFrame(); line.setFixedHeight(1)   # divider: BORDER_SUBTLE
  ```

- `MonoGroupBox` is reserved for full-page module surfaces that mount in the
  companion ADDON slot AND genuinely need collapsible groups (CONNECT is the
  canonical case). Even there, prefer sections; never nest a group box in a
  group box.
- **Inset surfaces (added 2026-07-08, same day):** inner feeds, lists, inputs
  and tab panes inside a panel use the shared `panelInset` QSS rule
  (`core/theme_engine.py`): `bg_input` fill, `border_subtle` hairline, rounded
  corners (6px surfaces / 4px inputs), accent focus ring. Opt in with
  `widget.setProperty("panelInset", True)`; keep only font styling inline.
  Never hand-style an inner surface with `BORDER_DARK` + square corners — the
  dark sharp box on the pane surface is the second face of "panel within a
  panel". Content widgets inside an inset tab pane are `background:transparent`
  (no box-in-box fill).

**Regions own color; content is transparent (root-cause law, 2026-07-08).**
The global stylesheet once set `QWidget {{ background: bg_main }}`, so every
plain container repainted the darkest color over its region's surface — THE
root cause of the nested-panel look, and why per-panel fixes kept "not
taking" (Config/Audio/Stats stayed dark: their bodies repainted bg_main).
Now the global QWidget default is `background: transparent`; only region
containers (`#MainFrame`, `#companion_pane`, `#icon_rail`, sidebar/footer)
and floating windows (`QDialog`, `QMenu`, `QToolTip`, overseer) paint fills.
A new panel needs NO background styling at all — transparency is correct by
default. Related: scroll areas and their viewports/bodies are transparent
(they are windows onto surfaces, never surfaces themselves).

## 3. Tokens — no hardcoded color, ever

- Every color, font, and spacing value in `ui/**` comes from `core.style`
  constants (`_s.*`) or live theme tokens via `core/theme_engine.py` QSS.
- **No hex literals in widget code.** If a semantic color is missing (success /
  warn / error verdict colors, grounded-green, …), add a token to `core.style`
  and the theme schema — do not inline `#88cc88`.
- Prefer objectName + centralized QSS in `theme_engine.py` over per-widget
  `setStyleSheet` f-strings: inline f-strings freeze token values at
  construction and go stale on theme change (the companion header needed an
  explicit `_apply_chrome_theme` re-subscribe workaround for exactly this).
  If a per-widget stylesheet is unavoidable, it must re-apply on
  `sig_theme_changed`.

## 4. Entry points and dispatch

- The omni bar is the universal opener. Panels dispatch as `panel:<STATE>`
  where `<STATE>` is a `CompanionState` member; `main_window._handle_omni_action`
  routes generically — adding a panel requires NO new dispatch code, only:
  1. a `CompanionState` member + entry in `CompanionPane._panels`,
  2. an omni catalog `_Result` (+ intent keywords, + optional `>command`).
- The rail is a launcher, not a state display for unpinned panels. Rail
  additions are governed by the cap in §1.
- Closing: panels toggle from the rail if pinned; otherwise collapse from the
  pane. (Open question logged 2026-07-08: a `>close` command or
  repeat-command-toggles — decide before adding any more unpinned panels.)

## 5. Compliance worklist (audit of 2026-07-08)

Violations found the day this contract was written — this is the flattening
worklist, in priority order. Each fix should delete chrome, not restyle it.

**Duplicate identity headers (§2)** — ✅ FIXED 2026-07-08 (same day): title
labels deleted from `expedition.py`, `self_maint.py`, `audit_log.py`,
`reasoning_tree.py`, `workshop_library.py`; `workshop.py` header reduced to
pure run-count status. `action_review.py`'s "ACTION PROPOSED" was judged
genuine status (an action IS proposed) and kept.

**Hardcoded hex (§3)** — ✅ FIXED 2026-07-08: added `fg_ok` theme token
(defaulted `#88cc88`, so presets and saved custom themes keep working) exposed
as `_s.FG_OK`; amber/red mapped to the existing `FG_WARN`/`FG_ERROR`;
audit-log "approved" blue mapped to `ACCENT_PRIMARY`. `ui/panels/**` is now
hex-free (enforce with: `git grep -nE "#[0-9a-fA-F]{6}" -- ui/panels/`).

**Group-box chrome inside the pane (§2)** — ✅ FIXED 2026-07-08 at the
component level: `MonoGroupBox` (`ui/components/atoms.py`) no longer paints a
bordered rectangle with a title cutout; it renders as a flat section — dim
Consolas label + `border_subtle` hairline — with the same API, so every host
(connections, databank, sd, audiogen, manager, theme) flattened at once with
zero call-site changes. Also converted the same day: `GenerationTracePanel`
(was on the dark `#trace_log` ID rule — NOTE: ID selectors out-rank
`[panelInset]`, so a widget must not carry a boxed objectName AND the
property), the `#archive_list`/`#archive_search` central rules, and the audio
mini-waveform frame. All inset fills are `bg_surface_1` — identical to the
pane — per the §2 inset-surface rule.

**Frozen inline styles (§3):** pervasive `setStyleSheet(f"...{_s.X}...")` in
panels — migrate opportunistically to objectName + `theme_engine.py` QSS
whenever a panel is edited for the items above.

## 6. History (why these laws)

- Multiple unversioned restyles: ATMOS companion toolkit (2026-06-29, reverted
  2026-07-08), ui_v2 cockpit + tokens (2026-07-01, shelved 2026-07-08 —
  its design language lives in `docs/specs/MONOLITH_UI_V2_SPEC.md` as a
  reference aesthetic, NOT as binding law).
- Per-feature specs in `docs/superpowers/specs/` each styled their own box;
  nobody owned composition — producing nested pane→panel→group-box chrome.
  This contract owns composition now.
