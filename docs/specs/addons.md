# Addons (Foundational Spec)

This document defines a minimal, open, and extensible foundation for Monolith addons. It is intentionally small, focused on the core integration points, and designed to evolve without breaking existing addons.

## Goals
- Simple in-process addons with clear hooks.
- Optional out-of-process addons via a small IPC protocol.
- Stable packaging format for distribution.
- Room for future UI extension points and DI.

## Current State (Baseline)
- Addons are registered in-process via `ui/addons/builtin.py`.
- Terminal UI is `ui/pages/chat.py`.
- LLM generation runs through `engine/llm.py`.
- Engine calls are bridged by `engine/bridge.py`.

## Message Interceptor API (Implemented)
Allows addons to modify messages just before generation.

### API
File: `core/message_interceptors.py`

```python
from core.message_interceptors import register_interceptor

def my_interceptor(messages, config):
    # return a new list, or None to keep unchanged
    return messages

register_interceptor(my_interceptor)
```

### Contract
- Signature: `intercept(messages: list[dict], config: dict) -> list[dict] | None`
- Ordering: registration order.
- Safety: interceptor exceptions are ignored (generation continues).

### Hook Point
Applied in `engine/llm.py` immediately before inference.

## Hot-Reload (Planned)
Goal: rapid addon development without full restart.

### Minimal v1 (Implemented)
- Watches `addons/manifest.json` and addon entry files.
- Auto reloads registry when changes are detected.
- Manual "Reload Addons" button and "Hot Reload" toggle in Addons page.
- Module refresh is opt-in via lifecycle hooks (see below).

## IPC for External Addons (Planned)
Goal: allow addons to run in external processes (local or remote).

### Minimal Protocol (Implemented)
- Transport: JSON-RPC 2.0 over stdout/stdin (newline-delimited JSON).
- External addon can:
  - `monolith.state` -> query state snapshot
  - `monolith.emit` -> emit log/trace signals
  - `monolith.register_verbs` -> publish verbs (recorded in module UI)
  - `monolith.engine_event` -> emitted as notification to addon

### Events
- `token` stream
- `trace` log messages
- `status` changes

## Addon Marketplace Format (Planned)
Goal: portable bundle format for distribution.

### Package Layout
```
addon.zip
  manifest.json
  code/
  assets/
```

### `manifest.json` (Draft)
```json
{
  "id": "my-addon",
  "version": "0.1.0",
  "title": "My Addon",
  "entrypoint": "code/main.py",
  "capabilities": ["intercept_messages"]
}
```

## UI Extension Points (Planned)
Goal: inject UI controls into existing pages.

### v1
- Register named slots in `ui/pages/chat.py` and `ui/pages/hub.py`
- Addon exposes a `render_controls()` hook to populate slots

## Dependency Injection (Planned)
Goal: replace engine implementations.

### v1
- Simple registry in `core/`
- Allow replacing the `EnginePort` implementation via config

## Versioning
- This spec starts at `v0.1`.
- Breaking changes should be accompanied by migration notes.
## Lifecycle Hooks (Implemented)
Widgets can opt into reload and state transfer:

```python
def on_save_state(self) -> dict: ...
def on_load_state(self, state: dict) -> None: ...
def on_reload(self) -> None: ...
sig_reload_requested = Signal()
```

Engine events are forwarded to widgets that implement:

```python
def on_engine_event(self, engine_key: str, event: str, payload: object) -> None: ...
```
