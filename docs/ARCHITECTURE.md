# Architecture overview

This is a public orientation to Monolith `v1.0.0`, not a stable API contract
for every internal module. The v1 source tree contains implemented,
experimental, partial, and stub paths side by side.

## Live startup path

The supported desktop entry point is:

```text
main.py
  -> bootstrap.main()
  -> QApplication and theme
  -> AppState and persistent stores
  -> MonoGuard / MonoDock / MonoBridge
  -> MonolithUI and Overseer
  -> addon registry and host
  -> configuration watcher, interceptors, traces, and optional workers
  -> Qt event loop
```

`start.bat` anchors itself to the repository directory and invokes `main.py`
with the repository's Python 3.11 virtual environment. It does not enable
experimental flags.

## Main turn path

The useful center of the application can be reduced to:

```text
User input
  -> PySide6 chat surface
  -> configured model adapter
  -> streamed model output
  -> turn/prompt interceptors and kernel boundaries
  -> optional tool, skill, or workflow execution
  -> rendered response
  -> archives, traces, and selected memory/state stores
```

Tools can create a follow-up model turn after returning results. This makes
tool permissions and output provenance part of the behavioral boundary, not
just UI decoration.

## Component map

| Area | Primary location | Responsibility | Maturity |
|---|---|---|---|
| Desktop shell and pages | `ui/` | Chat, configuration, files, connections, companion panels, media modules | Core, with some placeholder panels |
| Model runtime | `engine/llm.py`, model adapters | OpenAI-compatible HTTP and local GGUF generation | Core API path; optional local path |
| Kernel boundary | `monokernel/` | Engine envelopes, lifecycle/governance bridge, turn pipeline | Core |
| Application state and policy | `core/` | Configuration, paths, prompts, tools, traces, memory, policies, search | Mixed core and experimental |
| Addon system | `addons/`, `ui/addons/` | Built-in and manifest-driven UI/runtime surfaces | Available; manifests vary by addon |
| Skills | `skills/` plus `core/skill_runtime.py` | Model-discoverable tool descriptions and executors | Broad; not every directory is polished |
| Image engine | `engine/vision.py`, `engine/_workers/vision_worker.py` | Subprocess-isolated diffusion inference | Optional experimental |
| Audio module | `ui/modules/audiogen.py` | AudioCraft/MusicGen generation and export | Optional experimental; thread/UI path |
| CONNECT | `engine/agent_server.py`, `ui/pages/connections.py` | Local HTTP/MCP/peer bridge | Experimental network surface |
| Matrix | `engine/matrix_bridge.py` | Optional Matrix transport | Experimental network surface |
| Workshop/Monoline | `engine/monoline_bridge.py`, Workshop UI/tools | Author and run external Monoline workflows | Experimental external integration |

The old shorthand “Kernel + Engines + Addons, with every engine in a separate
process” is too absolute. Vision has a process-facing worker, audio uses a Qt
thread/module path, the HTTP model path is adapter-driven, and many cognition
and persistence systems run in process.

## Models and generation

The model configuration supports three user-visible modes:

- OpenAI-compatible API models;
- local GGUF through a local API/server path; and
- local GGUF through llama-cpp-python.

The UI persists selected endpoint/model/path fields, builds a model payload,
and submits generation through the engine/bridge path. Model weights and model
servers are outside the repository boundary.

Image generation uses a separate vision process so heavy Diffusers/Torch
imports and inference are isolated from the main UI path. Audio generation is
mounted through the audio module and imports AudioCraft/Torch lazily when used.
Neither stack belongs in the base dependency profile.

## Tools and skills

`core/skill_registry.py` discovers tool descriptions, while
`core/skill_runtime.py` validates and dispatches implemented executors. The
runtime divides tools by governance level:

- read-oriented leaf operations such as file inspection, search, calculation,
  traces, and model calls;
- worker operations that can write files, create notes, generate media, or
  spawn bounded subagents; and
- principal-only operations such as commands, tests, user prompts, Soundtrap,
  and Workshop authoring/execution.

Those levels are an application governance mechanism, not an operating-system
sandbox. A permitted file or command tool still acts with the permissions of
the Monolith process. Human review remains necessary for consequential actions.

## Persistence

`core/paths.py` resolves a runtime root and creates configuration, chat, log,
note, artifact, and addon directories. Windows defaults to
`%APPDATA%\Monolith`; `MONOLITH_ROOT` can isolate another profile.

Persistent data includes:

- YAML/JSON model and feature configuration;
- conversation archives;
- event/fault/turn traces;
- SQLite Acatalepsy/ACU and turn-state stores;
- continuity, identity, planning, curiosity, and self-maintenance records when
  their corresponding paths are used;
- Markdown notes; and
- generated artifacts.

Stored data is not equivalent to data injected into every model turn. Prompt
interceptors, retrieval gates, selected tools, flags, and active UI/runtime
paths determine what reaches the next model request.

## Experimental cognition surfaces

The repository contains real implementations for MonoSearch, ACU/Acatalepsy,
Bearing, Observer, Monothink, curiosity, identity emergence, planning,
prediction, and self-maintenance. These systems have different activation and
closure levels.

For the public release:

- their flags and schemas are unstable;
- their presence does not imply autonomous or verified reasoning;
- memory entries may be stored without being recalled into a later turn;
- proposal/detection paths are not the same as applied change paths; and
- `start.bat` does not opt into autonomous mutation or maintenance.

Open loops are allowed in the source release; they must not be described as
completed guarantees.

## Partial and external surfaces

- **MonoNote:** Markdown store, provenance/index, read/list tools, and search
  integration exist; a dedicated editor/workspace is not mounted.
- **Soundtrap:** headless loop/project/mix backend pieces exist; a finished DAW
  workspace does not.
- **Monoline:** the bridge can load a separate checkout, but Monoline is not
  bundled and the process-isolated worker path is incomplete.
- **UI v2:** a feature-flagged import hook exists, but the `ui_v2` package is
  absent from this release.
- **Producer adapters:** some newer CONNECT/local adapters identify themselves
  as migration stubs; they are not the main live chat transport.

## Network and trust boundaries

| Boundary | What crosses it | Release posture |
|---|---|---|
| OpenAI-compatible endpoint | Prompts, context, tool-related content, responses | User-configured external trust boundary |
| Tavily/web search | Search queries and returned web data | Optional network tool |
| CONNECT | Chat, state, events, tool/runtime information depending on route | Loopback-only recommended; remote exposure unsupported |
| Matrix | Messages, room metadata, credentials/tokens | Optional external service |
| External peers/webhooks | Requests, responses, event payloads | Explicitly configured and experimental |
| Monoline checkout | Imported code, subprocess calls, workflow/state exchange | Separate local code and license boundary |

CONNECT can be configured with a token, but this release does not provide TLS,
per-user roles, tenant isolation, or a hardened remote deployment model. See
[`SECURITY.md`](../SECURITY.md).

## Source-tree orientation

```text
addons/       first-party addon packages
assets/       small first-party workflow seed assets selected for release
core/         application policy, prompts, state, tools, traces, memory/search
engine/       model/media/network/workflow runtime surfaces
monokernel/   kernel bridge, guard, dock, envelopes, and turn pipeline
skills/       model-facing skill descriptions and executors
tools/        developer/runtime helper tools
ui/           PySide6 application shell, pages, panels, and modules
tests/        unit, contract, integration, and regression tests
docs/         public guides plus retained design/release evidence
```

## What this document does not promise

- stable Python APIs or persisted schemas;
- complete process isolation;
- a security sandbox around tools;
- deterministic behavior from third-party models;
- production-safe remote agent hosting;
- feature parity across model backends;
- completed UI for every backend; or
- that every internal design document represents shipped behavior.
