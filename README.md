# Monolith v0.28a

**Stop chatting with AI. Start commanding it.**

Monolith is a local-first AI workstation built for real execution, not just conversation. It combines a modular UI, a kernel-style routing layer, and multi-engine runtime support for coding, LLM chat, vision, audio, and relay workflows.

---

## What Is New in v0.28a

- Refined CODE agent loop with stronger runtime walls and clearer event traces
- Improved approval flow and redirect behavior while runs are active
- Better tool execution safety with boundary-aware path controls
- Expanded runtime observability via effect journaling and structured events
- Cleaner module orchestration through the MonoKernel task routing path

---

## Core Capabilities

- **CODE Agent Runtime**
  - Goal-driven execution loop with tool use, policy checks, and stop controls
  - Inline approvals for sensitive tool scopes
  - Timeline-first UI with cycle grouping and event visibility

- **Local LLM Chat**
  - GGUF model loading via `llama-cpp-python`
  - Streaming generation and persistent session behavior

- **Vision Generation**
  - Stable Diffusion module integration for image generation workflows

- **Audio Generation**
  - Audio generation module support for local creative pipelines

- **Relay Module**
  - Multi-agent style message relay and room coordination primitives

---

## Architecture (High Level)

Monolith follows a layered signal architecture:

1. **UI / Addons**
2. **MonoKernel (Guard + Dock + Bridge)**
3. **Engines (LLM / Loop / Vision / Audio / Relay)**

All execution commands are routed through the kernel path, not direct UI-to-engine calls.

---

## Quick Start

### Windows

1. Clone the repository
2. Run `install.bat`
3. Run `start.bat`

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

---

## Requirements

- Python 3.10+
- NVIDIA GPU recommended for Vision and Audio modules
- Sufficient VRAM for selected local models

---

## Project Layout

```text
monokernel/   # Guard, queue, and execution arbitration
engine/       # Loop runtime, LLM engine, tools, process control
ui/           # Main window, pages, modules, addon registry
core/         # Shared state, config, paths, style, theme logic
docs/         # Specs, architecture notes, prompts, contracts
```

---

## Docs Pointers

- `docs/architecture.md`
- `docs/AGENT_SYSTEM_SPEC_V1.md`
- `docs/MONOLITH_LOOP_REFACTOR_SPEC.md`
- `docs/RUNTIME_AWARENESS_SPEC.md`

---

## Status

**Version:** `0.28a`  
**Stage:** Active experimental development

Monolith is built for rapid iteration. Interfaces and internals can evolve quickly between versions.

