# MONOLITH ARCHITECTURE — Agent Reference

**Purpose**: This document maps Monolith's complete architecture with explicit signal flows, contracts, and implementation patterns. Everything an agent needs to understand the system in one read.

**Philosophy**: Sovereignty-focused local-first AI workstation with deterministic core (MonoKernel) and probabilistic periphery (engines/modules).

---

## TABLE OF CONTENTS

1. [System Overview](#system-overview)
2. [Component Hierarchy](#component-hierarchy)
3. [Signal Flow Architecture](#signal-flow-architecture)
4. [MonoKernel Contract (v1 — FROZEN)](#monokernel-contract-v1--frozen)
5. [Engine Architecture](#engine-architecture)
6. [Addon System](#addon-system)
7. [Task Queue System](#task-queue-system)
8. [Bootstrap Sequence](#bootstrap-sequence)
9. [Critical Patterns](#critical-patterns)
10. [Implementation Details](#implementation-details)

---

## SYSTEM OVERVIEW

### Core Principle
**MonoKernel decides WHEN things happen — never WHAT happens.**

Monolith enforces strict separation:
- **Deterministic Core**: MonoGuard + MonoDock (arbitration only)
- **Probabilistic Periphery**: Engines (LLM, Vision, Audio) + UI/Modules (execution)

### Component Layers
```
UI/Addons (presentation + user interaction)
    ↕ signals only
MonoKernel (arbitration + routing)
    ↕ signals only  
Engines (execution + computation)
```

**Rule**: UI never calls engines directly. Engines never emit to UI directly. Everything routes through MonoKernel.

---

## COMPONENT HIERARCHY

### Directory Structure
```
monolith-main/
├── monokernel/          # Core arbitration layer (FROZEN v1)
│   ├── guard.py         # Signal router + engine orchestrator
│   ├── dock.py          # Task queue + cancellation
│   └── bridge.py        # UI→Kernel API
├── engine/              # Execution layer
│   ├── base.py          # EnginePort protocol
│   ├── bridge.py        # EngineBridge (generation gating)
│   ├── llm.py           # LLMEngine implementation
│   └── vision.py        # VisionEngine implementation
├── ui/                  # Presentation layer
│   ├── main_window.py   # Main chrome + global signals
│   ├── pages/           # Page addons (full screen)
│   │   └── chat.py      # Chat interface (Terminal addon)
│   ├── modules/         # Module addons (stackable)
│   │   ├── sd.py        # Vision module
│   │   └── injector.py  # Context injector module
│   ├── addons/          # Addon system infrastructure
│   │   ├── spec.py      # AddonSpec (id, kind, factory)
│   │   ├── registry.py  # AddonRegistry (addon storage)
│   │   ├── host.py      # AddonHost (lifecycle manager)
│   │   ├── context.py   # AddonContext (dependency injection)
│   │   └── builtin.py   # Built-in addon factories + wiring
│   └── components/      # Reusable UI components
├── core/                # Shared state + utilities
│   ├── state.py         # AppState + SystemStatus enum
│   ├── task.py          # Task + TaskStatus (kernel commands)
│   └── llm_config.py    # LLM configuration + behavior tags
└── bootstrap.py         # Application entry point
```

---

## SIGNAL FLOW ARCHITECTURE

### Signal Flow: User Prompt → LLM Response

This is the canonical signal chain every agent should understand:

```
┌─────────────┐
│  User types │ "Hello"
│  in input   │
└─────┬───────┘
      │
      ↓ QLineEdit.returnPressed or btn_send.clicked
┌─────────────────┐
│  PageChat       │
│  .handle_send   │──┐
│    _click()     │  │ Emits sig_generate
└─────────────────┘  │
                     ↓
              ┌──────────────┐
              │ terminal_    │  (in builtin.py)
              │  factory     │  Wiring layer
              │  lambda      │
              └──────┬───────┘
                     │
                     ↓ ctx.bridge.submit(Task)
              ┌──────────────┐
              │  MonoBridge  │
              │  .submit()   │
              └──────┬───────┘
                     │
                     ↓ dock.enqueue(Task)
              ┌──────────────┐
              │  MonoDock    │
              │  .enqueue()  │──┐ Queues task
              └──────────────┘  │
                                ↓ _try_submit()
                         ┌──────────────┐
                         │  MonoGuard   │
                         │  .submit()   │──┐ Routes to engine
                         └──────────────┘  │
                                           ↓ handler(payload)
                                    ┌──────────────┐
                                    │ EngineBridge │
                                    │  .generate() │──┐ Generation gating
                                    └──────────────┘  │
                                                      ↓ impl.generate(payload)
                                               ┌──────────────┐
                                               │  LLMEngine   │
                                               │  .generate() │──┐ Creates worker thread
                                               └──────────────┘  │
                                                                 ↓
                                                          ┌─────────────┐
                                                          │ Generator   │
                                                          │   Worker    │ (QThread)
                                                          └─────┬───────┘
                                                                │
                              ┌─────────────────────────────────┴─────────────────────────┐
                              │  Token stream from llama.cpp                              │
                              └─────────┬─────────────────────────────────────────────────┘
                                        │
                                        ↓ worker.token.emit(text)
                                 ┌──────────────┐
                                 │  LLMEngine   │
                                 │  sig_token   │
                                 └──────┬───────┘
                                        │
                                        ↓ Connected via EngineBridge
                                 ┌──────────────┐
                                 │ EngineBridge │
                                 │  sig_token   │──┐ Generation-gated
                                 └──────────────┘  │
                                                   ↓ Only if _active_gid matches
                                            ┌──────────────┐
                                            │  MonoGuard   │
                                            │  sig_token   │──┐ Verbatim re-emit
                                            └──────────────┘  │
                                                              ↓ Connected in builtin.py
                                                       ┌──────────────┐
                                                       │  PageChat    │
                                                       │  .append_    │
                                                       │   token()    │
                                                       └──────────────┘
                                                              │
                                                              ↓ Accumulates tokens
                                                        [Display in UI]
```

### Key Signal Chains

#### 1. SEND Signal Chain (User → Kernel → Engine)
```
PageChat.sig_generate
  → builtin.terminal_factory lambda
    → MonoBridge.submit(Task)
      → MonoDock.enqueue(Task)
        → MonoDock._try_submit()
          → MonoGuard.submit(Task)
            → engine.generate(payload)
```

#### 2. TOKEN Signal Chain (Engine → Kernel → UI)
```
GeneratorWorker.token.emit(str)
  → LLMEngine.sig_token
    → EngineBridge.sig_token (gated)
      → MonoGuard.sig_token (verbatim)
        → PageChat.append_token()
```

#### 3. STATUS Signal Chain (Engine → Kernel → UI)
```
LLMEngine.set_status(SystemStatus)
  → LLMEngine.sig_status.emit(status)
    → EngineBridge.sig_status
      → MonoGuard.sig_status
        → MonolithUI.update_status()
        → PageChat.update_status()
```

#### 4. STOP Signal Chain (UI → Kernel → Engine)
```
PageChat.sig_stop
  → builtin.terminal_factory lambda
    → MonoBridge.stop("llm")
      → MonoDock.on_stop("llm")
        → MonoGuard.stop("llm")
          → engine.stop_generation()
            → GeneratorWorker.requestInterruption()
```

---

## MONOKERNEL CONTRACT (v1 — FROZEN)

### Purpose (Non-Negotiable)
MonoGuard is the **sole authority** between UI and engines. Its role is **arbitration, not computation**.

### Authority Rules

#### 1.1 Single Ingress
- All user commands affecting execution **MUST** pass through MonoGuard
- UI **MUST NOT** call engine methods directly
- Addons **MUST NOT** call engine methods directly

#### 1.2 Single Egress
- All execution state, tokens, traces, usage metrics **MUST** pass through MonoGuard
- UI **MUST NOT** subscribe to engine signals directly
- Engines **MUST NEVER** emit directly to UI

### Kernel Scope

**The kernel MAY:**
- Route commands
- Gate execution by system state
- Preempt execution via STOP
- Queue at most one pending command
- Re-emit engine signals verbatim
- Observe system state transitions

**The kernel MUST NOT:**
- Execute business logic
- Perform blocking operations
- Sleep, wait, or poll
- Contain UI logic
- Contain engine logic
- Know what "chat", "LLM", or "RAG" is
- Accumulate feature-specific state

### STOP Semantics (Hard Law)

#### 3.1 STOP Always Wins
When STOP is issued:
- Current execution is interrupted immediately (non-blocking)
- Any pending command is cleared
- Control returns to UI instantly

#### 3.2 Truthful State
`SystemStatus.READY` **MUST** only be emitted when:
- No execution is running
- No engine work is active
- No pending command is executing

The kernel **must never emit READY prematurely**.

### Pending Command Rule

#### 4.1 Single Pending Slot
- Kernel may hold **at most one** pending command
- Pending commands exist **only** to resume after STOP-based preemption

#### 4.2 Replay on READY
- Pending command may execute once when system transitions to READY
- Pending commands are discarded if STOP is explicitly invoked
- No scheduling, prioritization, or batching exists in v1

### Engine Isolation
The engine:
- Is execution-only
- Knows nothing about UI
- Knows nothing about kernel rules
- Knows nothing about addons
- Accepts commands and emits signals **only**

**The kernel adapts the engine; the engine never adapts to the kernel.**

### UI Restrictions
The UI:
- May emit commands freely
- **MUST NOT** assume commands will execute
- **MUST NOT** block waiting for execution
- **MUST** treat kernel signals as authoritative truth

**UI correctness depends on kernel truth, not intent.**

---

## ENGINE ARCHITECTURE

### EnginePort Protocol
All engines implement this protocol (defined in `engine/base.py`):

```python
@runtime_checkable
class EnginePort(Protocol):
    # Required signals
    sig_status: Signal  # SystemStatus transitions
    sig_trace: Signal   # Debug/status messages
    sig_token: Signal   # Text output stream
    
    # Required methods
    def set_model_path(self, path: str) -> None: ...
    def load_model(self) -> None: ...
    def unload_model(self) -> None: ...
    def generate(self, payload: dict) -> None: ...
    def stop_generation(self) -> None: ...
    def shutdown(self) -> None: ...
```

**Optional signals** (check with `hasattr` before use):
- `sig_usage`: Token/step count tracking (LLM-specific)
- `sig_image`: Image output (Vision engines)
- `sig_audio`: Audio output (Audio engines)
- `sig_finished`: Optional completion notification

### EngineBridge Pattern

**Purpose**: Generation gating to prevent signals from stale generations reaching the UI.

**Mechanism**:
```python
# Each generate() call gets a unique generation ID
self._gen_id += 1
gid = self._gen_id
self._active_gid = gid
self._connect_gated_handlers(gid)

# Signals only emit if gid matches _active_gid
lambda t, gid=gid: self.sig_token.emit(t) if self._active_gid == gid else None
```

**Result**: When `stop_generation()` is called, the generation ID increments, and all signals from the previous generation are automatically ignored.

### Engine Implementations

#### LLMEngine (`engine/llm.py`)
**State Machine**:
```
READY → set_model_path() → READY
READY → load_model() → LOADING → READY | ERROR
READY → generate() → RUNNING → READY | ERROR
RUNNING → stop_generation() → READY
READY → unload_model() → UNLOADING → READY
```

**Worker Threads**:
- `ModelLoader`: Loads GGUF via llama-cpp-python
- `GeneratorWorker`: Streaming inference with interruption support

**Conversation Management**:
- Maintains `conversation_history: list[dict]` with roles: system, user, assistant
- System prompt injection: `[{role: system, content: prompt}, {role: system, content: CONTEXT: ...}, ...]`
- Pending user index tracking for UPDATE semantics

#### VisionEngine (`engine/vision.py`)
**State Machine**: Same as LLMEngine

**Worker Threads**:
- `PipelineLoader`: Loads Stable Diffusion via diffusers
- `GenerationWorker`: Step-based image generation with callbacks

**Signals**:
- `sig_image`: Emits PIL Image objects
- `sig_trace`: Step progress messages
- No `sig_token` (not text-based)

---

## ADDON SYSTEM

### Addon Types

**Two kinds**:
1. **Page addons** (`kind="page"`): Full-screen views in main content area
2. **Module addons** (`kind="module"`): Stackable floating modules with icons in module strip

### AddonSpec
```python
@dataclass(frozen=True)
class AddonSpec:
    id: str                                    # Unique identifier
    kind: Literal["page", "module"]           # Addon type
    title: str                                # Display name
    icon: str | None                          # Unicode icon or None
    factory: Callable[[AddonContext], QWidget] # Widget constructor
```

### AddonContext (Dependency Injection)
```python
@dataclass
class AddonContext:
    state: AppState        # Shared application state
    guard: MonoGuard       # Kernel signal router
    bridge: MonoBridge     # Task submission API
    ui: MonolithUI | None  # Main window (for modules)
    host: AddonHost | None # Addon lifecycle manager
```

### Addon Lifecycle

#### Page Addon
```python
# Mounted once, lives forever
def addons_page_factory(ctx: AddonContext) -> QWidget:
    w = PageAddons(ctx.state)
    # Wire signals
    return w

# Registration
registry.register(AddonSpec(
    id="addons",
    kind="page",
    title="ADDONS",
    icon=None,
    factory=addons_page_factory
))

# Mounting
host.mount_page("addons")  # Returns cached widget if exists
```

#### Module Addon
```python
# Created fresh each launch, gets unique instance ID
def terminal_factory(ctx: AddonContext) -> QWidget:
    w = PageChat(ctx.state)
    
    # OUTGOING (addon → kernel)
    w.sig_generate.connect(
        lambda prompt: ctx.bridge.submit(
            ctx.bridge.wrap("terminal", "generate", "llm", 
                          payload={"prompt": prompt, "config": w.config})
        )
    )
    w.sig_stop.connect(lambda: ctx.bridge.stop("llm"))
    
    # INCOMING (kernel → addon)
    ctx.guard.sig_token.connect(w.append_token)
    ctx.guard.sig_trace.connect(w.append_trace)
    ctx.guard.sig_status.connect(w.update_status)
    
    return w

# Registration
registry.register(AddonSpec(
    id="terminal",
    kind="module",
    title="TERMINAL",
    icon="⌖",
    factory=terminal_factory
))

# Launching
instance_id = host.launch_module("terminal")  # Creates new instance
```

### Signal Wiring Pattern

**Critical**: Addon factories are responsible for **all signal wiring** between:
- Addon widgets and MonoBridge (outgoing commands)
- MonoGuard and addon widgets (incoming signals)

**Why**: Keeps wiring logic co-located with addon definition, making signal flows explicit.

---

## TASK QUEUE SYSTEM

### Task Structure
```python
@dataclass
class Task:
    id: UUID              # Unique task identifier
    addon_pid: str        # Addon instance that created task
    target: str           # Engine key ("llm", "vision", etc.)
    command: str          # Command verb ("generate", "load", etc.)
    payload: dict         # Command-specific data
    priority: int         # 1=STOP, 2=normal, 3+=low priority
    status: TaskStatus    # PENDING/RUNNING/DONE/FAILED/CANCELLED
    timestamp: float      # Creation time
```

### Task Priority System
- **Priority 1**: STOP commands (preempt everything)
- **Priority 2**: Normal commands (FIFO within priority)
- **Priority 3+**: Low priority (future use)

### MonoDock Queue Behavior

**Per-engine queues**:
```python
queues: dict[str, Deque[Task]] = {
    "llm": deque([task1, task2, task3]),
    "vision": deque([task4, task5])
}
```

**Insertion logic** (`_insert_task`):
- Priority 2 tasks: Insert at end of priority-2 group (maintain FIFO within priority)
- Priority 3+ tasks: Append to end of queue
- Priority 1 tasks: Don't queue, immediately call `on_stop()`

**Execution logic** (`_try_submit`):
1. Check if engine has active task (via `MonoGuard.active_tasks`)
2. If busy, wait for `sig_engine_ready` signal
3. When ready, pop front of queue, check if cancelled
4. Submit to `MonoGuard.submit()`, which calls engine method
5. If submission fails, leave task in queue

**Cancellation**:
```python
cancelled_task_ids: set[str]   # Specific task cancellation
cancelled_addons: set[str]     # All tasks from this addon

def _is_cancelled(task: Task) -> bool:
    return (str(task.id) in cancelled_task_ids or 
            task.addon_pid in cancelled_addons)
```

### MonoGuard Task Routing

**Dispatch table**:
```python
ENGINE_DISPATCH = {
    "set_path": "set_model_path",
    "load": "load_model",
    "unload": "unload_model",
    "generate": "generate",
}
```

**Submission flow**:
```python
def submit(self, task: Task) -> bool:
    engine = self.engines.get(task.target)
    if self.active_tasks[task.target] is not None:
        return False  # Engine busy
    
    self.active_tasks[task.target] = task
    task.status = TaskStatus.RUNNING
    
    method_name = ENGINE_DISPATCH[task.command]
    handler = getattr(engine, method_name)
    handler(task.payload)  # or handler() for no-arg commands
    return True
```

---

## BOOTSTRAP SEQUENCE

### Startup Order (bootstrap.py)
```python
1. QApplication()
2. AppState()
3. LLMEngine(state) + VisionEngine(state)
4. EngineBridge(llm) + EngineBridge(vision)
5. MonoGuard(state, engines)
6. MonoDock(guard)
7. MonoBridge(dock)
8. MonolithUI(state)
9. AddonRegistry + build_builtin_registry()
10. AddonContext(state, guard, bridge, ui, host=None)
11. AddonHost(registry, ctx)
12. ui.attach_host(host)
13. Wire global signals (guard → ui)
14. ui.show()
15. app.exec()
```

### Global Signal Wiring
```python
# System-wide status updates
guard.sig_status.connect(ui.update_status)

# Token usage tracking (LLM only)
guard.sig_usage.connect(ui.update_ctx)

# Shutdown sequence
app.aboutToQuit.connect(guard.stop)
app.aboutToQuit.connect(engine.shutdown)
app.aboutToQuit.connect(vision_engine.shutdown)
```

---

## CRITICAL PATTERNS

### Pattern 1: Generation Gating (EngineBridge)
**Problem**: After `stop_generation()`, stale tokens from cancelled generation still arrive
**Solution**: Generation ID gating
```python
# Old generation (gid=5) emits token after stop (gid=6)
lambda t, gid=5: self.sig_token.emit(t) if self._active_gid == 6 else None
# Result: Signal dropped, UI sees nothing
```

### Pattern 2: Signal Re-emission (MonoGuard)
**Problem**: Kernel must route signals without transforming them
**Solution**: Verbatim re-emit
```python
engine.sig_token.connect(self.sig_token)  # Direct passthrough
```

### Pattern 3: Factory-based Wiring (Addons)
**Problem**: Signal connections need to be explicit and traceable
**Solution**: All wiring happens in addon factory functions
```python
def terminal_factory(ctx):
    w = PageChat(ctx.state)
    # ALL WIRING HERE — visible in one place
    w.sig_generate.connect(lambda p: ctx.bridge.submit(...))
    ctx.guard.sig_token.connect(w.append_token)
    return w
```

### Pattern 4: UPDATE Semantics (Chat)
**Problem**: User types new prompt while generation is running
**Solution**: Tri-state SEND/STOP/UPDATE
```python
if not self._is_running:
    self.send()  # Normal send
elif not txt:
    self.sig_stop.emit()  # STOP button
else:
    self._pending_update_text = txt
    self._awaiting_update_restart = True
    self.sig_stop.emit()
    # On sig_engine_ready, re-submit with UPDATE prompt
```

### Pattern 5: Non-blocking Kernel (MonoGuard)
**Problem**: Kernel must never block
**Solution**: All engine calls are async via QThread
```python
# ✅ CORRECT
worker = GeneratorWorker(...)
worker.start()  # Returns immediately

# ❌ WRONG
result = model.generate(...)  # Blocks kernel!
```

### Pattern 6: Status Truth (MonoGuard)
**Problem**: UI needs reliable state
**Solution**: Kernel is single source of truth
```python
# UI must never assume:
self.send()  # ❌ Assumes execution will happen

# UI must react to signals:
def update_status(self, engine_key, status):
    if status == SystemStatus.READY:
        self.btn_send.setEnabled(True)  # ✅ React to truth
```

---

## IMPLEMENTATION DETAILS

### SystemStatus Enum
```python
class SystemStatus(Enum):
    READY = "READY"         # Engine idle, can accept commands
    LOADING = "LOADING"     # Model loading in progress
    RUNNING = "RUNNING"     # Generation in progress
    ERROR = "ERROR"         # Error occurred (auto-transitions to READY)
    UNLOADING = "UNLOADING" # Model unloading in progress
```

### TaskStatus Enum
```python
class TaskStatus(Enum):
    PENDING = "PENDING"       # Queued, not yet submitted
    RUNNING = "RUNNING"       # Submitted to engine
    DONE = "DONE"             # Completed successfully
    FAILED = "FAILED"         # Error occurred
    CANCELLED = "CANCELLED"   # Explicitly cancelled
```

### Behavior Tag System (LLM Config)
**Location**: `core/llm_config.py`

**Mechanism**: User selects tags → combined prompts injected into system message

```python
TAG_MAP = {
    "concise": "Be extremely concise. Omit pleasantries.",
    "technical": "Assume expert-level technical knowledge.",
    # ... more tags
}

# User selects ["concise", "technical"]
# → System prompt becomes:
# "You are Monolith. Be precise.\n\nBe extremely concise...\n\nAssume expert-level..."
```

### Context Injection (Injector Module)
**Location**: `ui/modules/injector.py`

**Mechanism**: Runtime context dynamically inserted into system messages before generation

```python
# In LLMEngine.generate():
if context_injection:
    self.conversation_history.insert(1, {
        "role": "system",
        "content": f"CONTEXT: {context_injection}"
    })
```

**Use case**: Add file contents, notes, or dynamic data to every generation without modifying system prompt.

### Module Strip (UI Component)
**Location**: `ui/components/module_strip.py`

**Features**:
- Horizontal scrollable icon strip
- Overflow arrows (left/right) when modules exceed width
- Module flashing on `sig_finished` (visual completion notification)
- Click to switch module, close button to remove

**Critical**: Module IDs are UUID strings, not addon IDs (allows multiple instances of same addon)

### Conversation History (LLM)
**Structure**:
```python
[
    {"role": "system", "content": "You are Monolith. Be precise."},
    {"role": "system", "content": "CONTEXT: <injector contents>"},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "How are you?"},
    # ... continues
]
```

**UPDATE semantics**:
- On normal send: append user message, create pending assistant slot
- On stop: remove pending user message if generation didn't complete
- On resume: inject special user message "You were interrupted mid-generation. Continue from: {last_text}"

---

## DEBUGGING SIGNAL FLOWS

### Trace Signal Path
To trace a signal from UI → Kernel → Engine:

1. **Find the UI trigger**
   - Example: `btn_send.clicked` → `handle_send_click()`

2. **Locate the signal emission**
   - Example: `self.sig_generate.emit(txt)`

3. **Find the addon factory wiring**
   - Example: In `terminal_factory()`: `w.sig_generate.connect(lambda p: ctx.bridge.submit(...))`

4. **Trace through MonoBridge/MonoDock**
   - `bridge.submit(task)` → `dock.enqueue(task)` → `dock._try_submit()`

5. **Trace through MonoGuard**
   - `guard.submit(task)` → `engine.generate(payload)`

6. **Trace engine implementation**
   - `LLMEngine.generate()` → `worker.start()`

7. **Trace return signal path**
   - `worker.token.emit()` → `LLMEngine.sig_token` → `EngineBridge.sig_token` → `MonoGuard.sig_token` → `PageChat.append_token()`

### Trace System Status Changes
```
User clicks LOAD MODEL
  → PageChat.toggle_load()
    → sig_load.emit()
      → bridge.submit(wrap("terminal", "load", "llm"))
        → guard.submit(task)
          → engine.load_model()
            → sig_status.emit(LOADING)
              → guard.sig_status.emit("llm", LOADING)
                → ui.update_status("llm", LOADING)
                  → PageChat.update_status("llm", LOADING)
                    → btn_load.setText("LOADING...")

ModelLoader finishes
  → LLMEngine._on_load_success()
    → set_status(READY)
      → sig_status.emit(READY)
        → guard.sig_status.emit("llm", READY)
          → ui.update_status("llm", READY)
            → PageChat.update_status("llm", READY)
              → btn_load.setText("UNLOAD MODEL")
```

---

## EXTENDING MONOLITH

### Adding a New Engine

1. **Implement EnginePort protocol** (`engine/your_engine.py`)
```python
class YourEngine(QObject):
    sig_status = Signal(SystemStatus)
    sig_trace = Signal(str)
    sig_token = Signal(str)
    sig_your_output = Signal(object)  # Optional custom signal
    
    def __init__(self, state: AppState):
        self.state = state
        self._status = SystemStatus.READY
    
    def set_model_path(self, path: str) -> None: ...
    def load_model(self) -> None: ...
    def unload_model(self) -> None: ...
    def generate(self, payload: dict) -> None: ...
    def stop_generation(self) -> None: ...
    def shutdown(self) -> None: ...
```

2. **Wrap in EngineBridge** (`bootstrap.py`)
```python
your_engine_impl = YourEngine(state)
your_engine = EngineBridge(your_engine_impl)
```

3. **Register with MonoGuard** (`bootstrap.py`)
```python
guard = MonoGuard(state, {
    "llm": engine,
    "vision": vision_engine,
    "your": your_engine,  # Add here
})
```

4. **Wire optional signals** (`monokernel/guard.py` → `__init__`)
```python
if hasattr(engine, "sig_your_output"):
    engine.sig_your_output.connect(self.sig_your_output)
```

5. **Create addon factory** (`ui/addons/builtin.py`)
```python
def your_module_factory(ctx: AddonContext):
    w = YourModuleWidget()
    w.sig_generate.connect(
        lambda prompt: ctx.bridge.submit(
            ctx.bridge.wrap("your_module", "generate", "your", payload={...})
        )
    )
    ctx.guard.sig_your_output.connect(w.on_output)
    return w
```

6. **Register addon**
```python
registry.register(AddonSpec(
    id="your_module",
    kind="module",
    title="YOUR MODULE",
    icon="★",
    factory=your_module_factory
))
```

### Adding a New Addon

1. **Create widget** (`ui/pages/` or `ui/modules/`)
```python
class YourWidget(QWidget):
    sig_action = Signal(str)  # Define outgoing signals
    
    def __init__(self):
        super().__init__()
        # Build UI
    
    def handle_input(self, data):
        # Incoming signal handler
        pass
```

2. **Create factory** (`ui/addons/builtin.py`)
```python
def your_factory(ctx: AddonContext):
    w = YourWidget()
    
    # OUTGOING
    w.sig_action.connect(
        lambda data: ctx.bridge.submit(
            ctx.bridge.wrap("your_addon", "command", "target", payload={...})
        )
    )
    
    # INCOMING
    ctx.guard.sig_some_signal.connect(w.handle_input)
    
    return w
```

3. **Register** (`ui/addons/builtin.py` → `build_builtin_registry()`)
```python
registry.register(AddonSpec(
    id="your_addon",
    kind="module",  # or "page"
    title="YOUR ADDON",
    icon="◆",
    factory=your_factory
))
```

4. **Add launcher** (if module, add to `ui/modules/manager.py`)
```python
btn_your = SkeetButton("YOUR ADDON")
btn_your.clicked.connect(lambda: self.sig_launch_addon.emit("your_addon"))
```

---

## AGENT QUICK REFERENCE

### When implementing UI changes:
- ✅ Emit signals from widgets
- ❌ Don't call kernel/engine methods directly
- ✅ Connect signals in addon factories
- ❌ Don't assume commands will execute

### When implementing engine changes:
- ✅ Follow EnginePort protocol
- ❌ Don't emit UI-specific signals
- ✅ Use QThread for blocking operations
- ❌ Don't know about kernel rules

### When implementing kernel changes:
- ⚠️ **STOP**: Kernel contract is FROZEN (v1)
- ⚠️ Any change requires architectural review
- ✅ If you must: version bump, not patch

### When debugging signal flows:
1. Find UI emission point
2. Locate addon factory wiring
3. Trace through bridge/dock/guard
4. Find engine handler
5. Trace return path backward

### When reading code:
- Signal chains: Follow `.connect()` calls
- Task flow: Start at `MonoBridge.submit()`
- Status changes: Track `sig_status.emit()`
- Generation flow: Start at `engine.generate()`

---

## APPENDIX: FILE LOCATIONS

### Core Contracts
- `monokernel/kernel_contract.md` — Kernel rules (FROZEN)
- `monokernel/Kernel_Contract_v2.txt` — Legacy contract (superseded)
- `engine/base.py` — EnginePort protocol

### Signal Routing
- `monokernel/guard.py` — MonoGuard (signal router)
- `monokernel/dock.py` — MonoDock (task queue)
- `monokernel/bridge.py` — MonoBridge (UI→Kernel API)
- `engine/bridge.py` — EngineBridge (generation gating)

### Engine Implementations
- `engine/llm.py` — LLM engine (llama-cpp-python)
- `engine/vision.py` — Vision engine (diffusers)

### Addon System
- `ui/addons/spec.py` — AddonSpec definition
- `ui/addons/registry.py` — AddonRegistry
- `ui/addons/host.py` — AddonHost (lifecycle)
- `ui/addons/context.py` — AddonContext (DI)
- `ui/addons/builtin.py` — Built-in addon factories + wiring

### UI Components
- `ui/main_window.py` — Main chrome + global signals
- `ui/pages/chat.py` — Chat/Terminal interface
- `ui/modules/sd.py` — Vision module
- `ui/modules/injector.py` — Context injector
- `ui/components/atoms.py` — Basic widgets
- `ui/components/complex.py` — Compound widgets
- `ui/components/module_strip.py` — Module icon strip

### Configuration
- `core/state.py` — AppState + SystemStatus
- `core/task.py` — Task + TaskStatus
- `core/llm_config.py` — LLM config + behavior tags
- `core/style.py` — UI styling constants

### Bootstrap
- `bootstrap.py` — Application entry point
- `main.py` — Thin wrapper around bootstrap

---

## FINAL NOTES FOR AGENTS

**This document captures implicit knowledge that would otherwise require:**
- Reading 15+ files
- Tracing signal chains across 5+ layers
- Understanding frozen contracts vs. extensible systems
- Discovering generation gating patterns
- Mapping task lifecycle through queue system

**Everything is here. No archaeology required.**

**When in doubt:**
1. Check kernel contract (FROZEN)
2. Trace signal chain (this doc)
3. Follow EnginePort protocol
4. Wire in addon factory
5. Test with VITALS pattern (sig_status transitions)

**Remember**: MonoKernel decides WHEN. Engines decide WHAT. UI reacts to truth.
