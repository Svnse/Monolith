from enum import Enum

# System Status Enum
class SystemStatus(Enum):
    READY = "READY"
    LOADING = "LOADING"
    RUNNING = "RUNNING"
    ERROR = "ERROR"
    UNLOADING = "UNLOADING"

# Shared Application State
class AppState:
    def __init__(self):
        # System
        self.gguf_path: str | None = None
        self.model_loaded: bool = False
        self.status: SystemStatus = SystemStatus.READY
        
        # Resources -- ctx_limit stays 0 until the engine resolves a window
        # via /v1/models -> inference table -> registry family default.
        # The vitals footer renders 0 distinctly so a missing window does
        # not silently masquerade as an arbitrary 8k default.
        self.ctx_limit: int = 0
        self.ctx_used: int = 0
