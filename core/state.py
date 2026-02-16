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
        
        # Resources
        self.ctx_limit: int = 8192
        self.ctx_used: int = 0
