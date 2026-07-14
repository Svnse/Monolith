from .action_review import ActionReviewPanel
from .audio_panel import AudioPanel
from .archive_browser import ArchiveBrowserPanel
from .audit_log import AuditLogPanel
from .expedition import ExpeditionPanel
from .generation_trace import GenerationTracePanel
from .model_config import ModelConfigPanel
from .question_panel import QuestionPanel
from .reasoning_tree import ReasoningTreePanel
from .self_maint import SelfMaintPanel
from .workshop import WorkshopPane
from .workshop_library import WorkshopLibraryPane

# VisionPanel removed — SDModule mounts directly in the companion pane
# via CompanionPane.attach_module(VISION, ...). The read-only proxy
# panel was a dead-end (its bind_module was never called for the sd
# addon, so user-visible buttons silently no-op'd).

__all__ = [
    "ActionReviewPanel",
    "AudioPanel",
    "ArchiveBrowserPanel",
    "AuditLogPanel",
    "ExpeditionPanel",
    "GenerationTracePanel",
    "ModelConfigPanel",
    "QuestionPanel",
    "ReasoningTreePanel",
    "SelfMaintPanel",
    "WorkshopPane",
    "WorkshopLibraryPane",
]
