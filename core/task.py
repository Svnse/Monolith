from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from time import time
from uuid import UUID, uuid4


class TaskStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class Task:
    id: UUID
    addon_pid: str
    target: str
    command: str
    payload: dict
    priority: int
    status: TaskStatus
    timestamp: float

    @classmethod
    def new(
        cls,
        addon_pid: str,
        target: str,
        command: str,
        payload: dict,
        priority: int = 2,
    ) -> "Task":
        return cls(
            id=uuid4(),
            addon_pid=addon_pid,
            target=target,
            command=command,
            payload=payload,
            priority=priority,
            status=TaskStatus.PENDING,
            timestamp=time(),
        )
