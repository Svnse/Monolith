from __future__ import annotations

from core.task import Task
from monokernel.dock import MonoDock


class MonoBridge:
    def __init__(self, dock: MonoDock):
        self.dock = dock

    def wrap(self, source: str, command: str, target: str, **kwargs) -> Task:
        payload = kwargs.pop("payload", kwargs)
        priority = int(kwargs.pop("priority", 2))
        return Task.new(
            addon_pid=source,
            target=target,
            command=command,
            payload=payload,
            priority=priority,
        )

    def submit(self, task: Task) -> None:
        self.dock.enqueue(task)

    def cancel(self, task_id: str) -> None:
        self.dock.cancel_task(task_id)

    def cancel_addon(self, addon_pid: str) -> None:
        self.dock.cancel_addon(addon_pid)

    def stop(self, target: str = "all") -> None:
        self.dock.on_stop(target)
