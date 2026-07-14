from __future__ import annotations

from core.task import Task
from monokernel.dock import MonoDock


class MonoBridge:
    def __init__(self, dock: MonoDock):
        self.dock = dock

    def wrap(self, source: str, command: str, target: str, **kwargs) -> Task:
        priority = int(kwargs.pop("priority", 2))
        payload = kwargs.pop("payload", kwargs)
        task = Task.new(
            addon_pid=source,
            target=target,
            command=command,
            payload=payload,
            priority=priority,
        )
        # turn_trace v1 spec join key: turn_id == str(Task.id). Stamp it into
        # the payload so engine/llm.py reads the canonical id instead of
        # minting its own uuid4().hex — that divergence silently broke
        # /rating → monothink (rating outcomes wrote dashed UUIDs while
        # frame_traces wrote hex; the join missed every time).
        if isinstance(task.payload, dict) and not task.payload.get("task_id"):
            task.payload["task_id"] = str(task.id)
        return task

    def submit(self, task: Task) -> None:
        self.dock.enqueue(task)

    def cancel(self, task_id: str) -> None:
        self.dock.cancel_task(task_id)

    def cancel_addon(self, addon_pid: str) -> None:
        self.dock.cancel_addon(addon_pid)

    def stop(self, target: str = "all") -> None:
        self.dock.on_stop(target)
