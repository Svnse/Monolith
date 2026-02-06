from __future__ import annotations

from collections import deque
from typing import Deque

from core.task import Task, TaskStatus
from monokernel.guard import MonoGuard


class MonoDock:
    def __init__(self, guard: MonoGuard):
        self.guard = guard
        self.queues: dict[str, Deque[Task]] = {}
        self.cancelled_task_ids: set[str] = set()
        self.cancelled_addons: set[str] = set()
        self._in_submit: dict[str, bool] = {}
        self.guard.sig_engine_ready.connect(self._on_engine_ready)

    def enqueue(self, task: Task) -> None:
        if task.priority == 1:
            self.on_stop(task.target)
            return
        queue = self.queues.setdefault(task.target, deque())
        self._insert_task(queue, task)
        self._try_submit(task.target)

    def cancel_task(self, task_id: str) -> None:
        self.cancelled_task_ids.add(task_id)
        for engine_key in self.guard.engines.keys():
            active = self.guard.get_active_task(engine_key)
            if active and str(active.id) == task_id:
                self.guard.stop(engine_key)

    def cancel_addon(self, addon_pid: str) -> None:
        self.cancelled_addons.add(addon_pid)
        for engine_key in self.guard.engines.keys():
            active = self.guard.get_active_task(engine_key)
            if active and active.addon_pid == addon_pid:
                self.guard.stop(engine_key)

    def on_stop(self, target: str = "all") -> None:
        self.guard.stop(target)
        if target == "all":
            for queue in self.queues.values():
                for task in queue:
                    self.cancelled_task_ids.add(str(task.id))
        else:
            queue = self.queues.get(target)
            if queue:
                for task in queue:
                    self.cancelled_task_ids.add(str(task.id))

    def _on_engine_ready(self, engine_key: str) -> None:
        self._try_submit(engine_key)

    def _try_submit(self, engine_key: str) -> None:
        if self._in_submit.get(engine_key):
            return
        queue = self.queues.get(engine_key)
        if not queue:
            return

        self._in_submit[engine_key] = True
        try:
            while queue:
                task = queue[0]
                if self._is_cancelled(task):
                    task.status = TaskStatus.CANCELLED
                    if queue:
                        queue.popleft()
                    self.cancelled_task_ids.discard(str(task.id))
                    continue
                accepted = self.guard.submit(task)
                if accepted and queue:
                    queue.popleft()
                break
        finally:
            self._in_submit[engine_key] = False

    def _is_cancelled(self, task: Task) -> bool:
        return str(task.id) in self.cancelled_task_ids or task.addon_pid in self.cancelled_addons

    def _insert_task(self, queue: Deque[Task], task: Task) -> None:
        if task.priority == 2:
            items = list(queue)
            insert_at = 0
            for existing in items:
                if existing.priority != 2:
                    break
                insert_at += 1
            items.insert(insert_at, task)
            queue.clear()
            queue.extend(items)
        else:
            queue.append(task)
