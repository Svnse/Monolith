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
        self.guard.sig_trace.emit("system", f"[DOCK] enqueue: task={task.id}, cmd={task.command}, target={task.target}, priority={task.priority}")
        if task.priority == 1 and task.command != "stop":
            self.on_stop(task.target)
            return
        queue = self.queues.setdefault(task.target, deque())
        if task.command == "stop":
            self.on_stop(task.target)
            queue.appendleft(task)
        else:
            self._insert_task(queue, task)
        self._sync_queue_state(task.target)
        self._try_submit(task.target)

    def cancel_task(self, task_id: str) -> None:
        self.cancelled_task_ids.add(task_id)
        for engine_key in self.guard.engines.keys():
            active = self.guard.get_active_task(engine_key)
            if active and str(active.id) == task_id:
                self.guard.stop(engine_key)
            self._sync_queue_state(engine_key)

    def cancel_addon(self, addon_pid: str) -> None:
        self.cancelled_addons.add(addon_pid)
        for engine_key in self.guard.engines.keys():
            active = self.guard.get_active_task(engine_key)
            if active and active.addon_pid == addon_pid:
                self.guard.stop(engine_key)
            self._sync_queue_state(engine_key)

    def on_stop(self, target: str = "all") -> None:
        self.guard.stop(target)
        if target == "all":
            for queue in self.queues.values():
                for task in queue:
                    self.cancelled_task_ids.add(str(task.id))
            for engine_key in self.guard.engines.keys():
                self._sync_queue_state(engine_key)
        else:
            queue = self.queues.get(target)
            if queue:
                for task in queue:
                    self.cancelled_task_ids.add(str(task.id))
            self._sync_queue_state(target)

    def _on_engine_ready(self, engine_key: str) -> None:
        self._try_submit(engine_key)

    def _try_submit(self, engine_key: str) -> None:
        if self._in_submit.get(engine_key):
            self.guard.sig_trace.emit("system", f"[DOCK] _try_submit: BLOCKED by _in_submit for {engine_key}")
            return
        queue = self.queues.get(engine_key)
        if not queue:
            self.guard.sig_trace.emit("system", f"[DOCK] _try_submit: empty queue for {engine_key}")
            return
        if self.guard.get_active_task(engine_key) is not None:
            next_task = queue[0] if queue else None
            if next_task and next_task.command == "stop":
                self.guard.sig_trace.emit("system", f"[DOCK] _try_submit: stop preempts busy engine {engine_key}")
            else:
                self.guard.sig_trace.emit("system", f"[DOCK] _try_submit: engine busy for {engine_key}")
                return

        self._in_submit[engine_key] = True
        try:
            while queue:
                # Pop the task immediately to avoid race conditions
                task = queue.popleft()
                
                if self._is_cancelled(task):
                    self.guard.sig_trace.emit("system", f"[DOCK] _try_submit: CANCELLED task={task.id}, cmd={task.command}")
                    task.status = TaskStatus.CANCELLED
                    self.cancelled_task_ids.discard(str(task.id))
                    self._sync_queue_state(engine_key)
                    continue
                
                accepted = self.guard.submit(task)
                self.guard.sig_trace.emit("system", f"[DOCK] _try_submit: guard.submit returned {accepted} for task={task.id}, cmd={task.command}")
                
                if not accepted:
                    # Re-queue the task at the front if not accepted
                    queue.appendleft(task)
                    self._sync_queue_state(engine_key)
                    break

                self._sync_queue_state(engine_key)
                # Immediate commands (set_path/set_history/set_ctx_limit) do not
                # occupy the engine. Keep draining until a non-immediate task is
                # running, otherwise queued generate/load tasks can get stranded.
                if self.guard.get_active_task(engine_key) is None:
                    continue
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

    def _sync_queue_state(self, engine_key: str) -> None:
        world_state = getattr(self.guard, "world_state", None)
        if world_state is None:
            return
        queue = self.queues.get(engine_key, deque())
        world_state.set_queue(engine_key, [str(task.id) for task in list(queue)])
