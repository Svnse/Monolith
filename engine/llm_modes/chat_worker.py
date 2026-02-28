from __future__ import annotations


class ChatWorkerExecutionMode:
    """
    Worker execution strategy for plain chat completion.

    Operates on the existing GeneratorWorker helper methods so the worker spine
    can delegate behavior without embedding mode-specific branching.
    """

    def run(self, worker):
        assistant_text = worker._chat_once_text(worker.messages)
        if assistant_text:
            worker._emit_text_stream(assistant_text)
        completed = not worker.isInterruptionRequested()
        loop_history = list(worker.messages)
        return completed, assistant_text, loop_history

