V2 Smoke Tests (Manual Checklist)

1) LLM generate + STOP (llm)
   - Submit a generate task through the UI terminal.
   - Verify tokens stream.
   - Issue STOP for llm and confirm it halts immediately.
   - Confirm READY is emitted and another task can run.

2) Vision generate + STOP (vision)
   - Load a vision model and submit a generate task.
   - Verify an image is emitted.
   - Issue STOP for vision and confirm it interrupts generation.

3) Concurrency
   - Start an LLM generate task and a vision generate task.
   - Confirm both run concurrently (one task per engine).

4) Cancellation
   - Queue multiple tasks, cancel one before it runs, and confirm it never executes.
   - Cancel an active task and confirm Dock triggers Guard.stop(target).

5) STOP(all)
   - Issue STOP(all) and confirm both engines stop and active tasks clear.
