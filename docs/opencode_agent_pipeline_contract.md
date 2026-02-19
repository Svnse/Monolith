---Contract Begin---
System: "OpenCode Agent Pipeline"
Inputs: (natural language + workspace)
Outputs: (planned diffs/results)
Steps:
  Step 1: RECEIVE_REQUEST
    - Entrypoint is the agent service method `Run(ctx, sessionID, content, attachments...)`, which gates concurrent execution per session and starts a cancellable generation context.
    - It then dispatches to `processGeneration(...)`, the core iterative loop controller.

  Step 2: BUILD_TASK_STATE
    - The natural-language request is converted into a persisted `user` message with structured content parts (text + optional binary attachments).
    - Existing session history is loaded and used as the task state; if a prior summary marker exists, history is truncated to summary-forward context before appending the new user message.
    - Effective task representation = ordered message list (`msgHistory`) + session metadata + configured system prompt + registered tool schemas.

  Step 3: CALL_LLM_GENERATOR
    - The provider is invoked through `StreamResponse(ctx, msgHistory, tools)`.
    - A new persisted assistant message is created before consuming stream events.
    - Streaming deltas update assistant state incrementally (reasoning/content/tool-call start/stop/complete/error).

  Step 4: MATERIALIZE_ACTIONS
    - On completion, assistant tool calls are resolved against registered tool inventory.
    - Each tool call executes via `tool.Run(ctx, ToolCall{ID, Name, Input})`.
    - Tool outputs are normalized as `tool_result` parts and persisted as a separate tool-role message.

  Step 5: VERIFY_AND_FEEDBACK
    - Verification is primarily tool-mediated (not a separate verifier model):
      - file edit/write/patch tools return diffs + metadata,
      - diagnostics-capable tools attach LSP diagnostics,
      - bash/tool outputs provide runtime/test evidence.
    - Tool result message is appended back into `msgHistory` so the same model can self-correct in the next iteration.

  Step 6: ITERATE_OR_TERMINATE
    - If assistant finish reason is `tool_use` and tool results exist, loop continues with updated history.
    - Otherwise, agent emits final assistant response and marks request done.

Actions:
  CALL_LLM_PLAN:
    Description: Implicit planning inside the main model turn where tool calls are proposed (no standalone planner service).
  CALL_LLM_GENERATE:
    Description: Stream assistant deltas/events from provider into persisted assistant message state.
  EXECUTE_TOOL:
    Description: Resolve tool name, run tool with structured JSON input, capture output/error/metadata.
  APPEND_TOOL_RESULT:
    Description: Persist tool outputs as a tool-role message and add to conversation history.
  VERIFY_OUTPUT:
    Description: Use returned diffs/diagnostics/command outputs as validation signals for next iteration.
  UPDATE_STATE:
    Description: Persist message/session updates (content deltas, tool calls, finish reason, usage/cost, summary pointer).
  HANDLE_ERROR:
    Description: Convert stream/tool/permission/cancellation failures into error events or finish reasons.
  TERMINATE:
    Description: Emit final `response` event when no further tool-use continuation is needed.

Loop Conditions:
  - ContinueIf:
      - context not canceled, AND
      - assistant finish reason == `tool_use`, AND
      - tool result message produced.
  - StopIf:
      - assistant finish reason is terminal (end turn / stop), OR
      - cancellation detected, OR
      - unrecoverable stream/provider error, OR
      - permission denial leading to permission-denied finish.

Context Management:
  - What context is stored?
      - Session-scoped message history (`user`, `assistant`, `tool` roles with typed parts),
      - assistant tool call state + finish reason,
      - usage/cost accounting,
      - optional summary checkpoint (`SummaryMessageID`) for context compression,
      - request cancellation handles in active-request map.
  - How is it passed between steps?
      - Message history array is passed into every provider call.
      - Tool/session/message identifiers are injected via context keys (`session_id`, `message_id`) for tool-side auditing/history/permission flows.
      - Persisted messages are reloaded and appended each iteration to form next prompt state.

Tool Integration:
  - What tools are called?
      - Core coding tools: bash, edit, patch, write, fetch, glob, grep, ls, view, sourcegraph, diagnostics, plus optional MCP tools.
      - Nested "agent" tool can spawn a constrained sub-agent (search/read-only toolset) for delegated discovery.
  - How are tool responses incorporated?
      - Each call returns normalized `ToolResponse{content, metadata, is_error}`.
      - Responses are transformed into `tool_result` content parts in a tool-role message.
      - That tool message is appended to history and fed back to the LLM on the next loop turn.

Verification:
  - How does the agent check its results?
      - There is no dedicated verifier LLM stage.
      - Verification is iterative and evidence-driven: file diffs, diagnostics blocks, and command outputs are returned by tools and then interpreted by the same generator model in subsequent loop cycles.
      - Completion is accepted when model stops requesting tools and emits terminal finish reason.

Errors/Recovery:
  - How are failures handled?
      - Request cancellation propagates through context; assistant message is finished as canceled and loop exits.
      - Missing tool names generate structured tool-result errors and allow continued iteration.
      - Permission-denied errors produce explicit denial tool results and mark finish as permission denied (remaining tool calls canceled).
      - Provider stream errors are surfaced as agent error events; unrecoverable errors terminate run.
      - Session-busy guard rejects concurrent run requests for same session.
---Contract End---
