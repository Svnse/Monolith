# Connecting external agents to Monolith

CONNECT is Monolith's experimental local HTTP/MCP/peer bridge. It normally
listens on `127.0.0.1:7821` after the user opens **Connections** and selects
**Start**.

CONNECT is intended for same-machine development. It does not provide TLS,
role-based access control, multi-user isolation, or a supported internet/LAN
deployment. Keep it on loopback.

## Authentication model

`MONOLITH_AGENT_TOKEN` is read when the AgentServer instance is created. Set it
before launching Monolith when authentication is required:

```bat
set "MONOLITH_AGENT_TOKEN=use-a-long-random-value"
start.bat
```

When a token is configured, protected HTTP routes accept either:

```http
Authorization: Bearer use-a-long-random-value
```

or:

```http
X-API-Key: use-a-long-random-value
```

The `/health` route remains an unauthenticated liveness check. State, chat,
participants, MCP HTTP/SSE, `/events`, traces, memory, logs, database queries,
hooks, and other functional routes require the token when configured.

Without a token, loopback operation remains available to local processes. Any
local process that can reach the port can then use the routes, so configure a
token even on loopback when the machine runs untrusted local software.

## Binding rules

- Default host: `127.0.0.1`.
- The Connections page can request `0.0.0.0` for all interfaces.
- A non-loopback bind is refused unless `MONOLITH_AGENT_TOKEN` is non-empty.
- Supplying a token does **not** make remote exposure supported or production
  safe.

Do not place CONNECT behind a public tunnel or port-forward it. See the root
[`SECURITY.md`](../../SECURITY.md).

## Loopback HTTP quick start

The examples below assume CONNECT is running on loopback without a token. If a
token is configured, add one of the authentication headers shown above to every
request except `/health`.

### Check liveness

```http
GET http://127.0.0.1:7821/health
```

Example response:

```json
{"ok": true, "server": "monolith", "port": 7821}
```

### Join

```http
POST http://127.0.0.1:7821/join
Content-Type: application/json

{"name": "LocalAgent"}
```

An optional `url` registers the caller as an external peer whose URL exposes a
compatible `/chat` endpoint:

```json
{"name": "LocalAgent", "url": "http://127.0.0.1:8300"}
```

Register only endpoints you control. A peer URL creates another trust boundary
for prompts and responses.

### Read current state

```http
GET http://127.0.0.1:7821/state
```

State can include model/runtime status, recent messages, and queue information.
Treat it as sensitive.

### Send a blocking chat request

```http
POST http://127.0.0.1:7821/chat
Content-Type: application/json

{"message": "Summarize the current task.", "agent": "LocalAgent"}
```

The request waits for the active Monolith model path and can time out after
approximately 120 seconds. CONNECT must be attached to a live chat/model
session for generation to succeed.

### Stream a chat response

```http
POST http://127.0.0.1:7821/chat/stream
Content-Type: application/json

{"message": "Explain the current state.", "agent": "LocalAgent"}
```

The response is a server-sent event stream. Event types can include:

| Event | Typical data |
|---|---|
| `generation_start` | engine identity |
| `token` | text and engine |
| `thinking` | internal-lane status when exposed by the active path |
| `tool_call` | tool name and arguments |
| `tool_result` | tool name and result |
| `done` | final response |
| `error` | sanitized error text |

Do not assume event payloads are free of private prompt, model, file, or tool
information.

### Subscribe to broadcast events

```http
GET http://127.0.0.1:7821/events
Accept: text/event-stream
```

`/events` requires the configured AgentServer token just like other protected
routes. It can carry response tokens and runtime events; do not expose it to an
untrusted client.

Example with `curl` and a token:

```bat
curl -N -H "Authorization: Bearer %MONOLITH_AGENT_TOKEN%" http://127.0.0.1:7821/events
```

### List participants and leave

```http
GET http://127.0.0.1:7821/who
```

```http
POST http://127.0.0.1:7821/leave
Content-Type: application/json

{"name": "LocalAgent"}
```

## MCP transports

### Stdio

For a local MCP client, launch the server module from the repository checkout:

```json
{
  "mcpServers": {
    "monolith": {
      "command": "C:/path/to/Monolith/venv/Scripts/python.exe",
      "args": ["C:/path/to/Monolith/engine/agent_server.py", "--stdio"],
      "cwd": "C:/path/to/Monolith"
    }
  }
}
```

Use the exact schema required by the selected MCP client. Stdio inherits the
trust of the local process that launches it; it does not create a listening
network socket by itself.

Available MCP tools include joining, sending a message, reading recent history,
listing participants, and leaving. Tool availability may expand in later
releases.

### HTTP JSON-RPC

```http
POST http://127.0.0.1:7821/mcp
Content-Type: application/json

{"jsonrpc":"2.0","id":1,"method":"tools/list"}
```

### MCP over SSE

Open the SSE session, then post JSON-RPC messages with the returned session ID:

```text
GET  http://127.0.0.1:7821/mcp/sse
POST http://127.0.0.1:7821/mcp/message?sessionId=<id>
```

Both routes require the configured AgentServer token. Clients must attach the
header to the SSE request and message requests.

## `@mention` peer routing

When CONNECT is active, the Monolith chat UI can route to registered peers:

```text
@localagent summarize this
@localagent @monolith compare your answers
```

A message without an `@` mention follows the normal Monolith model path.
Peer names and URLs are managed through **Connections -> External Peers** or
the optional `url` supplied to `/join`.

## Peer `/chat` protocol

An external peer exposes:

```http
POST /chat
Content-Type: application/json

{"message": "...", "agent": "monolith"}
```

and returns:

```json
{"ok": true, "response": "..."}
```

The bundled Claude peer helper defaults to loopback and requires a non-empty
`MONOLITH_PEER_TOKEN` for a non-loopback bind. Monolith sends
`MONOLITH_AGENT_TOKEN` to its AgentServer and `MONOLITH_PEER_TOKEN` to a peer
where the corresponding helper path supports them. Independently verify
authentication for any other peer implementation.

## Shutdown and data handling

Select **Stop** in Connections or close Monolith to stop the embedded server.
Participants are cleared when the server stops. Persistent chats, traces,
configuration, and other runtime data are not deleted.

Before sharing CONNECT logs or reproductions, remove tokens, prompts,
responses, endpoint URLs, peer identities, model details, paths, and tool
results. Follow [`SECURITY.md`](../../SECURITY.md) for vulnerabilities.
