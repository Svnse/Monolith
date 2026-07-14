# Configuration, secrets, and state

This document describes the public `v1.0.0` source profile. Monolith has many
development flags; their presence in source is not a support promise. The root
`start.bat` intentionally sets none of the experimental or network flags.

## Runtime root

On Windows, the default root is:

```text
%APPDATA%\Monolith
```

The application creates these primary directories:

| Path below the root | Purpose | May contain sensitive data |
|---|---|---|
| `config\` | YAML/JSON settings, provider configuration, identities, feature state | Yes - keys, endpoints, model paths, preferences |
| `chats\` | Conversation archives | Yes - prompts and model output |
| `logs\` | Traces, event records, SQLite databases | Yes - prompt previews, tool/runtime metadata, faults |
| `notes\` | Markdown notes and MonoNote data | Yes - user-authored content |
| `artifacts\` | Tool and generated artifacts | Yes - generated or copied content |
| `addons\` | Addon configuration and manifest state | Possibly |

The virtual environment and runtime root are independent. Deleting `venv`
does not reset Monolith data.

### Isolate a checkout

Set `MONOLITH_ROOT` to an absolute directory under the current user's profile:

```bat
set "MONOLITH_ROOT=%USERPROFILE%\.monolith-dev"
start.bat
```

PowerShell equivalent:

```powershell
$env:MONOLITH_ROOT = "$env:USERPROFILE\.monolith-dev"
.\start.bat
```

Relative paths, invalid paths, and paths outside the user's home/AppData
anchors fall back to the normal root. `MONOLITH_ALLOW_UNANCHORED_ROOT=1`
disables that anchor check; it is intended for controlled automation such as a
disposable CI runner, not normal desktop use.

Without an override, two Monolith checkouts share
`%APPDATA%\Monolith`. Use separate roots when comparing builds.

## Main configuration

The canonical configuration file is:

```text
%MONOLITH_ROOT%\config\config.yaml
```

It is created and migrated on first load. A reduced shape looks like this:

```yaml
version: 1
llm:
  backend: openai
  api_provider: openai
  api_base: http://localhost:8000/v1
  api_model: model-id
  api_key: ""
  gguf_path: null
  temp: 1.0
  top_p: 0.95
  ctx_limit: 0
theme:
  current: midnight
vision:
  model_path: ""
  model_root: ""
audio:
  model_id: facebook/musicgen-small
  duration: 5.0
  sample_rate: 32000
```

Prefer the UI for fields it exposes. Stop Monolith before hand-editing YAML,
keep indentation valid, and back up the file first. Invalid known values can
cause the loader to fall back to defaults.

Older `llm_config.json`, `theme.json`, `vision_config.json`, and
`audiogen_config.json` files may be migrated into the YAML configuration. Do
not commit any of these local files.

### Named overlays

Set `MONOLITH_ENV` to merge a named overlay after the base file:

```bat
set "MONOLITH_ENV=development"
start.bat
```

This reads:

```text
%MONOLITH_ROOT%\config\config.development.yaml
```

Nested mappings are merged. The overlay is useful for local profiles, but it
is not a secrets manager and should remain outside the repository.

## Model backends

The Config panel exposes three engine choices:

| UI choice | Configuration | Boundary |
|---|---|---|
| `Model (API)` | `api_base`, `api_model`, optional `api_key` | OpenAI-compatible HTTP; prompts leave the process and may leave the machine |
| `GGUF (API)` | GGUF path plus a local server endpoint selected/started by Monolith | Requires a compatible external/local llama server runtime |
| `GGUF (llama.cpp)` | `gguf_path` | Requires the optional llama-cpp-python stack |

Models and providers are not bundled. `ctx_limit=0` means the runtime should
resolve a ceiling; a positive user value is treated as an override where the
backend path supports it.

### API credentials

The current UI writes the general model `api_key` into `config.yaml` as
plaintext. Monolith does not provide encrypted credential storage in this
release.

- use the least-privileged key available;
- restrict provider-side spending and scope;
- protect the Windows account and runtime directory;
- never commit or attach the configuration file;
- redact endpoints, keys, prompts, and model paths from screenshots/logs; and
- rotate a key immediately if it was placed in a launcher, issue, patch, or
  public repository.

No general `OPENAI_API_KEY`-style environment override was confirmed in the
observed model configuration path. Do not assume setting one prevents local
credential persistence.

## Integration environment variables

Some integrations read credentials or machine-specific runtime paths directly
from environment variables:

| Variable | Used by | Guidance |
|---|---|---|
| `TAVILY_API_KEY` | Web search and optional Acatalepsy grounding | Prefer this over `config\tavily.json`; do not commit either value |
| `MATRIX_PASSWORD` | Optional Matrix login | Set only in the launching process; protect shell history and logs |
| `MONOLITH_AGENT_TOKEN` | CONNECT HTTP authentication | Use a high-entropy value; required for any all-interface bind |
| `MONOLITH_PEER_TOKEN` | External peer requests and bundled peer-helper authentication | Use a separate restricted value; required when the bundled helper binds beyond loopback |
| `MONOLITH_LLAMA_SERVER` | Local GGUF API mode | Absolute path to `llama-server`; otherwise the user-home build paths and `PATH` are checked |
| `MONOLITH_LLAMA_PY` | Python fallback for local GGUF API mode | Python executable containing `llama_cpp.server`; defaults to Monolith's active interpreter |

Environment variables reduce accidental repository commits but are still
visible to the current process and may be exposed by debugging tools.

## Conservative public launch profile

`start.bat` launches `venv\Scripts\python.exe main.py` and forces no feature
flags. This is intentional. The previous development launcher profile is not a
public default.

The following flags are opt-in experiments or network behavior:

| Variable | Effect | Public recommendation |
|---|---|---|
| `MONOLITH_AGENT_AUTOSTART=1` | Starts CONNECT with the UI | Leave off; start manually on loopback when needed |
| `MONOLITH_ACATALEPSY_AUDITOR_V1=1` | Starts the optional memory auditor | Experimental; may use configured model resources |
| `MONOLITH_ACATALEPSY_VERIFIER_V1=1` | Starts external grounding/verifier work | Experimental and network-capable; requires Tavily configuration |
| `MONOLITH_CURIOSITY_CAPTURE_V1=1` | Captures private curiosity blocks into self-memory | Experimental persistent-state mutation |
| `MONOLITH_CURIOSITY_V1=1` | Enables curiosity detection heartbeat | Experimental; default off |
| `MONOLITH_IDENTITY_EMERGENCE_V1=1` | Enables identity-emergence heartbeat | Experimental; default off |
| `MONOLITH_IDENTITY_ACU_SELF_MUTATE_V1=1` | Allows stable identity ACUs to rewrite the generated emergent block | Experimental identity mutation; leave off by default |
| `MONOLITH_USER_ALIASES` | Comma- or space-separated user aliases filtered from user-subject identity claims | Optional identity-projection hygiene; no maintainer-specific names are built in |
| `MONOLITH_SELF_MAINT_TRIGGER_V1=1` | Starts the self-maintenance trigger daemon | Experimental; leave off |
| `MONOLITH_SELF_MAINT_V1=1` | Permits self-maintenance apply actions | High-impact experimental path; do not enable unattended |
| `MONOLITH_FAULT_TELEMETRY_V1=1` | Injects prior verifier/fault feedback into a later turn | Diagnostic experiment; changes prompt content |

`MONOLITH_TURN_TRACE_V1` is different: its code default is on. Set it to `0` to
disable turn-trace writes. `MONOLITH_TURN_TRACE_TTL_DAYS` controls retention.
Inspect the active version before relying on retention for deletion guarantees.

This table is not an exhaustive environment-variable reference. Undocumented
flags are development surfaces and may disappear.

## Monoline

Monoline is not bundled. To enable the Workshop integration, point to a
separate checkout:

```bat
set "MONOLITH_MONOLINE_ROOT=C:\path\to\Monoline"
start.bat
```

The path must contain a compatible Monoline source tree. That project has its
own code, dependencies, version, and license. The process-isolated Monoline
worker path is incomplete in this Monolith release.

## CONNECT and Matrix

CONNECT is stopped by default and normally binds to `127.0.0.1:7821` when
started. Keep it on loopback. An all-interface bind requires a non-empty
`MONOLITH_AGENT_TOKEN`, but a token does not add TLS, per-user authorization,
or production hardening. Remote exposure is unsupported.

Matrix requires the `matrix` dependency profile and transmits room events,
messages, and credentials to the configured homeserver. Treat it as a separate
network trust boundary.

See the [CONNECT guide](agent-guides/CONNECT.md) and root
[security policy](../SECURITY.md).

## Backup and reset

1. Close Monolith and any CONNECT/Matrix/worker processes.
2. Record the active `MONOLITH_ROOT`.
3. Copy the complete runtime root to a private backup location.
4. Rename rather than immediately delete the original root.
5. Launch once with a fresh root and verify the intended profile.

SQLite databases and JSONL logs may be open while Monolith is running. Copying
individual files from a live process can produce an inconsistent backup.

## Bug-report redaction

Before sharing a config, log, trace, screenshot, or database, remove:

- API keys, Matrix credentials, CONNECT/peer tokens, and webhook secrets;
- prompts, responses, private notes, and generated artifacts;
- user names and absolute filesystem paths;
- private endpoint and peer URLs;
- model-library names when those are sensitive; and
- local IP addresses or room/user identifiers.

Security-sensitive reports should follow [`SECURITY.md`](../SECURITY.md), not a
public issue.
