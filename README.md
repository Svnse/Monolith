# Monolith

Monolith is a Windows-first, local-first desktop AI workstation for running
local or OpenAI-compatible language models with tools, files, media engines,
and experimental agent workflows in one PySide6 application.

**Release status:** `v1.0.0` - the first stable public source release. The v1
compatibility boundary is the documented Windows/Python base profile. Surfaces
explicitly labeled experimental, partial, external, or unsupported remain
outside that promise, and their flags, formats, or persisted schemas may change
without migration. Windows and Python 3.11 are the only tested base
configuration for this release.

> Monolith is local-first, not network-free. API model backends, web search,
> Matrix, CONNECT peers, model downloads, and other integrations can transmit
> prompts or data to external systems. File and command tools can change the
> local machine when used.

## What works now

| Surface | Status | Requirement or boundary |
|---|---|---|
| Desktop chat, streamed responses, sessions, and history | Core | Configure a model backend; model files are not bundled |
| OpenAI-compatible API models | Core | Endpoint, model ID, and a provider key when required |
| Local GGUF models | Optional | Compatible GGUF plus llama.cpp runtime; install the `local-llm` profile for in-process support |
| Model-callable file, search, note, trace, command, and test tools | Core/advanced | Tools have different permission and side-effect levels; review model-requested actions |
| Rich PDF, DOCX, XLSX, and image inspection | Optional | Install the `files` profile; basic/fallback behavior varies by format |
| Stable Diffusion image generation | Optional experimental | Separate Torch/diffusion install, compatible model, and suitable hardware |
| AudioCraft/MusicGen audio generation | Optional experimental | Separate Torch/audio install, compatible model, and suitable hardware |
| MonoSearch, ACU/Acatalepsy, Bearing, subagents, and self-maintenance | Experimental | Behavior, flags, schemas, and prompt effects may change |
| CONNECT and Matrix agent integration | Experimental network feature | Keep CONNECT on loopback; remote exposure is unsupported |
| Monoline Workshop workflows | Experimental external integration | Currently expects a separately installed Monoline checkout |
| MonoNote and Soundtrap | Partial/headless | Backends and tools exist; dedicated workspaces are not complete |

UI v2, a finished MonoNote editor, a finished Soundtrap studio, and every
Monoline execution path are **not** shipped as completed features.

## Supported configuration

| Item | Release boundary |
|---|---|
| Operating system | Windows 25H2 build 26200.8655 was used for release preparation; other Windows builds are unverified |
| Python | CPython 3.11.x; release preparation used 3.11.9 |
| Launch path | Root `install.bat`, then `start.bat` |
| Base model path | OpenAI-compatible HTTP endpoint; no provider is bundled or preconfigured |
| Local inference | Optional GGUF/llama.cpp path; compatibility depends on the selected model and runtime build |
| GPU | Not required for API-backed chat; local inference and media requirements depend on the selected model |
| Linux/macOS | Source may be portable in parts, but these platforms are not supported or tested for this release |

## Quick start

Clone or download the repository, open **Command Prompt** in its root, and run:

```bat
install.bat
start.bat
```

`install.bat` creates `venv`, installs the base dependencies from
`pyproject.toml`, and runs an import smoke check. It stops on failure. The
base profile does not install GGUF, file-reader, image, audio, Matrix, or
development extras.

After launch:

1. Open the **Config** model panel.
2. Choose **Model (API)** for an OpenAI-compatible service, or one of the GGUF
   modes for local inference.
3. For API mode, enter the API base URL, model ID, and key if that endpoint
   requires one. For GGUF mode, select a compatible local model and runtime.
4. Load the model and send a prompt.

Credentials entered in the current UI are persisted locally in configuration;
they are not encrypted by Monolith. Use a restricted key and protect the user
profile. See [configuration and data behavior](docs/CONFIGURATION.md).

For optional profiles, native build notes, clean-environment repair, and
developer setup, see [installation](docs/INSTALLATION.md).

## Default launch profile

The public `start.bat` deliberately forces no experimental `MONOLITH_*` flags.
It launches the source defaults. Features such as autonomous curiosity,
identity emergence, self-maintenance, CONNECT autostart, and external auditing
must be enabled deliberately and should not be treated as stable defaults.

## Data, privacy, and network behavior

On Windows, Monolith stores runtime state under `%APPDATA%\Monolith` by default:

- `config\` - model/provider settings and feature configuration;
- `chats\` - conversation archives;
- `logs\` - traces and SQLite state;
- `notes\` - MonoNote Markdown data;
- `artifacts\` - generated or tool-produced artifacts; and
- `addons\` - addon settings and manifests.

Set an absolute `MONOLITH_ROOT` to isolate a development or test profile. Two
Monolith checkouts otherwise share the same default state directory.
Back up data before changing versions. Removing `venv` does not remove user
state.

Local-model chat can stay local only when every selected tool and integration
also stays local. The following can use the network or disclose data:

- OpenAI-compatible API backends;
- Tavily-backed web search and optional grounding;
- Matrix and external CONNECT peers;
- webhook and external-agent integrations;
- model/package downloads; and
- any command or tool explicitly directed to a network client.

CONNECT defaults to `127.0.0.1`. It does not provide TLS or role-based access
control, and remote exposure remains unsupported in this release. Read the
[CONNECT guide](docs/agent-guides/CONNECT.md) and [security policy](SECURITY.md)
before enabling it.

## Architecture at a glance

```text
PySide6 shell
    -> chat and model adapter
    -> kernel/turn pipeline
    -> tools, skills, and optional workflows
    -> local configuration, archives, traces, and memory stores
```

Optional media engines, external peers, and Monoline sit beside that main path;
not every engine uses a separate process. See the
[architecture overview](docs/ARCHITECTURE.md).

## Test status and known limitations

The cache-free `v1.0.0` source-tree run on Python 3.11.9 collected 3,173 tests:

- 3,113 passed;
- 0 failed;
- 59 skipped;
- 1 xfailed; and
- 6 warnings.

The green suite is evidence for the tested release tree; it does not validate
unsupported platforms, every optional native stack, or production multi-user
deployment. Most skips are Monoline integration tests because that external
repository is intentionally not bundled. See
[known issues](KNOWN_ISSUES.md) for the exact test boundary and current
security, installation, and feature limitations.

Contributor setup and commands are in [CONTRIBUTING.md](CONTRIBUTING.md).
GitHub CI runs a required base smoke job and the complete test suite.

## Documentation

- [Documentation map](docs/README.md)
- [Installation and optional profiles](docs/INSTALLATION.md)
- [Configuration, secrets, and state](docs/CONFIGURATION.md)
- [Architecture overview](docs/ARCHITECTURE.md)
- [Known issues](KNOWN_ISSUES.md)
- [Security policy](SECURITY.md)
- [Support expectations](SUPPORT.md)
- [Release history](CHANGELOG.md)
- [Provenance](PROVENANCE.md) and [third-party boundary](THIRD_PARTY_NOTICES.md)

## License

First-party Monolith source selected for this release is provided under the
[MIT License](LICENSE), copyright 2026 Eryndel (Erick Ascano-Marin).
Dependencies, external Monoline code, model weights, datasets, provider
services, and generated outputs may have separate terms. No model weights are
bundled with this source release.

## Support and contributions

This is a maintainer-led personal project with best-effort support.
Reproducible bug reports and narrowly scoped pull requests are welcome. The v1
core compatibility boundary and the absence of response-time guarantees are
defined in [SUPPORT.md](SUPPORT.md). Read
[CONTRIBUTING.md](CONTRIBUTING.md), and [SECURITY.md](SECURITY.md) before filing.

Built by [Eryndel](https://eryndel.us).
