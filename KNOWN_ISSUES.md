# Known issues

This ledger describes the `v1.0.0` source release verified on 2026-07-14. It
separates confirmed defects and limitations from planned work. An item being
listed does not mean it blocks the documented v1 release boundary.

## Test baseline

Environment: Windows 25H2 build 26200.8655, CPython 3.11.9, pytest 9.0.2.

```text
3173 collected
3113 passed
0 failed
59 skipped
1 xfailed
6 warnings
```

The exact staged source tree was archived into a fresh cache-free directory and
completed in 200.67 seconds. The full-suite CI job is required. This green
result does not exercise unsupported platforms, every optional model/GPU/media
stack, or production multi-user deployment.

Skip breakdown:

- 55 tests in external-Monoline integration modules skip when
  `MONOLITH_MONOLINE_ROOT` is unavailable;
- 2 Second Opinion card tests skip for the same missing external integration;
- 1 Bearing ship-gate test is an explicit non-default validation profile; and
- 1 filesystem-security test skips because symlink creation is unavailable in
  the exercised Windows environment.

A pre-v1 release-preparation run with the maintainer's external Monoline
checkout configured collected 3,169 tests: 3,166 passed, 2 skipped, 1 xfailed,
and 0 failed. The four tests added for v1 verify release-version consistency
and the CONNECT MCP version handshake; they do not exercise Monoline. The
public-clone result above remains the v1 release baseline because it does not
rely on files outside the repository.

### Resolved release-audit failures

The earlier audit baseline had nine failures. Before this release was sealed,
the release pass corrected fresh-database initialization in the canonical-log
test, updated current Bearing/fault/tool-discovery contracts, removed a literal
checkout-name assumption, restored a monkeypatched review-act global, and made
the turn-retention fixture use a controlled clock. No failure was hidden with a
new `xfail` marker or a disabled test.

## Security and network limitations

- CONNECT is an experimental local integration, not a remotely hardened
  service.
- CONNECT defaults to loopback; all-interface startup requires a non-empty
  token and authenticated routes, including `/events`, use that token.
- CONNECT still has no TLS, role-based authorization, tenant isolation, or
  supported public deployment architecture. Remote exposure is unsupported.
- File and command tools run with the Windows permissions of the Monolith
  process. Tool governance is not an OS sandbox.
- General provider keys entered in the UI are stored in plaintext local
  configuration.
- Traces, archives, notes, screenshots, and SQLite state can contain private
  prompts, model output, paths, endpoints, and tool metadata.

See [`SECURITY.md`](SECURITY.md) before enabling network or high-impact tools.

## Installation and platform limitations

- Only Windows with CPython 3.11 is tested for this release.
- Linux/macOS are unverified and have no supported root launcher.
- The base environment does not include local GGUF, rich file readers, image,
  audio, Matrix, or development extras.
- llama-cpp-python, Torch, Diffusers, AudioCraft, and torchaudio compatibility
  depends on Python, native toolchain, GPU/CPU backend, and selected versions.
- Vision/audio one-click installation is intentionally not attempted.
- No model weights, standalone executable, code signing, updater, or lock file
  for every hardware stack is provided.

## Feature limitations

### Monoline Workshop

Monoline is not bundled. The bridge requires `MONOLITH_MONOLINE_ROOT` pointing
to a compatible external checkout. A process-isolated worker path is explicitly
unfinished, and path portability still needs work.

### MonoNote

Markdown storage, provenance/index operations, read/list tools, and search
integration exist. A dedicated mounted editor/workspace and the full intended
chat-to-note experience are not included.

### Soundtrap

Headless loop/project and mix backend pieces exist. A complete mounted studio,
transport, editing, and polished export workflow do not.

### UI v2

Bootstrap contains a feature-flagged `ui_v2.app` import, but no `ui_v2` package
ships in this release. Do not enable or advertise that hook.

### Stats

The Stats page is routed through the icon rail and is not entirely missing, but
several secondary widgets are placeholders.

### Producer adapters and probe tool

Some CONNECT/local producer adapters identify themselves as migration stubs,
and the `codex-probe-tool` executor remains a TODO. They are not supported
headline features.

### Experimental cognition

MonoSearch, ACU/Acatalepsy, Bearing, Observer, Monothink, curiosity, identity
emergence, planning, prediction, and self-maintenance have real code at
different maturity levels. Their existence does not guarantee:

- that stored memory is retrieved into every next turn;
- verified truth or model correctness;
- autonomous loop closure;
- stable schemas or prompt behavior; or
- safe unattended self-modification.

The public launcher leaves autonomous/identity-mutating flags off.

## Data and upgrade limitations

- Different Monolith checkouts share `%APPDATA%\Monolith` unless
  `MONOLITH_ROOT` is set separately.
- Data/schema migrations may be incomplete or one-way. Back up the
  complete runtime root before switching versions.
- A venv rebuild does not reset user state.
- Configuration and traces are not encrypted by the application.
- The default turn-trace retention behavior should not be treated as a secure
  deletion guarantee.

## Reporting another issue

Use the structured GitHub bug form with the exact commit, Python/Windows
versions, backend, flags, isolated-state result, and sanitized reproduction.
Do not include credentials or private content. Report vulnerabilities privately
under [`SECURITY.md`](SECURITY.md).
