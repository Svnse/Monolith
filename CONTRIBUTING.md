# Contributing

Monolith is a maintainer-led personal project. Reproducible issues and focused
pull requests are welcome, but review, acceptance, and response times remain
best-effort.

## Before starting

- Search existing issues and the current [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md).
- For a substantial change, open an issue describing the user problem and the
  smallest affected runtime path before investing in a broad rewrite.
- Never open a public issue for a vulnerability; use
  [`SECURITY.md`](SECURITY.md).
- Keep internal experiments distinct from the supported public profile.

## Development setup

The tested contributor path is Windows with CPython 3.11:

```bat
install.bat dev
```

For non-GPU optional integration tests:

```bat
venv\Scripts\python.exe -m pip install --no-build-isolation -e ".[dev,files,matrix]"
```

See [`docs/INSTALLATION.md`](docs/INSTALLATION.md) before adding local-LLM,
vision, or audio dependencies. Do not change a contributor's Torch build as a
side effect of an unrelated patch.

## Isolate runtime state

Monolith writes outside the checkout. Set an absolute development root so a
manual run cannot mutate another Monolith installation:

```bat
set "MONOLITH_ROOT=%USERPROFILE%\.monolith-contrib"
start.bat
```

Tests that touch paths, SQLite, environment variables, clocks, module globals,
or registries must isolate and restore them. Use temporary directories and
fixtures rather than a contributor's `%APPDATA%\Monolith` data.

## Change discipline

- Make one coherent change per pull request.
- Preserve existing user work in a dirty checkout; do not use destructive Git
  commands to make a patch easier.
- Trace behavior as `trigger -> producer -> consumer -> visible/next-turn
  effect` when changing tools, prompts, memory, workflows, or UI state.
- Distinguish stored state from state actually injected into a model request.
- Keep optional imports lazy and return a useful dependency-missing message.
- Do not hard-code user names, drive letters, Desktop paths, model libraries,
  LAN addresses, ports that already have configuration, or external checkout
  locations.
- New network listeners must default to loopback, authenticate consistently,
  and document their data boundary.
- New file/command tools must document side effects and fit the governance
  level intentionally.
- Feature flags must state their default, effect, persistent writes, network
  use, and removal/rollback path.
- Update public documentation and the changelog when behavior changes.

There is no repository-wide formatter configuration in this repository. Match
the surrounding Python style, use type hints where they clarify contracts, and
avoid formatting unrelated files.

## Tests

Run the smallest relevant tests while developing, then the full suite when the
change can affect shared runtime behavior:

```bat
venv\Scripts\python.exe -m pytest -q tests\path\to\test_file.py
venv\Scripts\python.exe -m pytest -q
```

The `v1.0.0` release tree's full suite was green at verification. Do not mark
failures `xfail`, weaken assertions, or delete tests solely to preserve that
status. State whether each failure is pre-existing, fixed, or new.

Tests that execute the separately installed Monoline integration are skipped
when `MONOLITH_MONOLINE_ROOT` is unavailable. Point that variable at a
compatible checkout to run them; a public-clone run must not silently discover
or depend on a maintainer machine path.

For documentation/packaging changes, also run:

```bat
venv\Scripts\python.exe -m pip check
venv\Scripts\python.exe -c "import bootstrap; print('bootstrap import: OK')"
```

Verify all relative Markdown links and inspect the rendered GitHub result.

## Pull-request requirements

A pull request should explain:

- the user-visible problem and mechanism changed;
- the trigger/producer/consumer path affected;
- exact tests and results;
- persistent-state or schema changes;
- new/changed feature flags and defaults;
- file, command, network, authentication, or privacy impact;
- new dependencies and their license/provenance boundary;
- screenshots for UI changes, scrubbed of private data; and
- work deliberately deferred.

Keep generated caches, virtual environments, databases, logs, model files,
local configuration, credentials, patches, and agent-private state out of the
pull request.

## Dependencies and assets

`pyproject.toml` is the dependency source of truth. Put unconditional startup
imports in the base group and heavy/native capabilities in an optional group.
Explain why a direct dependency is required and how absence is handled.

For copied/adapted source, prompts, screenshots, workflow assets, icons, fonts,
audio, models, or datasets, provide origin, permission/license, and required
notices. Unknown provenance blocks inclusion. Update
[`PROVENANCE.md`](PROVENANCE.md) or
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) when the public boundary
changes.

## License

Unless explicitly agreed otherwise, submitting a contribution means you have
the right to provide it and agree that it may be distributed under the
repository's [MIT License](LICENSE). Dependencies and externally licensed
material retain their own terms.
