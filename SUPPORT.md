# Support

Monolith `v1.0.0` is the first public major source release of a maintainer-led
personal project. Support is best-effort; there is no uptime, response-time,
migration, or model-quality guarantee, and compatibility guarantees exclude
surfaces explicitly labeled experimental, partial, external, or unsupported.

Within the `1.x` line, the documented Windows/Python base installation,
launcher, and core desktop path will not be intentionally broken without a
documented migration or a new major version. This commitment does not stabilize
internal modules, experimental feature flags, optional model stacks, or their
persisted schemas.

## Supported scope

The narrow supported setup is:

- Windows;
- CPython 3.11.x;
- installation through root `install.bat`;
- launch through root `start.bat`; and
- the base desktop UI with a user-configured OpenAI-compatible endpoint.

Local GGUF, document readers, image/audio generation, Matrix, CONNECT,
Monoline, cognition systems, subagents, and self-maintenance are optional or
experimental. Help may be limited to reproducing whether the base application
or an optional boundary is responsible.

Linux/macOS, public CONNECT hosting, production multi-user use, arbitrary model
stacks, and private forks are not supported in this release.

## Before asking for help

1. Read [`README.md`](README.md),
   [`docs/INSTALLATION.md`](docs/INSTALLATION.md), and
   [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md).
2. Confirm Python with `venv\Scripts\python.exe --version`.
3. Reproduce with the public `start.bat` and no extra `MONOLITH_*` flags.
4. Reproduce with an isolated `MONOLITH_ROOT` when persistent state may matter.
5. Record the exact commit, backend type, optional profiles, and error text.
6. Remove credentials, prompts, personal paths, and private data from evidence.

Use the structured GitHub bug form for reproducible non-security defects.
Feature requests are welcome as problem statements, but acceptance or scheduling
is not promised.

Security vulnerabilities must follow [`SECURITY.md`](SECURITY.md) and must not
be filed publicly.

## Useful bug evidence

- commit SHA and Monolith version;
- Windows build and Python version;
- fresh versus upgraded checkout;
- install command/profile;
- API versus local GGUF backend;
- enabled feature flags;
- minimal steps and frequency;
- expected and actual behavior; and
- a short sanitized traceback or log excerpt.

Do not upload model weights, full databases, complete runtime roots, provider
keys, access tokens, private conversations, or copyrighted third-party content.
