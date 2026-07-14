# Changelog

Notable public-facing changes are recorded here. Published releases use
`MAJOR.MINOR.PATCH` versioning for the documented core source boundary.
Surfaces explicitly labeled experimental, partial, external, or unsupported
remain outside compatibility promises unless stated otherwise.

## Unreleased

No changes recorded yet.

## 1.0.0 - 2026-07-14

First public v1 source release of the Monolith desktop
workstation: chat and model runtime, tool/skill execution, persistence,
optional media, CONNECT/Matrix, Workshop/Monoline integration, and experimental
cognition substrates.

### Added

- Public release manifest and verification workflow.
- Structured GitHub issue forms, pull-request template, and Windows CI.
- Public installation, configuration, architecture, security, support,
  provenance, third-party, and known-issues documentation.

### Changed

- Root `README.md` now describes the current capability/maturity boundary,
  tested platform, data/network behavior, and exact test status.
- `pyproject.toml` is the dependency source of truth, with separate base,
  files, local-LLM, vision, audio, Matrix, and developer groups.
- `install.bat` now validates Python 3.11, uses the environment interpreter,
  consumes package metadata, stops on failure, and runs a smoke check.
- `start.bat` now anchors to its own directory and uses a conservative public
  profile without autonomous, identity-mutation, or network flags.

### Security

- CONNECT all-interface startup is blocked unless
  `MONOLITH_AGENT_TOKEN` is non-empty.
- CONNECT `/events` applies the same authentication boundary as other protected
  routes.
- Public guidance now states that CONNECT has no TLS/RBAC and remains
  unsupported for remote exposure.

### Verification and known issues

- The cache-free v1 source tree collected 3,173 tests: 3,113 passed, 59
  skipped, 1 xfailed, and 0 failed. Most skips are the separately installed
  Monoline integration. See [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md).
- Linux/macOS and heavy native ML dependency combinations remain unverified.
- MonoNote and Soundtrap are backend-first/partial; Monoline remains an
  experimental external integration.

`v1.0.0` establishes the stable public core release line. It does not imply
production multi-user hosting, stable experimental schemas, complete optional
workspaces, or closure of every experimental feedback loop.
