# Monolith documentation

Start with the root [`README.md`](../README.md) for the product boundary and
supported quick start.

## User and operator guides

- [`INSTALLATION.md`](INSTALLATION.md) - Windows/Python support, base install,
  optional dependency profiles, model setup, repair, and troubleshooting.
- [`CONFIGURATION.md`](CONFIGURATION.md) - runtime root, model settings,
  credentials, feature flags, data isolation, backup, and redaction.
- [`agent-guides/CONNECT.md`](agent-guides/CONNECT.md) - loopback CONNECT and MCP
  usage, authentication, and unsupported remote-exposure boundary.

## Technical orientation

- [`ARCHITECTURE.md`](ARCHITECTURE.md) - live startup/turn paths, component map,
  persistence, experimental systems, partial surfaces, and trust boundaries.
- [`KNOWN_ISSUES.md`](../KNOWN_ISSUES.md) - exact test baseline, security limits,
  portability gaps, and incomplete features.

## Project and release policy

- [`SECURITY.md`](../SECURITY.md) - private vulnerability reporting and safe
  handling of logs/configuration.
- [`SUPPORT.md`](../SUPPORT.md) - supported scope and response
  expectations.
- [`CONTRIBUTING.md`](../CONTRIBUTING.md) - development setup, state isolation,
  tests, change scope, and pull-request requirements.
- [`CHANGELOG.md`](../CHANGELOG.md) - public release history.
- [`PROVENANCE.md`](../PROVENANCE.md) - source, asset, and AI-assistance
  provenance boundary.
- [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) - dependency, model, and
  external-integration license boundary.
- [`release/DEPENDENCY_LICENSE_INVENTORY.md`](release/DEPENDENCY_LICENSE_INVENTORY.md) -
  direct dependency ranges, observed base versions, upstream licenses, and
  binary/model/service release gates.

## Internal working material

Release-preparation evidence and retained design, research, report, snapshot,
and implementation-plan directories are maintainer working material. They are
not part of this public navigation surface. Their presence does not mean a
described feature is complete or supported.
