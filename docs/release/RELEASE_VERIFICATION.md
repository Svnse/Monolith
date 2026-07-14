# Release verification

Verification date: **2026-07-14**

Release: **Monolith v1.0.0**

This is the evidence record for the sanitized, source-only v1 GitHub release.
Acceptance is limited to the documented release boundary; it is not a claim
that every experimental loop or optional integration is complete.

## Release identity

| Field | Verified result |
|---|---|
| Public branch | `release/public-main` |
| Public tag | `v1.0.0` |
| Version metadata | `monolith-workstation==1.0.0` |
| Supported baseline | Windows with 64-bit CPython 3.11 |
| History model | One parentless public root commit; private development ancestry excluded |
| Distribution model | Source only; no Python environment, model, dataset, credential, database, or generated user state bundled |

The tag identifies the exact final release commit. A commit hash is not embedded
in this file because doing so would change the commit it is meant to identify.

## Release-tree construction

The release tree contains exactly 880 tracked files and approximately 6.49 MB
of source and documentation. It is assembled from the Git index, not from an
unfiltered working-directory copy. Ignored maintainer state remains local and
is not reachable from the public root commit.

The executable/test tree accepted by the authoritative v1 full-suite and clean
install gates was:

```text
59fb8606d361195d0bf469249a0878e84dc28442
```

Changes after that tree are confined to recording the final v1 gate results in
release documentation. No runtime, package, dependency, test, or workflow file
changed after the authoritative gates.

## Verification gates

| Gate | Result |
|---|---|
| Release payload inventory | PASS - public manifest applied; excluded/private paths absent from the staged payload |
| High-confidence credential scan | PASS - no credential value or private-key material found; one intentional credential-shaped URL remains in a security test fixture |
| Maintainer path/name scan | PASS - no maintainer absolute path found; personal attribution remains only where intentional in licensing/provenance |
| Private runtime-state scan | PASS - no databases, journals, logs, prompt backups, local agent state, model weights, or generated corpora included |
| Filesystem safety | PASS - no symlinks, reparse points, gitlinks, nested repositories, unsafe Windows names, case collisions, or file over 1 MiB |
| Python static check | PASS - all 729 tracked Python files parsed; no unresolved local-import target found by the release audit |
| Structured-data parse | PASS - tracked JSON, YAML, TOML, and Monoline files parsed by their applicable release-audit readers |
| Focused release tests | PASS - four v1 runtime/package/EventLedger/MCP-version consistency tests passed; security, portability, and changed-area tests remain covered by the full suite |
| Full isolated pytest suite | PASS - 3,173 collected: 3,113 passed, 59 skipped, 1 expected failure, 0 failed; 6 warnings; 200.67 s |
| Fresh base install | PASS - isolated Python 3.11.9 environment, editable install, `monolith-workstation==1.0.0` assertion, and `pip check` |
| Documented base imports | PASS - `PySide6`, `pydantic`, `yaml`, `markdown`, and `pygments` imported from the fresh environment |
| Headless startup smoke | PASS - offscreen `bootstrap.main()` created one top-level window, closed cleanly, and returned `0` |
| Markdown relative-link check | PASS - no missing local target in the public Markdown corpus |
| Git patch hygiene | PASS - staged diff passed Git whitespace/error checking |

The full suite ran against an archive materialized from the exact staged Git
tree, with bytecode and test caches absent before and after the run. The process
used Python 3.11.9 and pytest 9.0.2. Wall-clock time was 203.148 seconds.

### Skips and expected failure

The 59 skips are disclosed rather than counted as passes:

- 55 tests require a separately installed Monoline checkout;
- 2 second-opinion tests require that same external integration;
- 1 real model A/B ship gate is opt-in; and
- 1 symbolic-link test is unavailable on the tested Windows configuration.

The single expected failure records an unresolved Qwen 3.5 model-family
registry decision. It is retained as visible debt instead of being silently
disabled.

The six warnings were one Torch/PyNVML future warning, four Qt signal-disconnect
runtime warnings, and one deprecated Qt mouse-event-constructor warning. None
caused a test failure, but they remain maintenance work.

## Clean-environment evidence

The base install was performed in a new virtual environment with system site
packages disabled. It resolved these direct packages:

| Package | Resolved version |
|---|---:|
| PySide6 | 6.11.1 |
| pydantic | 2.13.4 |
| PyYAML | 6.0.3 |
| Markdown | 3.10.2 |
| Pygments | 2.20.0 |

The install, distribution-version assertion, `pip check`, import smoke, and
offscreen application bootstrap all completed successfully. Pytest was absent
from the base environment as intended; installing the declared `dev` extra and
running the three focused version tests produced three passes. The offscreen
bootstrap created exactly one visible `MonolithUI` and returned `0`.

## What this gate did not prove

- No real model inference was exercised, and no provider credential or model is
  bundled.
- Optional file, local-LLM, vision, audio, and Matrix dependency profiles were
  not installed as part of the base clean-room gate.
- External Monoline tests require `MONOLITH_MONOLINE_ROOT`; their absence is
  represented by explicit skips.
- GPU/CUDA/ROCm variants, model licenses, native redistribution payloads, and
  frozen installers require separate artifact-specific verification.
- The GitHub-hosted rendering, issue settings, branch protection, and private
  vulnerability-reporting switch cannot be verified until a remote repository
  exists.

See [`KNOWN_ISSUES.md`](../../KNOWN_ISSUES.md) for the user-facing limitation
record and [`DEPENDENCY_LICENSE_INVENTORY.md`](DEPENDENCY_LICENSE_INVENTORY.md)
for third-party distribution boundaries.

## Decision

**GO for the source-only GitHub release, with the limitations above disclosed.**

Publish only `release/public-main` as the remote `main` branch and then publish
`v1.0.0`. Do not push all local branches or mirror this local repository: the
other branches retain private development ancestry by design.
