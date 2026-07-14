# Public file manifest

This document defines the source boundary for Monolith's first public v1 source
release. The public branch is assembled from an allowlist and has a new root
commit; the private development history is not part of the upload.

## Included

| Area | Purpose |
|---|---|
| Root launch, packaging, license, and community files | Installation, launch, legal boundary, support, and GitHub behavior |
| `addons/`, `core/`, `engine/`, `monokernel/`, `ui/` | Application/runtime source |
| `skills/`, `prompts/` | Model-callable capabilities and shipped prompt scaffolds |
| `tools/`, `scripts/` | Reviewed developer/runtime utilities |
| `assets/` | First-party Workshop seed workflows |
| `tests/` | Reproducible verification suite |
| `docs/specs/` | Current architecture and UI contracts |
| `docs/agent-guides/CONNECT.md` | Supported local CONNECT setup and safety boundary |
| `docs/release/` | Public payload and verification records |

## Deliberately excluded

These paths remain available in the maintainer's private checkout where useful,
but are ignored and absent from the public commit.

| Path/class | Reason |
|---|---|
| `.gitnexus/`, `.pytest_cache/`, `__pycache__/`, bytecode | Generated indexes and caches |
| `.claude/`, `.codex/`, `.superpowers/` | Machine-local agent hooks, permissions, goals, copied skills, and generated state |
| `.claude/worktrees/` | Nested working copies and Git metadata |
| `.chat_*.patch`, `temp_*.py`, logs | Local patch and scratch residue |
| `codex_peer_server.py`, `combiner.py`, `monolith_source.txt` | Unsafe local-only helpers: an unauthenticated LAN peer and a source concatenator without a public-file boundary |
| `prompts/reasoning/*.journal.jsonl` | Runtime reasoning/evolution history containing generated turn material |
| prompt `*.bak.*` files | Pre-migration prompt snapshots superseded by the maintained prompt files |
| `bugs/` | Historical bug ledger whose source reports are not present; current limitations live in `KNOWN_ISSUES.md` |
| `docs/audits/`, `docs/reports/`, `docs/research/`, `docs/snapshots/` | Internal evidence, live-state observations, and source material not intended as product documentation |
| `docs/roadmaps/`, `docs/superpowers/` | Private implementation planning/history; public status is summarized in maintained docs |
| private agent guides and handoffs under `docs/agent-guides/` | Maintainer-specific paths, tools, and live-state instructions |
| `tools/frame_drift/*.json`, `tools/friction_labels.jsonl` | Local calibration/training corpora and derived calibration outputs |
| model weights, generated media, databases, chats, credentials | User-owned runtime data rather than source |

## History boundary

The public release is represented by a root commit created from the reviewed
tree. Existing private branches and inherited development history are retained
locally but are not ancestors of the public branch. This prevents deleted local
state or historical credentials from becoming reachable through the public Git
history.

Push only the reviewed public branch (for example,
`git push origin release/public-main:main`) and its intended release tag. Do
not use `git push --all`, `git push --mirror`, or otherwise publish the retained
private branches.

## Review rule

Any future file added outside the included areas requires an explicit public,
private, or generated-state decision. `.gitignore` is a safety net, not a
substitute for reviewing the exact staged tree before release.
