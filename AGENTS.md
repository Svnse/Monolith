# Contributor agent guidance

Monolith is a Windows-first PySide6 application. Keep changes small,
evidence-backed, and compatible with the supported public setup documented in
`README.md` and `docs/INSTALLATION.md`. Treat surfaces explicitly labeled
experimental, partial, external, or unsupported according to their documented
boundaries rather than as guarantees of the v1 core profile.

Before changing a shared runtime symbol, inspect its callers and affected tests.
When GitNexus is available, use its impact analysis and change-detection tools;
otherwise use source search and focused tests. Do not assume contributors have
private plugins, sibling repositories, model files, API credentials, or the
maintainer's filesystem layout.

For UI work, read `docs/specs/UI_CONTRACT.md` first. Preserve theme-engine
colors and the one-frame-per-region chrome budget.

Never commit runtime state, credentials, chat history, model weights, local
agent configuration, generated indexes, or reasoning journals. Run focused
tests for the changed subsystem and the isolated full suite before a release
commit.
