# Dependency profiles

`pyproject.toml` is the dependency source of truth. These files are thin
compatibility wrappers for tooling that only accepts a requirements file.
Run them from the repository root.

- `base.txt` - desktop UI and OpenAI-compatible API chat foundation
- `files.txt` - richer PDF, DOCX, XLSX, and image inspection
- `local-llm.txt` - in-process llama.cpp/GGUF support
- `matrix.txt` - optional Matrix bridge
- `dev.txt` - test runner

Vision and audio are intentionally not represented by one-click requirements
files. Select the correct Torch build for the machine first, then follow
[`docs/INSTALLATION.md`](../docs/INSTALLATION.md).
