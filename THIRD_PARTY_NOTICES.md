# Third-party components and license boundary

Monolith's root [MIT License](LICENSE) covers first-party material the named
copyright holder has the right to license. It does not relicense dependencies,
model weights, datasets, external services, external Monoline code, or other
third-party material.

This file records the declared direct integration boundary for `v1.0.0`. It is
not a lock file, complete transitive dependency report, SBOM, or legal opinion.
The exact installed version's upstream metadata and license text control.
See the detailed [direct dependency license inventory](docs/release/DEPENDENCY_LICENSE_INVENTORY.md)
for declared ranges, observed base versions, primary sources, and release gates.

## Declared Python packages

`pyproject.toml` is authoritative.

| Group | Direct packages | Distribution boundary |
|---|---|---|
| Base | PySide6, Pydantic, PyYAML, Python-Markdown, Pygments | Installed from package indexes into the user's environment; not vendored into this repository |
| Rich files | PyMuPDF, pypdf, python-docx, openpyxl, Pillow | Optional format readers; document contents retain their own rights |
| Local LLM | llama-cpp-python | Optional native binding/runtime; GGUF model terms are separate |
| Vision | Torch, Diffusers, Transformers, Accelerate, Pillow | Optional ML software; checkpoints, tokenizers, LoRAs, and generated outputs have separate terms |
| Audio | Torch, torchaudio, AudioCraft | Optional ML/audio software; model weights, codecs, samples, and outputs have separate terms |
| Matrix | matrix-nio, aiohttp | Optional network client libraries; Matrix server/content terms remain separate |
| Development | pytest | Test-only dependency |

Minimum/maximum ranges in `pyproject.toml` are compatibility policy, not a
license inventory. Before redistributing a packaged executable or vendored
environment, generate an exact dependency tree and preserve all notices
required by the resolved packages and native components.

## Models, datasets, and generated output

No GGUF, diffusion checkpoint, AudioCraft/MusicGen weight, embedding model,
dataset, or tokenizer is licensed by Monolith merely because the application
can load it.

Users are responsible for:

- obtaining models from an authorized source;
- reviewing each model/dataset license and acceptable-use terms;
- preserving required attribution or notices;
- determining whether a model permits the intended use and output; and
- complying with provider terms for remote inference.

Monolith does not guarantee that generated output is free of third-party rights
or usage restrictions.

## External Monoline integration

Monoline is loaded from a separate checkout configured by
`MONOLITH_MONOLINE_ROOT`. It is not bundled by this repository and is not
automatically covered by Monolith's MIT license. Review the exact external
checkout's license, version, dependencies, and workflow assets independently.

## External services

OpenAI-compatible endpoints, Tavily, Matrix homeservers, external peers,
webhooks, model hosts, and package indexes are independently operated services.
Their software/content, privacy policies, terms, billing, and availability are
outside the Monolith license and support boundary.

## Repository assets

The release source includes small `.monoline` Workshop seed files under
`assets/workshop_seeds/`. They are treated as first-party Monolith release
assets under the root license. Their embedded prompts are part of those assets.

The public README intentionally includes no externally hosted screenshots or
logo assets in this release. Future images, icons, fonts, audio, sample data,
or generated media must be added to [`PROVENANCE.md`](PROVENANCE.md) with source
and permission before inclusion.

## Downstream redistributors

If you redistribute Monolith with dependencies, a Python environment, native
libraries, model weights, screenshots, sample content, or an installer:

1. resolve exact versions and transitive/native components;
2. collect each controlling license and notice;
3. confirm model/data redistribution rights separately;
4. preserve Monolith's MIT copyright/license notice;
5. document modifications; and
6. do not imply that this file is a complete legal review of the bundle.
