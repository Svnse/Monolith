# Direct dependency license inventory

- Inventory date: **2026-07-14**
- Release target: **Monolith v1.0.0** (`monolith-workstation` `1.0.0`)
- Declaration source: [`pyproject.toml`](../../pyproject.toml)

This is a release inventory of the direct Python package names
declared by Monolith. It covers 22 declaration entries representing 20 unique
packages: five base packages and 15 additional optional/development packages.
It is not legal advice, a license-compatibility conclusion, an exact lock file,
a transitive dependency report, or an SBOM.

The ranges below are Monolith's compatibility policy. The resolved-version
column reports only artifacts actually observed in the clean-room base-install
pass. Optional and development dependencies were not installed in that pass,
so no exact version is claimed for them. License labels are taken from current
upstream project or package-index metadata on the inventory date. The license
and notice files shipped in the exact resolved artifact control.

## Clean-room base resolution

The base-only release verification used Python `3.11.9` and installed the local
project as `monolith-workstation==1.0.0`. The observed direct packages were:

| Direct package | Declared range | Observed resolved version |
|---|---:|---:|
| PySide6 | `>=6.10,<7` | `6.11.1` |
| pydantic | `>=2.6,<3` | `2.13.4` |
| PyYAML | `>=6.0.1,<7` | `6.0.3` |
| Markdown | `>=3.4,<4` | `3.10.2` |
| Pygments | `>=2.17,<3` | `2.20.0` |

The clean-room install, `pip check`, base imports, and offscreen application
bootstrap passed. That confirms the versions observed; it does not turn this
base-only result into a lock for optional profiles or future installations.

## Base dependencies

| Package and primary sources | Declared range | Clean-room resolved | Upstream-declared license or choice | Redistribution, notice, and native boundary |
|---|---:|---:|---|---|
| **PySide6** ([PyPI](https://pypi.org/project/PySide6/), [Qt source](https://code.qt.io/cgit/pyside/pyside-setup.git/), [Qt licensing](https://www.qt.io/development/qt-framework/qt-licensing)) | `>=6.10,<7` | `6.11.1` | `LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only`; Qt also publishes a commercial-license option | Native Qt boundary. The PyPI package is an alias for `PySide6_Essentials` and `PySide6_Addons`. Before bundling, select and document the applicable licensing route and collect the exact Qt, Shiboken, Essentials, Addons, plugin, and native-library notices. Monolith's MIT license does not replace them. |
| **pydantic** ([PyPI](https://pypi.org/project/pydantic/), [upstream](https://github.com/pydantic/pydantic)) | `>=2.6,<3` | `2.13.4` | `MIT` | Preserve the upstream license in a redistributed environment. The native `pydantic-core` artifact is transitive and therefore outside this direct-only table; include it in an artifact-specific SBOM and notice set. |
| **PyYAML** ([PyPI](https://pypi.org/project/PyYAML/), [upstream](https://github.com/yaml/pyyaml)) | `>=6.0.1,<7` | `6.0.3` | `MIT` | Preserve the upstream license. Inspect the chosen wheel/source build for its C-extension and native-component boundary when redistributing an environment. |
| **Markdown / Python-Markdown** ([PyPI](https://pypi.org/project/Markdown/), [upstream](https://github.com/Python-Markdown/markdown)) | `>=3.4,<4` | `3.10.2` | `BSD-3-Clause` | Preserve the copyright, conditions, and disclaimer supplied with the exact distribution. It does not license rendered input content. |
| **Pygments** ([PyPI](https://pypi.org/project/Pygments/), [upstream](https://github.com/pygments/pygments)) | `>=2.17,<3` | `2.20.0` | `BSD-2-Clause` | Preserve the copyright, conditions, and disclaimer supplied with the exact distribution. Lexer names and highlighted input remain outside Monolith's first-party license grant. |

## Optional and development dependencies

Every resolved-version cell in this section says **Not resolved** because the
clean-room verification installed the base project only. A current PyPI release
number is not substituted for an installed version.

| Group | Package and primary sources | Declared range | Clean-room resolved | Upstream-declared license or choice | Redistribution, notice, model, service, and native boundary |
|---|---|---:|---:|---|---|
| `files` | **PyMuPDF** ([PyPI](https://pypi.org/project/PyMuPDF/), [upstream](https://github.com/pymupdf/PyMuPDF)) | `>=1.24,<2` | Not resolved | Upstream wording: GNU Affero GPL 3.0 **or** Artifex commercial license | High-attention native MuPDF boundary. Do not treat this dependency as MIT. Select and document an applicable AGPL/commercial route before distributing PyMuPDF, MuPDF, or a bundled executable, and retain the controlling license and notices for the exact artifact. |
| `files` | **pypdf** ([PyPI](https://pypi.org/project/pypdf/), [upstream](https://github.com/py-pdf/pypdf)) | `>=4,<7` | Not resolved | `BSD-3-Clause` | Preserve the exact distribution's copyright, conditions, and disclaimer. PDF contents, embedded fonts, and attachments keep their own rights. Optional crypto/image extras would expand the dependency inventory. |
| `files` | **python-docx** ([PyPI](https://pypi.org/project/python-docx/), [upstream](https://github.com/python-openxml/python-docx)) | `>=1.1,<2` | Not resolved | `MIT` | Preserve the license. Capture its actual transitive XML stack, including native artifacts if present, in a packaged-build SBOM. `.docx` contents and embedded media are not relicensed. |
| `files` | **openpyxl** ([PyPI](https://pypi.org/project/openpyxl/), [upstream](https://foss.heptapod.net/openpyxl/openpyxl)) | `>=3.1,<4` | Not resolved | `MIT` | Preserve the license. Workbook contents, formulas, macros, images, and templates are separate content, not Monolith assets. |
| `files`, `vision` | **Pillow** ([PyPI](https://pypi.org/project/Pillow/), [upstream](https://github.com/python-pillow/Pillow)) | `>=10,<13` in both groups | Not resolved | `MIT-CMU` | Compiled image/codec boundary. Preserve Pillow's license and inspect the selected wheel for bundled or dynamically used image libraries and their notices. Input images, fonts, metadata, and generated outputs remain separate. |
| `local-llm` | **llama-cpp-python** ([PyPI](https://pypi.org/project/llama-cpp-python/), [upstream](https://github.com/abetlen/llama-cpp-python)) | `>=0.3,<0.4` | Not resolved | `MIT` | Native binding/runtime boundary: the package builds or carries `llama.cpp` components. Inventory the exact wheel/build and native notices. A package license never grants rights to a GGUF model, tokenizer, prompt corpus, or model output. |
| `vision`, `audio` | **torch** ([PyPI](https://pypi.org/project/torch/), [upstream](https://github.com/pytorch/pytorch), [upstream NOTICE](https://github.com/pytorch/pytorch/blob/main/NOTICE)) | `>=2.4` in both groups | Not resolved | Current PyPI expression: `Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT`; upstream describes the core license as BSD-style | High-attention native CPU/GPU boundary. The selected wheel/index/build can change bundled components and CUDA, ROCm, MKL, LLVM, or vendor-runtime exposure. Preserve the exact artifact's license/NOTICE material and produce a full native/transitive inventory before binary redistribution. Models and datasets are separate. |
| `vision` | **diffusers** ([PyPI](https://pypi.org/project/diffusers/), [upstream](https://github.com/huggingface/diffusers)) | `>=0.35,<1` | Not resolved | `Apache-2.0` | Preserve the Apache license and notices for the exact package. Pipelines can download third-party checkpoints, schedulers, tokenizers, safety components, and LoRAs; each model repository/model card controls those artifacts. |
| `vision` | **transformers** ([PyPI](https://pypi.org/project/transformers/), [upstream](https://github.com/huggingface/transformers)) | `>=4.57,<6` | Not resolved | `Apache-2.0` | Preserve the Apache license and notices for the exact package. Model weights, tokenizers, chat templates, remote code, datasets, and model-host terms are separate and must be reviewed per artifact. |
| `vision` | **accelerate** ([PyPI](https://pypi.org/project/accelerate/), [upstream](https://github.com/huggingface/accelerate)) | `>=1.11,<2` | Not resolved | `Apache-2.0` | Preserve the Apache license and notices. Hardware backends and optional integrations can introduce additional packages, native runtimes, and service terms that are not represented by this direct declaration. |
| `audio` | **torchaudio** ([PyPI](https://pypi.org/project/torchaudio/), [upstream](https://github.com/pytorch/audio), [upstream LICENSE](https://github.com/pytorch/audio/blob/main/LICENSE)) | `>=2.4` | Not resolved | `BSD-2-Clause` (upstream LICENSE; PyPI metadata says BSD) | Native audio/PyTorch boundary. Preserve the license for the selected wheel and inventory codec/native integrations. Torchaudio states that datasets and pretrained models may have separate terms; review each one rather than inheriting the library license. |
| `audio` | **AudioCraft** ([PyPI](https://pypi.org/project/audiocraft/), [upstream](https://github.com/facebookresearch/audiocraft), [weights license](https://github.com/facebookresearch/audiocraft/blob/main/LICENSE_weights)) | `>=1.3,<2` | Not resolved | `MIT` for code; upstream publishes its model weights separately under `CC-BY-NC 4.0` | High-attention model boundary. Do not bundle or describe MusicGen/AudioGen/EnCodec/MAGNeT weights as MIT. Record the exact model card, weight license, source, hash, and permitted use. FFmpeg and other codec/runtime tools are separate if supplied. |
| `matrix` | **matrix-nio** ([PyPI](https://pypi.org/project/matrix-nio/), [upstream](https://github.com/poljar/matrix-nio)) | `>=0.24,<1` | Not resolved | ISC license | Preserve the ISC notice. Matrix homeserver accounts, room content, federation, and server policies are separate service/content boundaries. The undeclared E2EE extra introduces `python-olm`/`libolm` and requires a new native/license inventory if enabled. |
| `matrix` | **aiohttp** ([PyPI](https://pypi.org/project/aiohttp/), [upstream](https://github.com/aio-libs/aiohttp)) | `>=3.9,<4` | Not resolved | `Apache-2.0 AND MIT` | Preserve both controlling license/notice sets from the exact distribution. Wheels and optional speedups can add compiled or transitive artifacts; network payloads and remote-service terms are not licensed by aiohttp. |
| `dev` | **pytest** ([PyPI](https://pypi.org/project/pytest/), [upstream](https://github.com/pytest-dev/pytest)) | `>=9,<10` | Not resolved | `MIT` | Development-only declaration. Preserve the license if a developer environment, test appliance, or bundled diagnostic distribution includes it; otherwise it is not part of the runtime dependency set. |

## High-attention release decisions

### 1. PySide6 and Qt

The base install necessarily crosses a non-MIT native-framework boundary.
PySide6 publishes open-source alternatives (`LGPL-3.0-only`, `GPL-2.0-only`, or
`GPL-3.0-only`) and a separate commercial option. The alias package also pulls
Essentials and Addons wheels. A source-only GitHub release may declare PySide6
without shipping those wheels, but a frozen application, installer, portable
environment, or wheel cache must be reviewed as an exact Qt payload. Record the
chosen licensing route and the actual Qt/Shiboken modules, plugins, license
files, and notices before that artifact is released.

### 2. PyMuPDF and MuPDF

PyMuPDF is explicitly dual licensed under GNU Affero GPL 3.0 or an Artifex
commercial license and wraps the native MuPDF engine. Its presence as an
optional declaration does not make it MIT-compatible by assertion. A future
bundle needs an explicit licensing decision and an artifact-specific review;
until then, keep it user-installed and do not include its binaries under the
Monolith license banner.

### 3. Torch, torchaudio, and native compute/audio stacks

The `torch>=2.4` declaration has no upper bound and does not identify a CPU,
CUDA, ROCm, or package-index variant. Those choices materially change the
binary payload. The current PyPI metadata already reports a multi-license
expression, and PyTorch publishes a substantial NOTICE file. Before shipping
any ML-enabled environment, resolve one exact platform/profile, capture hashes,
enumerate all transitive and native libraries, and preserve every controlling
license and notice. Apply the same rule to torchaudio, codecs, and vendor GPU
runtimes.

### 4. AudioCraft and all model artifacts

AudioCraft's MIT code license and its `CC-BY-NC 4.0` model-weight license are
different grants. Likewise, the Diffusers, Transformers, llama.cpp, and
torchaudio library licenses do not license downloaded checkpoints, GGUF files,
tokenizers, LoRAs, datasets, or pretrained models. The source release should
contain no model weights unless each artifact has a provenance record, exact
hash, controlling model/data license, notice/attribution plan, and confirmed
redistribution basis.

### 5. External services

OpenAI-compatible endpoints, Tavily, Matrix homeservers, webhooks, external
peers, model hosts, and package indexes are not covered by any package license
in this inventory. The operator's current terms, privacy policy, data-use
rules, acceptable-use policy, account/billing terms, and content rules control
each connection. Do not ship provider credentials. A deployment owner must
review the actual endpoint selected; "OpenAI-compatible" describes a protocol,
not one service or one set of terms.

## Release handling

For the current source-only GitHub release:

1. publish Monolith's root `LICENSE`, [`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md),
   and this inventory together;
2. keep dependency ranges in `pyproject.toml` clearly separate from observed
   versions and from any future lock;
3. do not state or imply that third-party packages, native libraries, models,
   datasets, user documents, generated outputs, or services are covered by
   Monolith's MIT license; and
4. repeat this inventory pass whenever a direct range, optional group, model,
   vendored component, installer, or distribution format changes.

Before distributing a wheelhouse, portable environment, frozen executable,
installer, container, or model bundle:

1. resolve exact versions for each supported profile and platform;
2. save artifact names, sources, hashes, and package-index/build variants;
3. generate a complete transitive SBOM, including native libraries;
4. extract and preserve the exact distributions' license and NOTICE files;
5. close the PySide6/Qt and PyMuPDF licensing choices in writing;
6. inventory every model, tokenizer, dataset, codec, font, sample, and external
   executable separately; and
7. update this document with the resolved artifact evidence instead of relying
   on today's project-level metadata.
