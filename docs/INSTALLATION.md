# Installation

Monolith `v1.0.0` is distributed as a source checkout. The supported
installation path for this release is 64-bit CPython 3.11 on Windows. There is
no wheel, executable installer, model bundle, or supported Linux/macOS path.

## Tested v1 release environment

| Component | Observed version |
|---|---|
| Windows | 25H2, build 26200.8655 |
| Python | CPython 3.11.9, 64-bit expected |
| Git | 2.54.0.windows.1 |
| Test runner | pytest 9.0.2 |

The packaging contract intentionally requires `>=3.11,<3.12`. That narrow
range records what was tested; it is not a claim that the source cannot be made
to work elsewhere.

## Base installation

Open **Command Prompt** in the repository root:

```bat
install.bat
start.bat
```

The installer:

1. finds Python 3.11 through the Windows `py` launcher or `python`;
2. creates or reuses `venv`;
3. upgrades pip, setuptools, and wheel inside that environment;
4. installs the base project in editable mode from `pyproject.toml`; and
5. imports the startup/configuration/rendering modules in an isolated temporary
   Monolith state directory.

The script exits non-zero on failure. It does not ask for credentials and does
not install model files.

### Manual equivalent

From **Command Prompt** in the repository root:

```bat
py -3.11 -m venv venv
venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
venv\Scripts\python.exe -m pip install --no-build-isolation -e .
venv\Scripts\python.exe main.py
```

`pyproject.toml` is authoritative. Root `requirements.txt` and files under
`requirements\` are compatibility wrappers; they do not maintain separate
package lists.

## Base profile contents

The base install contains only packages needed by the normal UI/configuration
and Markdown rendering path:

- PySide6;
- Pydantic;
- PyYAML;
- Python-Markdown; and
- Pygments.

It is sufficient to launch the application and configure an OpenAI-compatible
HTTP endpoint. It does not include a provider account or key.

## Optional profiles

Rerunning `install.bat` reuses the same `venv` and adds one selected profile:

```bat
install.bat files
install.bat local-llm
install.bat matrix
install.bat dev
```

| Profile | Adds | Important boundary |
|---|---|---|
| `files` | PyMuPDF, pypdf, python-docx, openpyxl, Pillow | Enables richer document/image inspection; archives and plain text need no profile |
| `local-llm` | llama-cpp-python | Native builds vary by compiler, CPU, and GPU backend; GGUF files are user-supplied |
| `matrix` | matrix-nio and aiohttp | Network integration; credentials and room data leave the base local-only boundary |
| `dev` | pytest | Adds the test runner, not every optional runtime stack |

Equivalent pip syntax is:

```bat
venv\Scripts\python.exe -m pip install --no-build-isolation -e ".[files]"
venv\Scripts\python.exe -m pip install --no-build-isolation -e ".[local-llm]"
venv\Scripts\python.exe -m pip install --no-build-isolation -e ".[matrix]"
venv\Scripts\python.exe -m pip install --no-build-isolation -e ".[dev]"
```

Extras include the base dependencies automatically. Install only profiles you
intend to use.

## Vision profile

Image generation depends on Torch, Diffusers, Transformers, Accelerate,
Pillow, a compatible diffusion model, and hardware appropriate to that model.
`install.bat vision` deliberately refuses to guess a Torch/CUDA build.

1. Choose and install the correct Torch build for the machine using the
   upstream PyTorch installation guidance.
2. Install the remaining declared extra:

   ```bat
   venv\Scripts\python.exe -m pip install --no-build-isolation -e ".[vision]"
   ```

3. Configure a local model path in Monolith.
4. Run a small generation before relying on the setup.

The extra may ask pip to resolve Torch again. Inspect the proposed transaction
before accepting it when preserving a specific CUDA build matters. No model is
downloaded or licensed by Monolith itself.

## Audio profile

Audio generation depends on a mutually compatible Python, Torch, torchaudio,
AudioCraft, native codec stack, and MusicGen model. `install.bat audio`
deliberately refuses a blind install.

1. Review AudioCraft's supported environment and install the matching Torch and
   torchaudio builds.
2. Install the declared extra only after that environment is coherent:

   ```bat
   venv\Scripts\python.exe -m pip install --no-build-isolation -e ".[audio]"
   ```

3. Verify imports and a short audio generation before enabling long jobs.

AudioCraft compatibility can lag current Python/Torch releases. Do not repair
an audio install by silently replacing a working vision or local-inference
stack. Separate virtual environments may be safer for incompatible profiles.

## Model setup

### OpenAI-compatible endpoint

In the Config model panel:

1. select **Model (API)**;
2. enter the base URL, normally ending in `/v1` when required by the server;
3. enter the exact model ID exposed by that endpoint;
4. enter a key if required; and
5. load the model.

The observed configuration path stores the API key in local YAML as plaintext.
Use a restricted key and protect `%APPDATA%\Monolith\config`. See
[configuration](CONFIGURATION.md).

### Local GGUF

Monolith exposes **GGUF (API)** and **GGUF (llama.cpp)** modes. Both require a
compatible model file and runtime. The repository does not include either.
Native acceleration and model compatibility depend on how llama.cpp or
llama-cpp-python was built. Start with a small known-compatible model and keep
its license alongside the model library.

For the local GGUF server path, Monolith checks an explicit
`MONOLITH_LLAMA_SERVER` executable first, then common builds below the current
user's home directory, then `llama-server` on `PATH`. If no native server is
found, it can fall back to `python -m llama_cpp.server`. Set
`MONOLITH_LLAMA_PY` to the Python executable that contains that module; the
active Monolith interpreter is the final fallback.

```bat
set "MONOLITH_LLAMA_SERVER=C:\path\to\llama-server.exe"
rem Or, for the Python server fallback:
set "MONOLITH_LLAMA_PY=C:\path\to\python.exe"
start.bat
```

These overrides are local machine configuration. Do not commit their absolute
paths in launchers or examples.

## Developer and test installation

Install the base developer profile:

```bat
install.bat dev
venv\Scripts\python.exe -m pytest -q
```

Some tests exercise optional integrations or file formats. For the broadest
non-GPU test environment:

```bat
venv\Scripts\python.exe -m pip install --no-build-isolation -e ".[dev,files,matrix]"
venv\Scripts\python.exe -m pytest -q
```

Set a disposable state root before manual development runs so another Monolith
checkout does not share state:

```bat
set "MONOLITH_ROOT=%USERPROFILE%\.monolith-dev"
start.bat
```

The verified v1 baseline is documented in
[`KNOWN_ISSUES.md`](../KNOWN_ISSUES.md); the full public-tree suite is green
within the disclosed skip and expected-failure boundary.

## Repair and removal

If the environment has the wrong Python version or incompatible native wheels,
close Monolith, delete only the repository's `venv` directory, and rerun
`install.bat`. A virtual-environment rebuild does not delete chats or settings.

Before deleting user state, identify the active root:

- default Windows root: `%APPDATA%\Monolith`;
- override: the absolute path in `MONOLITH_ROOT`.

Back up that directory before resetting it. Different checkouts share the
default root unless each launch sets a distinct `MONOLITH_ROOT`.

## Troubleshooting

### `Python 3.11` is not found

Install 64-bit CPython 3.11 with the Windows `py` launcher, reopen Command
Prompt, and verify:

```bat
py -3.11 --version
```

### The existing venv is rejected

The environment was created by another Python version. Delete the repository's
`venv` directory and rerun `install.bat`.

### `llama-cpp-python` fails to build

This is a native extension. Use a wheel/build path compatible with the target
CPU or GPU backend and the active Python 3.11 interpreter. Base API chat does
not require this package.

### Torch, Diffusers, or AudioCraft conflict

Do not keep layering packages onto an unknown environment. Record the current
Torch/CUDA versions, rebuild the venv if necessary, install the hardware-
appropriate Torch stack first, and then install one optional profile at a time.

### The UI launches but no response is generated

The base install does not configure a model. Confirm that the selected backend
has a model path or reachable API base/model ID, then use **LOAD MODEL** in the
Config panel. Never paste credentials into an issue report.

### Optional document support reports a missing package

Run `install.bat files`. Monolith intentionally returns dependency-missing
messages for formats whose richer readers are not installed.

## Not provided by this release

- automatic CUDA/ROCm/Metal selection;
- model downloads or model-license management;
- locked wheels for every hardware configuration;
- a standalone Windows executable or updater;
- supported Linux/macOS installation; or
- production deployment guidance for CONNECT or other servers.
