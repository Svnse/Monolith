<p align="center">
<img width="642" height="118" alt="monolith" src="https://github.com/user-attachments/assets/17bc1107-9fdd-4d9d-9c7d-6c9a1f00fed6" />
</p>

<p align="center">
<b>Stop chatting with AI. Start commanding it.</b><br/>
A local-first AI workstation for running LLMs, Stable Diffusion, and audio generation through a modular kernel.
</p>

---

## Screenshots

<table style="border: none;">
<tr>
<td width="50%" align="center" style="border:none;">
<img src="https://github.com/user-attachments/assets/817d57a3-fb4a-4210-80a4-511116faad0b" width="98%">
</td>
<td width="50%" align="center" style="border:none;">
<img src="https://github.com/user-attachments/assets/f6f2af83-3038-42a0-b0aa-dbfa79489485" width="98%">
</td>
</tr>
</table>

<p align="center">
<img src="https://github.com/user-attachments/assets/7bb47e3d-0b12-413b-8fa7-24c174a4ddc4" width="88%">
</p>

<p align="center">
<i>Top: idle kernel + LLM chat · Bottom: Vision tab generating an image</i>
</p>

---

## Quick Start

### Windows
1. Clone repo  
2. Run `install.bat`  
3. Run `start.bat`

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```


## Core Overview

**Status**
v0.2.1a — Early alpha. Built for personal use and active experimentation. Shared publicly for builders & creators.

**Features**
* **Local LLM Chat:** GGUF models via llama.cpp
* **Image Generation:** Stable Diffusion
* **Audio Generation:** AudioCraft
* **System:** Persistent conversation history, modular kernel architecture, dark mode

**Architecture**
Kernel + Engines + Addons model. Engines run isolated processes; Addons control them. The Kernel enforces contracts and lifecycle boundaries.

See [Kernel Contract (V2)](/monokernel/kernel_contract.md) for details.

---

## Requirements

* **Python:** 3.10+
* **GPU:** NVIDIA GPU with 8GB+ VRAM recommended (required for SD/Audio)
* **Storage:** ~10GB disk space for base models

---

<p align="center">
  Built by <a href="https://eryndel.us">Eryndel</a>
</p>
