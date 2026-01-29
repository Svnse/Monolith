<img width="1915" height="38" alt="image" src="https://github.com/user-attachments/assets/1a3d1889-4a23-4e93-86a0-2380ffc4235a" />

# Monolith

Stop chatting with AI. Start commanding it.

A local-first AI workstation for running LLMs, Stable Diffusion, and audio generation through a modular kernel.
--- 

> Screenshot 1
<img width="1102" height="702" alt="image" src="https://github.com/user-attachments/assets/817d57a3-fb4a-4210-80a4-511116faad0b" />
- Monolith Application Open with no Addons.
(More screenshots below)

--- 
## Quick Start

**Windows:**
1. Clone repo
2. Run `install.bat`
3. Run `start.bat`

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Status

Early alpha. Built for myself. Sharing in case others want it.

## Features

- Local LLM chat (GGUF models via llama.cpp)
- Stable Diffusion image generation
- Audio generation (via AudioCraft)
- Persistent chat history
- Modular kernel architecture
- Dark mode

## Requirements

- Python 3.10+
- CUDA GPU recommended (for SD/Audio)
- ~10GB disk space for models

## Architecture

Kernel + Engines + Addons model.

Engines run isolated processes. Addons control them. Kernel enforces contracts.

See `/monokernel/kernel_contract.md` for details or V2 (current).

---

** Screenshot - 2**
<img width="1919" height="1032" alt="image" src="https://github.com/user-attachments/assets/f6f2af83-3038-42a0-b0aa-dbfa79489485" />
- Terminal/LLM chat

** Screenshot - 3**
<img width="1918" height="1032" alt="image" src="https://github.com/user-attachments/assets/7bb47e3d-0b12-413b-8fa7-24c174a4ddc4" />
- Vision Tab loaded with a model and with a image generated.

--- 

Built by Eryndel | eryndel.us
