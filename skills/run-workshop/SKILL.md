---
name: run_workshop
description: Run a saved Workshop workflow (a Monoline LLM pipeline) by name and receive its output, so a pre-built multi-step pipeline becomes one step of your reasoning. Runs off-thread; you get a PENDING marker, then the result folds back. Only the principal turn may call it (workflows cannot call workflows).
---

{"tool":"run_workshop","name":"Two-Step Assistant","input":"draft a reply about the deployment plan"}
