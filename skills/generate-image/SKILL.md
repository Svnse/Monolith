---
name: generate-image
description: Generate an image via the vision engine (SD 1.5 / SDXL / Flux). Uses the model selected in the VISION tab unless `model` is given. Async — returns "submitted" immediately; the image arrives in the same tool-result bubble when ready.
---

{"tool":"generate_image","prompt":"a cat in a forest"}
Optional: negative_prompt, width, height, steps (default 25), seed (-1=random), guidance_scale, model (name or absolute path)

Default model is read from config.vision.model_path (set via the VISION tab MODEL card). If no default is set and no `model` arg is given, the skill returns the list of scanned models in MONOLITH/models/vision/ — pick one and retry with `model=<name>`. Batches (batch_size > 1) arrive as a single bubble with a thumb strip.
