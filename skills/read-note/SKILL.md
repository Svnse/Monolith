---
name: read_note
description: Read a saved MonoNote note by title when the user asks for note access, and emit canonical note-read provenance.
---

{"tool":"read_note","title":"project-ideas","max_chars":8000}

Use this only when the user asks to read, inspect, use, or bring in notes. The
tool logs a `mononote_note_read` canonical event and returns a `[NOTE_READ ...]`
metatag before the note text. Title is matched against the saved filename stem
after the same normalization `save_note` applies.
