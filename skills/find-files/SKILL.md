---
name: find-files
description: Recursively find files (and optionally directories) by name or glob pattern. Returns absolute paths.
---

Recursively searches a directory tree for entries whose **basename**
matches `pattern`. Pattern is a glob, matched against the filename
component — not the full path. Examples:

  - `pattern="config.yaml"` — exact filename match
  - `pattern="*.py"` — any Python file
  - `pattern="test_*.py"` — Python files starting with `test_`
  - `pattern="*/migrations/*"` — does NOT work; pattern is basename-scoped

By default, only files are matched. Set `include_dirs: true` to also
match directory entries (useful when you know a directory by name but
not its location). Directory results are rendered with a trailing
slash; the parser strips it from `data.matches`.

Set `recursive: false` to scan only the immediate children of `path`.

{"tool":"find_files","path":"C:/project","pattern":"*.py","recursive":true,"max_results":20}
{"tool":"find_files","path":"C:/project","pattern":"prompts","include_dirs":true,"max_results":5}
