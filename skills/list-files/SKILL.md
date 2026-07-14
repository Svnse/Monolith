---
name: list-files
description: List the immediate contents of a directory — files and subdirectories. Directories appear with a trailing slash.
---

Lists the immediate contents of a directory (one level, not recursive).
The result header shows the file count, directory count, and the glob
pattern applied. Pattern defaults to `*` and filters both files and
directories. Subdirectories are rendered as `name/` (trailing slash, no
size). Files are rendered as `name (size)`. Use `find_files` for
recursive search.

Returns `data.files` (list of filenames) and `data.dirs` (list of
subdirectory names, without the trailing slash). Both keys are omitted
when their list would be empty.

{"tool":"list_files","path":"C:/project","pattern":"*.py"}
{"tool":"list_files","path":"C:/project"}
