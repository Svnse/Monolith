from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

from core import paths
from core.paths import ensure_safe_local_path

from .model import NoteRecord, NoteTextWindow


_SAFE_TITLE_RE = re.compile(r"[^\w\-]")
_MAX_TITLE_CHARS = 64
_DEFAULT_MAX_CHARS = 8000
_HARD_MAX_CHARS = 50000


def safe_title(title: str) -> str:
    return _SAFE_TITLE_RE.sub("-", str(title or ""))[:_MAX_TITLE_CHARS].strip("-") or "note"


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def vault_dir(vault: str | Path | None = None, *, create: bool = False) -> Path:
    root = Path(vault).expanduser() if vault is not None else paths.NOTES_DIR
    root = root.resolve(strict=False)
    if create:
        root.mkdir(parents=True, exist_ok=True)
    if root.exists() and root.is_symlink():
        raise OSError(f"Refusing to use symlinked MonoNote vault: {root}")
    return root


def ensure_note_path(path: str | Path, vault: str | Path | None = None, *, create: bool = False) -> Path:
    root = vault_dir(vault, create=create)
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve(strict=False)
    if not _is_relative_to(candidate, root):
        raise ValueError(f"MonoNote path escapes vault: {candidate}")
    if create:
        return ensure_safe_local_path(candidate)
    if candidate.exists():
        if candidate.is_symlink():
            raise OSError(f"Refusing to use symlinked MonoNote file: {candidate}")
        if not candidate.is_file():
            raise OSError(f"Refusing to use non-file MonoNote path: {candidate}")
    return candidate


def note_path_for_title(title: str, vault: str | Path | None = None, *, create: bool = False) -> Path:
    return ensure_note_path(f"{safe_title(title)}.md", vault, create=create)


def _record_for_path(path: Path, root: Path, *, title: str | None = None) -> NoteRecord:
    stat = path.stat()
    data = path.read_bytes()
    relative = path.resolve(strict=False).relative_to(root.resolve(strict=False)).as_posix()
    note_id = hashlib.sha256(relative.lower().encode("utf-8")).hexdigest()[:16]
    return NoteRecord(
        note_id=note_id,
        title=(title or path.stem),
        safe_title=path.stem,
        path=path,
        relative_path=relative,
        size=int(stat.st_size),
        mtime_ns=int(stat.st_mtime_ns),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def write_note(title: str, body: str, vault: str | Path | None = None) -> NoteRecord:
    if not str(title or "").strip():
        raise ValueError("title must be non-empty")
    path = note_path_for_title(title, vault, create=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(str(body), encoding="utf-8")
    os.replace(tmp_path, path)
    return _record_for_path(path, vault_dir(vault), title=str(title).strip())


def load_note(title: str, vault: str | Path | None = None) -> tuple[NoteRecord, str]:
    if not str(title or "").strip():
        raise ValueError("title must be non-empty")
    root = vault_dir(vault, create=False)
    if not root.exists():
        raise FileNotFoundError(str(root))
    path = note_path_for_title(title, root, create=False)
    if not path.exists():
        raise FileNotFoundError(str(path))
    body = path.read_text(encoding="utf-8")
    return _record_for_path(path, root, title=str(title).strip()), body


def list_notes(
    pattern: str = "",
    *,
    max_results: int = 50,
    vault: str | Path | None = None,
) -> list[NoteRecord]:
    root = vault_dir(vault, create=False)
    if not root.exists():
        return []
    needle = str(pattern or "").strip().lower()
    limit = max(1, min(200, int(max_results or 50)))
    records: list[NoteRecord] = []
    for note_file in sorted(root.glob("*.md")):
        if note_file.is_symlink() or not note_file.is_file():
            continue
        if needle and needle not in note_file.stem.lower():
            continue
        records.append(_record_for_path(note_file.resolve(strict=False), root))
        if len(records) >= limit:
            break
    return records


def _int_or_default(value: object, default: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def normalize_max_chars(value: object, default: int = _DEFAULT_MAX_CHARS) -> int:
    parsed = _int_or_default(value, default)
    return max(1, min(_HARD_MAX_CHARS, parsed))


def slice_note_text(
    body: str,
    *,
    offset: object = 0,
    max_chars: object = _DEFAULT_MAX_CHARS,
    selection_start: object | None = None,
    selection_end: object | None = None,
) -> NoteTextWindow:
    text = str(body)
    length = len(text)
    cap = normalize_max_chars(max_chars)
    if selection_start is not None or selection_end is not None:
        start = max(0, min(length, _int_or_default(selection_start, 0)))
        end_default = length if selection_end is None else start
        end = max(start, min(length, _int_or_default(selection_end, end_default)))
        original_end = end
        if end - start > cap:
            end = start + cap
        return NoteTextWindow(
            text=text[start:end],
            start=start,
            end=end,
            truncated=end < original_end,
        )

    start = max(0, min(length, _int_or_default(offset, 0)))
    end = min(length, start + cap)
    return NoteTextWindow(
        text=text[start:end],
        start=start,
        end=end,
        truncated=end < length,
    )
