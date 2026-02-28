TOOL_ARGUMENT_SCHEMAS = {
    "read_file": {
        "required": {"path": str},
        "optional": {"offset": int, "limit": int},
    },
    "write_file": {
        "required": {"path": str, "content": str},
        "optional": {},
    },
    "list_dir": {
        "required": {"path": str},
        "optional": {"pattern": str},
    },
    "grep_search": {
        "required": {"pattern": str},
        "optional": {"path": str},
    },
    "glob_files": {
        "required": {"pattern": str},
        "optional": {"path": str, "include_dirs": bool, "limit": int},
    },
    "mkdir": {
        "required": {"path": str},
        "optional": {"parents": bool, "exist_ok": bool},
    },
    "move_path": {
        "required": {"src": str, "dst": str},
        "optional": {"overwrite": bool},
    },
    "copy_path": {
        "required": {"src": str, "dst": str},
        "optional": {"overwrite": bool, "recursive": bool},
    },
    "delete_path": {
        "required": {"path": str},
        "optional": {"recursive": bool, "missing_ok": bool, "trash": bool},
    },
    "zip_path": {
        "required": {"src": str, "dst": str},
        "optional": {"recursive": bool, "overwrite": bool},
    },
    "unzip_archive": {
        "required": {"src": str, "dst": str},
        "optional": {"overwrite": bool},
    },
    "run_cmd": {
        "required": {"command": str},
        "optional": {"timeout": int, "pty_enabled": bool},
    },
    "run_tests": {
        "required": {},
        "optional": {"command": str, "timeout": int, "pty_enabled": bool},
    },
    "apply_patch": {
        "required": {"path": str, "old": str, "new": str},
        "optional": {},
    },
    "run_python": {
        "required": {"code": str},
        "optional": {"timeout": int},
    },
    "git_status": {
        "required": {},
        "optional": {"path": str, "timeout": int},
    },
    "git_diff": {
        "required": {},
        "optional": {
            "path": str,
            "file": str,
            "staged": bool,
            "base": str,
            "context": int,
            "max_lines": int,
            "timeout": int,
        },
    },
    "http_fetch": {
        "required": {"url": str},
        "optional": {
            "timeout": int,
            "method": str,
            "headers": dict,
            "max_bytes": int,
            "follow_redirects": bool,
        },
    },
}
