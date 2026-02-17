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
    "run_cmd": {
        "required": {"command": str},
        "optional": {"timeout": int, "pty_enabled": bool},
    },
    "apply_patch": {
        "required": {"path": str, "old": str, "new": str},
        "optional": {},
    },
}
