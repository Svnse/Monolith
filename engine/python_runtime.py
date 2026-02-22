"""
Standalone Python execution runtime for Monolith agent tools.

Runs in subprocess — fully privileged OS-level execution.
No sandboxing; the capability manifest + OFAC contract handle authorization.
"""

import json
import sys
import traceback
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
import time

# Hard limits
MAX_OUTPUT_BYTES = 100_000
MAX_EXECUTION_TIME = 300  # 5 minutes absolute ceiling


def truncate_output(text: str, max_bytes: int = MAX_OUTPUT_BYTES) -> str:
    """Truncate output to prevent token bombs."""
    encoded = text.encode('utf-8')
    if len(encoded) <= max_bytes:
        return text
    # Find safe truncation point
    truncated = encoded[:max_bytes].decode('utf-8', errors='ignore')
    return truncated + f"\n[OUTPUT TRUNCATED - exceeded {max_bytes} bytes]"


def execute(contract: dict) -> dict:
    """Execute Python code according to JSON contract. Fully privileged."""
    code = contract.get('code', '')
    workspace = contract.get('workspace_root', '/tmp')

    # Execution namespace - clean slate
    namespace = {
        'workspace_root': workspace,
        'result': None,  # Convention: assign to result for return_value capture
    }

    stdout_buf = StringIO()
    stderr_buf = StringIO()

    start = time.perf_counter()
    exception_info = None
    status = 'ok'
    return_value = None

    try:
        # Compile first to catch syntax errors
        try:
            compiled = compile(code, '<agent>', 'exec')
        except SyntaxError as e:
            status = 'error'
            exception_info = {
                'type': 'SyntaxError',
                'message': str(e),
                'traceback': traceback.format_exc(),
                'line': e.lineno,
            }
            compiled = None

        if compiled:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(compiled, namespace)

            # Extract result if set
            return_value = namespace.get('result')

    except Exception as e:
        status = 'error'
        return_value = None
        exc_type = type(e).__name__
        exc_msg = str(e)
        exc_traceback = traceback.format_exc()

        # Extract line number from traceback
        line_no = None
        if e.__traceback__:
            tb = e.__traceback__
            while tb.tb_next:
                tb = tb.tb_next
            line_no = tb.tb_lineno

        if hasattr(e, 'lineno') and e.lineno:
            line_no = e.lineno

        exception_info = {
            'type': exc_type,
            'message': exc_msg,
            'traceback': exc_traceback,
            'line': line_no,
        }

    elapsed = int((time.perf_counter() - start) * 1000)

    # Truncate outputs
    stdout_text = truncate_output(stdout_buf.getvalue())
    stderr_text = truncate_output(stderr_buf.getvalue())

    # Ensure return_value is JSON serializable
    try:
        json.dumps(return_value, default=str)
    except (TypeError, ValueError):
        return_value = f"[unserializable: {type(return_value).__name__}]"

    return {
        'status': status,
        'return_value': return_value,
        'stdout': stdout_text,
        'stderr': stderr_text,
        'exception': exception_info,
        'execution_time_ms': elapsed,
    }


def main():
    """CLI entrypoint: read JSON from stdin, write JSON to stdout."""
    try:
        contract = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        result = {
            'status': 'error',
            'return_value': None,
            'stdout': '',
            'stderr': f'Invalid JSON input: {e}',
            'exception': {
                'type': 'JSONDecodeError',
                'message': str(e),
                'traceback': traceback.format_exc(),
                'line': None,
            },
            'execution_time_ms': 0,
        }
        print(json.dumps(result))
        sys.exit(1)

    result = execute(contract)
    print(json.dumps(result, default=str))
    sys.exit(0 if result['status'] == 'ok' else 1)


if __name__ == '__main__':
    main()
