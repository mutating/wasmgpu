"""Run a Python experiment with an independent watchdog (no GPU in this process).

Example: venv/bin/python -m tests.gpu_guard -- -m pytest tests/test_codegen.py
On macOS, account for the worker's Metal XPC compiler, including compressed
memory. Only services belonging to the worker's launchd domain may be killed.
This bounds development experiments; it cannot prevent a GPU driver panic.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import ClassVar


class Usage(ctypes.Structure):
    _fields_: ClassVar[list[tuple[str, type]]] = [('uuid', ctypes.c_uint8 * 16), *[
        (name, ctypes.c_uint64) for name in (
            'user', 'system', 'idle', 'interrupts', 'pageins', 'wired',
            'resident', 'footprint', 'start', 'exit',
        )
    ]]


def memory(pid):
    """Physical footprint and process identity, or None after process exit."""
    if sys.platform == 'darwin':
        usage = Usage()
        library = ctypes.CDLL('/usr/lib/libproc.dylib')
        if library.proc_pid_rusage(pid, 0, ctypes.byref(usage)) == 0:
            return max(usage.footprint, usage.resident), usage.start
        return None
    try:
        status = Path('/proc/{}/status'.format(pid)).read_text()
        resident = re.search(r'^VmRSS:\s+(\d+)', status, re.MULTILINE)
        swapped = re.search(r'^VmSwap:\s+(\d+)', status, re.MULTILINE)
        identity = Path('/proc/{}/stat'.format(pid)).read_text().rsplit(')', 1)[1].split()[19]
        return (int(resident[1]) + (int(swapped[1]) if swapped else 0)) * 1024, identity
    except (OSError, TypeError):
        return None


def compiler_pids(domain):
    services = re.search(r'^\s*services = \{(.*?)^\s*\}', domain, re.MULTILINE | re.DOTALL)
    if services is None:
        return set()
    return {int(match[1]) for match in re.finditer(
        r'^\s*(\d+)\s+[^\n]*\scom\.apple\.MTLCompilerService(?:\.[\w-]+)?\s*$',
        services[1], re.MULTILINE,
    ) if int(match[1]) > 0}


def owned_compilers(pid):
    if sys.platform != 'darwin':
        return set()
    result = subprocess.run(['launchctl', 'print', 'pid/{}'.format(pid)], capture_output=True, text=True, timeout=1, check=False)
    return compiler_pids(result.stdout)


def terminate(worker, compilers):
    try:
        os.killpg(worker.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    for pid, identity in compilers.items():
        current = memory(pid)
        if current is not None and current[1] == identity:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    worker.wait(timeout=5)


def run(command, *, seconds=180, memory_mib=1536, compile_seconds=30, interval=0.1):
    if sys.platform not in ('darwin', 'linux'):
        raise RuntimeError('development watchdog currently supports macOS and Linux')
    started = time.monotonic()
    peak = 0
    worker_peak = compiler_peak = 0
    compilers = {}
    reason = None
    with tempfile.TemporaryDirectory(prefix='wasmgpu-guard-') as directory:
        status = Path(directory) / 'phase.json'
        env = dict(os.environ, WASMGPU_GUARD_STATUS=str(status), PYTHONUNBUFFERED='1')
        worker = subprocess.Popen([sys.executable, *command], env=env, start_new_session=True)
        try:
            while worker.poll() is None:
                for pid in owned_compilers(worker.pid):
                    usage = memory(pid)
                    if usage is not None:
                        compilers[pid] = usage[1]
                usage = memory(worker.pid)
                total = usage[0] if usage else 0
                worker_peak = max(worker_peak, total)
                compiler_total = 0
                for pid, identity in compilers.items():
                    usage = memory(pid)
                    if usage is not None and usage[1] == identity:
                        total += usage[0]
                        compiler_total += usage[0]
                compiler_peak = max(compiler_peak, compiler_total)
                peak = max(peak, total)
                now = time.monotonic()
                if total > memory_mib * 1024 * 1024:
                    reason = 'memory limit'
                elif now - started > seconds:
                    reason = 'experiment timeout'
                if status.exists():
                    phase = json.loads(status.read_text())
                    if phase['compiling'] and now - phase['started'] > compile_seconds:
                        reason = 'shader compilation timeout'
                if reason:
                    terminate(worker, compilers)
                    break
                time.sleep(interval)
        finally:
            if worker.poll() is None:
                terminate(worker, compilers)
        report = {'guard': reason or 'completed', 'peak_mib': round(peak / 1048576, 1),
                  'worker_peak_mib': round(worker_peak / 1048576, 1), 'compiler_peak_mib': round(compiler_peak / 1048576, 1),
                  'seconds': round(time.monotonic() - started, 3), 'metal_compilers': len(compilers)}
        print(json.dumps(report), file=sys.stderr)  # noqa: T201 - CLI diagnostics.
        return 124 if reason else worker.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=180)
    parser.add_argument('--memory-mib', type=int, default=1536)
    parser.add_argument('--compile-seconds', type=float, default=30)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or min(args.seconds, args.memory_mib, args.compile_seconds) <= 0:
        parser.error('a Python command and positive limits are required')
    return run(command, seconds=args.seconds, memory_mib=args.memory_mib, compile_seconds=args.compile_seconds)


if __name__ == '__main__':
    raise SystemExit(main())
