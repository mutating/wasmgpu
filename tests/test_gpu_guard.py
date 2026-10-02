from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

from tests import gpu_guard
from tests.gpu_guard import compiler_pids, memory, run

pytestmark = pytest.mark.skipif(sys.platform not in ('darwin', 'linux'), reason='development watchdog supports macOS/Linux')


def test_compiler_ownership():
    domain = '''services = {
        0 - com.apple.MTLCompilerService
        123 (pe) com.apple.MTLCompilerService.0000-ABCD
        987 - com.apple.AnotherService
    }
    service stubs = {
        456 - com.apple.MTLCompilerService
    }'''
    assert compiler_pids(domain) == {123}
    assert compiler_pids('services = {\n}') == set()


def test_physical_memory():
    size, identity = memory(os.getpid())
    assert size > 0
    assert identity
    assert memory(2147483647) is None


def test_normal_exit():
    assert run(['-c', 'raise SystemExit(7)']) == 7


def test_timeout():
    assert run(['-c', 'import time; time.sleep(20)'], seconds=0.3) == 124


def test_memory_limit():
    assert run(['-c', 'import time; data = bytearray(80 * 1024 * 1024); time.sleep(20)'], memory_mib=64) == 124


def test_compile_timeout():
    code = 'from wasmgpu.runtime import _compile_phase; import time; _compile_phase(True); time.sleep(20)'
    assert run(['-c', code], seconds=10, compile_seconds=0.3) == 124


def test_termination_only_signals_confirmed_owned_compilers(monkeypatch):
    groups, killed = [], []
    worker = SimpleNamespace(pid=123, wait=lambda **_options: 0)
    monkeypatch.setattr(gpu_guard.os, 'killpg', lambda pid, _signal: groups.append(pid))
    monkeypatch.setattr(gpu_guard.os, 'kill', lambda pid, _signal: killed.append(pid))
    # A recycled PID must never receive a signal intended for an old compiler.
    monkeypatch.setattr(gpu_guard, 'memory', lambda pid: (100, 9) if pid == 200 else (100, 4))
    gpu_guard.terminate(worker, {200: 8, 300: 4})
    assert groups == [123]
    assert killed == [300]
