from __future__ import annotations

import os
import time

import pytest

import wasmgpu

from .conftest import binary

pytestmark = pytest.mark.gpu

READ_WRITE = '''(module
 (import "wasi_snapshot_preview1" "path_open" (func $open (param i32 i32 i32 i32 i32 i64 i64 i32 i32) (result i32)))
 (import "wasi_snapshot_preview1" "fd_read" (func $read (param i32 i32 i32 i32) (result i32)))
 (import "wasi_snapshot_preview1" "fd_write" (func $write (param i32 i32 i32 i32) (result i32)))
 (import "wasi_snapshot_preview1" "fd_seek" (func $seek (param i32 i64 i32 i32) (result i32)))
 (import "wasi_snapshot_preview1" "fd_close" (func $close (param i32) (result i32)))
 (memory (export "memory") 1)
 (data (i32.const 0) "input.txt")
 (data (i32.const 32) "\\80\\00\\00\\00\\05\\00\\00\\00")
 (func (export "run") (param i32) (result i32) (local $fd i32)
   i32.const 3 i32.const 0 i32.const 0 i32.const 9 i32.const 0
   i64.const 102 i64.const 0 i32.const 0 i32.const 16 call $open
   if unreachable end
   i32.const 16 i32.load local.set $fd
   local.get $fd i32.const 32 i32.const 1 i32.const 24 call $read if unreachable end
   i32.const 1 i32.const 32 i32.const 1 i32.const 24 call $write if unreachable end
   i32.const 128 local.get 0 i32.store8
   local.get $fd i64.const 0 i32.const 0 i32.const 48 call $seek if unreachable end
   local.get $fd i32.const 32 i32.const 1 i32.const 24 call $write if unreachable end
   local.get $fd call $close if unreachable end
   i32.const 128 i32.load8_u))'''


def test_embedded_files_run_on_gpu_and_are_isolated(monkeypatch):
    module = wasmgpu.Module(binary(READ_WRITE), files={'input.txt': b'hello'})
    with module.spawn(3) as instances:
        # Instantiation has finished. Executing file operations must not use the OS.
        def forbidden(*_args, **_kwargs):
            raise AssertionError('guest operation escaped to the host')

        monkeypatch.setattr(os, 'open', forbidden)
        monkeypatch.setattr(os, 'urandom', forbidden)
        monkeypatch.setattr(time, 'time_ns', forbidden)
        assert instances.call('run', [65, 66, 67]) == [65, 66, 67]
        assert instances.stdout == [b'hello'] * 3
        assert [instances.read_file('input.txt', instance=i) for i in range(3)] == [b'Aello', b'Bello', b'Cello']
        assert instances.call('run', [88, 89, 90]) == [88, 89, 90]
        assert instances.stdout == [b'helloAello', b'helloBello', b'helloCello']


def test_args_environment_and_stdin():
    wat = '''(module
      (import "wasi_snapshot_preview1" "args_sizes_get" (func $sizes (param i32 i32) (result i32)))
      (import "wasi_snapshot_preview1" "args_get" (func $args (param i32 i32) (result i32)))
      (import "wasi_snapshot_preview1" "environ_get" (func $env (param i32 i32) (result i32)))
      (import "wasi_snapshot_preview1" "fd_read" (func $read (param i32 i32 i32 i32) (result i32)))
      (memory 1) (data (i32.const 128) "\\00\\01\\00\\00\\03\\00\\00\\00")
      (func (export "run") (result i32)
        i32.const 0 i32.const 4 call $sizes drop
        i32.const 16 i32.const 32 call $args drop
        i32.const 64 i32.const 80 call $env drop
        i32.const 0 i32.const 128 i32.const 1 i32.const 8 call $read drop
        i32.const 0 i32.load))'''
    with wasmgpu.Module(binary(wat)).spawn(2, wasi=wasmgpu.Wasi(args=['worker', 'abc'], env={'KEY': 'value'}, stdin=b'xyz')) as instances:
        assert instances.call('run') == [2, 2]
        assert instances.read_memory(32, 11) == b'worker\0abc\0'
        assert instances.read_memory(80, 10) == b'KEY=value\0'
        assert instances.read_memory(256, 3) == b'xyz'


def test_emulated_clock_and_random_are_deterministic(monkeypatch):
    wat = '''(module
      (import "wasi_snapshot_preview1" "clock_time_get" (func $clock (param i32 i64 i32) (result i32)))
      (import "wasi_snapshot_preview1" "random_get" (func $random (param i32 i32) (result i32)))
      (memory 1)
      (func (export "run") (result i64)
        i32.const 32 i32.const 16 call $random drop
        i32.const 1 i64.const 0 i32.const 0 call $clock drop
        i32.const 0 i64.load))'''
    def forbidden(*_args, **_kwargs):
        raise AssertionError('OS service called')
    with wasmgpu.Module(binary(wat)).spawn(2, wasi=wasmgpu.Wasi(seed=81, clock_epoch_ns=1000)) as instances:
        monkeypatch.setattr(os, 'urandom', forbidden)
        monkeypatch.setattr(time, 'time_ns', forbidden)
        first = instances.call('run')
        first_random = instances.read_memory(32, 16)
        second = instances.call('run')
        assert first[0] == first[1]
        assert second[0] > first[0] >= 1000
        assert instances.read_memory(32, 16) != first_random
    with wasmgpu.Module(binary(wat)).spawn(2, wasi=wasmgpu.Wasi(seed=81, clock_epoch_ns=1000)) as instances:
        assert instances.call('run') == first
        assert instances.read_memory(32, 16) == first_random


def test_proc_exit_is_per_instance():
    wat = '''(module (import "wasi_snapshot_preview1" "proc_exit" (func $exit (param i32)))
      (func (export "run") (param i32) local.get 0 call $exit))'''
    with wasmgpu.Module(binary(wat)).spawn(2) as instances:
        with pytest.raises(wasmgpu.Trap, match='proc_exit'):
            instances.call('run', [0, 7])
        assert instances.exit_codes == [0, 7]


def test_wasi_invalid_pointers_return_errno():
    wat = '''(module (import "wasi_snapshot_preview1" "random_get" (func $random (param i32 i32) (result i32)))
      (memory 1) (func (export "run") (param i32) (result i32) local.get 0 i32.const 8 call $random))'''
    with wasmgpu.Module(binary(wat)).spawn(3) as instances:
        assert instances.call('run', [65536, -1, 65529]) == [21, 21, 21]


def test_initial_files_are_copied_and_not_host_paths():
    content = bytearray(b'hello')
    module = wasmgpu.Module(binary(READ_WRITE), files={'input.txt': content})
    content[:] = b'wrong'
    with module.spawn(1) as instances:
        assert instances.read_file('./input.txt') == b'hello'
    with pytest.raises(TypeError, match='bytes'):
        wasmgpu.Wasi(files={'input.txt': '/tmp/host-file'})
    with pytest.raises(ValueError, match='escapes'):
        wasmgpu.Wasi(files={'../secret': b'value'})
