"""Reproducible, end-to-end GPU / Wasmtime comparison; run as a module."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import struct
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import wasmtime

import wasmgpu

WAT = '''(module
  (memory (export "memory") 0 64)
  (func $process (export "process") (param $x i32) (param $rounds i32) (result i32)
    block $done loop $again
      local.get $rounds i32.eqz br_if $done
      local.get $x i32.const 1664525 i32.mul i32.const 1013904223 i32.add
      local.get $x i32.const 13 i32.rotl i32.xor local.set $x
      local.get $rounds i32.const 1 i32.sub local.set $rounds br $again
    end end local.get $x)
  (func (export "batch") (param $count i32) (param $rounds i32) (result i32)
    (local $i i32) (local $sum i32) (local $value i32)
    block $done loop $again
      local.get $i local.get $count i32.ge_u br_if $done
      local.get $i local.get $rounds call $process local.set $value
      local.get $i i32.const 4 i32.mul local.get $value i32.store
      local.get $sum local.get $value i32.xor local.set $sum
      local.get $i i32.const 1 i32.add local.set $i br $again
    end end local.get $sum))'''


def median_seconds(function, repetitions):
    times = []
    for _ in range(repetitions):
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def run(count, rounds, repetitions):
    wasm = bytes(wasmtime.wat2wasm(WAT))
    engine = wasmtime.Engine()
    store = wasmtime.Store(engine)
    reference = wasmtime.Instance(store, wasmtime.Module(engine, wasm), [])
    function = reference.exports(store)['process']
    batched = reference.exports(store)['batch']
    memory = reference.exports(store)['memory']
    memory.grow(store, (count * 4 + 65535) // 65536)

    def cpu_batch():
        batched(store, count, rounds)
        return list(struct.unpack('<' + 'i' * count, memory.read(store, 0, count * 4)))
    rows = [(i, rounds) for i in range(count)]
    expected = [function(store, *row) for row in rows]
    start = time.perf_counter()
    # The workload has no state: reusing a Wasmtime instance avoids charging it
    # artificial setup costs. Both CPU paths execute exactly the same WASM code.
    with wasmgpu.Module(wasm).spawn(count, stack_size=32, call_depth=4, quantum=65536, memory_pages=0) as instances:
        spawn_seconds = time.perf_counter() - start
        actual = instances.call('process', rows)
        if actual != expected:
            raise AssertionError('GPU results differ from Wasmtime')
        checksum = 0
        for value in actual:
            checksum ^= value
        if batched(store, count, rounds) != checksum:
            raise AssertionError('batched CPU results differ')
        # Warm up all paths before measuring. GPU timing includes Python input
        # packing, upload, every dispatch/synchronization, readback and decoding.
        [function(store, *row) for row in rows]
        if cpu_batch() != expected:
            raise AssertionError('CPU batched output differs')
        gpu_seconds = median_seconds(lambda: instances.call('process', rows), repetitions)
        cpu_seconds = median_seconds(lambda: [function(store, *row) for row in rows], repetitions)
        cpu_batch_seconds = median_seconds(cpu_batch, repetitions)
        return {
            'instances': count, 'rounds': rounds, 'repetitions': repetitions,
            'gpu_call_seconds': gpu_seconds, 'wasmtime_python_loop_seconds': cpu_seconds,
            'wasmtime_wasm_loop_seconds': cpu_batch_seconds,
            'speedup_vs_python_loop': cpu_seconds / gpu_seconds,
            'speedup_vs_wasm_loop': cpu_batch_seconds / gpu_seconds,
            'spawn_seconds': spawn_seconds, 'resident_bytes': instances.resident_bytes,
            'checksum': checksum, 'adapter': instances.adapter_info,
            'wasm_sha256': hashlib.sha256(wasm).hexdigest(),
            'gpu_config': {'stack_size': 32, 'call_depth': 4, 'quantum': 65536, 'memory_pages': 0},
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--counts', type=int, nargs='+', default=[1000, 10000, 100000])
    parser.add_argument('--rounds', type=int, nargs='+', default=[1, 100])
    parser.add_argument('--repetitions', type=int, default=5)
    parser.add_argument('--output', type=Path, default=Path('tests/benchmark-results.json'))
    args = parser.parse_args()
    if min(*args.counts, *args.rounds, args.repetitions) < 1:
        parser.error('counts, rounds and repetitions must be positive')
    report = {
        'platform': platform.platform(), 'python': platform.python_version(),
        'wgpu': version('wgpu'), 'wasmtime': version('wasmtime'),
        'workload': 'i32 LCG + rotate/xor; stateless; medians after warmup',
        'timing': 'GPU call includes input packing/upload/dispatch/readback/decoding; CPU wasm loop includes reading every output into a Python list; spawn reported separately',
        'results': [run(count, rounds, args.repetitions) for rounds in args.rounds for count in args.counts],
    }
    report['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
