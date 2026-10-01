from __future__ import annotations

import struct

import pytest
import wasmtime

import wasmgpu


@pytest.fixture(scope='session')
def engine():
    return wasmtime.Engine()


def binary(wat):
    return bytes(wasmtime.wat2wasm(wat))


def oracle(engine, wat, name, rows):
    module = wasmtime.Module(engine, binary(wat))
    store = wasmtime.Store(engine)
    instance = wasmtime.Instance(store, module, [])
    function = instance.exports(store)[name]
    results = []
    for row in rows:
        try:
            results.append(function(store, *row))
        except wasmtime.Trap as error:
            results.append(error)
    return results


def gpu(wat, rows, name='run', **options):
    with wasmgpu.Module(binary(wat)).spawn(len(rows), **options) as instances:
        return instances.call(name, rows)


def float_bits(value, ty):
    return struct.pack('<f' if ty == 'f32' else '<d', value)
