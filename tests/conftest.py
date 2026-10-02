from __future__ import annotations

import struct

import pytest
import wasmtime

import wasmgpu


def pytest_addoption(parser):
    parser.addoption('--wasmgpu-execution', choices=['auto', 'compiled', 'interpreter'], default='auto',
                     help='Default backend for the existing GPU conformance suite')


@pytest.fixture(scope='session', autouse=True)
def execution_mode(request):
    mode = request.config.getoption('--wasmgpu-execution')
    original = wasmgpu.Module.__init__
    if mode != 'auto':
        def configured(self, *args, **kwargs):
            kwargs.setdefault('execution', mode)
            original(self, *args, **kwargs)
        wasmgpu.Module.__init__ = configured
    yield
    wasmgpu.Module.__init__ = original


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
