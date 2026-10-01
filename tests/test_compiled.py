from __future__ import annotations

from pathlib import Path

import pytest
import wasmtime

import wasmgpu

pytestmark = pytest.mark.gpu

FIXTURE = Path(__file__).parent / 'fixtures' / 'worker.wasm'
RUST_FIXTURE = FIXTURE.with_name('worker-rust.wasm')


@pytest.mark.parametrize('fixture', [FIXTURE, RUST_FIXTURE], ids=['C', 'Rust'])
def test_malloc_and_double_arithmetic(engine, fixture):
    data = fixture.read_bytes()
    store = wasmtime.Store(engine)
    linker = wasmtime.Linker(engine)
    linker.define_wasi()
    store.set_wasi(wasmtime.WasiConfig())
    reference = linker.instantiate(store, wasmtime.Module(engine, data))
    if '_initialize' in reference.exports(store):
        reference.exports(store)['_initialize'](store)
    values = [0, 1, 2, 5, 10, 33, 100]
    expected = [reference.exports(store)['process'](store, n) for n in values]
    with wasmgpu.Module(data).spawn(len(values), stack_size=2048, call_depth=128, memory_pages=32) as instances:
        if '_initialize' in instances.module.exports:
            instances.call('_initialize')
        assert instances.call('process', values) == expected


def test_c_stdio_strtod_printf_and_embedded_files(engine, tmp_path):
    data = FIXTURE.read_bytes()
    content = b'1.25\n2.5\n-0.125\n'
    (tmp_path / 'numbers.txt').write_bytes(content)
    store = wasmtime.Store(engine)
    wasi = wasmtime.WasiConfig()
    wasi.preopen_dir(str(tmp_path), '.')
    wasi.stdout_file = str(tmp_path / 'stdout')
    store.set_wasi(wasi)
    linker = wasmtime.Linker(engine)
    linker.define_wasi()
    reference = linker.instantiate(store, wasmtime.Module(engine, data))
    reference.exports(store)['_initialize'](store)
    expected = reference.exports(store)['file_process'](store)
    expected_file = (tmp_path / 'result.txt').read_bytes()
    with wasmgpu.Module(data, files={'numbers.txt': content}).spawn(2, stack_size=2048, call_depth=128, memory_pages=32) as instances:
        instances.call('_initialize')
        assert instances.call('file_process') == [expected] * 2
        assert instances.read_file('result.txt') == expected_file
        assert instances.read_file('result.txt', instance=1) == expected_file
        assert instances.stdout == [b'processed\n'] * 2


def test_rust_std_files_parsing_and_formatting(engine, tmp_path):
    data = RUST_FIXTURE.read_bytes()
    content = b'1.25\n2.5\n-0.125\n'
    (tmp_path / 'numbers.txt').write_bytes(content)
    store = wasmtime.Store(engine)
    wasi = wasmtime.WasiConfig()
    wasi.preopen_dir(str(tmp_path), '.')
    store.set_wasi(wasi)
    linker = wasmtime.Linker(engine)
    linker.define_wasi()
    reference = linker.instantiate(store, wasmtime.Module(engine, data))
    expected = reference.exports(store)['file_process'](store)
    expected_file = (tmp_path / 'rust-result.txt').read_bytes()
    with wasmgpu.Module(data, files={'numbers.txt': content}).spawn(2, stack_size=4096, call_depth=128, memory_pages=32) as instances:
        assert instances.call('file_process') == [expected] * 2
        assert instances.read_file('rust-result.txt') == expected_file
        assert instances.read_file('rust-result.txt', instance=1) == expected_file
