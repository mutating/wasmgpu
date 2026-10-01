from __future__ import annotations

from types import SimpleNamespace

import pytest

import wasmgpu
from wasmgpu import runtime
from wasmgpu.binary import Reader

from .conftest import binary


@pytest.mark.parametrize('data', [
    b'', b'\0asm', b'\0asm\x02\0\0\0',
    b'\0asm\x01\0\0\0\x01\xff',
    b'\0asm\x01\0\0\0\x01\x01\xff',
    b'\0asm\x01\0\0\0\x01\x01\x00\x01\x01\x00',
    b'\0asm\x01\0\0\0\x03\x01\x00\x01\x01\x00',
    b'\0asm\x01\0\0\0\x00\x02\x01\xff',
    b'\0asm\x01\0\0\0\x01\x02\x00\x00',
    b'\0asm\x01\0\0\0\x0c\x01\x01',
])
def test_malformed_binary(data):
    with pytest.raises(wasmgpu.ValidationError):
        wasmgpu.Module(data)


@pytest.mark.parametrize('wat', [
    '(module (func (result i32)))',
    '(module (func i32.const 1))',
    '(module (func f32.const 1 i32.eqz drop))',
    '(module (func br 1))',
    '(module (func local.get 1 drop))',
    '(module (global i32 (i32.const 0)) (func i32.const 1 global.set 0))',
    '(module (func i32.const 0 i32.load drop))',
    '(module (memory 1) (func i32.const 0 i32.load align=8 drop))',
    '(module (func i32.const 1 if (result i32) i32.const 0 end drop))',
    '(module (func (result i32) i32.const 0) (start 0))',
    '(module (memory 2 1))',
    '(module (func (export "x")) (func (export "x")))',
    '(module (table 1 externref) (func i32.const 0 call_indirect))',
    '(module (table 1 funcref) (table 1 externref) (func i32.const 0 i32.const 0 i32.const 0 table.copy 0 1))',
])
def test_invalid_modules_rejected_before_gpu_execution(wat):
    with pytest.raises(wasmgpu.ValidationError):
        wasmgpu.Module(binary(wat))


@pytest.mark.parametrize('wat', [
    '(module (memory 1) (memory 1))',
    '(module (memory 1 1 shared))',
    '(module (memory i64 1))',
    '(module (import "host" "function" (func)))',
    '(module (import "host" "memory" (memory 1)))',
    '(module (func v128.const i32x4 0 0 0 0 drop))',
    '(module (type (struct (field i32))))',
])
def test_unsupported_features_rejected_explicitly(wat):
    with pytest.raises(wasmgpu.UnsupportedFeatureError):
        wasmgpu.Module(binary(wat))


def test_wrong_wasi_signature():
    with pytest.raises(wasmgpu.ValidationError, match='signature'):
        wasmgpu.Module(binary('(module (import "wasi_snapshot_preview1" "random_get" (func)))'))


@pytest.mark.parametrize(('data', 'bits', 'signed', 'expected'), [
    (b'\x80\x00', 32, False, 0), (b'\xff\x7f', 32, True, -1),
    (b'\xff\xff\xff\xff\x0f', 32, False, 2**32 - 1),
    (b'\x80\x80\x80\x80\x78', 32, True, -(2**31)),
])
def test_leb_padded_and_boundary_values(data, bits, signed, expected):
    assert Reader(data).leb(bits, signed=signed) == expected


@pytest.mark.parametrize('data', [b'\x80' * 5 + b'\x00', b'\xff\xff\xff\xff\x7f'])
def test_leb_invalid_encodings(data):
    with pytest.raises(wasmgpu.ValidationError):
        Reader(data).leb()


@pytest.mark.parametrize('source', [12, None, object()])
def test_invalid_source_type(source):
    with pytest.raises(TypeError, match='Module requires'):
        wasmgpu.Module(source)


def test_module_path_and_bytes_are_equivalent(tmp_path):
    data = binary('(module (func (export "answer") (result i32) i32.const 42))')
    path = tmp_path / 'worker.wasm'
    path.write_bytes(data)
    assert wasmgpu.Module(path).exports == wasmgpu.Module(bytearray(data)).exports == {'answer': 'function'}
    assert wasmgpu.Module(str(path)).exports == wasmgpu.Module(memoryview(data)).exports


@pytest.mark.parametrize('options', [
    {'count': -1}, {'count': True}, {'count': 2**32}, {'stack_size': 0}, {'fuel': 0},
    {'quantum': False}, {'memory_pages': 65536}, {'table_elements': -1},
    {'wasi': {}}, {'max_resident_bytes': 1},
])
def test_spawn_checks_inputs_before_gpu_allocation(options, monkeypatch):
    def forbidden():
        raise AssertionError('GPU allocation reached with invalid options')
    monkeypatch.setattr(runtime, '_context', forbidden)
    module = wasmgpu.Module(binary('(module)'))
    with pytest.raises((TypeError, ValueError, wasmgpu.ResourceLimitError)):
        module.spawn(**{'count': 1, **options})


@pytest.mark.parametrize('options', [
    {'files': {'': b'x'}}, {'files': {'a/../b': b'x', 'b': b'y'}},
    {'files': {'a\0b': b'x'}}, {'files': {'x' * 256: b'x'}},
    {'args': ['x\0y']}, {'env': {'a=b': 'c'}}, {'env': {'a': 'b\0'}},
    {'storage_size': True}, {'max_files': 3}, {'max_fds': 0},
    {'seed': 0}, {'seed': b'short'}, {'seed': False},
    {'clock_epoch_ns': -1}, {'clock_resolution_ns': 0},
])
def test_wasi_configuration_validation(options):
    with pytest.raises((TypeError, ValueError)):
        wasmgpu.Wasi(**options)


@pytest.mark.parametrize('options', [
    {'files': {'dir': b'x', 'dir/file': b'y'}},
    {'files': {'a': b'x'}, 'max_files': 4},
    {'files': {'a': b'long file'}, 'storage_size': 4},
])
def test_embedded_files_validate_before_allocation(options, monkeypatch):
    def forbidden():
        raise AssertionError('GPU was initialized')
    monkeypatch.setattr(runtime, '_context', forbidden)
    with pytest.raises((ValueError, wasmgpu.ResourceLimitError)):
        wasmgpu.Module(binary('(module)')).spawn(1, wasi=wasmgpu.Wasi(**options))


@pytest.mark.parametrize('adapter', [None, SimpleNamespace(info={'adapter_type': 'CPU'})])
def test_no_cpu_fallback_even_when_wgpu_offers_cpu(adapter, monkeypatch):
    backend = SimpleNamespace(gpu=SimpleNamespace(request_adapter_sync=lambda **_kwargs: adapter))
    monkeypatch.setattr(runtime.importlib, 'import_module', lambda _name: backend)
    with pytest.raises(wasmgpu.GPUUnavailableError, match='hardware GPU'):
        runtime._Context()


def test_adapter_failure_is_explicit(monkeypatch):
    def fail(**_kwargs):
        raise RuntimeError('driver unavailable')
    backend = SimpleNamespace(gpu=SimpleNamespace(request_adapter_sync=fail))
    monkeypatch.setattr(runtime.importlib, 'import_module', lambda _name: backend)
    with pytest.raises(wasmgpu.GPUUnavailableError, match='driver unavailable'):
        runtime._Context()


@pytest.mark.parametrize(('sections', 'error'), [
    (b'\x01\x01\x01', wasmgpu.ValidationError),  # type vector with no entries
    (b'\x01\x05\x01\x60\x01\x7b\x00', wasmgpu.UnsupportedFeatureError),
    (b'\x0d\x01\x00', wasmgpu.UnsupportedFeatureError),  # tag section
    (b'\x01\x04\x01\x60\x00\x00\x03\x02\x01\x00', wasmgpu.ValidationError),
    (b'\x04\x04\x01\x7f\x00\x01', wasmgpu.ValidationError),  # numeric table
    (b'\x06\x06\x01\x70\x00\xd0\x7f\x0b', wasmgpu.ValidationError),
    (b'\x06\x06\x01\x7f\x00\x23\x00\x0b', wasmgpu.UnsupportedFeatureError),
    (b'\x06\x06\x01\x7e\x00\x41\x00\x0b', wasmgpu.ValidationError),
    (b'\x06\x06\x01\x7f\x00\x41\x00\x01', wasmgpu.ValidationError),
    (b'\x06\x06\x01\x7f\x02\x41\x00\x0b', wasmgpu.ValidationError),
    (b'\x0b\x02\x01\x03', wasmgpu.ValidationError),  # data flags
    (b'\x0b\x03\x01\x02\x01', wasmgpu.ValidationError),  # memory index
    (b'\x09\x02\x01\x08', wasmgpu.ValidationError),  # element flags
    (b'\x09\x05\x01\x00\x41\x00\x0b', wasmgpu.ValidationError),
    (b'\x09\x03\x01\x01\x01', wasmgpu.ValidationError),  # elemkind
    (b'\x09\x04\x01\x05\x7f\x00', wasmgpu.ValidationError),
])
def test_invalid_section_encodings(sections, error):
    with pytest.raises(error):
        wasmgpu.Module(b'\0asm\x01\0\0\0' + sections)


@pytest.mark.parametrize(('instructions', 'message'), [
    (b'\x02\x01\x0b', 'block type'),
    (b'\x05', 'unexpected else'),
    (b'\xd0\x7f\x1a', 'reference type'),
    (b'\x41\x00\xd1\x1a', 'requires reference'),
    (b'\x3f\x01\x1a', 'memory index'),
])
def test_invalid_instruction_immediates(instructions, message):
    body = b'\x00' + instructions + b'\x0b'
    sections = b'\x01\x04\x01\x60\x00\x00\x03\x02\x01\x00\x05\x03\x01\x00\x01'
    code = b'\x01' + bytes([len(body)]) + body
    with pytest.raises(wasmgpu.ValidationError, match=message):
        wasmgpu.Module(b'\0asm\x01\0\0\0' + sections + b'\x0a' + bytes([len(code)]) + code)


def test_reference_declaration_and_table_init_types():
    for wat in [
        '(module (func $f) (func ref.func $f drop))',
        '(module (table 1 funcref) (elem $e externref (ref.null extern)) (func i32.const 0 i32.const 0 i32.const 1 table.init $e))',
    ]:
        with pytest.raises(wasmgpu.ValidationError):
            wasmgpu.Module(binary(wat))


def test_locals_limit_and_unknown_prefixed_opcode():
    header = b'\0asm\x01\0\0\0\x01\x04\x01\x60\x00\x00\x03\x02\x01\x00'
    for body in [b'\x01\x81\x80\x04\x7f\x0b', b'\x00\xfc\x7f\x0b']:
        code = b'\x01' + bytes([len(body)]) + body
        with pytest.raises(wasmgpu.UnsupportedFeatureError):
            wasmgpu.Module(header + b'\x0a' + bytes([len(code)]) + code)


@pytest.mark.parametrize('budget', [0, False, 1.5])
def test_invalid_resident_budget(budget):
    with pytest.raises((TypeError, ValueError)):
        wasmgpu.Module(binary('(module)')).spawn(1, max_resident_bytes=budget)


def test_oversized_filesystem_rejected_before_constructing_heap(monkeypatch):
    def forbidden():
        raise AssertionError('GPU allocation reached')
    monkeypatch.setattr(runtime, '_context', forbidden)
    with pytest.raises(wasmgpu.ResourceLimitError):
        wasmgpu.Module(binary('(module)'), files={'x': b'x'}).spawn(1, wasi=wasmgpu.Wasi(max_files=2**30))
