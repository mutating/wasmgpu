from __future__ import annotations

import pytest

import wasmgpu
from wasmgpu import runtime

from .conftest import binary

pytestmark = pytest.mark.gpu


@pytest.fixture
def instances():
    wat = '''(module (memory (export "memory") 1 2)
      (func (export "none"))
      (func (export "int") (param i32) (result i32) local.get 0)
      (func (export "float") (param f32) (result f32) local.get 0)
      (func (export "reference") (param externref) (result externref) local.get 0))'''
    with wasmgpu.Module(binary(wat)).spawn(2) as guest:
        yield guest


@pytest.mark.parametrize(('name', 'rows', 'error'), [
    ('int', None, TypeError), ('int', [1], ValueError), ('int', [1, True], TypeError),
    ('int', [1, 2**32], OverflowError), ('int', [1, -2**31 - 1], OverflowError),
    ('int', [(1, 2), (3, 4)], TypeError), ('int', ['x', 'y'], TypeError),
    ('none', [1, 2], TypeError), ('float', [True, 0], TypeError),
    ('float', [1e100, 0], OverflowError), ('reference', [1, None], TypeError),
    ('missing', None, KeyError), ('memory', None, KeyError),
])
def test_call_validation(instances, name, rows, error):
    with pytest.raises(error):
        instances.call(name, rows)
    assert instances.call('int', [12, 34]) == [12, 34]


def test_reference_and_float_values(instances):
    assert instances.call('reference', [None, None]) == [None, None]
    assert instances.call('float', [1, -2.5]) == [1.0, -2.5]
    assert instances.call('int', [2**32 - 1, -2**31]) == [-1, -2**31]
    assert instances.call('none') == [None, None]


@pytest.mark.parametrize(('offset', 'size', 'index', 'error'), [
    (-1, 1, 0, ValueError), (0, -1, 0, ValueError), (False, 1, 0, TypeError),
    (0, 1, -1, ValueError), (0, 1, 2, IndexError), (65536, 1, 0, IndexError),
    (65537, 0, 0, IndexError), (1, 65536, 0, IndexError),
])
def test_memory_read_validation(instances, offset, size, index, error):
    with pytest.raises(error):
        instances.read_memory(offset, size, instance=index)


def test_zero_length_read_write_and_absent_files(instances):
    assert instances.read_memory(65536, 0) == b''
    instances.write_memory(65536, b'')
    assert instances.stdout == [b'', b'']
    assert instances.stderr == [b'', b'']
    with pytest.raises(FileNotFoundError):
        instances.read_file('missing')
    with pytest.raises(IndexError):
        instances.write_memory(65536, b'x')


@pytest.mark.parametrize(('wat', 'options', 'error'), [
    ('(module (memory 1 2))', {'memory_pages': 0}, ValueError),
    ('(module (memory 1 2))', {'memory_pages': 3}, ValueError),
    ('(module (table 10 funcref))', {'table_elements': 5}, ValueError),
    ('(module (func (local i32 i32)))', {'stack_size': 1}, wasmgpu.ResourceLimitError),
    ('(module)', {'batch_size': 2**32 - 1}, wasmgpu.ResourceLimitError),
    ('(module)', {'batch_size': 0}, ValueError),
    ('(module (memory 0 10000))', {'memory_pages': 10000, 'max_resident_bytes': 2**32 - 1}, wasmgpu.ResourceLimitError),
    ('(module (memory 0) (data (i32.const 0) "x"))', {}, wasmgpu.Trap),
    ('(module (table 0 funcref) (func $x) (elem (i32.const 0) $x))', {}, wasmgpu.Trap),
    ('(module (func $x unreachable) (start $x))', {}, wasmgpu.Trap),
])
def test_instance_resource_and_initialization_failures(wat, options, error):
    with pytest.raises(error):
        wasmgpu.Module(binary(wat)).spawn(1, **options)


def test_gpu_allocation_failure_releases_partial_buffers(monkeypatch):
    context = runtime._context()
    original = context.buffer
    created = []
    destroyed = []

    def allocate(*args, **kwargs):
        if len(created) == 3:
            raise RuntimeError('simulated GPU allocation failure')
        buffer = original(*args, **kwargs)
        created.append(buffer)
        destroy = buffer.destroy

        def record_destroy():
            destroyed.append(buffer)
            destroy()
        monkeypatch.setattr(buffer, 'destroy', record_destroy)
        return buffer

    monkeypatch.setattr(context, 'buffer', allocate)
    with pytest.raises(RuntimeError, match='simulated GPU allocation failure'):
        wasmgpu.Module(binary('(module)')).spawn(1)
    assert set(created) == set(destroyed)
    assert len(destroyed) == len(created)


def test_environment_and_batch_buffers_count_toward_budget(monkeypatch):
    module = wasmgpu.Module(binary('(module)'))
    with module.spawn(1) as base:
        budget = base.resident_bytes
    with pytest.raises(wasmgpu.ResourceLimitError, match='configuration buffers'):
        module.spawn(1, max_resident_bytes=budget - 1)
    with pytest.raises(wasmgpu.ResourceLimitError):
        module.spawn(1, max_resident_bytes=budget, wasi=wasmgpu.Wasi(args=['x' * 4096]))
    context = runtime._context()
    monkeypatch.setitem(context.limits, 'max-buffer-size', 128)
    with pytest.raises(wasmgpu.ResourceLimitError, match='program and guest environment'):
        module.spawn(1, wasi=wasmgpu.Wasi(args=['x' * 4096]))


def test_program_buffer_allocation_failure_closes_cleanly(monkeypatch):
    def fail(*_args, **_kwargs):
        raise RuntimeError('device allocation failed')
    monkeypatch.setattr(runtime._context(), 'buffer', fail)
    with pytest.raises(RuntimeError, match='device allocation failed'):
        wasmgpu.Module(binary('(module)')).spawn(1)
