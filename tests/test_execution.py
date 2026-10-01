from __future__ import annotations

import ast
import struct
from pathlib import Path

import pytest

import wasmgpu

from .conftest import binary, gpu, oracle

pytestmark = pytest.mark.gpu


def test_recursive_calls(engine):
    wat = '''(module (func $factorial (export "run") (param i32) (result i64)
      local.get 0 i32.eqz if (result i64) i64.const 1
      else local.get 0 i64.extend_i32_u local.get 0 i32.const 1 i32.sub call $factorial i64.mul end))'''
    rows = [(n,) for n in range(21)]
    assert gpu(wat, rows) == oracle(engine, wat, 'run', rows)


def test_loop_branch_and_locals(engine):
    wat = '''(module (func (export "run") (param $n i32) (result i32) (local $acc i32)
      block $done loop $again
        local.get $n i32.eqz br_if $done
        local.get $acc local.get $n i32.add local.set $acc
        local.get $n i32.const 1 i32.sub local.set $n br $again
      end end local.get $acc))'''
    rows = [(n,) for n in range(50)]
    assert gpu(wat, rows, quantum=7) == oracle(engine, wat, 'run', rows)


def test_br_table(engine):
    wat = '''(module (func (export "run") (param i32) (result i32)
      block $out (result i32)
        block $two block $one block $zero
          local.get 0 br_table $zero $one $two
        end i32.const 10 br $out
        end i32.const 20 br $out
        end i32.const 30
      end))'''
    rows = [(0,), (1,), (2,), (3,), (-1,)]
    assert gpu(wat, rows) == oracle(engine, wat, 'run', rows)


def test_multi_value_blocks_calls_and_returns():
    wat = '''(module
      (type $pair (func (param i32 i64) (result i64 i32)))
      (func $swap (type $pair) local.get 1 local.get 0)
      (func (export "run") (param i32 i64) (result i64 i32)
        local.get 0 local.get 1 block (type $pair) call $swap end))'''
    assert gpu(wat, [(7, 99), (-2, -100)]) == [(99, 7), (-100, -2)]


def test_branch_carries_results_and_discards_operands():
    wat = '''(module (func (export "run") (result i32)
      block (result i32) i32.const 999 i32.const 42 br 0 end))'''
    assert gpu(wat, [(), ()]) == [42, 42]


def test_indirect_calls_and_equivalent_type_indices():
    wat = '''(module
      (type $a (func (param i32) (result i32)))
      (type $b (func (param i32) (result i32)))
      (table 3 funcref)
      (func $double (type $a) local.get 0 i32.const 2 i32.mul)
      (func $wrong (result i64) i64.const 1)
      (elem (i32.const 0) $double $wrong)
      (func (export "run") (param i32 i32) (result i32)
        local.get 0 local.get 1 call_indirect (type $b)))'''
    with wasmgpu.Module(binary(wat)).spawn(4) as instances:
        with pytest.raises(wasmgpu.Trap) as info:
            instances.call('run', [(21, 0), (1, 1), (1, 2), (1, 3)])
        assert info.value.results == [42, None, None, None]
        assert info.value.traps == {1: 'indirect call type mismatch', 2: 'uninitialized element', 3: 'out of bounds table access'}


def test_persistent_and_isolated_memory_globals():
    wat = '''(module (memory (export "memory") 1 2) (global $g (mut i32) (i32.const 1))
      (data (i32.const 3) "abcd")
      (func (export "run") (param i32) (result i32)
        global.get $g local.get 0 i32.add global.set $g
        i32.const 3 global.get $g i32.store
        i32.const 3 i32.load))'''
    module = wasmgpu.Module(binary(wat))
    with module.spawn(3, batch_size=2, memory_pages=2) as first, module.spawn(1) as second:
        assert first.read_memory(3, 4) == b'abcd'
        assert first.call('run', [2, 10, 20]) == [3, 11, 21]
        assert first.call('run', [2, 10, 20]) == [5, 21, 41]
        assert first.read_memory(3, 4, instance=2) == struct.pack('<I', 41)
        assert second.call('run', [2]) == [3]
        first.write_memory(4, b'xyz', instance=1)
        assert first.read_memory(3, 4, instance=1) == b'\x15xyz'
        assert first.read_memory(3, 4, instance=0) == struct.pack('<I', 5)


@pytest.mark.parametrize(('ty', 'store', 'load', 'value'), [
    ('i32', 'store', 'load', -2147483648), ('i32', 'store8', 'load8_s', -128),
    ('i32', 'store16', 'load16_s', -32768), ('i64', 'store', 'load', -(1 << 63)),
    ('i64', 'store8', 'load8_s', -128), ('i64', 'store16', 'load16_s', -32768),
    ('i64', 'store32', 'load32_s', -2147483648), ('f32', 'store', 'load', 1.25),
    ('f64', 'store', 'load', 1e-310),
])
def test_memory_load_store(engine, ty, store, load, value):
    wat = f'''(module (memory 1)
      (func (export "run") (param {ty}) (result {ty})
        i32.const 3 local.get 0 {ty}.{store} align=1 i32.const 3 {ty}.{load} align=1))'''
    assert gpu(wat, [(value,), (value,)]) == oracle(engine, wat, 'run', [(value,), (value,)])


def test_grow_and_bounds():
    wat = '''(module (memory 1 2)
      (func (export "grow") (param i32) (result i32) local.get 0 memory.grow)
      (func (export "size") (result i32) memory.size)
      (func (export "read") (param i32) (result i32) local.get 0 i32.load))'''
    with wasmgpu.Module(binary(wat)).spawn(3) as instances:
        assert instances.call('size') == [1, 1, 1]
        assert instances.call('grow', [1, 0, 2]) == [1, 1, -1]
        assert instances.call('size') == [2, 1, 1]
        assert instances.call('read', [65536, 0, 65532]) == [0, 0, 0]
        with pytest.raises(wasmgpu.Trap) as info:
            instances.call('read', [131069, -1, 65535])
        assert set(info.value.traps) == {0, 1, 2}
        assert instances.call('size') == [2, 1, 1]


def test_bulk_memory_and_dropped_segments():
    wat = '''(module (memory 1) (data $passive "abcdef")
      (func (export "run") (result i32)
        i32.const 8 i32.const 0 i32.const 6 memory.init $passive
        i32.const 9 i32.const 8 i32.const 5 memory.copy
        i32.const 14 i32.const 90 i32.const 2 memory.fill
        i32.const 8 i32.load)
      (func (export "drop") data.drop $passive))'''
    with wasmgpu.Module(binary(wat)).spawn(2) as instances:
        assert instances.call('run') == [int.from_bytes(b'aabc', 'little')] * 2
        assert instances.read_memory(8, 8) == b'aabcdeZZ'
        instances.call('drop')
        with pytest.raises(wasmgpu.Trap, match='out of bounds memory'):
            instances.call('run')


def test_start_function_runs_on_gpu_once():
    wat = '''(module (global $x (mut i32) (i32.const 4))
      (func $start global.get $x i32.const 3 i32.mul global.set $x) (start $start)
      (func (export "run") (result i32) global.get $x))'''
    with wasmgpu.Module(binary(wat)).spawn(2) as instances:
        assert instances.call('run') == [12, 12]
        assert instances.call('run') == [12, 12]


def test_fuel_and_stack_exhaustion_are_recoverable():
    wat = '''(module (func $recurse (export "recurse") call $recurse)
      (func (export "forever") loop br 0 end)
      (func (export "ok") (result i32) i32.const 42))'''
    with wasmgpu.Module(binary(wat)).spawn(2, call_depth=8, fuel=20, quantum=3) as instances:
        with pytest.raises(wasmgpu.Trap, match='stack exhausted'):
            instances.call('recurse')
        with pytest.raises(wasmgpu.Trap, match='fuel exhausted'):
            instances.call('forever')
        assert instances.call('ok') == [42, 42]


def test_hundred_thousand_instances():
    wat = '(module (func (export "run") (param i32) (result i32) local.get 0 i32.const 3 i32.mul))'
    count = 100_003
    with wasmgpu.Module(binary(wat)).spawn(count) as instances:
        assert instances.adapter_info['adapter_type'].lower() != 'cpu'
        assert instances.call('run', range(count)) == [value * 3 for value in range(count)]


def test_zero_instances_and_close():
    module = wasmgpu.Module(binary('(module (func (export "run") (result i32) i32.const 42))'))
    with module.spawn(0) as instances:
        assert len(instances) == 0
        assert instances.call('run') == []
    instances.close()
    with pytest.raises(RuntimeError, match='closed'):
        instances.call('run')


def test_no_cpu_runtime_dependency_in_package():
    for path in (Path(__file__).parents[1] / 'wasmgpu').glob('*.py'):
        parsed = ast.parse(path.read_text())
        for node in ast.walk(parsed):
            if isinstance(node, ast.Import):
                assert all(not alias.name.startswith(('wasmtime', 'wasmer')) for alias in node.names)
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or '').startswith(('wasmtime', 'wasmer'))


def test_multiple_tables_bulk_growth_copy_and_drop():
    wat = '''(module
      (type $sig (func (result i32)))
      (func $value (type $sig) i32.const 73)
      (table $a 2 4 funcref) (table $b 1 4 funcref)
      (elem $values func $value)
      (elem declare func $value)
      (func (export "run") (result i32 i32 i32)
        i32.const 0 i32.const 0 i32.const 1 table.init $a $values
        ref.func $value i32.const 2 table.grow $b
        i32.const 1 i32.const 0 i32.const 1 table.copy $b $a
        i32.const 1 call_indirect $b (type $sig)
        table.size $b)
      (func (export "clear") i32.const 0 ref.null func i32.const 3 table.fill $b)
      (func (export "check") (result i32) i32.const 1 table.get $b ref.is_null)
      (func (export "drop") elem.drop $values))'''
    with wasmgpu.Module(binary(wat)).spawn(2) as instances:
        assert instances.call('run') == [(1, 73, 3)] * 2
        assert instances.call('check') == [0, 0]
        instances.call('clear')
        assert instances.call('check') == [1, 1]
        instances.call('drop')
        with pytest.raises(wasmgpu.Trap, match='out of bounds table'):
            instances.call('run')


def test_reference_globals_active_expression_segments_and_select():
    wat = '''(module
      (func $f (result i32) i32.const 27)
      (global $g (mut funcref) (ref.func $f))
      (global $n externref (ref.null extern))
      (table $a 1 funcref) (table $b 2 funcref)
      (elem (table $b) (i32.const 1) funcref (ref.func $f))
      (func (export "run") (result i32 i32)
        i32.const 0 global.get $g table.set $a
        i32.const 0 call_indirect $a (result i32)
        global.get $n ref.null extern i32.const 1 select (result externref) ref.is_null))'''
    assert gpu(wat, [(), ()]) == [(27, 1), (27, 1)]
