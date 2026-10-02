from __future__ import annotations

import gc
import struct
import weakref
from array import array

import pytest

import wasmgpu
from tests.conftest import binary, oracle
from wasmgpu import compiler, runtime
from wasmgpu.compiler import MAX_COMPILED_INSTRUCTIONS, MAX_GENERATED_BYTES

LOOP = '''(module (memory 1 2) (global $g (mut i32) (i32.const 0))
  (func (export "run") (param $n i32) (result i32)
    block $done loop $next
      local.get $n i32.eqz br_if $done
      global.get $g i32.const 1 i32.add global.set $g
      i32.const 0 global.get $g i32.store
      local.get $n i32.const 1 i32.sub local.set $n br $next
    end end global.get $g))'''


def test_codegen_limits_and_cache():
    wasm = binary(LOOP)
    first = wasmgpu.Module(wasm, execution='compiled')
    second = wasmgpu.Module(wasm, execution='compiled')
    assert first.compiled is second.compiled
    assert first._binary is second._binary
    assert first.compiled.instructions > 0
    assert 'let v' in first.compiled.source
    assert 'invoke(' not in first.compiled.source
    assert wasmgpu.Module(wasm, execution='auto').compiled is first.compiled
    assert wasmgpu.Module(wasm, execution='interpreter').compiled is None
    with pytest.raises(wasmgpu.ResourceLimitError, match='limited'):
        wasmgpu.Module(wasm, execution='compiled', compile_limit=0)
    assert wasmgpu.Module(wasm, execution='compiled', compile_functions=[0, 0]).compiled.functions == (0,)
    with pytest.raises(wasmgpu.ResourceLimitError, match='limited'):
        wasmgpu.Module(wasm, execution='compiled', compile_limit=MAX_COMPILED_INSTRUCTIONS + 1)
    with pytest.raises(ValueError, match='function index'):
        wasmgpu.Module(wasm, execution='compiled', compile_functions=[1])
    with pytest.raises(ValueError, match='execution'):
        wasmgpu.Module(wasm, execution='cpu')


def test_source_budget_splits_units_without_losing_functions(monkeypatch):
    functions = ' '.join(f'(func (export "f{i}") (param f64) (result f64) local.get 0 f64.sqrt)' for i in range(60))
    module = wasmgpu.Module(binary('(module ' + functions + ')'), execution='interpreter')
    monkeypatch.setattr(compiler, 'MAX_GENERATED_BYTES', 32768)
    plan = compiler.compile_module(module._binary)
    assert len(plan.source.encode()) <= min(32768, MAX_GENERATED_BYTES)
    assert plan.complete
    assert len(plan.functions) == 60
    index = 0
    while index < len(plan.regions):
        assert len(plan.source_for(index).encode()) <= 32768
        index += 1
    assert len(plan.regions) > 1


@pytest.mark.parametrize('index', [True, 1.5, '0', -1])
def test_invalid_function_selection(index):
    with pytest.raises((ValueError, TypeError), match='function index'):
        wasmgpu.Module(binary(LOOP), execution='compiled', compile_functions=[index])


def test_evicted_module_is_not_retained_by_codegen_cache():
    def make(value):
        return wasmgpu.Module(binary(f'(module (func (result i32) i32.const {value}))'), execution='compiled')
    module = make(701)
    reference = weakref.ref(module._binary)
    del module
    assert reference() is not None
    for value in range(702, 707):
        make(value)
    gc.collect()
    assert reference() is None


def outcome(instances, values, fuel):
    try:
        result, traps = instances.call('run', values, fuel=fuel), {}
    except wasmgpu.Trap as error:
        result, traps = error.results, error.traps
    state = instances._batches[0].state
    # Compare continuation, memory size, trap, fuel and proc_exit, excluding counters.
    return result, traps, list(state[:6]) + list(state[7:11]), instances.read_memory(0, 4)


@pytest.mark.gpu
@pytest.mark.parametrize('quantum', [1, 3, 17, 4096])
def test_exact_fuel_prefix(quantum):
    modules = [wasmgpu.Module(binary(LOOP), execution=mode) for mode in ('compiled', 'interpreter')]
    for fuel in [1, 2, 5, 13, 37, 100]:
        with modules[0].spawn(1, quantum=quantum) as compiled, modules[1].spawn(1, quantum=quantum) as interpreted:
            assert outcome(compiled, [3], fuel) == outcome(interpreted, [3], fuel)
            assert compiled.last_call.compiled_instructions + compiled.last_call.interpreted_instructions == interpreted.last_call.interpreted_instructions
            if quantum == 4096 and fuel == 100:
                assert compiled.last_call.compiled_instructions > 0


@pytest.mark.gpu
def test_recursion_indirect_and_pipeline_cache(engine):
    wat = '''(module (type $f (func (param i32) (result i64)))
      (table 1 funcref) (elem (i32.const 0) $f)
      (func $f (export "run") (type $f)
        local.get 0 i32.eqz if (result i64) i64.const 1 else
        local.get 0 i64.extend_i32_u local.get 0 i32.const 1 i32.sub
        i32.const 0 call_indirect (type $f) i64.mul end))'''
    values = [(n,) for n in range(15)]
    module = wasmgpu.Module(binary(wat), execution='compiled')
    with module.spawn(len(values)) as instances:
        assert instances.call('run', values) == oracle(engine, wat, 'run', values)
        assert instances.last_call.interpreted_instructions == 0
        assert instances.last_call.compiled_instructions > 100
        pipeline = instances._pipeline
    with module.spawn(1) as instances:
        assert instances._pipeline is pipeline
        assert instances.pipeline_cache_hit


@pytest.mark.gpu
@pytest.mark.parametrize('indirect', [False, True])
def test_wasi_clock_and_import_continuation(indirect):
    call = 'i32.const 0 call_indirect (type $clock)' if indirect else 'call $clock'
    wat = f'''(module (type $clock (func (param i32 i64 i32) (result i32)))
      (import "wasi_snapshot_preview1" "clock_time_get" (func $clock (type $clock)))
      (memory 1) (table 1 funcref) (elem (i32.const 0) $clock)
      (func (export "run") (result i64)
        i32.const 0 i64.const 0 i32.const 0 {call} drop i32.const 0 i64.load))'''
    outputs = []
    for mode in ('compiled', 'interpreter'):
        with wasmgpu.Module(binary(wat), execution=mode).spawn(1, wasi=wasmgpu.Wasi(clock_resolution_ns=7)) as instances:
            outputs.append(instances.call('run'))
            assert instances.read_memory(0, 8) == struct.pack('<Q', outputs[-1][0])
            if mode == 'compiled':
                assert instances.last_call.compiled_instructions > 0
                assert instances.last_call.interpreted_instructions == 0
    assert outputs[0] == outputs[1]


@pytest.mark.gpu
def test_cancel_retains_metrics_and_effects():
    module = wasmgpu.Module(binary(LOOP), execution='compiled')
    with module.spawn(1, quantum=17) as instances:
        with pytest.raises(InterruptedError):
            instances.call('run', [1000], cancel=lambda: instances.last_call.dispatches == 2)
        assert instances.last_call.dispatches == 2
        assert instances.last_call.compiled_instructions > 0
        assert instances.read_memory(0, 4) != b'\0' * 4
        assert instances.call('run', [0])[0] > 0
        with pytest.raises(TypeError, match='cancel'):
            instances.call('run', [0], cancel=True)


@pytest.mark.gpu
@pytest.mark.parametrize('count', [1, 5])
def test_reset_reuses_buffers_and_restores_state(count, monkeypatch):
    wat = '''(module (memory 1 2) (data (i32.const 1) "init")
      (global $g (mut i32) (i32.const 3))
      (table 1 2 funcref) (elem (i32.const 0) $start)
      (func $start global.get $g i32.const 4 i32.add global.set $g) (start $start)
      (func (export "run") (result i32) global.get $g)
      (func (export "mutate")
        i32.const 1 memory.grow drop i32.const 1 i32.const 999 i32.store
        i32.const 0 ref.null func table.set i32.const 40 global.set $g))'''
    with wasmgpu.Module(binary(wat), execution='compiled', files={'a': b'initial'}).spawn(count, batch_size=2) as instances:
        buffers = [id(buffer) for batch in instances._batches for buffer in batch.buffers]
        assert instances.call('run') == [7] * count
        instances.call('mutate')
        assert instances.call('run') == [40] * count

        def forbidden(*_args, **_kwargs):
            raise AssertionError('reset allocated a new GPU buffer')
        monkeypatch.setattr(instances._context, 'buffer', forbidden)
        instances.reset()
        assert buffers == [id(buffer) for batch in instances._batches for buffer in batch.buffers]
        assert instances.call('run') == [7] * count
        for index in range(count):
            assert instances.read_memory(1, 4, instance=index) == b'init'
            assert instances.read_file('a', instance=index) == b'initial'
            with pytest.raises(IndexError):
                instances.read_memory(65536, 1, instance=index)
        instances.call('mutate')


@pytest.mark.gpu
def test_all_functions_and_indirect_traps(engine):
    wat = '''(module (type $t (func (param i32) (result i32))) (table 3 funcref)
      (func $f (type $t) local.get 0 i32.const 7 i32.mul)
      (func $wrong (result i64) i64.const 1)
      (elem (i32.const 0) $f $wrong)
      (func (export "run") (param i32) (result i32)
        i32.const 6 local.get 0 call_indirect (type $t)))'''
    module = wasmgpu.Module(binary(wat), execution='compiled', compile_functions=[0])
    assert module.compiled.complete
    assert oracle(engine, wat, 'run', [(0,)]) == [42]
    with module.spawn(4) as instances:
        with pytest.raises(wasmgpu.Trap) as error:
            instances.call('run', [0, 1, 2, 3])
        assert error.value.results == [42, None, None, None]
        assert error.value.traps == {1: 'indirect call type mismatch', 2: 'uninitialized element', 3: 'out of bounds table access'}
        assert instances.last_call.compiled_instructions > 4
        assert instances.last_call.interpreted_instructions == 0


@pytest.mark.gpu
@pytest.mark.parametrize(('ty', 'store', 'load', 'value'), [
    ('i32', 'store', 'load', -2147483648), ('i32', 'store8', 'load8_s', -128),
    ('i32', 'store16', 'load16_s', -32768), ('i64', 'store', 'load', -(1 << 63)),
    ('i64', 'store8', 'load8_s', -128), ('i64', 'store16', 'load16_s', -32768),
    ('i64', 'store32', 'load32_s', -2147483648), ('f32', 'store', 'load', 1.25),
    ('f64', 'store', 'load', 1e-310),
])
def test_compiled_word_loads(engine, ty, store, load, value):
    wat = f'''(module (memory 1) (func $compiled)
      (func (export "run") (param i32 {ty}) (result {ty})
        local.get 0 local.get 1 {ty}.{store} align=1 local.get 0 {ty}.{load} align=1))'''
    rows = [(offset, value) for offset in [0, 1, 2, 3, 7, 65528]]
    expected = oracle(engine, wat, 'run', rows)
    with wasmgpu.Module(binary(wat), execution='compiled', compile_functions=[0]).spawn(len(rows)) as instances:
        assert instances.call('run', rows) == expected
        assert instances.last_call.compiled_instructions > 0
        assert instances.last_call.interpreted_instructions == 0
        with pytest.raises(wasmgpu.Trap, match='out of bounds memory'):
            instances.call('run', [(-1, value)] * len(rows))


@pytest.mark.gpu
@pytest.mark.parametrize('mode', ['compiled', 'interpreter'])
@pytest.mark.parametrize('fuel', [2**32 - 1, 2**32, 2**32 + 3, 2**64 - 1])
def test_u64_fuel_borrow_and_bulk_charge(mode, fuel):
    wat = '''(module (memory 1)
      (func (export "run") (result i32)
        i32.const 0 i32.const 7 i32.const 16 memory.fill i32.const 0 i32.load))'''
    with wasmgpu.Module(binary(wat), execution=mode).spawn(2, fuel=fuel) as instances:
        assert instances.call('run') == [0x07070707] * 2
        state = instances._batches[0].state
        assert [state[base + 9] + (state[base + 12] << 32) for base in (0, 16)] == [fuel - 23] * 2
        assert instances.last_call.compiled_instructions + instances.last_call.interpreted_instructions == 14
        with pytest.raises(wasmgpu.Trap, match='fuel exhausted'):
            instances.call('run', fuel=4)
        with pytest.raises(ValueError, match='fuel'):
            instances.call('run', fuel=2**64)


def test_u64_fuel_overflow_rejected_before_gpu(monkeypatch):
    def forbidden():
        raise AssertionError('invalid fuel reached the GPU')
    monkeypatch.setattr(runtime, '_context', forbidden)
    with pytest.raises(ValueError, match='fuel'):
        wasmgpu.Module(binary(LOOP), execution='interpreter').spawn(1, fuel=2**64)


@pytest.mark.gpu
def test_hardlink_readback_and_filesystem_reset():
    wat = r'''(module
      (import "wasi_snapshot_preview1" "path_link" (func $link (param i32 i32 i32 i32 i32 i32 i32) (result i32)))
      (import "wasi_snapshot_preview1" "fd_write" (func $write (param i32 i32 i32 i32) (result i32)))
      (memory 1) (data (i32.const 0) "aalias")
      (data (i32.const 8) "\10\00\00\00\01\00\00\00X")
      (func (export "link") (result i32)
        i32.const 3 i32.const 0 i32.const 0 i32.const 1 i32.const 3 i32.const 1 i32.const 5 call $link)
      (func (export "write") (result i32)
        i32.const 1 i32.const 8 i32.const 1 i32.const 24 call $write))'''
    with wasmgpu.Module(binary(wat), execution='compiled', files={'a': b'content'}).spawn(2) as instances:
        assert instances.call('link') == [0, 0]
        assert instances.call('write') == [0, 0]
        assert instances.read_file('alias', instance=1) == b'content'
        assert instances.stdout == [b'X', b'X']
        instances.reset()
        assert instances.stdout == [b'', b'']
        assert instances.read_file('a') == b'content'
        with pytest.raises(FileNotFoundError):
            instances.read_file('alias')
        assert instances.call('link') == [0, 0]


@pytest.mark.gpu
@pytest.mark.parametrize('mode', ['compiled', 'interpreter'])
def test_instruction_counter_carry(mode):
    module = wasmgpu.Module(binary('(module (func (export "run") (result i32) i32.const 42))'), execution=mode)
    with module.spawn(1) as instances:
        with pytest.raises(InterruptedError):
            instances.call('run', cancel=lambda: True)
        batch = instances._batches[0]
        low, high = (11, 14) if mode == 'compiled' else (6, 13)
        # Seed a valid continuation near overflow to exercise the GPU carry
        # without running four billion instructions in a unit test.
        batch.state[low], batch.state[high] = 0xfffffffe, 7
        device = instances._context.device
        device.queue.write_buffer(batch.buffers[5], 0, batch.state)
        encoder = device.create_command_encoder(label='counter carry test')
        compute = encoder.begin_compute_pass()
        compute.set_pipeline(instances._pipeline)
        compute.set_bind_group(0, batch.group)
        compute.dispatch_workgroups(1)
        compute.end()
        device.queue.submit([encoder.finish()])
        state = array('I')
        state.frombytes(device.queue.read_buffer(batch.buffers[5]))
        assert state[low] == 0
        assert state[high] == 8
        assert state[7] == 1


@pytest.mark.gpu
def test_large_function_has_no_bytecode_or_interpreter(engine, monkeypatch):
    operations = 'i32.const 3 i32.add ' * 400
    wat = f'(module (func (export "run") (param i32) (result i32) local.get 0 {operations}))'
    module = wasmgpu.Module(binary(wat), compile_limit=64)
    reference = wasmgpu.Module(binary(wat), execution='interpreter')
    assert module.compiled.complete
    assert module.compiled.instructions > MAX_COMPILED_INSTRUCTIONS
    assert len(module.compiled.regions) > 1
    # No instruction stream is uploaded: the segment directory immediately
    # follows function metadata and the bytecode pointer is absent.
    assert module._program[0] == 0
    assert module._program[1] == 16 + len(module._binary.functions) * 8
    assert len(reference._program) - len(module._program) == module.compiled.total_instructions * 4

    def forbidden(_self):
        raise AssertionError('compiled execution requested the interpreter')
    monkeypatch.setattr(runtime._Context, 'pipeline', property(forbidden))
    sources = []
    context = runtime._context()
    create = context.device.create_shader_module

    def capture(**kwargs):
        sources.append(kwargs['code'])
        return create(**kwargs)
    monkeypatch.setattr(context.device, 'create_shader_module', capture)
    rows = [(0,), (7,), (-100,), (2147483640,)]
    with module.spawn(len(rows)) as instances:
        assert instances.call('run', rows) == oracle(engine, wat, 'run', rows)
        assert instances.last_call.interpreted_instructions == 0
        assert instances.last_call.compiled_instructions == module.compiled.instructions * len(rows)
    assert sources
    for source in sources:
        assert 'interpret_step' not in source
        assert 'switch op' not in source
        assert 'program[0]' not in source


@pytest.mark.gpu
def test_mutual_recursion_across_pipelines(engine):
    wat = '''(module (type $t (func (param i32) (result i32)))
      (table 2 funcref) (elem (i32.const 0) $even $odd)
      (func $even (export "run") (type $t)
        local.get 0 i32.eqz if (result i32) i32.const 1 else
          local.get 0 i32.const 1 i32.sub i32.const 1 call_indirect (type $t) end)
      (func $odd (type $t)
        local.get 0 i32.eqz if (result i32) i32.const 0 else
          local.get 0 i32.const 1 i32.sub call $even end))'''
    rows = [(n,) for n in range(12)]
    module = wasmgpu.Module(binary(wat), compile_limit=5)
    with module.spawn(len(rows), call_depth=32) as instances:
        assert instances.call('run', rows) == oracle(engine, wat, 'run', rows)
        assert instances.last_call.interpreted_instructions == 0
        assert instances.last_call.dispatches > 10


@pytest.mark.gpu
def test_exact_stack_exhaustion_in_compiled_prefix():
    wat = '''(module (memory 1)
      (func $run (export "run") (param i32) (result i32)
        i32.const 0 local.get 0 i32.store
        local.get 0 i32.const 1 i32.add call $run))'''
    results = []
    for mode in ('compiled', 'interpreter'):
        with wasmgpu.Module(binary(wat), execution=mode, compile_limit=3).spawn(1, stack_size=8, call_depth=64) as instances:
            results.append(outcome(instances, [0], 1000))
            if mode == 'compiled':
                assert instances.last_call.interpreted_instructions == 0
    assert results[0] == results[1]
    assert results[0][1] == {0: 'stack exhausted'}


@pytest.mark.gpu
def test_exported_import_is_a_compiled_continuation():
    wat = '''(module (import "wasi_snapshot_preview1" "args_sizes_get"
        (func $sizes (param i32 i32) (result i32)))
      (memory 1) (export "run" (func $sizes)))'''
    results = []
    for mode in ('compiled', 'interpreter'):
        with wasmgpu.Module(binary(wat), execution=mode).spawn(1, wasi=wasmgpu.Wasi(args=['a', 'bb'])) as instances:
            results.append((instances.call('run', [(0, 4)]), instances.read_memory(0, 8)))
            if mode == 'compiled':
                assert instances.last_call.interpreted_instructions == 0
    assert results == [([0], struct.pack('<II', 2, 5))] * 2


@pytest.mark.gpu
def test_large_native_jump_table_is_data(engine):
    targets = '$done ' * 10000
    wat = f'''(module (func (export "run") (param i32) (result i32)
      block $done (result i32) i32.const 42 local.get 0 br_table {targets}$done end))'''
    module = wasmgpu.Module(binary(wat))
    assert module.compiled.total_instructions > 10000
    assert module.compiled.instructions < 10
    assert len(module.compiled.source) < 10000
    assert module._program[0] == 0
    rows = [(0,), (1,), (9999,), (10000,), (-1,)]
    with module.spawn(len(rows)) as instances:
        assert instances.call('run', rows) == oracle(engine, wat, 'run', rows)
        assert instances.last_call.interpreted_instructions == 0


@pytest.mark.gpu
def test_small_loop_stays_in_one_native_unit():
    prefix = 'i32.const 0 drop ' * 27
    work = 'i32.const 1 drop ' * 15
    wat = f'''(module (func (export "run") (param i32) (result i32)
      {prefix} loop $next {work}
        local.get 0 i32.const 1 i32.sub local.tee 0 br_if $next
      end local.get 0))'''
    with wasmgpu.Module(binary(wat), compile_limit=64).spawn(1) as instances:
        assert instances.call('run', [1000]) == [0]
        assert instances.last_call.interpreted_instructions == 0
        # The backedge must not require a pipeline transition on every iteration.
        assert instances.last_call.dispatches < 30


@pytest.mark.gpu
def test_native_pipeline_eviction_and_compile_failure(monkeypatch):
    monkeypatch.setattr(runtime, '_PIPELINE_CACHE_ENTRIES', 2)
    modules = [wasmgpu.Module(binary(f'(module (func (export "run") (result i32) i32.const {n}))')) for n in (70123, 70124, 70125)]
    with modules[0].spawn(1) as first:
        for module in modules[1:]:
            with module.spawn(1) as instances:
                instances.call('run')
        assert len(first._context._compiled) == 2
        assert first._context._compiled_bytes == sum(first._context._compiled_sizes.values())
        assert first.call('run') == [70123]
        assert first.last_call.pipelines_created == 1
        assert first.last_call.interpreted_instructions == 0

    def failed(*_args):
        raise RuntimeError('native compiler rejected shader')
    def forbidden(_self):
        raise AssertionError('native failure fell back to the interpreter')
    monkeypatch.setattr(runtime._Context, 'compiled_pipeline', failed)
    monkeypatch.setattr(runtime._Context, 'pipeline', property(forbidden))
    with pytest.raises(RuntimeError, match='native compiler rejected'):
        modules[0].spawn(1)


@pytest.mark.gpu
def test_source_limit_can_split_one_basic_block(monkeypatch):
    monkeypatch.setattr(compiler, 'MAX_GENERATED_BYTES', 2500)
    wat = '(module (func (export "run") (result i32) i32.const 0 ' + 'i32.const 1 i32.add ' * 25 + '))'
    module = wasmgpu.Module(binary(wat))
    with module.spawn(1) as instances:
        assert instances.call('run') == [25]
        assert instances.last_call.interpreted_instructions == 0
    assert len(module.compiled.regions) > 2
