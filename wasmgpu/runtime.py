"""GPU execution and Python API. No CPU WebAssembly executor is used here."""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import struct
import sys
import threading
import time
from array import array
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence, Tuple, cast

from .binary import BRANCH_TABLE, F32, F64, I32, I64, BinaryModule
from .compiler import (
    MAX_COMPILED_INSTRUCTIONS,
    CompiledModule,
    compile_module,
    vm_support,
)
from .errors import GPUUnavailableError, ResourceLimitError, Trap
from .types import (
    AdapterInfo,
    Backend,
    Blob,
    Buffer,
    Pipeline,
    RequestAdapter,
    RequestDevice,
    Result,
    Row,
    Scalar,
)
from .wasi import WASI_IDS, Wasi, normalize_path

_TRAPS: dict[int, str] = {
    1: 'unreachable', 2: 'out of bounds memory access', 3: 'out of bounds table access',
    4: 'integer divide by zero', 5: 'integer overflow', 6: 'invalid conversion to integer',
    7: 'stack exhausted', 8: 'fuel exhausted', 9: 'uninitialized element',
    10: 'indirect call type mismatch', 11: 'invalid runtime state', 12: 'WASI proc_exit', 13: 'WASI proc_raise',
}
_CONTEXT: _Context | None = None
_CONTEXT_LOCK = threading.Lock()
_MODULE_CACHE: OrderedDict[bytes, tuple[BinaryModule, array[int]]] = OrderedDict()
_CODE_CACHE: OrderedDict[tuple[BinaryModule, tuple[int, ...] | None, int, int], CompiledModule] = OrderedDict()
_MODULE_LOCK = threading.RLock()
_STATE_WORDS = 16
_PIPELINE_CACHE_BYTES = 16 * 1024 * 1024
_PIPELINE_CACHE_ENTRIES = 128


def _compile_phase(active: bool) -> None:
    """Notify the external development watchdog without executing guest code."""
    destination = os.environ.get('WASMGPU_GUARD_STATUS')
    if destination:
        path = Path(destination)
        temporary = path.with_suffix('.tmp')
        status: dict[str, bool | float] = {'compiling': active, 'started': time.monotonic()}
        temporary.write_text(json.dumps(status))
        temporary.replace(path)


@dataclass
class CallMetrics:
    """Wall-clock phases; execution includes dispatch and completion-state synchronization."""

    codegen_seconds: float = 0.0
    compile_seconds: float = 0.0
    pipelines_created: int = 0
    pipeline_cache_hits: int = 0
    prepare_seconds: float = 0.0
    upload_seconds: float = 0.0
    execute_seconds: float = 0.0
    max_dispatch_seconds: float = 0.0
    readback_seconds: float = 0.0
    decode_seconds: float = 0.0
    dispatches: int = 0
    compiled_instructions: int = 0
    interpreted_instructions: int = 0
    function_samples: dict[int, int] = field(default_factory=dict)


def _words(blob: Blob) -> array[int]:
    words = array('I')
    words.frombytes(blob)
    return words


def _positive(value: int, name: str, zero: bool = False, *, bits: int = 32) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f'{name} must be an integer')
    if value < (0 if zero else 1) or value >= 1 << bits:
        raise ValueError(f'{name} out of range')
    return value


def _encode(value: Scalar, ty: int) -> int:
    if ty in (I32, I64):
        bits = 32 if ty == I32 else 64
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError('integer WASM arguments require Python int')
        if not -(1 << (bits - 1)) <= value < (1 << bits):
            raise OverflowError(f'integer does not fit i{bits}')
        return value & ((1 << bits) - 1)
    if ty in (F32, F64):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError('floating WASM arguments require int or float')
        try:
            return int.from_bytes(struct.pack('<f' if ty == F32 else '<d', value), 'little')
        except (OverflowError, struct.error) as error:
            raise OverflowError('floating argument out of range') from error
    if value is not None:
        raise TypeError('only null references may cross the Python API')
    return 0


def _decode(value: int, ty: int) -> Scalar:
    if ty in (I32, I64):
        bits = 32 if ty == I32 else 64
        value &= (1 << bits) - 1
        return value - (1 << bits) if value >> (bits - 1) else value
    if ty in (F32, F64):
        size = 4 if ty == F32 else 8
        return cast(Tuple[float, ...], struct.unpack('<f' if size == 4 else '<d', (value & ((1 << (size * 8)) - 1)).to_bytes(size, 'little')))[0]
    return None if value == 0 else value - 1


class _Context:
    def __init__(self) -> None:
        wgpu = cast(Backend, importlib.import_module('wgpu'))
        self.wgpu = wgpu
        try:
            request_adapter = cast(RequestAdapter, cast(object, getattr(wgpu.gpu, 'request_adapter_sync', None)) or cast(object, wgpu.gpu.request_adapter))
            self.adapter = request_adapter(power_preference='high-performance')
            if self.adapter is None or str(self.adapter.info.get('adapter_type', '')).lower() == 'cpu':
                raise GPUUnavailableError('a hardware GPU is required; CPU adapters are rejected')
            request_device = cast(RequestDevice, cast(object, getattr(self.adapter, 'request_device_sync', None)) or cast(object, self.adapter.request_device))
            requested = {key: min(256 * 1024 * 1024, value) for key, value in self.adapter.limits.items()
                         if key.replace('_', '-') in ('max-storage-buffer-binding-size', 'max-buffer-size')}
            self.device = request_device(required_limits=requested)
            self.limits = {key.replace('_', '-'): value for key, value in self.device.limits.items()}
        except GPUUnavailableError:
            raise
        except Exception as error:
            raise GPUUnavailableError(f'could not initialize a hardware GPU: {error}') from error
        constants = '\n'.join(f'const WASI_{name.upper()}: u32 = {index}u;' for name, index in WASI_IDS.items())
        source = constants + '\n' + '\n'.join((Path(__file__).parent / name).read_text() for name in ('numeric.wgsl', 'operations.wgsl', 'vm.wgsl', 'filesystem.wgsl'))
        self._interpreter_source = source
        self._interpreter: Pipeline | None = None
        self._initializer: Pipeline | None = None
        self._services: Pipeline | None = None
        self._compiled: OrderedDict[str, Pipeline] = OrderedDict()
        self._compiled_sizes: dict[str, int] = {}
        self._compiled_bytes = 0
        self._compile_lock = threading.Lock()
        self._constants = constants
        self._binding_layout = self.device.create_bind_group_layout(entries=[
            {'binding': index, 'visibility': 4, 'buffer': {'type': 'uniform' if index == 1 else 'read-only-storage' if index == 0 else 'storage'}}
            for index in range(7)
        ])
        self._pipeline_layout = self.device.create_pipeline_layout(bind_group_layouts=[self._binding_layout])

    @property
    def pipeline(self) -> Pipeline:
        with self._compile_lock:
            if self._interpreter is None:
                _compile_phase(True)
                try:
                    shader = self.device.create_shader_module(label='wasmgpu interpreter', code=self._interpreter_source)
                    self._interpreter = self.device.create_compute_pipeline(layout=self._pipeline_layout, compute={'module': shader, 'entry_point': 'run'})
                finally:
                    _compile_phase(False)
            return self._interpreter

    def compiled_pipeline(self, generated: str) -> tuple[Pipeline, bool]:
        key = hashlib.sha256(generated.encode()).hexdigest()
        with self._compile_lock:
            if key in self._compiled:
                self._compiled.move_to_end(key)
                return self._compiled[key], True
            root = Path(__file__).parent
            vm = vm_support()
            filesystem = '''
fn wasi_dispatch(syscall: u32, base: u32) -> u32 {
    output[lane] = vec2u(syscall, base); vm.status = 5u; return 0u;
}
fn fs_charge(amount: u32) -> bool {
    if !fuel_available(amount) { fail(8u); return false; }
    consume_fuel(amount); return true;
}'''
            source = '\n'.join([self._constants, (root / 'numeric.wgsl').read_text(), 'struct NumericResult { value: vec2u, trap: u32 }', vm,
                                filesystem, generated, (root / 'compiled.wgsl').read_text()])
            _compile_phase(True)
            try:
                shader = self.device.create_shader_module(label='wasmgpu compiled module', code=source)
                pipeline = self.device.create_compute_pipeline(layout=self._pipeline_layout, compute={'module': shader, 'entry_point': 'run'})
            finally:
                _compile_phase(False)
            self._compiled[key] = pipeline
            cost = len(generated.encode())
            self._compiled_sizes[key] = cost
            self._compiled_bytes += cost
            while len(self._compiled) > 1 and (len(self._compiled) > _PIPELINE_CACHE_ENTRIES or self._compiled_bytes > _PIPELINE_CACHE_BYTES):
                evicted, _ = self._compiled.popitem(last=False)
                self._compiled_bytes -= self._compiled_sizes.pop(evicted)
            return pipeline, False

    @property
    def services(self) -> Pipeline:
        with self._compile_lock:
            if self._services is None:
                root = Path(__file__).parent
                vm = vm_support()
                source = '\n'.join([self._constants, (root / 'numeric.wgsl').read_text(), vm,
                                    (root / 'filesystem.wgsl').read_text(), (root / 'services.wgsl').read_text()])
                _compile_phase(True)
                try:
                    shader = self.device.create_shader_module(label='wasmgpu GPU services', code=source)
                    self._services = self.device.create_compute_pipeline(layout=self._pipeline_layout, compute={'module': shader, 'entry_point': 'service'})
                finally:
                    _compile_phase(False)
            return self._services

    @property
    def initializer(self) -> Pipeline:
        with self._compile_lock:
            if self._initializer is None:
                _compile_phase(True)
                try:
                    shader = self.device.create_shader_module(label='wasmgpu initialization', code=Path(__file__).with_name('initialize.wgsl').read_text())
                    self._initializer = self.device.create_compute_pipeline(layout='auto', compute={'module': shader, 'entry_point': 'initialize'})
                finally:
                    _compile_phase(False)
            return self._initializer

    def buffer(self, size: int = 0, data: Blob | array[int] | None = None, uniform: bool = False) -> Buffer:
        flags = self.wgpu.BufferUsage
        usage = flags.COPY_DST | flags.COPY_SRC | (flags.UNIFORM if uniform else flags.STORAGE)
        if data is not None:
            return self.device.create_buffer_with_data(data=data, usage=usage)
        return self.device.create_buffer(size=max(16, size), usage=usage)


def _context() -> _Context:
    global _CONTEXT  # noqa: PLW0603 - Lazy process-wide device, protected by a lock.
    with _CONTEXT_LOCK:
        if _CONTEXT is None:
            _CONTEXT = _Context()
        return _CONTEXT


def _program(module: BinaryModule, *, bytecode: bool = True) -> array[int]:
    canonical = [module.types.index(signature) for signature in module.types]
    words = [0] * 16
    code: list[int] = []
    offset = 0
    for fn in module.functions:
        params, results = module.types[fn.type_index]
        instructions = fn.instructions
        if fn.imported is not None:
            fn.locals = list(params)
            instructions = [[0x20, index, 0, 0] for index in range(len(params))] + [[0x10, module.functions.index(fn), 0, 0], [0x0F, 0, 0, 0]]
        fn.offset = offset
        offset += len(instructions)
        words.extend([fn.offset, len(params), len(results), len(fn.locals), canonical[fn.type_index], WASI_IDS[fn.imported[1]] if fn.imported else 0, 0, 0])
        for op, operand, b, c in instructions if bytecode else ():
            a = operand
            if 0x100 <= op <= 0x104:
                a += fn.offset
            elif op == 0x11:
                a = canonical[a]
            code.extend([op, a & 0xFFFFFFFF, b & 0xFFFFFFFF, c & 0xFFFFFFFF])
    words[0] = len(words) if bytecode else 0
    words.extend(code)
    words[1] = len(words)
    words.extend([0] * (len(module.data) * 2))
    words[2] = len(words)
    words.extend([0] * (len(module.elements) * 2))
    words[3] = len(module.functions)
    for i, (_, blob) in enumerate(module.data):
        words[words[1] + i * 2:words[1] + i * 2 + 2] = [len(words), len(blob)]
        words.extend(_words(blob + b'\0' * (-len(blob) % 4)))
    for i, (_, elements, _, _, _) in enumerate(module.elements):
        words[words[2] + i * 2:words[2] + i * 2 + 2] = [len(words), len(elements)]
        words.extend(elements)
    if not bytecode:
        # Native jump tables contain continuation addresses and stack moves,
        # never opcodes. Keeping them in data avoids enormous shader switches.
        words[5] = len(words)
        for function in module.functions:
            for op, start, count, _ in function.instructions:
                if op == BRANCH_TABLE:
                    for _, target, height, arity in function.instructions[start:start + count]:
                        words.extend([function.offset + target, height, arity])
    return array('I', words)


def _load_module(data: bytes) -> tuple[BinaryModule, array[int]]:
    with _MODULE_LOCK:
        if data not in _MODULE_CACHE:
            binary = BinaryModule(data)
            Wasi.validate_imports(binary)
            _MODULE_CACHE[data] = binary, _program(binary)
        _MODULE_CACHE.move_to_end(data)
        while len(_MODULE_CACHE) > 1 and (len(_MODULE_CACHE) > 4 or sum(map(len, _MODULE_CACHE)) > 64 * 1024 * 1024):
            _, (evicted, _) = _MODULE_CACHE.popitem(last=False)
            # Code-cache keys must not keep large evicted syntax trees alive.
            for key in list(_CODE_CACHE):
                if key[0] is evicted:
                    del _CODE_CACHE[key]
        return _MODULE_CACHE[data]


def _compile_module(binary: BinaryModule, functions: tuple[int, ...] | None, limit: int) -> CompiledModule:
    with _MODULE_LOCK:
        # The older Metal backend used by Python 3.8 exhausted compiler
        # memory on a larger single shader. Keep each of its units smaller.
        source_limit = 16384 if sys.version_info < (3, 9) else 65536
        key = binary, functions, limit, source_limit
        if key not in _CODE_CACHE:
            _CODE_CACHE[key] = compile_module(binary, functions, instruction_limit=limit, source_limit=source_limit)
            _CODE_CACHE[key].program = _program(binary, bytecode=False)
        _CODE_CACHE.move_to_end(key)
        while len(_CODE_CACHE) > 16:
            _CODE_CACHE.popitem(last=False)
        return _CODE_CACHE[key]


class Module:
    """Validate a binary WASM file. Function bodies execute only on the GPU.

    ``source`` is a path or binary bytes. WAT conversion is intentionally not a
    runtime dependency; tests use Wasmtime's assembler.
    """

    def __init__(self, source: str | os.PathLike[str] | Blob, *, files: Mapping[str, Blob] | None = None,
                 execution: str = 'auto', compile_functions: Iterable[int] | None = None, compile_limit: int = MAX_COMPILED_INSTRUCTIONS) -> None:
        started = time.perf_counter()
        if execution not in ('auto', 'compiled', 'interpreter'):
            raise ValueError('execution must be auto, compiled or interpreter')
        _positive(compile_limit, 'compile_limit', zero=True)
        if isinstance(source, (str, os.PathLike)):
            data = Path(source).read_bytes()
        elif isinstance(source, (bytes, bytearray, memoryview)):
            data = bytes(source)
        else:
            raise TypeError('Module requires a filesystem path or WASM bytes')
        self._binary, self._program = _load_module(data)
        self._files = Wasi(files=files).files
        self.load_seconds = time.perf_counter() - started
        started = time.perf_counter()
        self.execution = execution
        selection = tuple(_positive(index, 'compiled function index', zero=True) for index in compile_functions) if compile_functions is not None else None
        self.compiled = _compile_module(self._binary, selection, compile_limit) if execution != 'interpreter' else None
        if self.compiled is not None:
            self._program = self.compiled.program
        self.codegen_seconds = time.perf_counter() - started

    @property
    def exports(self) -> dict[str, str]:
        """Export names mapped to kind names."""
        kinds = ('function', 'table', 'memory', 'global')
        return {name: kinds[kind] for name, (kind, _) in self._binary.exports.items()}

    def spawn(self, count: int, *, memory_pages: int | None = None, table_elements: int | None = None,  # noqa: PLR0913 - Explicit independent resource budgets.
              stack_size: int = 256, call_depth: int = 64, fuel: int = 10_000_000, quantum: int = 4096,
              batch_size: int | None = None, max_resident_bytes: int = 512 * 1024 * 1024,
              wasi: Wasi | None = None) -> Instances:
        """Create isolated, persistent instances on a hardware GPU.

        memory_pages/table_elements bound growth, and do not alter initial sizes.
        The resident allocation budget is checked before any GPU allocation.
        """
        return Instances(self, count, memory_pages=memory_pages, table_elements=table_elements,
                         stack_size=stack_size, call_depth=call_depth, fuel=fuel, quantum=quantum,
                         batch_size=batch_size, max_resident_bytes=max_resident_bytes, wasi=wasi)


class _Batch:
    def __init__(self, owner: Instances, count: int, first: int) -> None:
        self.owner, self.count, self.first = owner, count, first
        self.context = owner._context
        device = self.context.device
        binary = owner.module._binary
        self.buffers: list[Buffer] = []
        self.initialization_config: Buffer | None = None
        try:
            assert owner._program_buffer is not None
            self.buffers = [owner._program_buffer]
            config = array('I', [count, owner.stack_size, owner.call_depth, owner.memory_pages,
                                owner._memory_offset, owner._table_offset, owner.table_elements, owner._data_flags,
                                owner._element_flags, owner.quantum, owner._output_slots, 0,
                                owner._fs_offset, owner._wasi.max_files, owner._wasi.max_fds, owner._wasi.storage_size])
            self.buffers.append(self.context.buffer(data=config, uniform=True))
            self.buffers.append(self.context.buffer(size=count * owner.stack_size * 8))
            self.buffers.append(self.context.buffer(size=owner._heap_words * count * 4))
            self.buffers.append(self.context.buffer(size=count * owner.call_depth * 16))
            state = array('I', [0, 0, 0, 0, 0, binary.memory[0] if binary.memory else 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) * count
            self.buffers.append(self.context.buffer(data=state))
            self.buffers.append(self.context.buffer(size=count * owner._output_slots * 8))
            self.state = state
            self.group = device.create_bind_group(layout=owner._pipeline.get_bind_group_layout(0), entries=[
                {'binding': index, 'resource': {'buffer': buffer, 'offset': 0, 'size': buffer.size}}
                for index, buffer in enumerate(self.buffers)
            ])
            if owner._template_buffer is not None:
                self.initialization_config = self.context.buffer(data=array('I', [count, owner._heap_words,
                    owner._fs_offset + 18 if owner._uses_wasi else 0xffffffff, first]), uniform=True)
                self.initialization_group = device.create_bind_group(layout=self.context.initializer.get_bind_group_layout(0), entries=[
                    {'binding': index, 'resource': {'buffer': buffer, 'offset': 0, 'size': buffer.size}}
                    for index, buffer in enumerate((owner._template_buffer, self.buffers[3], self.initialization_config))
                ])
            self.initialize_heap()
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        for buffer in self.buffers[1:]:
            buffer.destroy()
        self.buffers = []
        if self.initialization_config is not None:
            self.initialization_config.destroy()
            self.initialization_config = None

    def initialize_heap(self) -> None:
        if self.owner._template_buffer is None:
            self.context.device.queue.write_buffer(self.buffers[3], 0, self.owner._initial_words)
        else:
            encoder = self.context.device.create_command_encoder(label='wasmgpu initialize batch')
            compute = encoder.begin_compute_pass()
            compute.set_pipeline(self.context.initializer)
            compute.set_bind_group(0, self.initialization_group)
            compute.dispatch_workgroups(min(65535, (self.count * self.owner._heap_words + 255) // 256))
            compute.end()
            self.context.device.queue.submit([encoder.finish()])

    def execute(self, function_index: int, inputs: Sequence[tuple[int, ...]], fuel: int, raw: bool = False,  # noqa: PLR0915 - Dispatch, phase timing and decoding.
                cancel: Callable[[], bool] | None = None) -> tuple[list[Result], dict[int, str]]:
        owner = self.owner
        device = self.context.device
        fn = owner.module._binary.functions[function_index]
        _, returns = owner.module._binary.signature(function_index)
        metrics = owner.last_call
        started = time.perf_counter()
        arguments = array('Q', [0]) * (self.count * max(1, len(fn.locals)))
        for lane, row in enumerate(inputs):
            for index, value in enumerate(row):
                arguments[index * self.count + lane] = value
            base = lane * _STATE_WORDS
            pages = self.state[base + 5]
            self.state[base:base + _STATE_WORDS] = array('I', [fn.offset, len(fn.locals), 0, function_index, 0, pages, 0, 0, 0, fuel & 0xffffffff, 0, 0, fuel >> 32, 0, 0, 0])
        metrics.prepare_seconds += time.perf_counter() - started
        started = time.perf_counter()
        device.queue.write_buffer(self.buffers[2], 0, arguments)
        device.queue.write_buffer(self.buffers[5], 0, self.state)
        metrics.upload_seconds += time.perf_counter() - started
        previous_compiled = previous_interpreted = 0
        last_region = -1
        while True:
            if cancel is not None and cancel():
                raise InterruptedError('GPU invocation cancelled between dispatches')
            if any(status == 5 for status in self.state[7::_STATE_WORDS]):
                pipeline = self.context.services
            elif owner.module.compiled is not None:
                plan = owner.module.compiled
                regions = sorted({plan.region_for(self.state[base]) for base in range(0, len(self.state), _STATE_WORDS) if self.state[base + 7] == 0})
                region = next((index for index in regions if index > last_region), regions[0])
                pipeline = owner._native_pipeline(region, metrics)
                last_region = region
            else:
                pipeline = owner._pipeline
            if cancel is not None and cancel():
                raise InterruptedError('GPU invocation cancelled between dispatches')
            started = time.perf_counter()
            encoder = device.create_command_encoder(label='wasmgpu dispatch')
            compute = encoder.begin_compute_pass()
            compute.set_pipeline(pipeline)
            compute.set_bind_group(0, self.group)
            compute.dispatch_workgroups((self.count + 63) // 64)
            compute.end()
            device.queue.submit([encoder.finish()])
            self.state = array('I')
            self.state.frombytes(device.queue.read_buffer(self.buffers[5]))
            elapsed = time.perf_counter() - started
            metrics.execute_seconds += elapsed
            metrics.max_dispatch_seconds = max(metrics.max_dispatch_seconds, elapsed)
            metrics.dispatches += 1
            compiled = sum(self.state[11::_STATE_WORDS]) + (sum(self.state[14::_STATE_WORDS]) << 32)
            interpreted = sum(self.state[6::_STATE_WORDS]) + (sum(self.state[13::_STATE_WORDS]) << 32)
            metrics.compiled_instructions += compiled - previous_compiled
            metrics.interpreted_instructions += interpreted - previous_interpreted
            previous_compiled, previous_interpreted = compiled, interpreted
            for function in self.state[3::_STATE_WORDS]:
                metrics.function_samples[function] = metrics.function_samples.get(function, 0) + 1
            statuses = self.state[7::_STATE_WORDS]
            if all(status in (1, 2) for status in statuses):
                break
        started = time.perf_counter()
        output = array('Q')
        output.frombytes(device.queue.read_buffer(self.buffers[6]))
        metrics.readback_seconds += time.perf_counter() - started
        started = time.perf_counter()
        results: list[Result] = []
        traps: dict[int, str] = {}
        for lane in range(self.count):
            if self.state[lane * _STATE_WORDS + 7] == 2:
                code = self.state[lane * _STATE_WORDS + 8]
                if code == 12:
                    owner.exit_codes[self.first + lane] = self.state[lane * _STATE_WORDS + 10]
                traps[self.first + lane] = _TRAPS[code]
                results.append(None)
            else:
                output_row = tuple(output[index * self.count + lane] if raw else _decode(output[index * self.count + lane], ty) for index, ty in enumerate(returns))
                results.append(output_row[0] if len(output_row) == 1 else output_row if output_row else None)
        metrics.decode_seconds += time.perf_counter() - started
        return results, traps



class Instances:
    """A collection of isolated GPU WASM instances. Calls preserve memory/state."""

    def __init__(self, module: Module, count: int, *, memory_pages: int | None, table_elements: int | None,  # noqa: PLR0913, PLR0915 - Checked instance allocation.
                 stack_size: int, call_depth: int, fuel: int, quantum: int, batch_size: int | None,
                 max_resident_bytes: int, wasi: Wasi | None) -> None:
        self.module = module
        self.last_call = CallMetrics()
        self.count = _positive(count, 'count', zero=True)
        self.stack_size = _positive(stack_size, 'stack_size')
        self.call_depth = _positive(call_depth, 'call_depth')
        self.fuel = _positive(fuel, 'fuel', bits=64)
        self.quantum = _positive(quantum, 'quantum')
        self._closed = False
        self._lock = threading.RLock()
        self._batches: list[_Batch] = []
        self._program_buffer: Buffer | None = None
        self._template_buffer: Buffer | None = None
        binary = module._binary
        initial_pages, declared_pages = binary.memory or (0, 0)
        self.memory_pages = min(declared_pages if declared_pages is not None else 65535, max(initial_pages, 16)) if memory_pages is None else _positive(memory_pages, 'memory_pages', zero=True)
        if self.memory_pages < initial_pages or self.memory_pages > 65535 or (declared_pages is not None and self.memory_pages > declared_pages):
            raise ValueError('memory_pages must be within the declared memory limits and at most 65535')
        if table_elements is not None:
            _positive(table_elements, 'table_elements', zero=True)
        self._table_capacities = [min(maximum if maximum is not None else 0xFFFFFFFF,
                                     max(minimum, 256) if table_elements is None else table_elements)
                                  for _, (minimum, maximum) in binary.tables]
        if any(capacity < limits[0] for capacity, (_, limits) in zip(self._table_capacities, binary.tables)):
            raise ValueError('table_elements is below a declared table minimum')
        self.table_elements = sum(self._table_capacities)
        self._table_lengths = len(binary.globals) * 2
        self._table_offset = self._table_lengths + len(binary.tables)
        self._table_offsets = []
        next_offset = self._table_offset
        for capacity in self._table_capacities:
            self._table_offsets.append(next_offset)
            next_offset += capacity
        self._data_flags = self._table_offset + self.table_elements
        self._element_flags = self._data_flags + len(binary.data)
        self._memory_offset = self._element_flags + len(binary.elements)
        self._fs_offset = self._memory_offset + self.memory_pages * 16384
        self._wasi = copy.deepcopy(wasi) if wasi is not None else Wasi()
        if not isinstance(self._wasi, Wasi):
            raise TypeError('wasi must be a Wasi configuration')
        uses_wasi = any(fn.imported is not None for fn in binary.functions) or bool(module._files) or bool(self._wasi.files)
        self._uses_wasi = uses_wasi
        filesystem_words = 48 + self._wasi.max_files * 76 + self._wasi.max_fds * 8 + (self._wasi.storage_size + 3) // 4 if uses_wasi else 0
        self._heap_words = max(4, self._fs_offset + filesystem_words)
        self._output_slots = max([8] + [max(len(args), len(results)) for args, results in binary.types])
        for fn in binary.functions:
            if len(fn.locals) + fn.max_stack > self.stack_size:
                raise ResourceLimitError('stack_size is smaller than a function requires')
        per_instance = self.stack_size * 8 + self.call_depth * 16 + self._heap_words * 4 + _STATE_WORDS * 4 + self._output_slots * 8
        if isinstance(max_resident_bytes, bool) or not isinstance(max_resident_bytes, int):
            raise TypeError('max_resident_bytes must be an integer')
        if max_resident_bytes <= 0:
            raise ValueError('max_resident_bytes must be positive')
        budget = max_resident_bytes
        environment_strings = [*self._wasi.args, *(f'{key}={value}' for key, value in self._wasi.env.items())]
        program_bytes = 4 * (len(module._program) + len(binary.tables) * 4 + sum(2 + (len(value.encode()) + 4) // 4 for value in environment_strings))
        self.resident_bytes = per_instance * self.count + program_bytes + (self._heap_words * 4 if count > 1 else 0)
        if self.resident_bytes > budget:
            raise ResourceLimitError(f'{self.count} instances require approximately {self.resident_bytes} bytes; budget is {budget}')
        if uses_wasi:
            entries = self._wasi._initial(module._files)
            if sum((len(content) + 3) & ~3 for _, _, content in entries) > self._wasi.storage_size:
                raise ResourceLimitError('embedded files exceed WASI storage_size')
        self.exit_codes: list[int | None] = [None] * count
        self._context = _context()
        limits = self._context.limits
        if program_bytes > min(limits['max-storage-buffer-binding-size'], limits['max-buffer-size']):
            raise ResourceLimitError('program and guest environment exceed the device buffer limit')
        largest = max(self.stack_size * 8, self.call_depth * 16, self._heap_words * 4, _STATE_WORDS * 4, self._output_slots * 8)
        max_batch = min(limits['max-storage-buffer-binding-size'] // largest,
                        limits['max-compute-workgroups-per-dimension'] * 64,
                        (128 * 1024 * 1024) // per_instance)
        if max_batch < 1:
            raise ResourceLimitError('a single instance exceeds device buffer limits')
        self.batch_size = max_batch if batch_size is None else _positive(batch_size, 'batch_size')
        if self.batch_size > max_batch:
            raise ResourceLimitError(f'batch_size exceeds device/runtime limit {max_batch}')
        self.resident_bytes += (80 if count > 1 else 64) * ((count + self.batch_size - 1) // self.batch_size)
        if self.resident_bytes > budget:
            raise ResourceLimitError('batch configuration buffers exceed the resident allocation budget')
        self.codegen_seconds = 0.0
        started = time.perf_counter()
        if count > 1:
            _ = self._context.initializer
        if module.compiled is not None:
            if module.compiled.wasi:
                _ = self._context.services
            generated_at = time.perf_counter()
            generated = module.compiled.source
            self.codegen_seconds = time.perf_counter() - generated_at
            self._pipeline, self.pipeline_cache_hit = self._context.compiled_pipeline(generated)
        else:
            self.pipeline_cache_hit = self._context._interpreter is not None
            self._pipeline = self._context.pipeline
        self.pipeline_seconds = time.perf_counter() - started - self.codegen_seconds
        self._initial_words = self._build_initial_heap() if count else array('I')
        try:
            program = array('I', module._program)
            program[12] = int(uses_wasi)
            program[4] = len(program)
            for index, (offset, capacity) in enumerate(zip(self._table_offsets, self._table_capacities)):
                program.extend([self._table_lengths + index, offset, capacity, 0])
            for slot, strings in ((8, self._wasi.args), (10, [f'{key}={value}' for key, value in self._wasi.env.items()])):
                program[slot], program[slot + 1] = len(program), len(strings)
                info_offset = len(program)
                program.extend([0] * (len(strings) * 2))
                for index, string in enumerate(strings):
                    blob = string.encode() + b'\0'
                    program[info_offset + index * 2:info_offset + index * 2 + 2] = array('I', [len(program), len(blob)])
                    program.extend(_words(blob + b'\0' * (-len(blob) % 4)))
            self._program_buffer = self._context.buffer(data=program)
            if count > 1:
                self._template_buffer = self._context.buffer(data=self._initial_words)
                self._initial_words = array('I')
            for first in range(0, count, self.batch_size):
                self._batches.append(_Batch(self, min(self.batch_size, count - first), first))
            self._synchronize_initialization()
            if binary.start is not None:
                self._call(binary.start, [()] * count, self.fuel)
        except Exception:
            self.close()
            raise

    def _native_pipeline(self, region: int, metrics: CallMetrics) -> Pipeline:
        assert self.module.compiled is not None
        started = time.perf_counter()
        source = self.module.compiled.source_for(region)
        metrics.codegen_seconds += time.perf_counter() - started
        started = time.perf_counter()
        pipeline, hit = self._context.compiled_pipeline(source)
        metrics.compile_seconds += time.perf_counter() - started
        metrics.pipelines_created += int(not hit)
        metrics.pipeline_cache_hits += int(hit)
        return pipeline

    @property
    def adapter_info(self) -> AdapterInfo:
        assert self._context.adapter is not None
        return dict(self._context.adapter.info)

    def _build_initial_heap(self) -> array[int]:
        binary = self.module._binary
        words = array('I', [0]) * self._heap_words
        for index, (_, _, value) in enumerate(binary.globals):
            words[index * 2:index * 2 + 2] = array('I', [value & 0xffffffff, value >> 32])
        memory = bytearray((binary.memory[0] if binary.memory else 0) * 65536)
        for index, (offset, blob) in enumerate(binary.data):
            if offset is not None:
                if offset > len(memory) or len(blob) > len(memory) - offset:
                    raise Trap({0: 'out of bounds memory access during instantiation'}, [])
                memory[offset:offset + len(blob)] = blob
                words[self._data_flags + index] = 1
        words[self._memory_offset:self._memory_offset + len(memory) // 4] = _words(memory)
        for index, (_, (size, _)) in enumerate(binary.tables):
            words[self._table_lengths + index] = size
        for index, (offset, elements, declarative, _, table) in enumerate(binary.elements):
            if offset is not None:
                size = binary.tables[table][1][0]
                if offset > size or len(elements) > size - offset:
                    raise Trap({0: 'out of bounds table access during instantiation'}, [])
                start = self._table_offsets[table] + offset
                words[start:start + len(elements)] = array('I', elements)
            if offset is not None or declarative:
                words[self._element_flags + index] = 1
        if self._uses_wasi:
            filesystem = self._build_filesystem()
            words[self._fs_offset:self._fs_offset + len(filesystem)] = filesystem
        return words

    def _synchronize_initialization(self) -> None:
        if self._batches:
            self._context.device.queue.read_buffer(self._batches[-1].buffers[5], 0, 4)

    def reset(self) -> None:
        """Restore fresh guest state and rerun its start function, reusing GPU buffers."""
        with self._lock:
            self._check_open()
            binary = self.module._binary
            for batch in self._batches:
                batch.initialize_heap()
                batch.state = array('I', [0, 0, 0, 0, 0, binary.memory[0] if binary.memory else 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) * batch.count
                self._context.device.queue.write_buffer(batch.buffers[5], 0, batch.state)
            self._synchronize_initialization()
            self.exit_codes = [None] * self.count
            self.last_call = CallMetrics()
            if binary.start is not None:
                self._call(binary.start, [()] * self.count, self.fuel)

    @property
    def stdout(self) -> list[bytes]:
        return [self._read_file_index(1, instance) for instance in range(self.count)]

    @property
    def stderr(self) -> list[bytes]:
        return [self._read_file_index(2, instance) for instance in range(self.count)]

    def _build_filesystem(self) -> array[int]:
        config = self._wasi
        entries = config._initial(self.module._files)
        names_offset = 48 + config.max_files * 12
        fds_offset = names_offset + config.max_files * 64
        data_offset = fds_offset + config.max_fds * 8
        words = array('I', [0]) * (data_offset + (config.storage_size + 3) // 4)
        cursor = 0
        for index, (kind, name, content) in enumerate(entries):
            capacity = (len(content) + 3) & ~3
            encoded = name.encode()
            words[48 + index * 12:48 + index * 12 + 6] = array('I', [kind, len(encoded), len(content), cursor, capacity, index])
            for start, blob in ((names_offset + index * 64, encoded), (data_offset + cursor // 4, content)):
                packed = blob + b'\0' * (-len(blob) % 4)
                values = array('I')
                values.frombytes(packed)
                words[start:start + len(values)] = values
            cursor += capacity
        words[0] = cursor
        words[1] = config.clock_epoch_ns & 0xFFFFFFFF
        words[2] = config.clock_epoch_ns >> 32
        words[8:16] = _words(config.random_key)
        words[37] = 64
        words[4] = config.clock_resolution_ns
        for fd in range(4):
            rights = (1 << 1) | (1 << 21) | (1 << 27) if fd == 0 else (1 << 6) | (1 << 21) | (1 << 27) if fd in (1, 2) else (1 << 30) - 1
            words[fds_offset + fd * 8:fds_offset + (fd + 1) * 8] = array('I', [fd + 1, 0, 0, 0, rights, 0, (1 << 30) - 1, 0])
        return words

    def _read_file_index(self, index: int, instance: int) -> bytes:
        with self._lock:
            self._check_open()
            if not self._uses_wasi:
                return b''
            batch, lane = self._locate(instance)
            entry = self._fs_offset + 48 + index * 12
            metadata = self._read_heap(batch, lane, entry, 12)
            inode = metadata[5]
            if inode != index:
                metadata = self._read_heap(batch, lane, self._fs_offset + 48 + inode * 12, 12)
            length, start = metadata[2], metadata[3]
            if length == 0:
                return b''
            data_offset = self._fs_offset + 48 + self._wasi.max_files * 76 + self._wasi.max_fds * 8
            content = self._read_heap(batch, lane, data_offset + start // 4, (start % 4 + length + 3) // 4).tobytes()
            return content[start % 4:start % 4 + length]

    def _read_heap(self, batch: _Batch, lane: int, start: int, count: int) -> array[int]:
        raw = self._context.device.queue.read_buffer(batch.buffers[3], start * batch.count * 4, count * batch.count * 4)
        return _words(raw)[lane::batch.count]

    def read_file(self, path: str, *, instance: int = 0) -> bytes:
        """Read a file from one instance's GPU filesystem."""
        with self._lock:
            self._check_open()
            encoded_path = normalize_path(path).encode()
            batch, lane = self._locate(instance)
            if self._uses_wasi:
                words = self._read_heap(batch, lane, self._fs_offset, 48 + self._wasi.max_files * 76)
                names = 48 + self._wasi.max_files * 12
                for index in range(4, self._wasi.max_files):
                    entry = 48 + index * 12
                    name = array('I', words[names + index * 64:names + (index + 1) * 64]).tobytes()[:words[entry + 1]]
                    if words[entry] == 1 and name == encoded_path:
                        return self._read_file_index(index, instance)
            raise FileNotFoundError(path)

    def __len__(self) -> int:
        return self.count

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError('instances are closed')

    def call(self, name: str, inputs: Iterable[Row] | None = None, *, fuel: int | None = None,
             cancel: Callable[[], bool] | None = None) -> list[Result]:
        """Invoke one export per instance, in input order.

        A scalar per instance is accepted for a single argument. For multiple
        arguments use tuples; no-argument functions accept omitted inputs.
        The whole batch is validated before executing any instance.
        """
        with self._lock:
            self._check_open()
            started = time.perf_counter()
            if cancel is not None and not callable(cancel):
                raise TypeError('cancel must be a callable')
            binary = self.module._binary
            if name not in binary.exports or binary.exports[name][0] != 0:
                raise KeyError(f'no exported function {name!r}')
            index = binary.exports[name][1]
            params, _ = binary.signature(index)
            if inputs is None:
                if params:
                    raise TypeError('inputs are required for a function with parameters')
                rows: list[Row] = [()] * self.count
            else:
                rows = list(inputs)
                if len(rows) != self.count:
                    raise ValueError(f'expected {self.count} input rows, got {len(rows)}')
            encoded: list[tuple[int, ...]] = []
            for row in rows:
                normalized = (row,) if len(params) == 1 and not isinstance(row, (tuple, list)) else row
                if not isinstance(normalized, (tuple, list)) or len(normalized) != len(params):
                    raise TypeError(f'each input row must contain {len(params)} arguments')
                encoded.append(tuple(_encode(value, ty) for value, ty in zip(cast(Sequence[Scalar], normalized), params)))
            budget = self.fuel if fuel is None else _positive(fuel, 'fuel', bits=64)
            preparation = time.perf_counter() - started
            try:
                return self._call(index, encoded, budget, cancel=cancel)
            finally:
                self.last_call.prepare_seconds += preparation

    def _call(self, index: int, inputs: Sequence[tuple[int, ...]], fuel: int, raw: bool = False,
              cancel: Callable[[], bool] | None = None) -> list[Result]:
        self.last_call = CallMetrics()
        results: list[Result] = []
        traps: dict[int, str] = {}
        for batch in self._batches:
            output, errors = batch.execute(index, inputs[batch.first:batch.first + batch.count], fuel, raw=raw, cancel=cancel)
            results.extend(output)
            traps.update(errors)
        if traps:
            raise Trap(traps, results)
        return results

    def _locate(self, instance: int) -> tuple[_Batch, int]:
        _positive(instance, 'instance', zero=True)
        if instance >= self.count:
            raise IndexError('instance index out of range')
        batch = self._batches[instance // self.batch_size]
        return batch, instance - batch.first

    def read_memory(self, offset: int, size: int, *, instance: int = 0) -> bytes:
        """Read bytes from an instance's current linear memory."""
        with self._lock:
            self._check_open()
            _positive(offset, 'offset', zero=True)
            _positive(size, 'size', zero=True)
            batch, lane = self._locate(instance)
            length = batch.state[lane * _STATE_WORDS + 5] * 65536
            if offset > length or size > length - offset:
                raise IndexError('memory range out of bounds')
            if size == 0:
                return b''
            start, end = offset // 4, (offset + size + 3) // 4
            data = self._context.device.queue.read_buffer(batch.buffers[3], (self._memory_offset + start) * batch.count * 4, (end - start) * batch.count * 4)
            words = array('I')
            words.frombytes(data)
            return words[lane::batch.count].tobytes()[offset % 4:offset % 4 + size]

    def write_memory(self, offset: int, data: Blob, *, instance: int = 0) -> None:
        """Write bytes without executing WASM on the host."""
        with self._lock:
            self._check_open()
            data = bytes(data)
            self.read_memory(offset, len(data), instance=instance)
            if not data:
                return
            batch, lane = self._locate(instance)
            start, end = offset // 4, (offset + len(data) + 3) // 4
            address = (self._memory_offset + start) * batch.count * 4
            words = array('I')
            words.frombytes(self._context.device.queue.read_buffer(batch.buffers[3], address, (end - start) * batch.count * 4))
            existing = bytearray(words[lane::batch.count].tobytes())
            existing[offset % 4:offset % 4 + len(data)] = data
            words[lane::batch.count] = _words(existing)
            self._context.device.queue.write_buffer(batch.buffers[3], address, words)

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                for batch in self._batches:
                    batch.close()
                if self._program_buffer is not None:
                    self._program_buffer.destroy()
                if self._template_buffer is not None:
                    self._template_buffer.destroy()
                self._initial_words = array('I')
                self._closed = True

    def __enter__(self) -> Instances:
        self._check_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
