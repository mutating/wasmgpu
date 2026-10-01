"""GPU execution and Python API. No CPU WebAssembly executor is used here."""
from __future__ import annotations

import copy
import importlib
import os
import struct
import threading
from array import array
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Tuple, cast

from .binary import F32, F64, I32, I64, BinaryModule
from .errors import GPUUnavailableError, ResourceLimitError, Trap
from .types import (
    AdapterInfo,
    Backend,
    Blob,
    Buffer,
    RequestAdapter,
    RequestDevice,
    Result,
    Row,
    Scalar,
)
from .wasi import WASI_IDS, Wasi, normalize_path

_TRAPS = {
    1: 'unreachable', 2: 'out of bounds memory access', 3: 'out of bounds table access',
    4: 'integer divide by zero', 5: 'integer overflow', 6: 'invalid conversion to integer',
    7: 'stack exhausted', 8: 'fuel exhausted', 9: 'uninitialized element',
    10: 'indirect call type mismatch', 11: 'invalid runtime state', 12: 'WASI proc_exit', 13: 'WASI proc_raise',
}
_CONTEXT: _Context | None = None
_CONTEXT_LOCK = threading.Lock()


def _words(blob: Blob) -> array[int]:
    words = array('I')
    words.frombytes(blob)
    return words


def _positive(value: int, name: str, zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f'{name} must be an integer')
    if value < (0 if zero else 1) or value > 0xFFFFFFFF:
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
        shader = self.device.create_shader_module(label='wasmgpu interpreter', code=source)
        self.pipeline = self.device.create_compute_pipeline(layout='auto', compute={'module': shader, 'entry_point': 'run'})

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


def _program(module: BinaryModule) -> array[int]:
    canonical = [module.types.index(signature) for signature in module.types]
    words = [0] * 16
    code: list[int] = []
    for fn in module.functions:
        params, results = module.types[fn.type_index]
        instructions = fn.instructions
        if fn.imported is not None:
            fn.locals = list(params)
            instructions = [[0x20, index, 0, 0] for index in range(len(params))] + [[0x10, module.functions.index(fn), 0, 0], [0x0F, 0, 0, 0]]
        fn.offset = len(code) // 4
        words.extend([fn.offset, len(params), len(results), len(fn.locals), canonical[fn.type_index], WASI_IDS[fn.imported[1]] if fn.imported else 0, 0, 0])
        for op, operand, b, c in instructions:
            a = operand
            if 0x100 <= op <= 0x104:
                a += fn.offset
            elif op == 0x11:
                a = canonical[a]
            code.extend([op, a & 0xFFFFFFFF, b & 0xFFFFFFFF, c & 0xFFFFFFFF])
    words[0] = len(words)
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
    return array('I', words)


class Module:
    """Validate a binary WASM file. Function bodies execute only on the GPU.

    ``source`` is a path or binary bytes. WAT conversion is intentionally not a
    runtime dependency; tests use Wasmtime's assembler.
    """

    def __init__(self, source: str | os.PathLike[str] | Blob, *, files: Mapping[str, Blob] | None = None) -> None:
        if isinstance(source, (str, os.PathLike)):
            data = Path(source).read_bytes()
        elif isinstance(source, (bytes, bytearray, memoryview)):
            data = bytes(source)
        else:
            raise TypeError('Module requires a filesystem path or WASM bytes')
        self._binary = BinaryModule(data)
        Wasi.validate_imports(self._binary)
        self._program = _program(self._binary)
        self._files = Wasi(files=files).files

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
        words = array('I', [0]) * (owner._heap_words * count)

        def repeat(offset: int, value: int) -> None:
            words[offset * count:(offset + 1) * count] = array('I', [value]) * count

        for index, (_, _, value) in enumerate(binary.globals):
            repeat(index * 2, value & 0xFFFFFFFF)
            repeat(index * 2 + 1, value >> 32)
        initial_memory = bytearray((binary.memory[0] if binary.memory else 0) * 65536)
        for index, (offset, blob) in enumerate(binary.data):
            if offset is not None:
                if offset > len(initial_memory) or len(blob) > len(initial_memory) - offset:
                    raise Trap({first: 'out of bounds memory access during instantiation'}, [])
                initial_memory[offset:offset + len(blob)] = blob
                repeat(owner._data_flags + index, 1)
        for index, value in enumerate(_words(initial_memory)):
            if value:
                repeat(owner._memory_offset + index, value)
        for index, (_, (initial_size, _)) in enumerate(binary.tables):
            repeat(owner._table_lengths + index, initial_size)
        for index, (offset, elements, declarative, _, table) in enumerate(binary.elements):
            if offset is not None:
                initial_size = binary.tables[table][1][0]
                if offset > initial_size or len(elements) > initial_size - offset:
                    raise Trap({first: 'out of bounds table access during instantiation'}, [])
                for element_index, value in enumerate(elements):
                    repeat(owner._table_offsets[table] + offset + element_index, value)
            if offset is not None or declarative:
                repeat(owner._element_flags + index, 1)
        for offset, value in enumerate(owner._filesystem):
            if value:
                repeat(owner._fs_offset + offset, value)
        if owner._filesystem:
            for lane in range(count):
                words[(owner._fs_offset + 18) * count + lane] = first + lane
        self.buffers: list[Buffer] = []
        try:
            assert owner._program_buffer is not None
            self.buffers = [owner._program_buffer]
            config = array('I', [count, owner.stack_size, owner.call_depth, owner.memory_pages,
                                owner._memory_offset, owner._table_offset, owner.table_elements, owner._data_flags,
                                owner._element_flags, owner.quantum, owner._output_slots, 0,
                                owner._fs_offset, owner._wasi.max_files, owner._wasi.max_fds, owner._wasi.storage_size])
            self.buffers.append(self.context.buffer(data=config, uniform=True))
            self.buffers.append(self.context.buffer(size=count * owner.stack_size * 8))
            self.buffers.append(self.context.buffer(data=words))
            self.buffers.append(self.context.buffer(size=count * owner.call_depth * 16))
            state = array('I', [0]) * (count * 12)
            for lane in range(count):
                state[lane * 12 + 5] = binary.memory[0] if binary.memory else 0
                state[lane * 12 + 6] = 0
            self.buffers.append(self.context.buffer(data=state))
            self.buffers.append(self.context.buffer(size=count * owner._output_slots * 8))
            self.state = state
            self.group = device.create_bind_group(layout=self.context.pipeline.get_bind_group_layout(0), entries=[
                {'binding': index, 'resource': {'buffer': buffer, 'offset': 0, 'size': buffer.size}}
                for index, buffer in enumerate(self.buffers)
            ])
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        for buffer in self.buffers[1:]:
            buffer.destroy()
        self.buffers = []

    def execute(self, function_index: int, inputs: Sequence[tuple[int, ...]], fuel: int, raw: bool = False) -> tuple[list[Result], dict[int, str]]:
        owner = self.owner
        device = self.context.device
        fn = owner.module._binary.functions[function_index]
        _, returns = owner.module._binary.signature(function_index)
        arguments = array('Q', [0]) * (self.count * max(1, len(fn.locals)))
        for lane, row in enumerate(inputs):
            for index, value in enumerate(row):
                arguments[index * self.count + lane] = value
            base = lane * 12
            pages, table_len = self.state[base + 5], self.state[base + 6]
            self.state[base:base + 12] = array('I', [fn.offset, len(fn.locals), 0, function_index, 0, pages, table_len, 0, 0, fuel, 0, 0])
        device.queue.write_buffer(self.buffers[2], 0, arguments)
        device.queue.write_buffer(self.buffers[5], 0, self.state)
        while True:
            encoder = device.create_command_encoder(label='wasmgpu dispatch')
            compute = encoder.begin_compute_pass()
            compute.set_pipeline(self.context.pipeline)
            compute.set_bind_group(0, self.group)
            compute.dispatch_workgroups((self.count + 63) // 64)
            compute.end()
            device.queue.submit([encoder.finish()])
            self.state = array('I')
            self.state.frombytes(device.queue.read_buffer(self.buffers[5]))
            statuses = self.state[7::12]
            if all(status in (1, 2) for status in statuses):
                break
        output = array('Q')
        output.frombytes(device.queue.read_buffer(self.buffers[6]))
        results: list[Result] = []
        traps: dict[int, str] = {}
        for lane in range(self.count):
            if self.state[lane * 12 + 7] == 2:
                code = self.state[lane * 12 + 8]
                if code == 12:
                    owner.exit_codes[self.first + lane] = self.state[lane * 12 + 10]
                traps[self.first + lane] = _TRAPS[code]
                results.append(None)
            else:
                output_row = tuple(output[index * self.count + lane] if raw else _decode(output[index * self.count + lane], ty) for index, ty in enumerate(returns))
                results.append(output_row[0] if len(output_row) == 1 else output_row if output_row else None)
        return results, traps



class Instances:
    """A collection of isolated GPU WASM instances. Calls preserve memory/state."""

    def __init__(self, module: Module, count: int, *, memory_pages: int | None, table_elements: int | None,  # noqa: PLR0913, PLR0915 - Checked instance allocation.
                 stack_size: int, call_depth: int, fuel: int, quantum: int, batch_size: int | None,
                 max_resident_bytes: int, wasi: Wasi | None) -> None:
        self.module = module
        self.count = _positive(count, 'count', zero=True)
        self.stack_size = _positive(stack_size, 'stack_size')
        self.call_depth = _positive(call_depth, 'call_depth')
        self.fuel = _positive(fuel, 'fuel')
        self.quantum = _positive(quantum, 'quantum')
        self._closed = False
        self._lock = threading.RLock()
        self._batches: list[_Batch] = []
        self._program_buffer: Buffer | None = None
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
        filesystem_words = 48 + self._wasi.max_files * 76 + self._wasi.max_fds * 8 + (self._wasi.storage_size + 3) // 4 if uses_wasi else 0
        self._heap_words = max(4, self._fs_offset + filesystem_words)
        self._output_slots = max([8] + [max(len(args), len(results)) for args, results in binary.types])
        for fn in binary.functions:
            if len(fn.locals) + fn.max_stack > self.stack_size:
                raise ResourceLimitError('stack_size is smaller than a function requires')
        per_instance = self.stack_size * 8 + self.call_depth * 16 + self._heap_words * 4 + 48 + self._output_slots * 8
        if isinstance(max_resident_bytes, bool) or not isinstance(max_resident_bytes, int):
            raise TypeError('max_resident_bytes must be an integer')
        if max_resident_bytes <= 0:
            raise ValueError('max_resident_bytes must be positive')
        budget = max_resident_bytes
        environment_strings = [*self._wasi.args, *(f'{key}={value}' for key, value in self._wasi.env.items())]
        program_bytes = 4 * (len(module._program) + len(binary.tables) * 4 + sum(2 + (len(value.encode()) + 4) // 4 for value in environment_strings))
        self.resident_bytes = per_instance * self.count + program_bytes
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
        largest = max(self.stack_size * 8, self.call_depth * 16, self._heap_words * 4, 48, self._output_slots * 8)
        max_batch = min(limits['max-storage-buffer-binding-size'] // largest,
                        limits['max-compute-workgroups-per-dimension'] * 64,
                        (128 * 1024 * 1024) // per_instance)
        if max_batch < 1:
            raise ResourceLimitError('a single instance exceeds device buffer limits')
        self.batch_size = max_batch if batch_size is None else _positive(batch_size, 'batch_size')
        if self.batch_size > max_batch:
            raise ResourceLimitError(f'batch_size exceeds device/runtime limit {max_batch}')
        self.resident_bytes += 64 * ((count + self.batch_size - 1) // self.batch_size)
        if self.resident_bytes > budget:
            raise ResourceLimitError('batch configuration buffers exceed the resident allocation budget')
        self._filesystem = self._build_filesystem() if uses_wasi else array('I')
        try:
            program = array('I', module._program)
            program[12] = int(bool(self._filesystem))
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
            for first in range(0, count, self.batch_size):
                self._batches.append(_Batch(self, min(self.batch_size, count - first), first))
            if binary.start is not None:
                self._call(binary.start, [()] * count, self.fuel)
        except Exception:
            self.close()
            raise

    @property
    def adapter_info(self) -> AdapterInfo:
        assert self._context.adapter is not None
        return dict(self._context.adapter.info)

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
            if not self._filesystem:
                return b''
            batch, lane = self._locate(instance)
            raw = self._context.device.queue.read_buffer(batch.buffers[3])
            words = _words(raw)[lane::batch.count]
            entry = self._fs_offset + 48 + index * 12
            inode = words[entry + 5]
            entry = self._fs_offset + 48 + inode * 12
            length, start = words[entry + 2], words[entry + 3]
            data_offset = self._fs_offset + 48 + self._wasi.max_files * 76 + self._wasi.max_fds * 8
            content = array('I', words[data_offset + start // 4:data_offset + (start + length + 3) // 4]).tobytes()
            return content[start % 4:start % 4 + length]

    def read_file(self, path: str, *, instance: int = 0) -> bytes:
        """Read a file from one instance's GPU filesystem."""
        with self._lock:
            self._check_open()
            encoded_path = normalize_path(path).encode()
            batch, lane = self._locate(instance)
            if self._filesystem:
                raw = self._context.device.queue.read_buffer(batch.buffers[3])
                words = _words(raw)[lane::batch.count]
                names = self._fs_offset + 48 + self._wasi.max_files * 12
                for index in range(4, self._wasi.max_files):
                    entry = self._fs_offset + 48 + index * 12
                    name = array('I', words[names + index * 64:names + (index + 1) * 64]).tobytes()[:words[entry + 1]]
                    if words[entry] == 1 and name == encoded_path:
                        return self._read_file_index(index, instance)
            raise FileNotFoundError(path)

    def __len__(self) -> int:
        return self.count

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError('instances are closed')

    def call(self, name: str, inputs: Iterable[Row] | None = None, *, fuel: int | None = None) -> list[Result]:
        """Invoke one export per instance, in input order.

        A scalar per instance is accepted for a single argument. For multiple
        arguments use tuples; no-argument functions accept omitted inputs.
        The whole batch is validated before executing any instance.
        """
        with self._lock:
            self._check_open()
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
            return self._call(index, encoded, self.fuel if fuel is None else _positive(fuel, 'fuel'))

    def _call(self, index: int, inputs: Sequence[tuple[int, ...]], fuel: int, raw: bool = False) -> list[Result]:
        results: list[Result] = []
        traps: dict[int, str] = {}
        for batch in self._batches:
            output, errors = batch.execute(index, inputs[batch.first:batch.first + batch.count], fuel, raw=raw)
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
            length = batch.state[lane * 12 + 5] * 65536
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
                self._closed = True

    def __enter__(self) -> Instances:
        self._check_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
