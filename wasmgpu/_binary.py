"""WASM binary reader, type validator and structured-control lowering.

This module never executes function bodies. It lowers them into fixed-width
instructions consumed by the compute shader. Values keep their original bits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple, TypeVar

from ._errors import UnsupportedFeatureError, ValidationError

T = TypeVar('T')
Signature = Tuple[List[int], List[int]]
Limits = Tuple[int, Optional[int]]
Element = Tuple[Optional[int], List[int], bool, int, int]


I32, I64, F32, F64, FUNCREF, EXTERNREF = 0x7F, 0x7E, 0x7D, 0x7C, 0x70, 0x6F
VALUE_TYPES = (I32, I64, F32, F64, FUNCREF, EXTERNREF)
JUMP, IF_ZERO, BRANCH, BRANCH_IF, BRANCH_TABLE = range(0x100, 0x105)
RETURN = 0x0F


class Reader:
    def __init__(self, data: bytes) -> None:
        self.data = bytes(data)
        self.pos = 0

    def take(self, count: int) -> bytes:
        if count < 0 or count > len(self.data) - self.pos:
            raise ValidationError('unexpected end of WebAssembly')
        result = self.data[self.pos:self.pos + count]
        self.pos += count
        return result

    def byte(self) -> int:
        return self.take(1)[0]

    def leb(self, bits: int = 32, signed: bool = False) -> int:
        value = 0
        for shift in range(0, bits, 7):
            byte = self.byte()
            value |= (byte & 127) << shift
            if byte < 128:
                if signed and byte & 64:
                    value -= 1 << (shift + 7)
                low = -(1 << (bits - 1)) if signed else 0
                high = (1 << (bits - (1 if signed else 0))) - 1
                if not low <= value <= high:
                    raise ValidationError('integer representation out of range')
                return value
        raise ValidationError('integer representation too long')

    def vec(self, read: Callable[[], T]) -> list[T]:
        count = self.leb()
        if count > len(self.data) - self.pos:
            raise ValidationError('vector length exceeds section size')
        return [read() for _ in range(count)]

    def name(self) -> str:
        try:
            return self.take(self.leb()).decode('utf-8')
        except UnicodeDecodeError as error:
            raise ValidationError('invalid UTF-8 name') from error

    def finish(self) -> None:
        if self.pos != len(self.data):
            raise ValidationError('trailing bytes in section or function')


def value_type(reader: Reader) -> int:
    result = reader.byte()
    if result not in VALUE_TYPES:
        raise UnsupportedFeatureError(f'unsupported value type 0x{result:02x}')
    return result


def limits(reader: Reader, memory: bool = False) -> Limits:
    flags = reader.leb()
    if flags not in (0, 1):
        raise UnsupportedFeatureError('shared memory, memory64 and custom page sizes are not supported')
    minimum = reader.leb()
    maximum = reader.leb() if flags else None
    if maximum is not None and minimum > maximum:
        raise ValidationError('minimum exceeds maximum')
    if memory and (minimum > 65536 or (maximum is not None and maximum > 65536)):
        raise ValidationError('memory32 exceeds 65536 pages')
    return minimum, maximum


@dataclass
class Function:
    type_index: int
    locals: list[int] = field(default_factory=list)
    instructions: list[list[int]] = field(default_factory=list)
    imported: tuple[str, str] | None = None
    offset: int = 0
    max_stack: int = 0


@dataclass
class Control:
    kind: int
    height: int
    params: list[int]
    results: list[int]
    start: int
    patches: list[int] = field(default_factory=list)
    unreachable: bool = False
    else_patch: int | None = None
    has_else: bool = False


class BinaryModule:
    def __init__(self, data: bytes) -> None:
        self.types: list[Signature] = []
        self.functions: list[Function] = []
        self.declared_functions: set[int] = set()
        self.exports: dict[str, tuple[int, int]] = {}
        self.globals: list[tuple[int, int, int]] = []
        self.memory: Limits | None = None
        self.tables: list[tuple[int, Limits]] = []
        self.data: list[tuple[int | None, bytes]] = []
        self.elements: list[Element] = []
        self.start: int | None = None
        self.data_count: int | None = None
        reader = Reader(data)
        if reader.take(8) != b'\0asm\x01\0\0\0':
            raise ValidationError('expected a WebAssembly 1 binary module')
        order = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 12: 10, 10: 11, 11: 12}
        last = 0
        bodies: list[Reader] = []
        while reader.pos < len(reader.data):
            section_id = reader.byte()
            section = Reader(reader.take(reader.leb()))
            if section_id == 0:
                section.name()
                continue
            if section_id not in order:
                raise UnsupportedFeatureError(f'unsupported section {section_id}')
            if order[section_id] <= last:
                raise ValidationError('duplicate or out-of-order section')
            last = order[section_id]
            parsed_bodies = self._read_section(section_id, section)
            if section_id == 10:
                bodies = parsed_bodies
            section.finish()
        defined = [fn for fn in self.functions if fn.imported is None]
        if len(bodies) != len(defined):
            raise ValidationError('function and code section lengths differ')
        if self.data_count is not None and self.data_count != len(self.data):
            raise ValidationError('data count does not match data section')
        for fn, body in zip(defined, bodies):
            params, _ = self.types[fn.type_index]
            fn.locals = list(params)
            for _ in range(body.leb()):
                count, ty = body.leb(), value_type(body)
                if count > 65536 - len(fn.locals):
                    raise UnsupportedFeatureError('more than 65536 locals per function')
                fn.locals.extend([ty] * count)
            Compiler(self, fn, body).compile()
            body.finish()

    def _read_section(self, section_id: int, section: Reader) -> list[Reader]:
        if section_id == 1:
            self.types = section.vec(lambda: self._type(section))
        elif section_id == 2:
            section.vec(lambda: self._import(section))
        elif section_id == 3:
            self.functions.extend(section.vec(lambda: Function(self._type_index(section))))
        elif section_id == 4:
            self.tables = section.vec(lambda: (value_type(section), limits(section)))
            if any(ty not in (FUNCREF, EXTERNREF) for ty, _ in self.tables):
                raise ValidationError('table element type must be a reference')
        elif section_id == 5:
            memories = section.vec(lambda: limits(section, memory=True))
            if len(memories) > 1:
                raise UnsupportedFeatureError('multiple memories are not supported')
            self.memory = memories[0] if memories else None
        elif section_id == 6:
            section.vec(lambda: self._global(section))
        elif section_id == 7:
            section.vec(lambda: self._export(section))
        elif section_id == 8:
            self.start = self._index(section, self.functions, 'start function')
            if self.signature(self.start) != ([], []):
                raise ValidationError('start function must have no arguments or results')
        elif section_id == 9:
            self.elements = section.vec(lambda: self._element(section))
        elif section_id == 10:
            return section.vec(lambda: Reader(section.take(section.leb())))
        elif section_id == 11:
            self.data = section.vec(lambda: self._data(section))
        else:  # The caller validated the section id; only data-count (12) remains.
            self.data_count = section.leb()
        return []

    @staticmethod
    def _index(reader: Reader, values: Sequence[T], name: str) -> int:
        index = reader.leb()
        if index >= len(values):
            raise ValidationError(f'{name} index out of range: {index}')
        return index

    def _type_index(self, reader: Reader) -> int:
        return self._index(reader, self.types, 'type')

    def signature(self, index: int) -> Signature:
        return self.types[self.functions[index].type_index]

    @staticmethod
    def _type(reader: Reader) -> Signature:
        if reader.byte() != 0x60:
            raise UnsupportedFeatureError('GC and recursive types are not supported')
        return reader.vec(lambda: value_type(reader)), reader.vec(lambda: value_type(reader))

    def _import(self, reader: Reader) -> None:
        module, name, kind = reader.name(), reader.name(), reader.byte()
        if kind != 0:
            raise UnsupportedFeatureError('only function imports are supported')
        self.functions.append(Function(self._type_index(reader), imported=(module, name)))

    def _constant(self, reader: Reader, expected: int) -> int:
        op = reader.byte()
        if op in (0x41, 0x42):
            ty = I32 if op == 0x41 else I64
            val = reader.leb(32 if ty == I32 else 64, signed=True)
            value = val & ((1 << (32 if ty == I32 else 64)) - 1)
        elif op in (0x43, 0x44):
            ty = F32 if op == 0x43 else F64
            value = int.from_bytes(reader.take(4 if ty == F32 else 8), 'little')
        elif op == 0xD0:
            ty, value = value_type(reader), 0
            if ty not in (FUNCREF, EXTERNREF):
                raise ValidationError('ref.null requires a reference type')
        elif op == 0xD2:
            ty, value = FUNCREF, self._index(reader, self.functions, 'function') + 1
            self.declared_functions.add(value - 1)
        else:
            raise UnsupportedFeatureError(f'unsupported constant expression opcode 0x{op:02x}')
        if ty != expected or reader.byte() != 0x0B:
            raise ValidationError('invalid constant expression')
        return value

    def _global(self, reader: Reader) -> None:
        ty, mutable = value_type(reader), reader.byte()
        if mutable > 1:
            raise ValidationError('invalid global mutability')
        self.globals.append((ty, mutable, self._constant(reader, ty)))

    def _export(self, reader: Reader) -> None:
        name, kind, index = reader.name(), reader.byte(), reader.leb()
        counts = {0: len(self.functions), 1: len(self.tables), 2: int(self.memory is not None), 3: len(self.globals)}
        if kind not in counts or index >= counts[kind] or name in self.exports:
            raise ValidationError('invalid or duplicate export')
        self.exports[name] = (kind, index)
        if kind == 0:
            self.declared_functions.add(index)

    def _data(self, reader: Reader) -> tuple[int | None, bytes]:
        flags = reader.leb()
        if flags not in (0, 1, 2):
            raise ValidationError('invalid data segment flags')
        if flags == 2 and reader.leb() != 0:
            raise ValidationError('memory index out of range')
        offset = None if flags == 1 else self._constant(reader, I32)
        if offset is not None and self.memory is None:
            raise ValidationError('data segment requires a memory')
        return offset, reader.take(reader.leb())

    def _element(self, reader: Reader) -> Element:
        flags = reader.leb()
        if flags > 7:
            raise ValidationError('invalid element segment flags')
        table_index = reader.leb() if flags in (2, 6) else 0
        offset = self._constant(reader, I32) if flags % 2 == 0 else None
        if offset is not None and table_index >= len(self.tables):
            raise ValidationError('active element segment requires a table')
        if flags & 4:
            ty = value_type(reader) if flags != 4 else FUNCREF
            values = reader.vec(lambda: self._constant(reader, ty))
        else:
            ty = FUNCREF
            if flags != 0 and reader.byte() != 0:
                raise ValidationError('invalid element kind')
            values = reader.vec(lambda: self._index(reader, self.functions, 'function') + 1)
        if ty not in (FUNCREF, EXTERNREF) or (offset is not None and ty != self.tables[table_index][0]):
            raise ValidationError('invalid element type')
        if ty == FUNCREF:
            self.declared_functions.update(value - 1 for value in values if value)
        return offset, values, flags in (3, 7), ty, table_index


def numeric_signature(op: int) -> Signature | None:  # noqa: PLR0911 - Opcode signature dispatch.
    if op == 0x45:
        return [I32], [I32]
    if 0x46 <= op <= 0x4F:
        return [I32, I32], [I32]
    if op == 0x50:
        return [I64], [I32]
    if 0x51 <= op <= 0x5A:
        return [I64, I64], [I32]
    if 0x5B <= op <= 0x60:
        return [F32, F32], [I32]
    if 0x61 <= op <= 0x66:
        return [F64, F64], [I32]
    for start, unary_end, end, ty in ((0x67, 0x69, 0x78, I32), (0x79, 0x7B, 0x8A, I64), (0x8B, 0x91, 0x98, F32), (0x99, 0x9F, 0xA6, F64)):
        if start <= op <= end:
            return [ty] * (1 if op <= unary_end else 2), [ty]
    conversions = {
        0xA7: (I64, I32), 0xA8: (F32, I32), 0xA9: (F32, I32), 0xAA: (F64, I32), 0xAB: (F64, I32),
        0xAC: (I32, I64), 0xAD: (I32, I64), 0xAE: (F32, I64), 0xAF: (F32, I64), 0xB0: (F64, I64), 0xB1: (F64, I64),
        0xB2: (I32, F32), 0xB3: (I32, F32), 0xB4: (I64, F32), 0xB5: (I64, F32), 0xB6: (F64, F32),
        0xB7: (I32, F64), 0xB8: (I32, F64), 0xB9: (I64, F64), 0xBA: (I64, F64), 0xBB: (F32, F64),
        0xBC: (F32, I32), 0xBD: (F64, I64), 0xBE: (I32, F32), 0xBF: (I64, F64),
        0xC0: (I32, I32), 0xC1: (I32, I32), 0xC2: (I64, I64), 0xC3: (I64, I64), 0xC4: (I64, I64),
    }
    if op in conversions:
        arg, result = conversions[op]
        return [arg], [result]
    if 0xFC00 <= op <= 0xFC07:
        offset = op - 0xFC00
        return [F32 if offset % 4 < 2 else F64], [I32 if offset < 4 else I64]
    return None


class Compiler:
    def __init__(self, module: BinaryModule, function: Function, reader: Reader) -> None:
        self.module, self.fn, self.reader = module, function, reader
        self.stack: list[int | None] = []
        self.controls: list[Control] = []

    def emit(self, op: int, a: int = 0, b: int = 0, c: int = 0) -> int:
        self.fn.instructions.append([op, a, b, c])
        return len(self.fn.instructions) - 1

    def pop(self, expected: int | None = None) -> int | None:
        control = self.controls[-1]
        if len(self.stack) == control.height and control.unreachable:
            return expected
        if len(self.stack) <= control.height:
            raise ValidationError('operand stack underflow')
        actual = self.stack.pop()
        if expected is not None and actual is not None and actual != expected:
            raise ValidationError('operand type mismatch')
        return actual

    def effect(self, params: Sequence[int | None], results: Sequence[int | None]) -> None:
        for ty in reversed(params):
            self.pop(ty)
        self.stack.extend(results)
        self.fn.max_stack = max(self.fn.max_stack, len(self.stack))

    def unreachable(self) -> None:
        frame = self.controls[-1]
        del self.stack[frame.height:]
        frame.unreachable = True

    def target(self, depth: int) -> Control:
        if depth >= len(self.controls):
            raise ValidationError('branch depth out of range')
        return self.controls[-1 - depth]

    def branch(self, target: Control, conditional: bool = False) -> list[int]:
        types = target.params if target.kind == 0x03 else target.results
        self.effect(types, types)
        index = self.emit(BRANCH_IF if conditional else BRANCH, target.start, target.height, len(types))
        if target.kind != 0x03:
            target.patches.append(index)
        return types

    def require_memory(self) -> None:
        if self.module.memory is None:
            raise ValidationError('instruction requires memory')

    def table_index(self) -> int:
        return self.module._index(self.reader, self.module.tables, 'table')

    def memory_index(self) -> None:
        self.require_memory()
        if self.reader.leb() != 0:
            raise ValidationError('memory index out of range')

    def compile(self) -> None:  # noqa: PLR0915 - One validation rule per opcode family.
        _, results = self.module.types[self.fn.type_index]
        self.controls.append(Control(0xFF, 0, [], results, 0))
        params: list[int]
        returns: list[int]
        while self.controls:
            op = self.reader.byte()
            if op in (0x02, 0x03, 0x04):
                if op == 0x04:
                    self.pop(I32)
                bt = self.reader.leb(33, signed=True)
                if bt == -64:
                    params, returns = [], []
                elif bt < 0 and (bt & 127) in VALUE_TYPES:
                    params, returns = [], [bt & 127]
                elif 0 <= bt < len(self.module.types):
                    params, returns = self.module.types[bt]
                else:
                    raise ValidationError('invalid block type')
                self.effect(params, [])
                control = Control(op, len(self.stack), params, returns, len(self.fn.instructions))
                self.stack.extend(params)
                if op == 0x04:
                    control.else_patch = self.emit(IF_ZERO)
                self.controls.append(control)
            elif op in (0x05, 0x0B):
                control = self.controls[-1]
                self.effect(control.results, [])
                if len(self.stack) != control.height:
                    raise ValidationError('incorrect block result arity')
                if op == 0x05:
                    if control.kind != 0x04 or control.has_else:
                        raise ValidationError('unexpected else')
                    control.has_else = True
                    end_jump = self.emit(JUMP)
                    assert control.else_patch is not None
                    self.fn.instructions[control.else_patch][1] = len(self.fn.instructions)
                    control.patches.append(end_jump)
                    control.else_patch = None
                    control.unreachable = False
                    self.stack.extend(control.params)
                else:
                    if control.kind == 0x04 and not control.has_else and control.params != control.results:
                        raise ValidationError('if without else must have matching parameter/result types')
                    end = len(self.fn.instructions)
                    for index in control.patches:
                        self.fn.instructions[index][1] = end
                    if control.else_patch is not None:
                        self.fn.instructions[control.else_patch][1] = end
                    self.controls.pop()
                    self.stack.extend(control.results)
                    if not self.controls:
                        self.emit(RETURN)
            elif op in (0x0C, 0x0D):
                if op == 0x0D:
                    self.pop(I32)
                self.branch(self.target(self.reader.leb()), op == 0x0D)
                if op == 0x0C:
                    self.unreachable()
            elif op == 0x0E:
                targets = self.reader.vec(lambda: self.target(self.reader.leb()))
                targets.append(self.target(self.reader.leb()))
                self.pop(I32)
                index = self.emit(BRANCH_TABLE, len(self.fn.instructions) + 1, len(targets))
                branch_signature = None
                for target in targets:
                    types = self.branch(target)
                    if branch_signature is not None and branch_signature != types:
                        raise ValidationError('br_table target types differ')
                    branch_signature = types
                self.unreachable()
            elif op == RETURN:
                self.effect(self.controls[0].results, [])
                self.emit(RETURN)
                self.unreachable()
            elif op in (0x10, 0x11):
                table = 0
                if op == 0x10:
                    index = self.module._index(self.reader, self.module.functions, 'function')
                    params, returns = self.module.signature(index)
                else:
                    index = self.module._type_index(self.reader)
                    table = self.table_index()
                    if self.module.tables[table][0] != FUNCREF:
                        raise ValidationError('call_indirect requires funcref table')
                    self.pop(I32)
                    params, returns = self.module.types[index]
                self.effect(params, returns)
                self.emit(op, index, table)
            elif op == 0x00:
                self.emit(op)
                self.unreachable()
            elif op == 0x01:
                pass
            elif op == 0x1A:
                self.pop()
                self.emit(op)
            elif op in (0x1B, 0x1C):
                select_types = self.reader.vec(lambda: value_type(self.reader)) if op == 0x1C else None
                if select_types is not None and len(select_types) != 1:
                    raise ValidationError('typed select requires one type')
                self.pop(I32)
                rhs = self.pop(select_types[0] if select_types else None)
                lhs = self.pop(rhs)
                if select_types is None and lhs in (FUNCREF, EXTERNREF):
                    raise ValidationError('reference select requires explicit type')
                self.stack.append(lhs if lhs is not None else rhs)
                self.emit(0x1B)
            elif 0x20 <= op <= 0x24:
                global_op = op >= 0x23
                index = self.module._index(self.reader, self.module.globals if global_op else self.fn.locals, 'variable')
                ty = self.module.globals[index][0] if global_op else self.fn.locals[index]
                if op == 0x24 and not self.module.globals[index][1]:
                    raise ValidationError('cannot set immutable global')
                self.effect([] if op in (0x20, 0x23) else [ty], [ty] if op in (0x20, 0x22, 0x23) else [])
                self.emit(op, index)
            elif op in (0x25, 0x26):
                table = self.table_index()
                ty = self.module.tables[table][0]
                self.effect([I32] if op == 0x25 else [I32, ty], [ty] if op == 0x25 else [])
                self.emit(op, table)
            elif 0x28 <= op <= 0x3E:
                self.require_memory()
                align, offset = self.reader.leb(), self.reader.leb()
                widths = [4, 8, 4, 8, 1, 1, 2, 2, 1, 1, 2, 2, 4, 4, 4, 8, 4, 8, 1, 2, 1, 2, 4]
                width = widths[op - 0x28]
                if (1 << min(align, 32)) > width:
                    raise ValidationError('memory alignment exceeds natural alignment')
                if op <= 0x35:
                    ty = I32 if op in (0x28, 0x2C, 0x2D, 0x2E, 0x2F) else F32 if op == 0x2A else F64 if op == 0x2B else I64
                    self.effect([I32], [ty])
                else:
                    ty = I32 if op in (0x36, 0x3A, 0x3B) else F32 if op == 0x38 else F64 if op == 0x39 else I64
                    self.effect([I32, ty], [])
                self.emit(op, offset, width)
            elif op in (0x3F, 0x40):
                self.memory_index()
                self.effect([I32] if op == 0x40 else [], [I32])
                self.emit(op)
            elif 0x41 <= op <= 0x44:
                ty = (I32, I64, F32, F64)[op - 0x41]
                if op <= 0x42:
                    val = self.reader.leb(32 if op == 0x41 else 64, signed=True) & ((1 << 64) - 1)
                else:
                    val = int.from_bytes(self.reader.take(4 if op == 0x43 else 8), 'little')
                self.effect([], [ty])
                self.emit(op, val & 0xFFFFFFFF, (val >> 32) if ty in (I64, F64) else 0)
            elif op == 0xD0:
                ty = value_type(self.reader)
                if ty not in (FUNCREF, EXTERNREF):
                    raise ValidationError('invalid reference type')
                self.effect([], [ty])
                self.emit(0x41)
            elif op == 0xD1:
                reference_type = self.pop()
                if reference_type not in (None, FUNCREF, EXTERNREF):
                    raise ValidationError('ref.is_null requires reference')
                self.effect([], [I32])
                self.emit(0x45)
            elif op == 0xD2:
                index = self.module._index(self.reader, self.module.functions, 'function')
                if index not in self.module.declared_functions:
                    raise ValidationError('ref.func requires a declared function reference')
                self.effect([], [FUNCREF])
                self.emit(0x41, index + 1)
            elif op == 0xFC:
                sub = self.reader.leb()
                if sub <= 7:
                    conversion = numeric_signature(0xFC00 + sub)
                    assert conversion is not None
                    self.effect(*conversion)
                    self.emit(0xFC00 + sub, 1)
                else:
                    self.bulk(sub)
            else:
                signature = numeric_signature(op)
                if signature is None:
                    raise UnsupportedFeatureError(f'unsupported opcode 0x{op:02x}')
                self.effect(*signature)
                self.emit(op, len(signature[0]))

    def bulk(self, sub: int) -> None:
        a, b = 0, 0
        if sub in (8, 9):
            if self.module.data_count is None:
                raise ValidationError('memory.init/data.drop requires data count section')
            a = self.module._index(self.reader, self.module.data, 'data segment')
            if sub == 8:
                self.memory_index()
                self.effect([I32, I32, I32], [])
        elif sub in (10, 11):
            self.memory_index()
            if sub == 10:
                self.memory_index()
            self.effect([I32, I32, I32], [])
        elif sub in (12, 13):
            a = self.module._index(self.reader, self.module.elements, 'element segment')
            if sub == 12:
                b = self.table_index()
                if self.module.elements[a][3] != self.module.tables[b][0]:
                    raise ValidationError('table.init element type mismatch')
                self.effect([I32, I32, I32], [])
        elif sub in (14, 15, 16, 17):
            a = self.table_index()
            if sub == 14:
                b = self.table_index()
                if self.module.tables[a][0] != self.module.tables[b][0]:
                    raise ValidationError('table.copy element type mismatch')
            ty = self.module.tables[a][0]
            args = {14: [I32, I32, I32], 15: [ty, I32], 16: [], 17: [I32, ty, I32]}[sub]
            self.effect(args, [I32] if sub in (15, 16) else [])
        else:
            raise UnsupportedFeatureError(f'unsupported 0xfc opcode {sub}')
        self.emit(0xFC00 + sub, a, b)
