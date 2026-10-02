"""Specialize validated WASM basic blocks into WGSL with local SSA values.

Calls use explicit frames and compiled continuations. Bounded compilation units
cover the whole module; no unit contains a bytecode interpreter. Native pipelines
are compiled on demand and cached without imposing a size limit on the module.
"""
from __future__ import annotations

import re
import threading
from array import array
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, cast

from .binary import (
    BRANCH,
    BRANCH_IF,
    BRANCH_TABLE,
    IF_ZERO,
    JUMP,
    RETURN,
    BinaryModule,
    Function,
    numeric_signature,
)
from .errors import ResourceLimitError

TERMINATORS = {0, RETURN, 0x10, 0x11, JUMP, IF_ZERO, BRANCH, BRANCH_IF, BRANCH_TABLE}
MAX_COMPILED_INSTRUCTIONS = 256
MAX_GENERATED_BYTES = 65536
TRAPPING_NUMERIC: set[int] = {0x6D, 0x6E, 0x6F, 0x70, 0x7F, 0x80, 0x81, 0x82}
TRAPPING_NUMERIC.update(range(0xA8, 0xAC))
TRAPPING_NUMERIC.update(range(0xAE, 0xB2))


@dataclass(frozen=True)
class Block:
    start: int
    end: int
    height: int
    peak: int


def _effect(module: BinaryModule, instruction: list[int]) -> int:
    op, a, _, _ = instruction
    if op in (0x10, 0x11):
        args, results = module.signature(a) if op == 0x10 else module.types[a]
        return len(results) - len(args) - int(op == 0x11)
    if op in (0x20, 0x23, 0x3F, 0x41, 0x42, 0x43, 0x44, 0xFC10):
        return 1
    if op in (0x1A, 0x21, 0x24, IF_ZERO, BRANCH_IF, BRANCH_TABLE, 0xFC0F):
        return -1
    if op in (0x1B, 0x26) or 0x36 <= op <= 0x3E:
        return -2
    if op in (0xFC08, 0xFC0A, 0xFC0B, 0xFC0C, 0xFC0E, 0xFC11):
        return -3
    signature = numeric_signature(op)
    return 1 - len(signature[0]) if signature else 0


def blocks(module: BinaryModule, function: Function, maximum: int = 32) -> list[Block]:
    """Infer reachable stack heights on the lowered CFG, excluding br_table data."""
    code = function.instructions
    if not code:
        return []
    heights: dict[int, int] = {}
    leaders = {0}
    pending = deque([(0, 0)])
    while pending:
        pc, height = pending.popleft()
        if pc in heights:
            if heights[pc] != height:
                raise ValueError('inconsistent validated operand stack at a branch')
            continue
        if not 0 <= pc < len(code):
            raise ValueError('validated branch outside function')
        heights[pc] = height
        op, a, b, c = code[pc]
        after = height + _effect(module, code[pc])
        edges = []
        if op in (JUMP, BRANCH, BRANCH_IF, IF_ZERO):
            edges.append((a, b + c if op in (BRANCH, BRANCH_IF) else after))
            leaders.add(a)
        if op == BRANCH_TABLE:
            for _, target, target_height, arity in code[a:a + b]:
                edges.append((target, target_height + arity))
                leaders.add(target)
        if op not in (0, RETURN, JUMP, BRANCH, BRANCH_TABLE):
            edges.append((pc + 1, after))
            if op in TERMINATORS or op >= 0xFC08:
                leaders.add(pc + 1)
        pending.extend(edges)
    result = []
    pc = 0
    while pc < len(code):
        if pc not in heights:
            pc += 1
            continue
        start, peak = pc, heights[pc]
        while True:
            op = code[pc][0]
            peak = max(peak, heights[pc], heights[pc] + _effect(module, code[pc]))
            pc += 1
            if op in TERMINATORS or op >= 0xFC08 or pc in leaders or pc not in heights or pc - start >= maximum:
                break
        result.append(Block(start, pc, heights[start], peak))
    return result


def _interval_width(interval: tuple[int, int]) -> int:
    return interval[1] - interval[0]


def clusters(module: BinaryModule, function: Function, limit: int) -> list[list[Block]]:
    """Keep bounded loops together so a backedge need not launch another kernel."""
    body = blocks(module, function, min(32, limit))
    positions = {block.start: index for index, block in enumerate(body)}
    totals = [0]
    intervals: set[tuple[int, int]] = set()
    for index, block in enumerate(body):
        totals.append(totals[-1] + block.end - block.start)
        op, a, b, _ = function.instructions[block.end - 1]
        targets = [a] if op in (JUMP, IF_ZERO, BRANCH, BRANCH_IF) else []
        if op == BRANCH_TABLE:
            targets = [target for _, target, _, _ in function.instructions[a:a + b]]
        for target in targets:
            if target < block.start:
                intervals.add((positions[target], index + 1))
    joined: set[int] = set()
    for interval in sorted(intervals, key=_interval_width):
        start, end = interval
        if totals[end] - totals[start] > limit:
            continue
        while start in joined:
            start -= 1
        while end in joined:
            end += 1
        if totals[end] - totals[start] <= limit:
            joined.update(range(start + 1, end))
    result: list[list[Block]] = []
    for index, block in enumerate(body):
        if index not in joined:
            result.append([])
        result[-1].append(block)
    return result


class BlockEmitter:
    def __init__(self, module: BinaryModule, function: Function, block: Block, canonical: list[int], tables: dict[int, int], *, checked: bool = False) -> None:  # noqa: PLR0913 - Static lowering context and checked-prefix mode.
        self.module = module
        self.function, self.block, self.canonical = function, block, canonical
        self.stack = [f's{i}' for i in range(block.height)]
        self.locals: dict[int, str] = {}
        self.dirty: set[int] = set()
        self.lines: list[str] = []
        self.numeric: set[int] = set()
        self.pc = block.start
        self.checked = checked
        self.tables = tables

    def emit(self, text: str) -> None:
        self.lines.append(text)

    def push(self, expression: str) -> None:
        if self.checked and len(self.stack) >= self.block.height:
            self.trap_if(f'vm.base + {len(self.function.locals) + len(self.stack)}u >= config.stack_cap', '7u')
        name = f'v{self.pc}'
        self.emit(f'let {name} = {expression};')
        self.stack.append(name)

    def local(self, index: int) -> str:
        if index not in self.locals:
            self.locals[index] = f'l{index}'
            self.emit(f'let l{index} = get_value(vm.base + {index}u);')
        return self.locals[index]

    def finish(self, action: str = '', *, consumed: int | None = None) -> str:
        """Spill only at a continuation or trap, with the exact executed prefix."""
        count = self.pc - self.block.start + 1 if consumed is None else consumed
        lines = [f'set_value(vm.base + {index}u, {self.locals[index]});' for index in sorted(self.dirty)]
        offset = len(self.function.locals)
        lines.extend(f'set_value(vm.base + {offset + index}u, {value});' for index, value in enumerate(self.stack) if value != f's{index}')
        lines.extend([f'vm.sp = vm.base + {offset + len(self.stack)}u;', f'vm.pc = {self.function.offset + self.block.start + count}u;', f'compiled_tick({count}u);'])
        if action:
            lines.append(action)
        lines.append(f'return {count}u;')
        return '\n'.join(lines)

    def trap_if(self, condition: str, code: str) -> None:
        self.emit(f'if {condition} {{\n{self.finish(f"fail({code});")}\n}}')

    def instruction(self, instruction: list[int]) -> bool:  # noqa: PLR0915 - Explicit instruction lowering.
        op, a, b, c = instruction
        pop = self.stack.pop
        if op == 0:
            self.emit(self.finish('fail(1u);'))
        elif op == RETURN:
            self.emit(self.finish('return_function();'))
        elif op == JUMP:
            self.emit(self.finish(f'vm.pc = {self.function.offset + a}u;'))
        elif op in (IF_ZERO, BRANCH, BRANCH_IF):
            condition = f'{pop()}.x {"==" if op == IF_ZERO else "!="} 0u' if op != BRANCH else 'true'
            action = f'vm.pc = {self.function.offset + a}u;' if op == IF_ZERO else f'branch({self.function.offset + a}u, {b}u, {c}u);'
            self.emit(self.finish(f'if {condition} {{ {action} }}'))
        elif op == BRANCH_TABLE:
            selector = pop()
            offset = self.tables[self.function.offset + self.pc]
            self.emit(self.finish(f'compiled_branch_table({offset}u, {b}u, {selector}.x);'))
        elif op == 0x10:
            self.emit(self.finish(f'invoke({a}u);'))
        elif op == 0x11:
            index = pop()
            self.emit(self.finish(f'compiled_indirect({self.canonical[a]}u, {b}u, {index}.x);'))
        elif op >= 0xFC08:
            self.emit(self.finish(f'bulk_operation({op - 0xFC00}u, {a}u, {b}u);'))
        elif op == 0x1A:
            pop()
        elif op == 0x1B:
            condition, rhs, lhs = pop(), pop(), pop()
            self.push(f'select({rhs}, {lhs}, {condition}.x != 0u)')
        elif op == 0x20:
            self.push(self.local(a))
        elif op in (0x21, 0x22):
            self.locals[a] = pop() if op == 0x21 else self.stack[-1]
            self.dirty.add(a)
        elif op == 0x23:
            self.push(f'vec2u(read_heap({a * 2}u), read_heap({a * 2 + 1}u))')
        elif op == 0x24:
            value = pop()
            self.emit(f'write_heap({a * 2}u, {value}.x); write_heap({a * 2 + 1}u, {value}.y);')
        elif op in (0x25, 0x26):
            value = pop() if op == 0x26 else ''
            index = pop()
            self.trap_if(f'{index}.x >= table_size({a}u)', '3u')
            if op == 0x25:
                self.push(f'vec2u(read_heap(table_offset({a}u) + {index}.x), 0u)')
            else:
                self.emit(f'write_heap(table_offset({a}u) + {index}.x, {value}.x);')
        elif 0x28 <= op <= 0x3E:
            value = pop() if op >= 0x36 else ''
            pointer = pop()
            address = f'address{self.pc}'
            self.emit(f'let {address} = {pointer}.x + {a}u;')
            self.trap_if(f'{address} < {pointer}.x || !bounds({address}, {b}u)', '2u')
            if op >= 0x36:
                self.emit(f'compiled_store({address}, {b}u, {value});')
            else:
                expression = f'compiled_load({address}, {b}u)'
                if op in (0x2C, 0x2E, 0x30, 0x32, 0x34):
                    expression = f'sar64(shl64({expression}, {64 - b * 8}u), {64 - b * 8}u)'
                if op in (0x28, 0x2A, 0x2C, 0x2D, 0x2E, 0x2F):
                    expression = f'vec2u(({expression}).x, 0u)'
                self.push(expression)
        elif op == 0x3F:
            self.push('vec2u(vm.pages, 0u)')
        elif op == 0x40:
            delta = pop()
            self.emit(f'let old{self.pc} = vm.pages;')
            self.emit(f'let fits{self.pc} = {delta}.x <= config.memory_cap - vm.pages;')
            self.emit(f'if fits{self.pc} {{ vm.pages += {delta}.x; }}')
            self.push(f'vec2u(select(0xffffffffu, old{self.pc}, fits{self.pc}), 0u)')
        elif 0x41 <= op <= 0x44:
            self.push(f'vec2u({a}u, {b}u)')
        else:
            signature = numeric_signature(op)
            if signature is None:
                raise ValueError(f'cannot compile opcode {op}')
            rhs = pop() if len(signature[0]) == 2 else 'vec2u(0u)'
            lhs = pop()
            self.numeric.add(op)
            self.emit(f'let answer{self.pc} = numeric_{op:x}({lhs}, {rhs});')
            if op in TRAPPING_NUMERIC:
                self.trap_if(f'answer{self.pc}.trap != 0u', f'answer{self.pc}.trap')
            self.push(f'answer{self.pc}.value')
        return op in TERMINATORS or op >= 0xFC08

    def render(self) -> str:
        block = self.block
        count = block.end - block.start
        if self.checked:
            self.emit('if !fuel_available(1u) { fail(8u); return 0u; }')
        else:
            self.emit(f'if !fuel_available({count}u) || vm.base + {len(self.function.locals) + block.peak}u > config.stack_cap {{ return prefix_{self.function.offset + block.start}(); }}')
        terminal = False
        for self.pc in range(block.start, block.end):
            terminal = self.instruction(self.function.instructions[self.pc])
        if not terminal:
            self.emit(self.finish())
        body = '\n'.join(self.lines[1:])
        # Untouched incoming stack slots already reside in storage and need no
        # load/spill. Omitting dead loads also keeps native compiler input small.
        used = sorted({int(match[1]) for match in re.finditer(r'\bs(\d+)\b', body)})
        loads = [f'let s{index} = get_value(vm.base + {len(self.function.locals) + index}u);' for index in used]
        return '\n'.join([self.lines[0], *loads, body])


def _numeric(opcodes: set[int]) -> str:
    """Reuse exact numeric implementations, removing the runtime opcode switch."""
    source = Path(__file__).with_name('operations.wgsl').read_text()
    body = source[source.index('    var result'):source.index('    switch op')]
    cases = list(re.finditer(r'^        (case [^\n]+|default): \{', source, re.MULTILINE))
    implementations: dict[int, str] = {}
    default = ''
    for index, match in enumerate(cases):
        end = cases[index + 1].start() if index + 1 < len(cases) else source.rindex('    }')
        code = source[match.end():end].rstrip()
        code = code[:code.rfind('}')]
        label = cast(str, match.group(1))
        if label == 'default':
            default = code
        else:
            for value in re.finditer(r'0x([0-9a-f]+)u', label):
                implementations[int(cast(str, value.group(1)), 16)] = code
    return '\n'.join(f'fn numeric_{op:x}(a: vec2u, b: vec2u) -> NumericResult {{\nlet op = {op}u;\n{body}{implementations.get(op, default)}\nreturn NumericResult(result, trap);\n}}' for op in sorted(opcodes))



def vm_support() -> str:
    """Shared memory/frames/services ABI, deliberately excluding bytecode execution."""
    source = Path(__file__).with_name('vm.wgsl').read_text().split('fn interpret_step()', 1)[0]
    start, end = source.index('fn memory_operation('), source.index('fn table_size(')
    return source[:start] + source[end:]


class CompiledModule:
    """Complete CFG with bounded, lazily generated native compilation units.

    The pc map is host scheduling metadata. GPU buffers contain function metadata
    and data/element segments only, never the WASM instruction stream.
    """

    def __init__(self, module: BinaryModule, order: list[int], instruction_limit: int, source_limit: int) -> None:
        self.module = module
        self.program: array[int] = array('I')
        self.source_limit = source_limit
        self.functions = tuple(range(len(module.functions)))
        self.wasi = any(fn.imported is not None for fn in module.functions)
        self.complete = True
        self.canonical = [module.types.index(signature) for signature in module.types]
        self.bodies = list(module.functions)
        for index, function in enumerate(self.bodies):
            if function.imported is not None:
                params, _ = module.signature(index)
                instructions = [[0x20, i, 0, 0] for i in range(len(params))] + [[0x10, index, 0, 0], [RETURN, 0, 0, 0]]
                self.bodies[index] = Function(function.type_index, list(params), instructions, function.imported, function.offset, len(params))
        self.total_instructions = sum(len(fn.instructions) for fn in self.bodies)
        self.branch_tables: dict[int, int] = {}
        offset = 0
        for function in self.bodies:
            for pc, (op, _, count, _) in enumerate(function.instructions):
                if op == BRANCH_TABLE:
                    self.branch_tables[function.offset + pc] = offset
                    offset += count
        self.opcodes = tuple(sorted({op for fn in self.bodies for op, _, _, _ in fn.instructions}))
        self.regions: list[list[tuple[int, Block]]] = [[]]
        size = 0
        self.blocks = self.instructions = 0
        self.locations = array('I', [0]) * self.total_instructions
        for index in order:
            function = self.bodies[index]
            groups = clusters(module, function, instruction_limit)
            function_count = sum(block.end - block.start for group in groups for block in group)
            if size and size + function_count > instruction_limit:
                self.regions.append([])
                size = 0
            for group in groups:
                count = sum(block.end - block.start for block in group)
                if size + count > instruction_limit:
                    self.regions.append([])
                    size = 0
                for block in group:
                    self.regions[-1].append((index, block))
                    self.locations[function.offset + block.start] = len(self.regions)
                self.blocks += len(group)
                self.instructions += count
                size += count
        self._sources: OrderedDict[int, str] = OrderedDict()
        self._lock = threading.RLock()

    @property
    def source(self) -> str:
        """First compilation unit, retained for diagnostics of small modules."""
        return self.source_for(0)

    def region_for(self, pc: int) -> int:
        if not 0 <= pc < len(self.locations) or not self.locations[pc]:
            raise RuntimeError(f'invalid compiled continuation {pc}')
        return self.locations[pc] - 1

    def _render(self, region: int) -> str:
        functions, dispatch = [], []
        numeric: set[int] = set()
        for index, block in self.regions[region]:
            function = self.bodies[index]
            address = function.offset + block.start
            emitter = BlockEmitter(self.module, function, block, self.canonical, self.branch_tables)
            functions.append(f'fn block_{address}() -> u32 {{\n{emitter.render()}\n}}')
            numeric.update(emitter.numeric)
            prefix = ['var consumed = 0u;']
            height = block.height
            for pc in range(block.start, block.end):
                after = height + _effect(self.module, function.instructions[pc])
                step = BlockEmitter(self.module, function, Block(pc, pc + 1, height, max(height, after)), self.canonical, self.branch_tables, checked=True)
                step_address = function.offset + pc
                functions.append(f'fn step_{step_address}() -> u32 {{\n{step.render()}\n}}')
                numeric.update(step.numeric)
                prefix.append(f'consumed += step_{step_address}(); if vm.status != 0u {{ return consumed; }}')
                height = after
            functions.append(f'fn prefix_{address}() -> u32 {{\n' + '\n'.join(prefix) + '\nreturn consumed;\n}')
            dispatch.append(f'case {address}u: {{ return block_{address}(); }}')
        return (_numeric(numeric) + '\n' + '\n'.join(functions) +
                '\nfn compiled_dispatch() -> u32 {\nswitch vm.pc {\n' + '\n'.join(dispatch) +
                '\ndefault: { return 0u; }\n}\n}\n')

    def _split(self, region: int) -> None:
        units = self.regions[region]
        if not units:
            raise ResourceLimitError('empty compilation unit exceeds the generated source budget')
        if len(units) == 1:
            index, block = units[0]
            if block.end - block.start == 1:
                raise ResourceLimitError('one compiled instruction exceeds the generated source budget')
            middle = (block.start + block.end) // 2
            height = block.height
            peak = height
            function = self.bodies[index]
            for pc in range(block.start, middle):
                height += _effect(self.module, function.instructions[pc])
                peak = max(peak, height)
            units = [(index, Block(block.start, middle, block.height, peak)),
                     (index, Block(middle, block.end, height, block.peak))]
            self.blocks += 1
        middle = len(units) // 2
        self.regions[region] = units[:middle]
        self.regions.append(units[middle:])
        for index, block in self.regions[-1]:
            self.locations[self.bodies[index].offset + block.start] = len(self.regions)

    def source_for(self, region: int) -> str:
        with self._lock:
            if region not in self._sources:
                source = self._render(region)
                while len(source.encode()) > min(MAX_GENERATED_BYTES, self.source_limit):
                    self._split(region)
                    source = self._render(region)
                self._sources[region] = source
            self._sources.move_to_end(region)
            while len(self._sources) > 16:
                self._sources.popitem(last=False)
            return self._sources[region]


def compile_module(module: BinaryModule, functions: Iterable[int] | None = None, *, instruction_limit: int = MAX_COMPILED_INSTRUCTIONS,
                   source_limit: int = MAX_GENERATED_BYTES) -> CompiledModule:
    """Compile every function; the limit bounds each unit, never module coverage.

    Optional function indices influence grouping order only. All other functions
    follow them and are compiled when reached, with no interpreter fallback.
    """
    if not 1 <= instruction_limit <= MAX_COMPILED_INSTRUCTIONS:
        raise ResourceLimitError(f'compilation units are limited to 1..{MAX_COMPILED_INSTRUCTIONS} instructions')
    if source_limit < 256:
        raise ResourceLimitError('generated source limit is too small')
    order: list[int] = []
    selected: set[int] = set()
    for index in functions or ():
        if index not in selected:
            selected.add(index)
            order.append(index)
    if any(not 0 <= index < len(module.functions) for index in order):
        raise ValueError('compiled function index out of range')
    order.extend(index for index in range(len(module.functions)) if index not in selected)
    return CompiledModule(module, order, instruction_limit, source_limit)
