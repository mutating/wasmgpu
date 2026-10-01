from __future__ import annotations

import math
import random
import struct

import pytest
import wasmtime

import wasmgpu

from .conftest import binary, float_bits, oracle

pytestmark = pytest.mark.gpu

INTEGER_BINARY = ['add', 'sub', 'mul', 'div_s', 'div_u', 'rem_s', 'rem_u', 'and', 'or', 'xor',
                  'shl', 'shr_s', 'shr_u', 'rotl', 'rotr', 'eq', 'ne', 'lt_s', 'lt_u', 'gt_s',
                  'gt_u', 'le_s', 'le_u', 'ge_s', 'ge_u']
FLOAT_BINARY = ['add', 'sub', 'mul', 'div', 'min', 'max', 'copysign', 'eq', 'ne', 'lt', 'gt', 'le', 'ge']


def compare(engine, wat, rows, result_type, **options):
    expected = oracle(engine, wat, 'run', rows)
    with wasmgpu.Module(binary(wat)).spawn(len(rows), **options) as instances:
        try:
            actual = instances.call('run', rows)
            traps = {}
        except wasmgpu.Trap as error:
            actual, traps = error.results, error.traps
    for index, (got, want) in enumerate(zip(actual, expected)):
        context = (index, rows[index], got, want, wat)
        if isinstance(want, wasmtime.Trap):
            assert index in traps, context
        else:
            assert index not in traps, context
            if result_type in ('f32', 'f64'):
                if math.isnan(want):
                    assert math.isnan(got), context
                else:
                    assert float_bits(got, result_type) == float_bits(want, result_type), context
            else:
                assert got == want, context


@pytest.mark.parametrize('ty', ['i32', 'i64'])
@pytest.mark.parametrize('op', INTEGER_BINARY)
def test_integer_binary(engine, ty, op):
    bits = int(ty[1:])
    limit = 1 << (bits - 1)
    edges = [-limit, -limit + 1, -1, 0, 1, 2, 31, 32, 63, 64, limit - 1]
    rng = random.Random(817)
    rows = [(a, b) for a in edges for b in edges]
    rows.extend((rng.randrange(-limit, limit), rng.randrange(-limit, limit)) for _ in range(100))
    result = 'i32' if op in INTEGER_BINARY[15:] else ty
    wat = f'(module (func (export "run") (param {ty} {ty}) (result {result}) local.get 0 local.get 1 {ty}.{op}))'
    compare(engine, wat, rows, result)


@pytest.mark.parametrize('ty', ['i32', 'i64'])
@pytest.mark.parametrize('op', ['eqz', 'clz', 'ctz', 'popcnt', 'extend8_s', 'extend16_s'])
def test_integer_unary(engine, ty, op):
    bits = int(ty[1:])
    values = [0, -1, -(1 << (bits - 1))] + [1 << bit for bit in range(bits - 1)] + [127, 128, 255, 32768, 65535]
    result = 'i32' if op == 'eqz' else ty
    compare(engine, f'(module (func (export "run") (param {ty}) (result {result}) local.get 0 {ty}.{op}))', [(x,) for x in values], result)


def floats(ty):
    size = 32 if ty == 'f32' else 64
    fmt = '<f' if size == 32 else '<d'
    rng = random.Random(139)
    values = [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 1.5, 2.5, -2.5, math.inf, -math.inf, math.nan]
    raw = [1, 2, (1 << (23 if size == 32 else 52)) - 1, 1 << (23 if size == 32 else 52),
           0x7f7fffff if size == 32 else 0x7fefffffffffffff]
    raw += [rng.getrandbits(size) for _ in range(180)]
    values += [struct.unpack(fmt, value.to_bytes(size // 8, 'little'))[0] for value in raw]
    return values


@pytest.mark.parametrize('ty', ['f32', 'f64'])
@pytest.mark.parametrize('op', FLOAT_BINARY)
def test_float_binary(engine, ty, op):
    values = floats(ty)
    rows = [(a, b) for a in values[:17] for b in values[:17]]
    rows.extend(zip(values[17:], reversed(values[17:])))
    result = 'i32' if op in FLOAT_BINARY[7:] else ty
    compare(engine, f'(module (func (export "run") (param {ty} {ty}) (result {result}) local.get 0 local.get 1 {ty}.{op}))', rows, result)


@pytest.mark.parametrize('ty', ['f32', 'f64'])
@pytest.mark.parametrize('op', ['abs', 'neg', 'ceil', 'floor', 'trunc', 'nearest', 'sqrt'])
def test_float_unary(engine, ty, op):
    compare(engine, f'(module (func (export "run") (param {ty}) (result {ty}) local.get 0 {ty}.{op}))', [(v,) for v in floats(ty)], ty)


@pytest.mark.parametrize('source', ['i32', 'i64'])
@pytest.mark.parametrize('destination', ['f32', 'f64'])
@pytest.mark.parametrize('sign', ['s', 'u'])
def test_integer_to_float(engine, source, destination, sign):
    bits = int(source[1:])
    rng = random.Random(717)
    values = [-1, 0, 1, -(1 << (bits - 1)), (1 << (bits - 1)) - 1]
    values += [rng.randrange(-(1 << (bits - 1)), 1 << (bits - 1)) for _ in range(200)]
    compare(engine, f'(module (func (export "run") (param {source}) (result {destination}) local.get 0 {destination}.convert_{source}_{sign}))', [(x,) for x in values], destination)


@pytest.mark.parametrize('source', ['f32', 'f64'])
@pytest.mark.parametrize('destination', ['i32', 'i64'])
@pytest.mark.parametrize('sign', ['s', 'u'])
@pytest.mark.parametrize('saturate', ['', '_sat'])
def test_float_to_integer(engine, source, destination, sign, saturate):
    values = floats(source)
    values += [float(value) for value in [-0.99, 0.99, 2**31 - 1, -(2**31), 2**32, 2**63, -(2**63), 2**64]]
    compare(engine, f'(module (func (export "run") (param {source}) (result {destination}) local.get 0 {destination}.trunc{saturate}_{source}_{sign}))', [(x,) for x in values], destination)


@pytest.mark.parametrize(('source', 'destination', 'op'), [('f32', 'f64', 'promote_f32'), ('f64', 'f32', 'demote_f64')])
def test_float_width(engine, source, destination, op):
    compare(engine, f'(module (func (export "run") (param {source}) (result {destination}) local.get 0 {destination}.{op}))', [(x,) for x in floats(source)], destination)
