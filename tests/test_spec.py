"""Selected official WebAssembly 2.0 core suites, executed on real GPU hardware.

The archive preserves the upstream assertions and modules; see fixtures/README.
Raw argument bits avoid Python's conversion of signalling f32 NaNs to quiet NaNs.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

import wasmgpu

pytestmark = pytest.mark.gpu

ARCHIVE = Path(__file__).parent / 'fixtures' / 'core-spec.zip'
with zipfile.ZipFile(ARCHIVE) as _archive:
    SUITES = sorted(name for name in _archive.namelist() if name.endswith('.json'))


def check_value(actual, expected):
    ty, value = expected['type'], expected['value']
    if ty in ('i32', 'i64'):
        width = 32 if ty == 'i32' else 64
        assert actual & ((1 << width) - 1) == int(value)
    elif ty in ('f32', 'f64'):
        size = 32 if ty == 'f32' else 64
        actual &= (1 << size) - 1
        if value.startswith('nan:'):
            fraction = 23 if size == 32 else 52
            exponent = ((1 << (size - fraction - 1)) - 1) << fraction
            assert actual & exponent == exponent
            assert actual & (1 << (fraction - 1))
            if value == 'nan:canonical':
                assert actual & ((1 << fraction) - 1) == 1 << (fraction - 1)
        else:
            assert actual == int(value)
    elif ty in ('funcref', 'externref'):
        assert actual == (0 if value == 'null' else int(value) + 1)
    else:
        raise AssertionError(f'unhandled reference type {ty}')



@pytest.mark.parametrize('suite', SUITES)
def test_official_core(suite):
    instances = None
    with zipfile.ZipFile(ARCHIVE) as archive:
        commands = json.loads(archive.read(suite))['commands']
        try:
            for command in commands:
                kind = command['type']
                context = f"{suite}:{command['line']} {kind}"
                try:
                    if kind in ('assert_invalid', 'assert_malformed'):
                        with pytest.raises((wasmgpu.ValidationError, wasmgpu.UnsupportedFeatureError)):
                            wasmgpu.Module(archive.read(command['filename']))
                    elif kind == 'module':
                        if instances is not None:
                            instances.close()
                        module = wasmgpu.Module(archive.read(command['filename']))
                        instances = module.spawn(1, stack_size=32768, call_depth=4096, fuel=10000000, memory_pages=min(1024 if suite == 'memory_grow.json' else 512, module._binary.memory[1] if module._binary.memory[1] is not None else 1024) if module._binary.memory else 0)
                    elif kind in ('assert_return', 'assert_trap', 'assert_exhaustion', 'action'):
                        assert instances is not None
                        action = command['action']
                        assert action['type'] == 'invoke'
                        index = instances.module._binary.exports[action['field']][1]
                        args = tuple(int(arg['value']) + int(arg['type'] in ('funcref', 'externref')) if arg['value'] != 'null' else 0 for arg in action['args'])
                        if kind in ('assert_trap', 'assert_exhaustion'):
                            with pytest.raises(wasmgpu.Trap):
                                instances._call(index, [args], instances.fuel, raw=True)
                        else:
                            actual = instances._call(index, [args], instances.fuel, raw=True)[0]
                            if kind == 'assert_return':
                                expected = command['expected']
                                values = actual if isinstance(actual, tuple) else (actual,) if expected else ()
                                assert len(values) == len(expected)
                                for value, wanted in zip(values, expected):
                                    check_value(value, wanted)
                    else:
                        raise AssertionError(f'unhandled command {kind}')
                except Exception as error:
                    raise AssertionError(context) from error
        finally:
            if instances is not None:
                instances.close()
