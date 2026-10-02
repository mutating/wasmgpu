"""Real CPython WASI and throng guest workloads; no scheduler overhead included.

Supply a runtime directory containing python.wasm and lib/, and (for linters)
a directory of pure-Python wheels. Downloads and input preparation are untimed.
Run GPU variants through tests.gpu_guard. Results are printed, never checked in.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import tempfile
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from zipfile import ZipFile

import wasmtime

import wasmgpu
from wasmgpu.binary import Reader

SOURCE = b'from typing import Iterable\n\ndef total(values: Iterable[int]) -> int:\n    return sum(values)\n\nanswer: int = total([1, 2, 3])\n'


def function_names(data):
    reader = Reader(data)
    reader.take(8)
    names = {}
    while reader.pos < len(reader.data):
        section_id = reader.byte()
        section = Reader(reader.take(reader.leb()))
        if section_id != 0 or section.name() != 'name':
            continue
        while section.pos < len(section.data):
            subsection_id = section.byte()
            subsection = Reader(section.take(section.leb()))
            if subsection_id == 1:
                for _ in range(subsection.leb()):
                    index = subsection.leb()
                    names[index] = subsection.name()
    return names


def workload(runtime, scenario, wheels, count):
    files = {'python/' + path.relative_to(runtime).as_posix(): path.read_bytes()
             for path in sorted((runtime / 'lib').rglob('*')) if path.is_file()}
    env = {'PYTHONHOME': '/python', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONHASHSEED': '0'}
    if scenario == 'smoke':
        args = ['-S', '-c', 'print(6 * 7)']
    elif scenario == 'startup':
        args = ['-c', 'pass']
    else:
        if not wheels or not list(wheels.glob('*.whl')):
            raise ValueError('linter scenarios require --wheels containing pure-Python wheels')
        for wheel in sorted(wheels.glob('*.whl')):
            with ZipFile(wheel) as archive:
                for entry in archive.infolist():
                    if not entry.is_dir():
                        files['packages/' + entry.filename] = archive.read(entry)
        env['PYTHONPATH'] = '/packages'
        for index in range(count):
            files[f'project/module_{index:03}.py'] = SOURCE
        # Only the project is scanned; stdlib and wheels share the embedded root.
        args = ['-m', scenario]
        if scenario == 'mypy':
            args += ['--no-incremental', '--cache-dir=/dev/null', '--no-site-packages', '--python-version=3.13', '--platform=linux']
        args += ['project']
    return files, ['python', *args], env


def cpu(data, files, args, env, *, count_fuel=False):
    with tempfile.TemporaryDirectory(prefix='wasmgpu-oracle-') as directory:
        root = Path(directory)
        for name, content in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        started = time.perf_counter()
        engine_config = wasmtime.Config()
        engine_config.consume_fuel = count_fuel
        engine = wasmtime.Engine(engine_config)
        module = wasmtime.Module(engine, data)
        compilation = time.perf_counter() - started
        started = time.perf_counter()
        store = wasmtime.Store(engine)
        if count_fuel:
            store.set_fuel(1 << 60)
        config = wasmtime.WasiConfig()
        config.argv, config.env = args, list(env.items())
        config.preopen_dir(directory, '.')
        config.stdout_file, config.stderr_file = str(root / 'stdout'), str(root / 'stderr')
        store.set_wasi(config)
        linker = wasmtime.Linker(engine)
        linker.define_wasi()
        instance = linker.instantiate(store, module)
        preparation = time.perf_counter() - started
        started = time.perf_counter()
        code = 0
        try:
            instance.exports(store)['_start'](store)
        except wasmtime.ExitTrap as error:
            code = error.code
        execution = time.perf_counter() - started
        return {'compile_seconds': compilation, 'prepare_seconds': preparation, 'execute_seconds': execution,
                'wasmtime_fuel': (1 << 60) - store.get_fuel() if count_fuel else None,
                'exit_code': code, 'stdout': (root / 'stdout').read_text(), 'stderr': (root / 'stderr').read_text()}


def gpu(data, files, args, env, *, options, names):  # noqa: PLR0913 - Workload and measurement configuration.
    if not os.environ.get('WASMGPU_GUARD_STATUS'):
        raise RuntimeError('run this development benchmark through python -m tests.gpu_guard')
    selected = [index for index, name in names.items() if name in options.functions]
    if options.engine == 'compiled' and len(selected) != len(options.functions):
        raise ValueError('selected function name missing or ambiguous in this runtime')
    module = wasmgpu.Module(data, execution=options.engine, compile_functions=selected)
    plan = module.compiled
    print(json.dumps({'stage': 'planned', 'functions': len(plan.functions) if plan else 0, 'instructions': plan.instructions if plan else 0,  # noqa: T201 - Benchmark CLI.
                      'compilation_units': len(plan.regions) if plan else 0}), flush=True)
    wasi = wasmgpu.Wasi(args=args, env=env, files=files, storage_size=48 * 1024 * 1024, max_files=4096)
    started = time.perf_counter()
    with module.spawn(1, memory_pages=1024, stack_size=8192, call_depth=1024, fuel=options.fuel,
                      quantum=262144, wasi=wasi) as instances:
        preparation = time.perf_counter() - started - instances.pipeline_seconds - instances.codegen_seconds
        print(json.dumps({'stage': 'pipeline', 'seconds': instances.pipeline_seconds}), flush=True)  # noqa: T201
        started = time.perf_counter()
        error = None
        reported = started

        def cancelled():
            nonlocal reported
            now = time.perf_counter()
            if now - reported >= 15:
                state = instances._batches[0].state
                print(json.dumps({'stage': 'execute', 'seconds': round(now - started, 2),  # noqa: T201 - Bounded-run progress.
                                  'fuel_used': options.fuel - state[9] - (state[12] << 32),
                                  'function': names.get(state[3], str(state[3])),
                                  'pipelines_created': instances.last_call.pipelines_created,
                                  'compile_seconds': round(instances.last_call.compile_seconds, 2),
                                  'execute_seconds': round(instances.last_call.execute_seconds, 2)}), flush=True)
                reported = now
            return now - started > options.call_seconds
        try:
            instances.call('_start', cancel=cancelled)
        except wasmgpu.Trap as trap:
            if any(value != 'WASI proc_exit' for value in trap.traps.values()):
                error = str(trap)
        except InterruptedError as interrupted:
            error = str(interrupted)
        wall = time.perf_counter() - started
        metrics = asdict(instances.last_call)
        metrics['function_samples'] = {names.get(index, str(index)): count for index, count in
                                      sorted(instances.last_call.function_samples.items(), key=lambda item: -item[1])[:20]}
        started = time.perf_counter()
        stdout, stderr = instances.stdout[0].decode(), instances.stderr[0].decode()
        stream_readback = time.perf_counter() - started
        exit_code = (instances.exit_codes[0] or 0) if error is None else None
        cached = wasmgpu.Module(data, execution=options.engine, compile_functions=selected)
        started = time.perf_counter()
        instances.reset()
        reset_seconds = time.perf_counter() - started
        return {'load_seconds': module.load_seconds, 'codegen_seconds': module.codegen_seconds + instances.codegen_seconds,
                'cached_load_seconds': cached.load_seconds, 'cached_codegen_seconds': cached.codegen_seconds,
                'reset_seconds': reset_seconds,
                'compiled_functions': [names.get(index, str(index)) for index in plan.functions[:32]] if plan else [],
                'compiled_function_count': len(plan.functions) if plan else 0,
                'compilation_units': len(plan.regions) if plan else 0,
                'compiled_static_instructions': plan.instructions if plan else 0,
                'compile_seconds': instances.pipeline_seconds, 'pipeline_cache_hit': instances.pipeline_cache_hit,
                'prepare_seconds': preparation, 'call_seconds': wall, 'stream_readback_seconds': stream_readback,
                'metrics': metrics, 'exit_code': exit_code, 'error': error,
                'stdout': stdout, 'stderr': stderr, 'adapter': instances.adapter_info}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', required=True, type=Path)
    parser.add_argument('--engine', required=True, choices=['wasmtime', 'interpreter', 'compiled'])
    parser.add_argument('--scenario', default='smoke', choices=['smoke', 'startup', 'pyflakes', 'mypy'])
    parser.add_argument('--wheels', type=Path)
    parser.add_argument('--files', type=int, default=1, choices=[1, 100])
    parser.add_argument('--functions', nargs='+', default=['_PyCode_Quicken', '_PyPegen_is_memoized', 'strlen', 'visit_decref'])
    parser.add_argument('--fuel', type=int, default=300_000_000)
    parser.add_argument('--call-seconds', type=float, default=120)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--reference', type=Path, help='assert observable results against a previous engine report')
    parser.add_argument('--count-fuel', action='store_true', help='instrument Wasmtime for a separate work-volume estimate; changes CPU timings')
    options = parser.parse_args()
    root = Path(wasmgpu.__file__).parent
    source_digest = hashlib.sha256()
    for path in sorted([*root.glob('*.py'), *root.glob('*.wgsl')]):
        source_digest.update(path.name.encode() + b'\0' + path.read_bytes())
    data = (options.runtime / 'python.wasm').read_bytes()
    files, args, env = workload(options.runtime, options.scenario, options.wheels, options.files)
    result = (cpu(data, files, args, env, count_fuel=options.count_fuel) if options.engine == 'wasmtime' else
              gpu(data, files, args, env, options=options, names=function_names(data)))
    result.update(engine=options.engine, scenario=options.scenario, files=options.files,
                  runtime_sha256=hashlib.sha256(data).hexdigest(), args=args, env=env,
                  engine_source_sha256=source_digest.hexdigest(), python=platform.python_version(),
                  system=platform.platform(), wgpu=version('wgpu'), wasmtime=version('wasmtime'))
    digest = hashlib.sha256()
    for name, content in sorted(files.items()):
        digest.update(name.encode() + b'\0' + len(content).to_bytes(8, 'little') + content)
    result['workload_sha256'] = digest.hexdigest()
    if options.reference:
        reference = json.loads(options.reference.read_text())
        keys = ('runtime_sha256', 'workload_sha256', 'scenario', 'files', 'args', 'env', 'exit_code', 'stdout', 'stderr')
        result['matches_reference'] = all(result[key] == reference.get(key) for key in keys) and result.get('error') is None
    output = json.dumps(result, indent=2)
    print(output, flush=True)  # noqa: T201
    if options.output:
        options.output.write_text(output + '\n')
    return int(result.get('error') is not None or result.get('matches_reference') is False)


if __name__ == '__main__':
    raise SystemExit(main())
