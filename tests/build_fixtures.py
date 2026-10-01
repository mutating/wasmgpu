"""Rebuild offline conformance fixtures with an explicitly supplied WABT/spec checkout."""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import zipfile
from pathlib import Path

SPEC_COMMIT = '05ca4182176763112561ae20153975c12bd689e4'  # WebAssembly/spec v2.0.0
SUITES = ['i32', 'i64', 'f32', 'f64', 'f32_cmp', 'f64_cmp', 'conversions', 'float_exprs', 'float_literals', 'float_memory', 'int_exprs', 'int_literals', 'block', 'loop', 'if', 'br', 'br_if', 'br_table', 'call', 'call_indirect', 'return', 'select', 'local_get', 'local_set', 'local_tee', 'memory', 'memory_copy', 'memory_fill', 'memory_init', 'memory_grow', 'memory_size', 'load', 'store', 'align', 'address', 'const', 'end', 'func', 'nop', 'unreachable', 'unwind']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--wast2json', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('tests/fixtures/core-spec.zip'))
    args = parser.parse_args()
    revision = subprocess.check_output(['git', '-C', str(args.spec), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != SPEC_COMMIT:
        parser.error(f'spec checkout must be at {SPEC_COMMIT}, found {revision}')
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        for name in SUITES:
            source = args.spec / 'test' / 'core' / (name + '.wast')
            if source.exists():
                subprocess.run([str(args.wast2json), str(source), '-o', str(output / (name + '.json'))], check=True)
        with zipfile.ZipFile(args.output, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sorted(output.iterdir()):
                contents = path.read_bytes()
                if path.suffix == '.json':
                    data = json.loads(contents)
                    data['source_filename'] = 'test/core/' + path.stem + '.wast'
                    contents = (json.dumps(data, separators=(',', ':')) + '\n').encode()
                info = zipfile.ZipInfo(path.name, (2020, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, contents)
            info = zipfile.ZipInfo('LICENSE', (2020, 1, 1, 0, 0, 0))
            archive.writestr(info, (args.spec / 'LICENSE').read_bytes())


if __name__ == '__main__':
    main()
