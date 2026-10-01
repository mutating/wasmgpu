# wasmgpu

Run independent WebAssembly instances on a hardware GPU from Python 3.8+.

```python
import wasmgpu

module = wasmgpu.Module("worker.wasm")
with module.spawn(100_000) as instances:
    results = instances.call("process", inputs)
```

The runtime is implemented in this repository:

```text
Python → wgpu-py → WGSL bytecode interpreter → wgpu-native → Metal / Vulkan / DX12
```

Python parses and validates the module, uploads initial state, dispatches work,
and reads results. **All guest instructions and WASI services run on the GPU.**
There is no CPU interpreter, CPU fallback, host WASI service loop, or dependency
on Wasmtime in the installed package. Software GPU adapters are rejected.

This is an experimental engine with the supported profile below, **not a
complete implementation of every WebAssembly proposal or a security sandbox**.
Apple M4 / Metal has been tested. Vulkan and DX12 use the same shader but have
not yet been verified on physical hardware in this project.

## Install

Use a virtual environment for installation, development and tests:

```sh
python3 -m venv venv
venv/bin/python -m pip install -e .
venv/bin/python -m pip install -r requirements_dev.txt
```

On Windows, use `venv\Scripts\python.exe` instead. Dependency markers select
wgpu-py 0.18.0 for Python 3.8, 0.24.0 for 3.9, 0.31.0 for 3.10, and 0.32.0 for
3.11+. Actual Metal execution was verified with Python 3.8, 3.9, 3.10 and 3.11. The other
Python/backend combinations still require hardware testing.

## Calls and state

`Module` accepts a path or binary WASM bytes. It does not compile Rust, C or WAT.
`module.exports` lists export names and kinds. Instantiate once, then reuse the
instances: linear memory, globals, tables, files and descriptors persist between
calls. Instances have independent state, including across internal GPU batches.

```python
with module.spawn(3) as instances:
    a = instances.call("one_argument", [10, 20, 30])
    b = instances.call("two_arguments", [(1, 2), (3, 4), (5, 6)])
    c = instances.call("no_arguments")
```

There must be exactly one input row per instance. Input validation completes
before any instance executes. Results are a list in instance order: scalars for
one result, tuples for multiple results, and `None` for no results. Integers
return as signed i32/i64; integer inputs accept signed or unsigned bit patterns.
Floating inputs/outputs are Python floats; Python may quiet signalling f32 NaNs
at this API boundary. Only null references can be supplied from Python.

```python
instances.write_memory(1024, b"payload", instance=0)
content = instances.read_memory(1024, 7, instance=0)
```

These are explicit data transfers. Reading or writing memory does not execute
guest code on the CPU. A core WASM start function executes during `spawn`.
For WASI reactor modules, call an exported `_initialize` once if the compiler
requires it; for command modules, explicitly call `_start`.

`Trap` provides `traps` (instance index → reason) and `results` (including
successful peers). Completed effects before a trap persist; calls are not
transactions. A later call can reuse the instances. `proc_exit` raises a trap
and records the per-instance code in `instances.exit_codes`. Explicitly close
instances, preferably with a context manager, to release GPU buffers.

## Embedded WASI Preview 1

Files are **byte contents embedded into each instance's GPU filesystem**.
There are no host directory mounts, host file operations or network access.

```python
module = wasmgpu.Module("worker.wasm", files={"data/input.txt": b"1.25\n2.5\n"})
wasi = wasmgpu.Wasi(
    args=["worker", "data/input.txt"],
    env={"MODE": "batch"},
    stdin=b"input stream\n",
    storage_size=256 * 1024,
    max_files=64,
    max_fds=64,
    seed=123,
)
with module.spawn(8, wasi=wasi, memory_pages=32, stack_size=4096) as instances:
    instances.call("process")
    output_file = instances.read_file("result.txt", instance=0)
    stdout = instances.stdout  # list of captured bytes, one per instance
    stderr = instances.stderr
```

`Wasi(files=...)` overrides same-named `Module(files=...)` entries. Configuration
and contents are copied at instantiation. Root is preopened at fd 3 as `.`;
fd 0/1/2 are emulated stdin/stdout/stderr. Paths use `/`, are limited to 255 UTF-8
bytes after resolution, and cannot escape their directory capability.
`max_files` includes directories, symlinks and four reserved entries;
`storage_size` includes stdin, stdout, stderr and all file contents.

The shader implements file creation, reads/writes and positioned I/O, seeks,
truncation/allocation, descriptor rights, metadata/timestamps, directory
enumeration, rename, hardlinks, symlinks and unlink. Deleted storage is reclaimed;
open descriptors and hardlinks keep their inode alive. `read_file` is an
inspection API for a regular file's stored path; guest code resolves symlinks.

Other operating-system services have explicit virtual semantics:

- Arguments/environment come from the embedded configuration.
- All clocks use a per-instance virtual counter. It starts at `clock_epoch_ns`
  (default 0) and advances by `clock_resolution_ns` (default 1) per interpreter
  instruction. It never reads the host clock.
- `poll_oneoff` reports ready virtual descriptors or advances virtual time to
  the earliest clock deadline, without sleeping on the host.
- `random_get` uses ChaCha20 on the GPU. The seed is a nonzero u32 or a 32-byte
  key; an instance index supplies its nonce. Streams are deterministic and
  independent of batch size. The default seed is public and provides **no
  unpredictable system entropy**.
- `sched_yield` is a no-op in the isolated instance model. Signals terminate the
  affected invocation; no host process is signalled.
- No virtual sockets are provisioned. Socket imports return `BADF` for invalid
  descriptors and `NOTSOCK` for existing non-socket descriptors. They never open
  host sockets. `sync`/`datasync` operate on the in-memory filesystem only.

All 46 Preview 1 imports have signature validation and GPU dispatch. This is an
emulated environment, not a promise of an ordinary operating system or complete
WASI conformance. Preview 2/3 and arbitrary host imports are unsupported.

## Supported WASM profile and limits

Supported: all scalar MVP numeric instructions; i32/i64; software IEEE-754 f32
and f64 including subnormals, signed zero and rounding; direct/indirect recursive
calls; blocks/loops/branches; multi-value; mutable globals; memory32; multiple
funcref/externref tables; reference instructions; sign extension; saturating
conversions; bulk memory/table operations and passive/declarative segments.

Currently unsupported: SIMD, threads/shared memory, exceptions, tail calls, GC,
typed function references, memory64, multiple linear memories, imported
memories/tables/globals, and module linking. Unsupported features fail explicitly.
They never trigger execution through a CPU engine. Tables and linear memory have
fixed GPU growth budgets; `grow` returns -1 when the reserved capacity is reached.
The current memory32 addressing implementation caps memory at 65,535 pages.

`spawn` exposes resource controls:

| Option | Default | Meaning |
|---|---:|---|
| `memory_pages` | up to 16, at least declared minimum | Reserved 64 KiB pages per instance, capped by the module maximum |
| `table_elements` | up to 256, at least each declared minimum | Growth capacity per table, capped by each declared maximum |
| `stack_size` | 256 | 64-bit value slots per instance, including locals |
| `call_depth` | 64 | Nested call frames per instance |
| `fuel` | 10,000,000 | Invocation budget; bulk work also consumes fuel |
| `quantum` | 4096 | Interpreter instructions per dispatch before resumption |
| `batch_size` | device-derived | Instances per dispatch/buffer group |
| `max_resident_bytes` | 512 MiB | Total resident buffer allocation budget |

All persistent instance state stays on the GPU. Internal batching respects device
buffer limits; it does not page state to a CPU runtime. Thus 100,000 tiny workers
are practical, but 100,000 workers with 1 MiB private memory require about 100 GiB
before stacks/files. Excessive allocations fail before allocation. `resident_bytes`
and `adapter_info` expose the allocation estimate and selected hardware.

A dispatch quantum is not a real-time deadline. Large individual bulk operations
and WASI operations can take longer than a scalar instruction. Tune resource
budgets for trusted workloads; this runtime is not suitable for hostile modules.

## Verification and benchmarks

```sh
venv/bin/python -m pytest tests -q           # hardware GPU required; absence fails
venv/bin/python -m pytest tests -m 'not gpu' # parser/configuration/failure tests only
WASMGPU_COVERAGE_BRANCH=true venv/bin/python -m coverage run -m pytest tests -q
venv/bin/python -m coverage combine
venv/bin/python -m coverage report -m
venv/bin/python -m tests.benchmark           # run alone, without concurrent GPU jobs
venv/bin/python -m build
```

All 358 tests passed on Apple M4 / Metal with Python 3.11, including 40 official
spec suites plus API, numeric, WASI and compiled-code cases. This run covered
100% of Python statements and branches; that percentage does not measure WGSL.
Python 3.8/3.9/3.10 each passed 315 cases with their pinned GPU backend; Python 3.11
also runs every official suite. Python 3.15rc2 passed the checks that require no GPU.
Tests compare scalar operations with Wasmtime, exercise 100,003 concurrent
instances, and run actual compiled C and Rust fixtures with allocation, internal
calls, f64, libc/Rust formatting and embedded files. Forty unmodified official
WebAssembly 2.0 core suites contribute over 20,000 module/action/assertion commands,
including malformed modules and precise NaN bit checks. See
[fixture provenance](tests/fixtures/README.md). They are a selected subset, not the
complete spec suite; Python coverage does not measure WGSL instruction coverage.

[Benchmark script](tests/benchmark.py) records medians after warmup, hardware and
versions in the local `tests/benchmark-results.json` file. This generated report
is ignored by Git and excluded from distributions; use `--output` to choose a
different path. GPU timing includes Python
packing, transfers, every dispatch/synchronization and result decoding. Two CPU
baselines use Wasmtime: one Python call per worker, and a single WASM loop that
writes every output followed by reading those outputs into a Python list. Module
compilation/instantiation is excluded from call timings; GPU spawn is separate.

Compare both CPU baselines when interpreting results: reducing Python call
overhead alone does not establish an acceleration of WASM computation over CPU
JIT execution. Performance depends on the workload, instance count and hardware.

CI runs only Python tests that require no GPU, static checks, and package builds,
as configured for ordinary hosted runners. The Linux / Python 3.11 job uploads its
coverage report to Coveralls with the `python-only` flag and saves it as a GitHub
artifact. This report does not claim GPU tests have run. Full conformance and
performance tests must be run locally with a hardware GPU.
