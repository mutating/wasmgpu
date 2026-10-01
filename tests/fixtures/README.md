# Test fixture provenance

`worker.wasm` is compiled from the adjacent `worker.c` with WASI SDK 34
(Clang 23.1). `worker-rust.wasm` is compiled from `worker.rs` with Rust 1.94.1,
its matching `wasm32-wasip1` standard library, and the same SDK linker. They are
checked in so tests do not download or require a C/Rust toolchain.

```sh
"$WASI_SDK/bin/clang" -O1 -mexec-model=reactor \
  -Wl,--export=process -Wl,--export=file_process \
  -Wl,--initial-memory=262144 -Wl,--max-memory=4194304 \
  tests/fixtures/worker.c -o tests/fixtures/worker.wasm

rustc --target wasm32-wasip1 --crate-type cdylib -C opt-level=1 -C panic=abort \
  -C linker="$WASI_SDK/bin/wasm-ld" \
  tests/fixtures/worker.rs -o tests/fixtures/worker-rust.wasm
```

When the matching Rust standard library is installed separately, supply its
prefix with `--sysroot`. The C reactor exports `_initialize`; the Rust cdylib
exports the two functions directly. Tests instantiate independent Wasmtime
oracles and compare returned values and file contents. OS files are used only
by Wasmtime's test oracle; the implementation under `wasmgpu/` embeds bytes in GPU
memory and does not use those OS files.

`core-spec.zip` contains unmodified binary modules/actions/assertions generated
from 40 scalar/control/memory suites in the official
[WebAssembly spec v2.0.0](https://github.com/WebAssembly/spec/tree/v2.0.0/test/core),
commit `05ca4182176763112561ae20153975c12bd689e4`, with
[WABT 1.0.42](https://github.com/WebAssembly/wabt/releases/tag/1.0.42).
The upstream Apache-2.0 license is included inside the archive. JSON filenames
are normalized to relative source paths; assertion content is unchanged. The
complete list is in `tests/build_fixtures.py`. This selection includes 674 valid
module declarations, 18,567 result assertions, 1,105 invalid-module assertions,
340 malformed-module assertions, 284 trap assertions, 4 exhaustion assertions
and 63 plain actions. It does not cover SIMD, host linking or all WASM proposals.

Rebuild the archive using a venv, an exact spec checkout, and WABT:

```sh
venv/bin/python -m tests.build_fixtures \
  --spec /path/to/WebAssembly-spec-v2.0.0 \
  --wast2json /path/to/wabt-1.0.42/bin/wast2json
```

The spec harness passes raw argument/result bits internally so that Python's
floating conversions cannot mask signalling-NaN behavior in the GPU interpreter.
It executes every command in the selected suites; a missing GPU or a failed
assertion fails the test. Larger stack, fuel and memory budgets allow official
recursion and memory-growth cases within the device's physical limits.

SHA-256 of the checked-in binaries:

```text
worker.wasm       aded4ccde3790c639d14e188bce196aff9e352ff68ffb8657c35e6885a3f588f
worker-rust.wasm  a0dd24fc0b0ca038000717fde8637d14e958783665b75ca36ab66affce7134e2
core-spec.zip     5d86ac725ea7681584bee48889987f62dbd681e130c87942bf2abd28e59d58c2
```
