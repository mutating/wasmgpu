"""Value types and the narrow wgpu interface used by the runtime.

wgpu releases supported on Python 3.8+ do not publish typing information. These
protocols describe the common API, including the pre-0.19 synchronous spelling.
"""
from __future__ import annotations

from array import array
from typing import Mapping, Protocol, Sequence, Tuple, Union

Blob = Union[bytes, bytearray, memoryview]
Scalar = Union[int, float, None]
Result = Union[Scalar, Tuple[Scalar, ...]]
Row = Union[Scalar, Sequence[Scalar]]
AdapterInfo = Mapping[str, Union[str, int]]


class Buffer(Protocol):
    size: int
    def destroy(self) -> None: ...


class Queue(Protocol):
    def write_buffer(self, buffer: Buffer, buffer_offset: int, data: Blob | array[int]) -> None: ...
    def read_buffer(self, buffer: Buffer, buffer_offset: int = 0, size: int | None = None) -> memoryview: ...
    def submit(self, command_buffers: Sequence[object]) -> None: ...


class Pipeline(Protocol):
    def get_bind_group_layout(self, index: int) -> object: ...


class ComputePass(Protocol):
    def set_pipeline(self, pipeline: Pipeline) -> None: ...
    def set_bind_group(self, index: int, bind_group: object) -> None: ...
    def dispatch_workgroups(self, workgroup_count_x: int) -> None: ...
    def end(self) -> None: ...


class Encoder(Protocol):
    def begin_compute_pass(self) -> ComputePass: ...
    def finish(self) -> object: ...


class Device(Protocol):
    queue: Queue
    limits: Mapping[str, int]

    def create_buffer(self, *, size: int, usage: int) -> Buffer: ...
    def create_buffer_with_data(self, *, data: Blob | array[int], usage: int) -> Buffer: ...
    def create_shader_module(self, *, label: str, code: str) -> object: ...
    def create_compute_pipeline(self, *, layout: str, compute: Mapping[str, object]) -> Pipeline: ...
    def create_bind_group(self, *, layout: object, entries: Sequence[Mapping[str, object]]) -> object: ...
    def create_command_encoder(self, *, label: str) -> Encoder: ...


class Adapter(Protocol):
    info: AdapterInfo
    limits: Mapping[str, int]

    def request_device(self, *, required_limits: Mapping[str, int]) -> Device: ...


class RequestDevice(Protocol):
    def __call__(self, *, required_limits: Mapping[str, int]) -> Device: ...


class RequestAdapter(Protocol):
    def __call__(self, *, power_preference: str) -> Adapter | None: ...


class BufferUsage(Protocol):
    COPY_DST: int
    COPY_SRC: int
    UNIFORM: int
    STORAGE: int


class GPU(Protocol):
    def request_adapter(self, *, power_preference: str) -> Adapter | None: ...


class Backend(Protocol):
    gpu: GPU
    BufferUsage: BufferUsage
