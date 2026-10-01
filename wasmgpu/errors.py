"""Public errors. GPU traps never cause execution to fall back to the CPU."""


from __future__ import annotations

from .types import Result


class WasmGPUError(Exception):
    """Base class for runtime errors."""


class ValidationError(WasmGPUError, ValueError):
    """Malformed or ill-typed WebAssembly."""


class UnsupportedFeatureError(WasmGPUError, NotImplementedError):
    """A valid feature outside this runtime's supported profile."""


class GPUUnavailableError(WasmGPUError):
    """No hardware GPU is available."""


class ResourceLimitError(WasmGPUError):
    """An explicit runtime or device resource limit was reached."""


class Trap(WasmGPUError):  # noqa: N818 - WebAssembly calls these traps.
    """One or more instances trapped; successful peers are in ``results``."""

    def __init__(self, traps: dict[int, str], results: list[Result]) -> None:
        self.traps = traps
        self.results = results
        super().__init__('; '.join(f'instance {index}: {reason}' for index, reason in traps.items()))
