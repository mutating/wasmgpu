"""Embedded WASI configuration. Files and their descriptors live on the GPU.

Clocks, random data, streams and filesystem operations are emulated in the
compute shader. No guest operation calls an operating-system service.
"""
from __future__ import annotations

from pathlib import PurePosixPath
from typing import Iterable, Mapping

from .binary import I32, I64, BinaryModule
from .errors import UnsupportedFeatureError, ValidationError
from .types import Blob

_SIGNATURES = {
    'args_get': ('ii', 'i'), 'args_sizes_get': ('ii', 'i'),
    'environ_get': ('ii', 'i'), 'environ_sizes_get': ('ii', 'i'),
    'clock_res_get': ('ii', 'i'), 'clock_time_get': ('iIi', 'i'),
    'fd_advise': ('iIIi', 'i'), 'fd_allocate': ('iII', 'i'), 'fd_close': ('i', 'i'),
    'fd_datasync': ('i', 'i'), 'fd_fdstat_get': ('ii', 'i'), 'fd_fdstat_set_flags': ('ii', 'i'),
    'fd_fdstat_set_rights': ('iII', 'i'), 'fd_filestat_get': ('ii', 'i'),
    'fd_filestat_set_size': ('iI', 'i'), 'fd_filestat_set_times': ('iIIi', 'i'),
    'fd_pread': ('iiiIi', 'i'), 'fd_prestat_get': ('ii', 'i'), 'fd_prestat_dir_name': ('iii', 'i'),
    'fd_pwrite': ('iiiIi', 'i'), 'fd_read': ('iiii', 'i'), 'fd_readdir': ('iiiIi', 'i'),
    'fd_renumber': ('ii', 'i'), 'fd_seek': ('iIii', 'i'), 'fd_sync': ('i', 'i'),
    'fd_tell': ('ii', 'i'), 'fd_write': ('iiii', 'i'),
    'path_create_directory': ('iii', 'i'), 'path_filestat_get': ('iiiii', 'i'),
    'path_filestat_set_times': ('iiiiIIi', 'i'), 'path_link': ('iiiiiii', 'i'),
    'path_open': ('iiiiiIIii', 'i'), 'path_readlink': ('iiiiii', 'i'),
    'path_remove_directory': ('iii', 'i'), 'path_rename': ('iiiiii', 'i'),
    'path_symlink': ('iiiii', 'i'), 'path_unlink_file': ('iii', 'i'),
    'poll_oneoff': ('iiii', 'i'), 'proc_exit': ('i', ''), 'proc_raise': ('i', 'i'),
    'random_get': ('ii', 'i'), 'sched_yield': ('', 'i'),
    'sock_accept': ('iii', 'i'), 'sock_recv': ('iiiiii', 'i'),
    'sock_send': ('iiiii', 'i'), 'sock_shutdown': ('ii', 'i'),
}

WASI_IDS = {name: index + 1 for index, name in enumerate(_SIGNATURES)}


def normalize_path(path: str) -> str:
    if not isinstance(path, str) or '\0' in path:
        raise ValueError('embedded file names must be strings without NUL')
    parts: list[str] = []
    for part in path.split('/'):
        if part in ('', '.'):
            continue
        if part == '..':
            if not parts:
                raise ValueError('path escapes the instance filesystem')
            parts.pop()
        else:
            parts.append(part)
    result = '/'.join(parts)
    if len(result.encode()) > 255:
        raise ValueError('embedded paths are limited to 255 UTF-8 bytes')
    return result


class Wasi:
    """WASI Preview 1 with a private filesystem resident in GPU memory.

    files maps guest paths to bytes; it never refers to host paths. Each instance
    receives its own copy. storage_size is the total file-data arena capacity.
    The root directory is preopened as descriptor 3, named '.'.
    """

    def __init__(self, *, files: Mapping[str, Blob] | None = None, args: Iterable[str] = (),  # noqa: PLR0913 - Guest environment and explicit quotas.
                 env: Mapping[str, str] | None = None, stdin: Blob = b'', storage_size: int = 65536,
                 max_files: int = 32, max_fds: int = 64, seed: int | bytes = 1, clock_epoch_ns: int = 0,
                 clock_resolution_ns: int = 1) -> None:
        self.files: dict[str, bytes] = {}
        for name, content in (files or {}).items():
            path = normalize_path(name)
            if not path or path in self.files:
                raise ValueError('empty or duplicate embedded file name')
            if not isinstance(content, (bytes, bytearray, memoryview)):
                raise TypeError('embedded file contents must be bytes, not host paths')
            self.files[path] = bytes(content)
        self.args = tuple(str(value) for value in args)
        self.env = {str(key): str(value) for key, value in (env or {}).items()}
        self.stdin = bytes(stdin)
        if any('\0' in arg for arg in self.args) or any('\0' in key + value or '=' in key for key, value in self.env.items()):
            raise ValueError('WASI arguments/environment contain invalid characters')
        for name, value in [('storage_size', storage_size), ('max_files', max_files), ('max_fds', max_fds)]:
            if isinstance(value, bool) or not isinstance(value, int) or not 4 <= value <= 0x7fffffff:
                raise ValueError(f'{name} must be an integer between 4 and 2147483647')
        self.storage_size, self.max_files, self.max_fds = storage_size, max_files, max_fds
        if isinstance(seed, bytes):
            if len(seed) != 32:
                raise ValueError('a byte seed must contain exactly 32 bytes')
            self.random_key = seed
        else:
            if isinstance(seed, bool) or not isinstance(seed, int) or not 0 < seed <= 0xFFFFFFFF:
                raise ValueError('seed must be a nonzero u32 or 32 bytes')
            self.random_key = seed.to_bytes(32, 'little')
        if isinstance(clock_epoch_ns, bool) or not isinstance(clock_epoch_ns, int) or not 0 <= clock_epoch_ns < 1 << 64:
            raise ValueError('clock_epoch_ns must fit u64')
        if isinstance(clock_resolution_ns, bool) or not isinstance(clock_resolution_ns, int) or not 0 < clock_resolution_ns < 1 << 32:
            raise ValueError('clock_resolution_ns must be a positive u32')
        self.seed, self.clock_epoch_ns, self.clock_resolution_ns = seed, clock_epoch_ns, clock_resolution_ns

    @staticmethod
    def validate_imports(module: BinaryModule) -> None:
        for fn in module.functions:
            if fn.imported is None:
                continue
            namespace, name = fn.imported
            if namespace != 'wasi_snapshot_preview1' or name not in _SIGNATURES:
                raise UnsupportedFeatureError(f'unsupported import {namespace}.{name}')
            args, results = _SIGNATURES[name]
            expected = ([I32 if char == 'i' else I64 for char in args], [I32 for _ in results])
            if module.types[fn.type_index] != expected:
                raise ValidationError(f'incorrect WASI signature for {name}')

    def _initial(self, extra_files: Mapping[str, bytes]) -> list[tuple[int, str, bytes]]:
        files = dict(extra_files)
        files.update(self.files)
        directories = {''}
        for path in files:
            for parent in PurePosixPath(path).parents:
                if str(parent) != '.':
                    directories.add(str(parent))
        if directories.intersection(files):
            raise ValueError('an embedded path is both a file and a directory')
        entries = [(1, '', self.stdin), (1, '', b''), (1, '', b''), (2, '', b'')]
        entries.extend((2, name, b'') for name in sorted(directories - {''}))
        entries.extend((1, name, content) for name, content in sorted(files.items()))
        if len(entries) > self.max_files:
            raise ValueError('max_files is too small for the embedded files and directories')
        return entries
