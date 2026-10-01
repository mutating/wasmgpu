from __future__ import annotations

import struct

import pytest

import wasmgpu
from wasmgpu.wasi import _SIGNATURES

from .conftest import binary

pytestmark = pytest.mark.gpu


def wasi_module():
    imports = []
    exports = []
    for index, (name, (args, results)) in enumerate(_SIGNATURES.items()):
        params = ' '.join('i32' if ty == 'i' else 'i64' for ty in args)
        signature = (f'(param {params})' if params else '') + ('(result i32)' if results else '')
        imports.append(f'(import "wasi_snapshot_preview1" "{name}" (func {signature}))')
        exports.append(f'(export "{name}" (func {index}))')
    return wasmgpu.Module(binary('(module ' + ''.join(imports + exports) + '(memory 1))'))


@pytest.fixture
def guest():
    with wasi_module().spawn(1, wasi=wasmgpu.Wasi(files={'dir/input': b'abcdef', 'outside': b'xyz'})) as instances:
        yield instances


def invoke(guest, name, *args):
    return guest.call(name, [args])[0]


def text(guest, value, offset=0):
    blob = value.encode()
    guest.write_memory(offset, blob)
    return offset, len(blob)


def u32(guest, offset):
    return int.from_bytes(guest.read_memory(offset, 4), 'little')


def open_file(guest, name='dir/input', *, rights=(1 << 30) - 1, flags=0, oflags=0, fd=3, follow=1):  # noqa: PLR0913 - Mirrors the WASI path_open arguments.
    pointer, size = text(guest, name)
    error = invoke(guest, 'path_open', fd, follow, pointer, size, oflags, rights, rights, flags, 512)
    assert error == 0
    return u32(guest, 512)


def test_descriptors_positioned_io_and_flags(guest):
    fd = open_file(guest)
    guest.write_memory(128, struct.pack('<II', 256, 3))
    assert invoke(guest, 'fd_pread', fd, 128, 1, 2, 520) == 0
    assert guest.read_memory(256, 3) == b'cde'
    assert invoke(guest, 'fd_tell', fd, 528) == 0
    assert u32(guest, 528) == 0
    guest.write_memory(256, b'XYZ')
    assert invoke(guest, 'fd_pwrite', fd, 128, 1, 1, 520) == 0
    assert guest.read_file('dir/input') == b'aXYZef'
    assert invoke(guest, 'fd_fdstat_set_flags', fd, 1) == 0
    assert invoke(guest, 'fd_write', fd, 128, 1, 520) == 0
    assert guest.read_file('dir/input') == b'aXYZefXYZ'
    assert invoke(guest, 'fd_seek', fd, -3, 2, 528) == 0
    assert u32(guest, 528) == 6
    assert invoke(guest, 'fd_renumber', fd, 12) == 0
    assert invoke(guest, 'fd_tell', fd, 528) == 8
    assert invoke(guest, 'fd_close', 12) == 0
    assert invoke(guest, 'fd_close', 12) == 8


def test_rights_can_only_be_reduced(guest):
    fd = open_file(guest)
    assert invoke(guest, 'fd_fdstat_set_rights', fd, 2, 0) == 0
    assert invoke(guest, 'fd_fdstat_set_rights', fd, 66, 0) == 76
    assert invoke(guest, 'fd_write', fd, 0, 0, 512) == 76
    assert invoke(guest, 'fd_sync', fd) == 76
    assert invoke(guest, 'fd_fdstat_get', fd, 128) == 0
    assert u32(guest, 136) == 2


def test_directory_capability_cannot_escape_to_parent(guest):
    fd = open_file(guest, 'dir', oflags=2)
    p, n = text(guest, '../outside')
    assert invoke(guest, 'path_open', fd, 1, p, n, 0, 2, 0, 0, 512) == 76
    p, n = text(guest, 'missing/../outside')
    assert invoke(guest, 'path_open', 3, 1, p, n, 0, 2, 0, 0, 512) == 44
    p, n = text(guest, 'outside/../dir')
    assert invoke(guest, 'path_open', 3, 1, p, n, 0, 2, 0, 0, 512) == 54


def test_symlinks_follow_readlink_and_cycle_detection(guest):
    old, size = text(guest, 'input')
    new, length = text(guest, 'dir/link', 32)
    assert invoke(guest, 'path_symlink', old, size, 3, new, length) == 0
    assert invoke(guest, 'path_readlink', 3, new, length, 128, 3, 512) == 0
    assert guest.read_memory(128, 3) == b'inp'
    assert u32(guest, 512) == 3
    fd = open_file(guest, 'dir/link')
    assert invoke(guest, 'fd_filestat_get', fd, 128) == 0
    assert u32(guest, 160) == 6
    p, n = text(guest, 'dir/link')
    assert invoke(guest, 'path_open', 3, 0, p, n, 0, 2, 0, 0, 512) == 32
    assert invoke(guest, 'path_filestat_get', 3, 0, p, n, 128) == 0
    assert guest.read_memory(144, 1) == b'\x07'
    p, n = text(guest, 'loop')
    assert invoke(guest, 'path_symlink', p, n, 3, p, n) == 0
    assert invoke(guest, 'path_open', 3, 1, p, n, 0, 2, 0, 0, 512) == 32


def test_stat_allocate_truncate_timestamps_and_advice(guest):
    fd = open_file(guest)
    assert invoke(guest, 'fd_allocate', fd, 10, 5) == 0
    assert guest.read_file('dir/input') == b'abcdef' + b'\0' * 9
    assert invoke(guest, 'fd_filestat_set_size', fd, 2) == 0
    assert guest.read_file('dir/input') == b'ab'
    assert invoke(guest, 'fd_filestat_set_times', fd, 123, 456, 5) == 0
    assert invoke(guest, 'fd_filestat_get', fd, 128) == 0
    assert u32(guest, 168) == 123
    assert u32(guest, 176) == 456
    assert invoke(guest, 'fd_filestat_set_times', fd, 123, 456, 3) == 28
    assert invoke(guest, 'fd_advise', fd, 0, 2, 0) == 0
    assert invoke(guest, 'fd_advise', fd, 0, 2, 8) == 28
    assert invoke(guest, 'fd_datasync', fd) == 0
    assert invoke(guest, 'fd_sync', fd) == 0


def test_hardlink_rename_unlink_with_open_descriptor(guest):
    fd = open_file(guest)
    old, length = text(guest, 'dir/input')
    new, size = text(guest, 'dir/linked', 32)
    assert invoke(guest, 'path_link', 3, 1, old, length, 3, new, size) == 0
    assert invoke(guest, 'fd_filestat_get', fd, 128) == 0
    assert u32(guest, 152) == 2
    assert invoke(guest, 'path_unlink_file', 3, old, length) == 0
    assert guest.read_file('dir/linked') == b'abcdef'
    assert invoke(guest, 'fd_filestat_set_size', fd, 2) == 0
    assert guest.read_file('dir/linked') == b'ab'
    assert invoke(guest, 'path_rename', 3, new, size, 3, old, length) == 0
    assert guest.read_file('dir/input') == b'ab'
    with pytest.raises(FileNotFoundError):
        guest.read_file('dir/linked')


def test_readdir_cookies_and_partial_buffer(guest):
    assert invoke(guest, 'fd_readdir', 3, 128, 256, 0, 512) == 0
    count = u32(guest, 512)
    raw = guest.read_memory(128, count)
    names = []
    cursor = 0
    while cursor < len(raw):
        cookie, _, size, _ = struct.unpack_from('<QQIB', raw, cursor)
        names.append(raw[cursor + 24:cursor + 24 + size])
        cursor += 24 + size
    assert set(names) == {b'dir', b'outside'}
    assert invoke(guest, 'fd_readdir', 3, 128, 256, cookie, 512) == 0
    assert u32(guest, 512) == 0
    assert invoke(guest, 'fd_readdir', 3, 128, 7, 0, 512) == 0
    assert u32(guest, 512) == 7


def test_poll_clock_advances_virtual_time_without_sleep(guest):
    subscription = bytearray(48)
    struct.pack_into('<Q', subscription, 0, 123)
    struct.pack_into('<I', subscription, 16, 1)
    struct.pack_into('<Q', subscription, 24, 10**12)
    guest.write_memory(128, subscription)
    assert invoke(guest, 'poll_oneoff', 128, 256, 1, 512) == 0
    assert u32(guest, 512) == 1
    assert guest.read_memory(256, 11) == (123).to_bytes(8, 'little') + b'\0\0\0'
    assert invoke(guest, 'clock_time_get', 1, 0, 520) == 0
    assert int.from_bytes(guest.read_memory(520, 8), 'little') >= 10**12
    subscription[8] = 1
    struct.pack_into('<I', subscription, 16, 99)
    guest.write_memory(128, subscription)
    assert invoke(guest, 'poll_oneoff', 128, 256, 1, 512) == 0
    assert guest.read_memory(264, 3) == b'\x08\0\x01'


@pytest.mark.parametrize('name', ['sock_accept', 'sock_recv', 'sock_send', 'sock_shutdown'])
def test_socket_imports_report_absent_virtual_socket(guest, name):
    args = [0] * len(_SIGNATURES[name][0])
    args[0] = 99
    assert invoke(guest, name, *args) == 8
    args[0] = 0
    assert invoke(guest, name, *args) == 57


def test_directory_rename_preserves_children_and_open_directory(guest):
    fd = open_file(guest, 'dir', oflags=2)
    old, length = text(guest, 'dir')
    new, size = text(guest, 'renamed', 32)
    assert invoke(guest, 'path_rename', 3, old, length, 3, new, size) == 0
    assert guest.read_file('renamed/input') == b'abcdef'
    child = open_file(guest, 'input', fd=fd)
    assert invoke(guest, 'fd_filestat_get', child, 128) == 0
    old, length = text(guest, 'renamed')
    new, size = text(guest, 'renamed/child', 32)
    assert invoke(guest, 'path_rename', 3, old, length, 3, new, size) == 28
    assert invoke(guest, 'path_remove_directory', 3, old, length) == 55


def test_random_known_chacha20_block_and_independent_streams():
    # RFC 8439, section 2.4.2's algorithm with all-zero key/nonce/counter.
    expected = bytes.fromhex(
        '76b8e0ada0f13d90405d6ae55386bd28bdd219b8a08ded1aa836efcc8b770dc7'
        'da41597c5157488d7724e03fb8d84a376a43b8f41518a11cc387b669b2ee6586',
    )
    with wasi_module().spawn(2, wasi=wasmgpu.Wasi(seed=bytes(32)), batch_size=1) as guest:
        assert guest.call('random_get', [(0, 17), (0, 17)]) == [0, 0]
        assert guest.call('random_get', [(17, 47), (17, 47)]) == [0, 0]
        assert guest.read_memory(0, 64) == expected
        assert guest.read_memory(0, 64, instance=1) != expected


def test_deleted_inode_and_storage_are_reclaimed():
    with wasi_module().spawn(1, wasi=wasmgpu.Wasi(max_files=5, storage_size=64)) as guest:
        for _ in range(12):
            fd = open_file(guest, 'file', oflags=1)
            assert invoke(guest, 'fd_filestat_set_size', fd, 64) == 0
            assert invoke(guest, 'fd_close', fd) == 0
            p, n = text(guest, 'file')
            assert invoke(guest, 'path_unlink_file', 3, p, n) == 0


def test_growing_early_file_preserves_later_files():
    config = wasmgpu.Wasi(files={'a': b'abcd', 'b': b'12345678'}, storage_size=24)
    with wasi_module().spawn(1, wasi=config) as guest:
        fd = open_file(guest, 'a')
        assert invoke(guest, 'fd_filestat_set_size', fd, 16) == 0
        assert guest.read_file('a') == b'abcd' + b'\0' * 12
        assert guest.read_file('b') == b'12345678'
        assert invoke(guest, 'fd_filestat_set_size', fd, 17) == 51
        assert guest.read_file('b') == b'12345678'


def test_environment_sizes_preopen_and_clock_resolution(guest):
    assert invoke(guest, 'environ_sizes_get', 512, 516) == 0
    assert u32(guest, 512) == 0
    assert u32(guest, 516) == 0
    assert invoke(guest, 'environ_sizes_get', 65535, 516) == 21
    assert invoke(guest, 'fd_prestat_get', 3, 128) == 0
    assert u32(guest, 128) == 0
    assert u32(guest, 132) == 1
    assert invoke(guest, 'fd_prestat_dir_name', 3, 256, 1) == 0
    assert guest.read_memory(256, 1) == b'.'
    assert invoke(guest, 'fd_prestat_dir_name', 3, 256, 0) == 37
    assert invoke(guest, 'clock_res_get', 1, 520) == 0
    assert u32(guest, 520) == 1
    assert invoke(guest, 'clock_res_get', 99, 520) == 28
    assert invoke(guest, 'fd_filestat_get', 1, 128) == 0
    assert guest.read_memory(144, 1) == b'\x02'
    assert u32(guest, 152) == 1


def test_directory_create_remove_and_path_timestamps(guest):
    pointer, size = text(guest, 'newdir')
    assert invoke(guest, 'path_create_directory', 3, pointer, size) == 0
    assert invoke(guest, 'path_create_directory', 3, pointer, size) == 20
    assert invoke(guest, 'path_filestat_set_times', 3, 0, pointer, size, 111, 222, 5) == 0
    assert invoke(guest, 'path_filestat_get', 3, 0, pointer, size, 128) == 0
    assert u32(guest, 168) == 111
    assert u32(guest, 176) == 222
    assert invoke(guest, 'path_filestat_set_times', 3, 0, pointer, size, 0, 0, 10) == 0
    assert invoke(guest, 'path_remove_directory', 3, pointer, size) == 0
    assert invoke(guest, 'path_filestat_get', 3, 0, pointer, size, 128) == 44


def test_virtual_yield_and_signals(guest):
    assert invoke(guest, 'sched_yield') == 0
    assert invoke(guest, 'proc_raise', 0) == 0
    assert invoke(guest, 'proc_raise', 31) == 28
    with pytest.raises(wasmgpu.Trap, match='proc_raise'):
        invoke(guest, 'proc_raise', 15)
