// Per-instance filesystem and WASI Preview 1 services, executed on the GPU.
// Names, descriptors, inode metadata and file contents are all in heap storage.
var<private> path_raw: array<u32, 512>;
var<private> path_tail: array<u32, 512>;
var<private> path_text: array<u32, 256>;
var<private> path_length: u32;
fn arg(base: u32, index: u32) -> u32 { return get_value(base + index).x; }
fn fs_entry(index: u32) -> u32 { return config.fs_offset + 48u + index * 12u; }
fn fs_names() -> u32 { return config.fs_offset + 48u + config.fs_files * 12u; }
fn fs_descriptor(fd: u32) -> u32 { return fs_names() + config.fs_files * 64u + fd * 8u; }
fn fs_data() -> u32 { return fs_descriptor(config.fs_fds); }
fn heap_byte(base: u32, index: u32) -> u32 { return (read_heap(base + index / 4u) >> ((index & 3u) * 8u)) & 255u; }
fn set_heap_byte(base: u32, index: u32, value: u32) {
    let address = base + index / 4u; let shift = (index & 3u) * 8u;
    write_heap(address, (read_heap(address) & ~(255u << shift)) | ((value & 255u) << shift));
}
fn load32(address: u32) -> u32 {
    return read_byte(address) | (read_byte(address + 1u) << 8u) | (read_byte(address + 2u) << 16u) | (read_byte(address + 3u) << 24u);
}
fn store32(address: u32, value: u32) { for (var i = 0u; i < 4u; i += 1u) { write_byte(address + i, value >> (8u * i)); } }
fn store64(address: u32, value: vec2u) { store32(address, value.x); store32(address + 4u, value.y); }
fn fs_file(fd: u32) -> u32 {
    if fd >= config.fs_fds { return 0xffffffffu; }
    return read_heap(fs_descriptor(fd)) - 1u;
}
fn fs_inode(index: u32) -> u32 { return fs_entry(read_heap(fs_entry(index) + 5u)); }
fn fs_right(fd: u32, right: u32) -> bool { return (read_heap(fs_descriptor(fd) + 4u) & right) == right; }
fn fs_charge(amount: u32) -> bool {
    if !fuel_available(amount) { fail(8u); return false; }
    consume_fuel(amount); return true;
}
fn fs_open_entry(index: u32) -> bool {
    for (var fd = 0u; fd < config.fs_fds; fd += 1u) { if fs_file(fd) == index { return true; } }
    return false;
}
fn fs_live_inode(index: u32) -> bool {
    for (var i = 0u; i < config.fs_files; i += 1u) {
        let entry = fs_entry(i); let kind = read_heap(entry);
        if kind == 0u || read_heap(entry + 5u) != index { continue; }
        if kind < 4u || fs_open_entry(i) { return true; }
    }
    return false;
}
fn fs_collect() {
    for (var i = 4u; i < config.fs_files; i += 1u) {
        if read_heap(fs_entry(i)) != 4u || fs_open_entry(i) { continue; }
        if read_heap(fs_entry(i) + 5u) == i && fs_live_inode(i) { continue; }
        write_heap(fs_entry(i), 0u); write_heap(fs_entry(i) + 4u, 0u);
    }
}
fn fs_compact() -> bool {
    fs_collect();
    var scan = 0u; var packed = 0u;
    for (var step = 0u; step < config.fs_files; step += 1u) {
        var chosen = 0xffffffffu; var position = 0xffffffffu;
        for (var i = 0u; i < config.fs_files; i += 1u) {
            let entry = fs_entry(i); let start = read_heap(entry + 3u);
            if read_heap(entry) == 0u || read_heap(entry + 5u) != i || read_heap(entry + 4u) == 0u { continue; }
            if start >= scan && start < position { position = start; chosen = i; }
        }
        if chosen == 0xffffffffu { break; }
        let entry = fs_entry(chosen); let length = read_heap(entry + 2u);
        if !fs_charge(length) { return false; }
        for (var i = 0u; i < length; i += 1u) { set_heap_byte(fs_data(), packed + i, heap_byte(fs_data(), position + i)); }
        write_heap(entry + 3u, packed); write_heap(entry + 4u, length);
        scan = position + 1u; packed += length;
    }
    write_heap(config.fs_offset, packed); return true;
}
fn fs_resize(index: u32, length: u32) -> u32 {
    let entry = fs_inode(index); let old_size = read_heap(entry + 2u);
    var start = read_heap(entry + 3u); var capacity = read_heap(entry + 4u);
    if length > config.fs_bytes { return 51u; }
    if length > capacity {
        var cursor = read_heap(config.fs_offset);
        if length - capacity > config.fs_bytes - cursor {
            if !fs_compact() { return 29u; }
            start = read_heap(entry + 3u); capacity = read_heap(entry + 4u); cursor = read_heap(config.fs_offset);
            if length - capacity > config.fs_bytes - cursor { return 51u; }
        }
        let wanted = min(capacity + config.fs_bytes - cursor, max(length, min(config.fs_bytes, max(64u, capacity * 2u))));
        if capacity == 0u { start = cursor; }
        let tail = start + capacity; let extra = wanted - capacity;
        if !fs_charge(cursor - tail) { return 29u; }
        for (var i = cursor; i > tail; i -= 1u) { set_heap_byte(fs_data(), i - 1u + extra, heap_byte(fs_data(), i - 1u)); }
        for (var i = 0u; i < config.fs_files; i += 1u) {
            let other = fs_entry(i);
            if other == entry || read_heap(other) == 0u || read_heap(other + 5u) != i || read_heap(other + 4u) == 0u { continue; }
            if read_heap(other + 3u) >= tail { write_heap(other + 3u, read_heap(other + 3u) + extra); }
        }
        write_heap(entry + 3u, start); write_heap(entry + 4u, wanted);
        write_heap(config.fs_offset, cursor + extra);
    }
    if length > old_size {
        if !fs_charge(length - old_size) { return 29u; }
        for (var i = old_size; i < length; i += 1u) { set_heap_byte(fs_data(), start + i, 0u); }
    }
    write_heap(entry + 2u, length);
    let now = vec2u(read_heap(config.fs_offset + 1u), read_heap(config.fs_offset + 2u));
    write_heap(entry + 8u, now.x); write_heap(entry + 9u, now.y);
    write_heap(entry + 10u, now.x); write_heap(entry + 11u, now.y);
    return 0u;
}
fn fs_resolve(fd: u32, address: u32, size: u32, follow_final: bool) -> u32 {
    let directory = fs_file(fd);
    if directory == 0xffffffffu { return 8u; }
    if read_heap(fs_entry(directory)) != 2u { return 54u; }
    if !bounds(address, size) { return 21u; }
    if size == 0u { return 44u; }
    if size > 511u { return 37u; }
    if read_byte(address) == 47u { return 76u; }
    let floor = read_heap(fs_entry(directory) + 1u);
    path_length = floor;
    for (var j = 0u; j < floor; j += 1u) { path_text[j] = heap_byte(fs_names() + directory * 64u, j); }
    var raw_length = size;
    for (var j = 0u; j < size; j += 1u) {
        let byte = read_byte(address + j);
        if byte == 0u { return 28u; }
        path_raw[j] = byte;
    }
    var i = 0u; var links = 0u;
    while i < raw_length {
        if path_raw[i] == 47u { i += 1u; continue; }
        let begin = i;
        while i < raw_length && path_raw[i] != 47u { i += 1u; }
        let length = i - begin;
        if length == 1u && path_raw[begin] == 46u { continue; }
        if length == 2u && path_raw[begin] == 46u && path_raw[begin + 1u] == 46u {
            if path_length <= floor { return 76u; }
            while path_length > floor && path_text[path_length - 1u] != 47u { path_length -= 1u; }
            if path_length > floor { path_length -= 1u; }
            continue;
        }
        let parent_length = path_length;
        if path_length + length + u32(path_length != 0u) > 255u { return 37u; }
        if path_length != 0u { path_text[path_length] = 47u; path_length += 1u; }
        for (var j = 0u; j < length; j += 1u) { path_text[path_length] = path_raw[begin + j]; path_length += 1u; }
        let index = fs_lookup();
        let intermediate = i < raw_length;
        if index == 0xffffffffu { if intermediate { return 44u; } continue; }
        let kind = read_heap(fs_entry(index));
        if kind == 3u && (intermediate || follow_final) {
            links += 1u; if links > 40u { return 32u; }
            let entry = fs_inode(index); let link_size = read_heap(entry + 2u); let start = read_heap(entry + 3u);
            if link_size == 0u { return 44u; }
            if heap_byte(fs_data(), start) == 47u { return 76u; }
            let tail_size = raw_length - i;
            if link_size + tail_size > 511u { return 37u; }
            for (var j = 0u; j < tail_size; j += 1u) { path_tail[j] = path_raw[i + j]; }
            for (var j = 0u; j < link_size; j += 1u) { path_raw[j] = heap_byte(fs_data(), start + j); }
            for (var j = 0u; j < tail_size; j += 1u) { path_raw[link_size + j] = path_tail[j]; }
            raw_length = link_size + tail_size; i = 0u; path_length = parent_length;
        } else if intermediate && kind != 2u { return 54u; }
    }
    return 0u;
}
fn fs_path(fd: u32, address: u32, size: u32) -> u32 { return fs_resolve(fd, address, size, false); }
fn fs_lookup() -> u32 {
    if path_length == 0u { return 3u; }
    for (var index = 4u; index < config.fs_files; index += 1u) {
        let entry = fs_entry(index); let kind = read_heap(entry);
        if kind == 0u || kind >= 4u || read_heap(entry + 1u) != path_length { continue; }
        var equal = true;
        for (var i = 0u; i < path_length; i += 1u) { if heap_byte(fs_names() + index * 64u, i) != path_text[i] { equal = false; break; } }
        if equal { return index; }
    }
    return 0xffffffffu;
}
fn fs_parent_exists() -> bool {
    let original = path_length;
    while path_length > 0u && path_text[path_length - 1u] != 47u { path_length -= 1u; }
    if path_length > 0u { path_length -= 1u; }
    let parent = fs_lookup(); path_length = original;
    return parent != 0xffffffffu && read_heap(fs_entry(parent)) == 2u;
}
fn fs_save_name(index: u32) {
    write_heap(fs_entry(index) + 1u, path_length);
    for (var i = 0u; i < path_length; i += 1u) { set_heap_byte(fs_names() + index * 64u, i, path_text[i]); }
}
fn fs_create(kind: u32) -> u32 {
    fs_collect();
    for (var index = 4u; index < config.fs_files; index += 1u) {
        if read_heap(fs_entry(index)) == 0u {
            let entry = fs_entry(index);
            for (var j = 0u; j < 12u; j += 1u) { write_heap(entry + j, 0u); }
            write_heap(entry, kind); write_heap(entry + 5u, index); fs_save_name(index);
            return index;
        }
    }
    return 0xffffffffu;
}
fn fs_child(index: u32, parent: u32) -> bool {
    let kind = read_heap(fs_entry(index)); let length = read_heap(fs_entry(parent) + 1u);
    if kind == 0u || kind >= 4u || read_heap(fs_entry(index) + 1u) <= length { return false; }
    if heap_byte(fs_names() + index * 64u, length) != 47u { return false; }
    for (var i = 0u; i < length; i += 1u) {
        if heap_byte(fs_names() + index * 64u, i) != heap_byte(fs_names() + parent * 64u, i) { return false; }
    }
    return true;
}
fn fs_rename(index: u32, existing: u32) -> u32 {
    if existing == index { return 0u; }
    if index == 3u || existing == 3u { return 10u; }
    let directory = read_heap(fs_entry(index)) == 2u;
    let old_length = read_heap(fs_entry(index) + 1u);
    if directory && path_length > old_length && path_text[old_length] == 47u {
        var descendant = true;
        for (var i = 0u; i < old_length; i += 1u) { descendant = descendant && path_text[i] == heap_byte(fs_names() + index * 64u, i); }
        if descendant { return 28u; }
    }
    if existing != 0xffffffffu {
        let existing_directory = read_heap(fs_entry(existing)) == 2u;
        if directory && !existing_directory { return 54u; }
        if !directory && existing_directory { return 31u; }
        if fs_inode(existing) == fs_inode(index) { return 0u; }
        if existing_directory {
            for (var i = 4u; i < config.fs_files; i += 1u) { if fs_child(i, existing) { return 55u; } }
        }
    }
    if directory {
        for (var i = 4u; i < config.fs_files; i += 1u) {
            if fs_child(i, index) && read_heap(fs_entry(i) + 1u) - old_length + path_length > 255u { return 37u; }
        }
        for (var i = 4u; i < config.fs_files; i += 1u) {
            if !fs_child(i, index) { continue; }
            let suffix = read_heap(fs_entry(i) + 1u) - old_length;
            for (var j = 0u; j < suffix; j += 1u) { path_tail[j] = heap_byte(fs_names() + i * 64u, old_length + j); }
            for (var j = 0u; j < path_length; j += 1u) { set_heap_byte(fs_names() + i * 64u, j, path_text[j]); }
            for (var j = 0u; j < suffix; j += 1u) { set_heap_byte(fs_names() + i * 64u, path_length + j, path_tail[j]); }
            write_heap(fs_entry(i) + 1u, path_length + suffix);
        }
    }
    if existing != 0xffffffffu { write_heap(fs_entry(existing), 4u); }
    fs_save_name(index); return 0u;
}
fn fs_stat(index: u32, address: u32) {
    let entry = fs_inode(index); let kind = read_heap(fs_entry(index));
    for (var i = 0u; i < 64u; i += 1u) { write_byte(address + i, 0u); }
    store64(address, vec2u(1u, 0u)); store64(address + 8u, vec2u(read_heap(fs_entry(index) + 5u) + 1u, 0u));
    write_byte(address + 16u, select(select(select(4u, 3u, kind == 2u), 7u, kind == 3u), 2u, index < 3u));
    var links = 0u;
    for (var i = 3u; i < config.fs_files; i += 1u) { if read_heap(fs_entry(i)) > 0u && read_heap(fs_entry(i)) < 4u && fs_inode(i) == entry { links += 1u; } }
    store64(address + 24u, vec2u(select(links, 1u, index < 3u), 0u)); store64(address + 32u, vec2u(read_heap(entry + 2u), 0u));
    for (var i = 0u; i < 6u; i += 1u) { store32(address + 40u + i * 4u, read_heap(entry + 6u + i)); }
}
fn fs_io(syscall: u32, base: u32) -> u32 {
    let fd = arg(base, 0u); let index = fs_file(fd);
    if index == 0xffffffffu { return 8u; }
    if read_heap(fs_entry(index)) == 2u { return 31u; }
    let writing = syscall == WASI_FD_WRITE || syscall == WASI_FD_PWRITE;
    if !fs_right(fd, select(2u, 64u, writing)) { return 76u; }
    let positioned = syscall == WASI_FD_PREAD || syscall == WASI_FD_PWRITE;
    let vectors = arg(base, 1u); let count = arg(base, 2u); let result_ptr = arg(base, select(3u, 4u, positioned));
    if count > 0x1fffffffu || !bounds(vectors, count * 8u) || !bounds(result_ptr, 4u) { return 21u; }
    var total = 0u;
    for (var i = 0u; i < count; i += 1u) {
        let pointer = load32(vectors + i * 8u); let length = load32(vectors + i * 8u + 4u);
        if !bounds(pointer, length) { return 21u; }
        if total + length < total { return 28u; }
        total += length;
    }
    if !fs_charge(total) { return 29u; }
    let entry = fs_inode(index); let descriptor = fs_descriptor(fd);
    var position = vec2u(read_heap(descriptor + 1u), read_heap(descriptor + 2u));
    if positioned { position = get_value(base + 3u); }
    if writing && (read_heap(descriptor + 3u) & 1u) != 0u { position = vec2u(read_heap(entry + 2u), 0u); }
    if position.y != 0u { if writing { return 27u; } store32(result_ptr, 0u); return 0u; }
    if writing && total != 0u {
        if position.x > config.fs_bytes || total > config.fs_bytes - position.x { return 51u; }
        let error = fs_resize(index, max(read_heap(entry + 2u), position.x + total));
        if error != 0u { return error; }
    }
    var transferred = 0u;
    let start = read_heap(entry + 3u); let file_length = read_heap(entry + 2u);
    for (var i = 0u; i < count; i += 1u) {
        let pointer = load32(vectors + i * 8u); var size = load32(vectors + i * 8u + 4u);
        if !bounds(pointer, size) { return 21u; }
        if !writing { size = min(size, file_length - min(position.x, file_length)); }
        for (var j = 0u; j < size; j += 1u) {
            if writing { set_heap_byte(fs_data(), start + position.x + j, read_byte(pointer + j)); }
            else { write_byte(pointer + j, heap_byte(fs_data(), start + position.x + j)); }
        }
        position.x += size; transferred += size;
    }
    if !positioned { write_heap(descriptor + 1u, position.x); write_heap(descriptor + 2u, position.y); }
    store32(result_ptr, transferred); return 0u;
}
fn fs_poll(base: u32) -> u32 {
    let input_ptr = arg(base, 0u); let output_ptr = arg(base, 1u); let count = arg(base, 2u); let result_ptr = arg(base, 3u);
    if count == 0u || count > 0x05555555u { return 28u; }
    if !bounds(input_ptr, count * 48u) || !bounds(output_ptr, count * 32u) || !bounds(result_ptr, 4u) { return 21u; }
    if !fs_charge(count) { return 29u; }
    let now = vec2u(read_heap(config.fs_offset + 1u), read_heap(config.fs_offset + 2u));
    var earliest = vec2u(0xffffffffu); var immediate = false;
    for (var i = 0u; i < count; i += 1u) {
        let subscription = input_ptr + i * 48u; let kind = read_byte(subscription + 8u);
        if kind > 2u { return 28u; }
        if kind != 0u { immediate = true; continue; }
        let clock = load32(subscription + 16u); let flags = read_byte(subscription + 40u) | (read_byte(subscription + 41u) << 8u);
        if clock > 3u || flags > 1u { immediate = true; continue; }
        var deadline = vec2u(load32(subscription + 24u), load32(subscription + 28u));
        if flags == 0u { deadline = add64(now, deadline); }
        if lt64(deadline, earliest) { earliest = deadline; }
    }
    let awake = select(earliest, now, immediate || lt64(earliest, now));
    write_heap(config.fs_offset + 1u, awake.x); write_heap(config.fs_offset + 2u, awake.y);
    var events = 0u;
    for (var i = 0u; i < count; i += 1u) {
        let subscription = input_ptr + i * 48u; let kind = read_byte(subscription + 8u);
        var error = 0u; var available = 0u;
        if kind == 0u {
            let flags = read_byte(subscription + 40u) | (read_byte(subscription + 41u) << 8u);
            if load32(subscription + 16u) > 3u || flags > 1u { error = 28u; }
            else {
                var deadline = vec2u(load32(subscription + 24u), load32(subscription + 28u));
                if flags == 0u { deadline = add64(now, deadline); }
                if lt64(awake, deadline) { continue; }
            }
        } else {
            let fd = load32(subscription + 16u); let index = fs_file(fd);
            if index == 0xffffffffu { error = 8u; }
            else if !fs_right(fd, 1u << 27u) { error = 76u; }
            else {
                let size = read_heap(fs_inode(index) + 2u); let position = read_heap(fs_descriptor(fd) + 1u);
                available = select(size - min(size, position), config.fs_bytes - min(config.fs_bytes, position), kind == 2u);
            }
        }
        let event = output_ptr + events * 32u;
        for (var j = 0u; j < 32u; j += 1u) { write_byte(event + j, 0u); }
        store64(event, vec2u(load32(subscription), load32(subscription + 4u)));
        write_byte(event + 8u, error); write_byte(event + 10u, kind); store64(event + 16u, vec2u(available, 0u));
        events += 1u;
    }
    store32(result_ptr, events); return 0u;
}
fn fs_set_times(index: u32, atime: vec2u, mtime: vec2u, flags: u32) -> u32 {
    if flags > 15u || (flags & 3u) == 3u || (flags & 12u) == 12u { return 28u; }
    let entry = fs_inode(index);
    let now = vec2u(read_heap(config.fs_offset + 1u), read_heap(config.fs_offset + 2u));
    if (flags & 3u) != 0u {
        let value = select(atime, now, (flags & 2u) != 0u);
        write_heap(entry + 6u, value.x); write_heap(entry + 7u, value.y);
    }
    if (flags & 12u) != 0u {
        let value = select(mtime, now, (flags & 8u) != 0u);
        write_heap(entry + 8u, value.x); write_heap(entry + 9u, value.y);
    }
    write_heap(entry + 10u, now.x); write_heap(entry + 11u, now.y); return 0u;
}
fn fs_readdir(base: u32) -> u32 {
    let fd = arg(base, 0u); let directory = fs_file(fd);
    if directory == 0xffffffffu { return 8u; }
    if read_heap(fs_entry(directory)) != 2u { return 54u; }
    if !fs_right(fd, 1u << 14u) { return 76u; }
    let pointer = arg(base, 1u); let size = arg(base, 2u); let cookie = get_value(base + 3u); let result_ptr = arg(base, 4u);
    if !bounds(pointer, size) || !bounds(result_ptr, 4u) { return 21u; }
    if !fs_charge(size) { return 29u; }
    let prefix = read_heap(fs_entry(directory) + 1u); var written = 0u;
    if cookie.y == 0u {
        for (var index = max(4u, cookie.x); index < config.fs_files; index += 1u) {
            let entry = fs_entry(index); let kind = read_heap(entry); let length = read_heap(entry + 1u);
            if kind == 0u || kind >= 4u || length <= prefix { continue; }
            let name_start = fs_names() + index * 64u;
            var child = true;
            for (var j = 0u; j < prefix; j += 1u) { child = child && heap_byte(name_start, j) == heap_byte(fs_names() + directory * 64u, j); }
            if prefix != 0u { child = child && heap_byte(name_start, prefix) == 47u; }
            let basename = prefix + u32(prefix != 0u);
            for (var j = basename; j < length; j += 1u) { if heap_byte(name_start, j) == 47u { child = false; } }
            if !child { continue; }
            let name_length = length - basename;
            for (var j = 0u; j < 24u + name_length && written < size; j += 1u) {
                var value = 0u;
                if j < 4u { value = (index + 1u) >> (j * 8u); }
                else if j >= 8u && j < 12u { value = (read_heap(entry + 5u) + 1u) >> ((j - 8u) * 8u); }
                else if j >= 16u && j < 20u { value = name_length >> ((j - 16u) * 8u); }
                else if j == 20u { value = select(select(4u, 3u, kind == 2u), 7u, kind == 3u); }
                else if j >= 24u { value = heap_byte(name_start, basename + j - 24u); }
                write_byte(pointer + written, value); written += 1u;
            }
            if written == size { break; }
        }
    }
    store32(result_ptr, written); return 0u;
}
// ChaCha20 (RFC 8439), deterministic keyed streams with an instance nonce.
// No entropy or clock is requested from the operating system.
fn rng_rotate(x: u32, n: u32) -> u32 { return (x << n) | (x >> (32u - n)); }
fn rng_quarter(v: vec4u) -> vec4u {
    var a = v.x; var b = v.y; var c = v.z; var d = v.w;
    a += b; d = rng_rotate(d ^ a, 16u); c += d; b = rng_rotate(b ^ c, 12u);
    a += b; d = rng_rotate(d ^ a, 8u); c += d; b = rng_rotate(b ^ c, 7u);
    return vec4u(a, b, c, d);
}
fn rng_block() {
    var original: array<u32, 16>;
    original[0] = 0x61707865u; original[1] = 0x3320646eu; original[2] = 0x79622d32u; original[3] = 0x6b206574u;
    for (var i = 0u; i < 8u; i += 1u) { original[4u + i] = read_heap(config.fs_offset + 8u + i); }
    original[12] = read_heap(config.fs_offset + 16u);
    for (var i = 0u; i < 3u; i += 1u) { original[13u + i] = read_heap(config.fs_offset + 18u + i); }
    var x = original;
    for (var round = 0u; round < 10u; round += 1u) {
        for (var i = 0u; i < 4u; i += 1u) {
            let q = rng_quarter(vec4u(x[i], x[i + 4u], x[i + 8u], x[i + 12u]));
            x[i] = q.x; x[i + 4u] = q.y; x[i + 8u] = q.z; x[i + 12u] = q.w;
        }
        for (var i = 0u; i < 4u; i += 1u) {
            let b = 4u + (i + 1u) % 4u; let c = 8u + (i + 2u) % 4u; let d = 12u + (i + 3u) % 4u;
            let q = rng_quarter(vec4u(x[i], x[b], x[c], x[d])); x[i] = q.x; x[b] = q.y; x[c] = q.z; x[d] = q.w;
        }
    }
    for (var i = 0u; i < 16u; i += 1u) { write_heap(config.fs_offset + 21u + i, x[i] + original[i]); }
    let counter = read_heap(config.fs_offset + 16u) + 1u;
    write_heap(config.fs_offset + 16u, counter);
    if counter == 0u { write_heap(config.fs_offset + 19u, read_heap(config.fs_offset + 19u) + 1u); }
    write_heap(config.fs_offset + 37u, 0u);
}
fn wasi_dispatch(syscall: u32, base: u32) -> u32 {
    if syscall == WASI_PROC_EXIT { vm.exit_code = arg(base, 0u); fail(12u); return 0u; }
    if syscall == WASI_PROC_RAISE {
        if arg(base, 0u) > 30u { return 28u; }
        if arg(base, 0u) != 0u { vm.exit_code = arg(base, 0u); fail(13u); }
        return 0u;
    }
    if syscall == WASI_SCHED_YIELD { return 0u; }
    if syscall == WASI_POLL_ONEOFF { return fs_poll(base); }
    if syscall == WASI_FD_READDIR { return fs_readdir(base); }
    if syscall == WASI_CLOCK_TIME_GET || syscall == WASI_CLOCK_RES_GET {
        if arg(base, 0u) > 3u { return 28u; }
        let pointer = arg(base, select(2u, 1u, syscall == WASI_CLOCK_RES_GET));
        if !bounds(pointer, 8u) { return 21u; }
        var clock = vec2u(read_heap(config.fs_offset + 1u), read_heap(config.fs_offset + 2u));
        if syscall == WASI_CLOCK_RES_GET { clock = vec2u(read_heap(config.fs_offset + 4u), 0u); }
        store64(pointer, clock); return 0u;
    }
    if syscall == WASI_RANDOM_GET {
        let pointer = arg(base, 0u); let size = arg(base, 1u);
        if !bounds(pointer, size) { return 21u; }
        if !fs_charge(size) { return 29u; }
        var cursor = read_heap(config.fs_offset + 37u);
        for (var i = 0u; i < size; i += 1u) {
            if cursor == 64u { rng_block(); cursor = 0u; }
            write_byte(pointer + i, heap_byte(config.fs_offset + 21u, cursor)); cursor += 1u;
        }
        write_heap(config.fs_offset + 37u, cursor); return 0u;
    }
    if syscall == WASI_FD_WRITE || syscall == WASI_FD_READ || syscall == WASI_FD_PWRITE || syscall == WASI_FD_PREAD { return fs_io(syscall, base); }
    if syscall == WASI_ARGS_GET || syscall == WASI_ARGS_SIZES_GET || syscall == WASI_ENVIRON_GET || syscall == WASI_ENVIRON_SIZES_GET {
        let environment = syscall == WASI_ENVIRON_GET || syscall == WASI_ENVIRON_SIZES_GET;
        let sizes = syscall == WASI_ARGS_SIZES_GET || syscall == WASI_ENVIRON_SIZES_GET;
        let slot = select(8u, 10u, environment); let info = program[slot]; let count = program[slot + 1u];
        let a = arg(base, 0u); let b = arg(base, 1u);
        var total = 0u;
        for (var i = 0u; i < count; i += 1u) { total += program[info + i * 2u + 1u]; }
        if sizes {
            if !bounds(a, 4u) || !bounds(b, 4u) { return 21u; }
            store32(a, count); store32(b, total);
        } else {
            if !bounds(a, count * 4u) || !bounds(b, total) { return 21u; }
            if !fs_charge(total) { return 29u; }
            var cursor = b;
            for (var i = 0u; i < count; i += 1u) {
                store32(a + i * 4u, cursor);
                let start = program[info + i * 2u]; let length = program[info + i * 2u + 1u];
                for (var j = 0u; j < length; j += 1u) { write_byte(cursor + j, (program[start + j / 4u] >> ((j & 3u) * 8u)) & 255u); }
                cursor += length;
            }
        }
        return 0u;
    }
    if syscall == WASI_PATH_OPEN {
        let fd = arg(base, 0u); let result_ptr = arg(base, 8u);
        if !bounds(result_ptr, 4u) { return 21u; }
        let error = fs_resolve(fd, arg(base, 2u), arg(base, 3u), arg(base, 1u) == 1u); if error != 0u { return error; }
        if !fs_right(fd, 1u << 13u) { return 76u; }
        let oflags = arg(base, 4u); let flags = arg(base, 7u);
        if oflags > 15u || flags > 31u || arg(base, 1u) > 1u { return 28u; }
        let rights = get_value(base + 5u); let inheriting = get_value(base + 6u);
        let allowed = vec2u(read_heap(fs_descriptor(fd) + 6u), read_heap(fs_descriptor(fd) + 7u));
        if any((rights & ~allowed) != vec2u(0u)) || any((inheriting & ~allowed) != vec2u(0u)) { return 76u; }
        var new_fd = 0xffffffffu;
        for (var i = 0u; i < config.fs_fds; i += 1u) { if read_heap(fs_descriptor(i)) == 0u { new_fd = i; break; } }
        if new_fd == 0xffffffffu { return 33u; }
        var index = fs_lookup();
        if index == 0xffffffffu {
            if (oflags & 1u) == 0u { return 44u; }
            if (oflags & 2u) != 0u { return 54u; }
            if !fs_right(fd, 1u << 10u) { return 76u; }
            if !fs_parent_exists() { return 44u; }
            index = fs_create(1u); if index == 0xffffffffu { return 51u; }
        } else if (oflags & 5u) == 5u { return 20u; }
        let kind = read_heap(fs_entry(index));
        if (oflags & 2u) != 0u && kind != 2u { return 54u; }
        if kind == 3u { return 32u; }
        if (oflags & 8u) != 0u {
            if (rights.x & 64u) == 0u { return 76u; }
            if kind == 2u { return 31u; }
            let resize_error = fs_resize(index, 0u); if resize_error != 0u { return resize_error; }
        }
        let descriptor = fs_descriptor(new_fd);
        write_heap(descriptor, index + 1u); write_heap(descriptor + 1u, 0u); write_heap(descriptor + 2u, 0u);
        write_heap(descriptor + 3u, flags); write_heap(descriptor + 4u, rights.x); write_heap(descriptor + 5u, rights.y);
        write_heap(descriptor + 6u, inheriting.x); write_heap(descriptor + 7u, inheriting.y);
        store32(result_ptr, new_fd); return 0u;
    }
    if syscall == WASI_PATH_CREATE_DIRECTORY || syscall == WASI_PATH_UNLINK_FILE || syscall == WASI_PATH_REMOVE_DIRECTORY {
        let fd = arg(base, 0u); let error = fs_path(fd, arg(base, 1u), arg(base, 2u)); if error != 0u { return error; }
        let right = select(select(1u << 25u, 1u << 26u, syscall == WASI_PATH_UNLINK_FILE), 1u << 9u, syscall == WASI_PATH_CREATE_DIRECTORY);
        if !fs_right(fd, right) { return 76u; }
        let index = fs_lookup();
        if syscall == WASI_PATH_CREATE_DIRECTORY {
            if index != 0xffffffffu { return 20u; }
            if !fs_parent_exists() { return 44u; }
            return select(0u, 51u, fs_create(2u) == 0xffffffffu);
        }
        if index == 0xffffffffu { return 44u; }
        let kind = read_heap(fs_entry(index));
        if syscall == WASI_PATH_UNLINK_FILE && kind == 2u { return 31u; }
        if syscall == WASI_PATH_REMOVE_DIRECTORY {
            if kind != 2u { return 54u; }
            if index == 3u { return 10u; }
            for (var i = 4u; i < config.fs_files; i += 1u) {
                if read_heap(fs_entry(i)) == 0u || read_heap(fs_entry(i)) >= 4u || read_heap(fs_entry(i) + 1u) <= path_length { continue; }
                var child = heap_byte(fs_names() + i * 64u, path_length) == 47u;
                for (var j = 0u; j < path_length; j += 1u) { child = child && heap_byte(fs_names() + i * 64u, j) == path_text[j]; }
                if child { return 55u; }
            }
        }
        write_heap(fs_entry(index), 4u); return 0u;
    }
    if syscall == WASI_PATH_FILESTAT_GET {
        if !bounds(arg(base, 4u), 64u) { return 21u; }
        if arg(base, 1u) > 1u { return 28u; }
        let error = fs_resolve(arg(base, 0u), arg(base, 2u), arg(base, 3u), arg(base, 1u) == 1u); if error != 0u { return error; }
        if !fs_right(arg(base, 0u), 1u << 18u) { return 76u; }
        let index = fs_lookup(); if index == 0xffffffffu { return 44u; }
        fs_stat(index, arg(base, 4u)); return 0u;
    }
    if syscall == WASI_PATH_FILESTAT_SET_TIMES {
        if arg(base, 1u) > 1u { return 28u; }
        let fd = arg(base, 0u); let error = fs_resolve(fd, arg(base, 2u), arg(base, 3u), arg(base, 1u) == 1u); if error != 0u { return error; }
        if !fs_right(fd, 1u << 20u) { return 76u; }
        let index = fs_lookup(); if index == 0xffffffffu { return 44u; }
        return fs_set_times(index, get_value(base + 4u), get_value(base + 5u), arg(base, 6u));
    }
    if syscall == WASI_PATH_SYMLINK {
        let old_ptr = arg(base, 0u); let old_size = arg(base, 1u); let fd = arg(base, 2u);
        if !bounds(old_ptr, old_size) { return 21u; }
        if old_size == 0u { return 44u; }
        for (var i = 0u; i < old_size; i += 1u) { if read_byte(old_ptr + i) == 0u { return 28u; } }
        let error = fs_path(fd, arg(base, 3u), arg(base, 4u)); if error != 0u { return error; }
        if !fs_right(fd, 1u << 24u) { return 76u; }
        if fs_lookup() != 0xffffffffu { return 20u; }
        if !fs_parent_exists() { return 44u; }
        let index = fs_create(3u); if index == 0xffffffffu { return 51u; }
        let resized = fs_resize(index, old_size);
        if resized != 0u { write_heap(fs_entry(index), 0u); return resized; }
        let start = read_heap(fs_inode(index) + 3u);
        for (var i = 0u; i < old_size; i += 1u) { set_heap_byte(fs_data(), start + i, read_byte(old_ptr + i)); }
        return 0u;
    }
    if syscall == WASI_PATH_READLINK {
        let fd = arg(base, 0u); let error = fs_path(fd, arg(base, 1u), arg(base, 2u)); if error != 0u { return error; }
        if !fs_right(fd, 1u << 15u) { return 76u; }
        let index = fs_lookup(); if index == 0xffffffffu { return 44u; }
        if read_heap(fs_entry(index)) != 3u { return 28u; }
        let pointer = arg(base, 3u); let size = arg(base, 4u); let result_ptr = arg(base, 5u);
        if !bounds(pointer, size) || !bounds(result_ptr, 4u) { return 21u; }
        let entry = fs_inode(index); let amount = min(size, read_heap(entry + 2u)); let start = read_heap(entry + 3u);
        if !fs_charge(amount) { return 29u; }
        for (var i = 0u; i < amount; i += 1u) { write_byte(pointer + i, heap_byte(fs_data(), start + i)); }
        store32(result_ptr, amount); return 0u;
    }
    if syscall == WASI_PATH_RENAME || syscall == WASI_PATH_LINK {
        let linking = syscall == WASI_PATH_LINK; let old_fd = arg(base, 0u);
        if linking && arg(base, 1u) > 1u { return 28u; }
        var error = fs_resolve(old_fd, arg(base, select(1u, 2u, linking)), arg(base, select(2u, 3u, linking)), linking && arg(base, 1u) == 1u);
        if error != 0u { return error; }
        if !fs_right(old_fd, select(1u << 16u, 1u << 11u, linking)) { return 76u; }
        let index = fs_lookup(); if index == 0xffffffffu { return 44u; }
        if linking && read_heap(fs_entry(index)) == 2u { return 63u; }
        let new_fd = arg(base, select(3u, 4u, linking));
        error = fs_path(new_fd, arg(base, select(4u, 5u, linking)), arg(base, select(5u, 6u, linking)));
        if error != 0u { return error; }
        if !fs_right(new_fd, select(1u << 17u, 1u << 12u, linking)) { return 76u; }
        if !fs_parent_exists() { return 44u; }
        let existing = fs_lookup();
        if linking {
            if existing != 0xffffffffu { return 20u; }
            let linked = fs_create(read_heap(fs_entry(index))); if linked == 0xffffffffu { return 51u; }
            write_heap(fs_entry(linked) + 5u, read_heap(fs_entry(index) + 5u));
        } else { return fs_rename(index, existing); }
        return 0u;
    }
    // Descriptor services.
    let fd = arg(base, 0u); let index = fs_file(fd);
    if index == 0xffffffffu { return 8u; }
    if syscall == WASI_SOCK_ACCEPT || syscall == WASI_SOCK_RECV || syscall == WASI_SOCK_SEND || syscall == WASI_SOCK_SHUTDOWN { return 57u; }
    let descriptor = fs_descriptor(fd); let entry = fs_inode(index);
    if syscall == WASI_FD_CLOSE { write_heap(descriptor, 0u); return 0u; }
    if syscall == WASI_FD_RENUMBER {
        let destination = arg(base, 1u); if destination >= config.fs_fds { return 8u; }
        if destination != fd {
            for (var i = 0u; i < 8u; i += 1u) { write_heap(fs_descriptor(destination) + i, read_heap(descriptor + i)); }
            write_heap(descriptor, 0u);
        }
        return 0u;
    }
    if syscall == WASI_FD_PRESTAT_GET || syscall == WASI_FD_PRESTAT_DIR_NAME {
        if fd != 3u || index != 3u { return 8u; }
        let pointer = arg(base, 1u);
        if syscall == WASI_FD_PRESTAT_GET {
            if !bounds(pointer, 8u) { return 21u; }
            store32(pointer, 0u); store32(pointer + 4u, 1u);
        } else {
            if arg(base, 2u) < 1u { return 37u; }
            if !bounds(pointer, 1u) { return 21u; }
            write_byte(pointer, 46u);
        }
        return 0u;
    }
    if syscall == WASI_FD_FDSTAT_GET {
        let pointer = arg(base, 1u); if !bounds(pointer, 24u) { return 21u; }
        for (var i = 0u; i < 24u; i += 1u) { write_byte(pointer + i, 0u); }
        write_byte(pointer, select(select(4u, 3u, read_heap(fs_entry(index)) == 2u), 2u, index < 3u));
        write_byte(pointer + 2u, read_heap(descriptor + 3u));
        store64(pointer + 8u, vec2u(read_heap(descriptor + 4u), read_heap(descriptor + 5u)));
        store64(pointer + 16u, vec2u(read_heap(descriptor + 6u), read_heap(descriptor + 7u)));
        return 0u;
    }
    if syscall == WASI_FD_FDSTAT_SET_FLAGS {
        if !fs_right(fd, 1u << 3u) { return 76u; }
        let flags = arg(base, 1u); if flags > 31u { return 28u; }
        write_heap(descriptor + 3u, flags); return 0u;
    }
    if syscall == WASI_FD_FDSTAT_SET_RIGHTS {
        let rights = get_value(base + 1u); let inheriting = get_value(base + 2u);
        if (rights.x & ~read_heap(descriptor + 4u)) != 0u || (rights.y & ~read_heap(descriptor + 5u)) != 0u || (inheriting.x & ~read_heap(descriptor + 6u)) != 0u || (inheriting.y & ~read_heap(descriptor + 7u)) != 0u { return 76u; }
        write_heap(descriptor + 4u, rights.x); write_heap(descriptor + 5u, rights.y);
        write_heap(descriptor + 6u, inheriting.x); write_heap(descriptor + 7u, inheriting.y); return 0u;
    }
    if syscall == WASI_FD_SEEK || syscall == WASI_FD_TELL {
        if !fs_right(fd, select(1u << 5u, 1u << 2u, syscall == WASI_FD_SEEK)) { return 76u; }
        let result_ptr = arg(base, select(1u, 3u, syscall == WASI_FD_SEEK));
        if !bounds(result_ptr, 8u) { return 21u; }
        var position = vec2u(read_heap(descriptor + 1u), read_heap(descriptor + 2u));
        if syscall == WASI_FD_SEEK {
            let whence = arg(base, 2u); let offset = get_value(base + 1u);
            if whence > 2u { return 28u; }
            if whence == 0u { position = vec2u(0u); }
            if whence == 2u { position = vec2u(read_heap(entry + 2u), 0u); }
            if (offset.y >> 31u) != 0u && lt64(position, neg64(offset)) { return 28u; }
            let next = add64(position, offset);
            if (offset.y >> 31u) == 0u && lt64(next, position) { return 61u; }
            position = next; write_heap(descriptor + 1u, position.x); write_heap(descriptor + 2u, position.y);
        }
        store64(result_ptr, position); return 0u;
    }
    if syscall == WASI_FD_FILESTAT_GET {
        if !fs_right(fd, 1u << 21u) { return 76u; }
        if !bounds(arg(base, 1u), 64u) { return 21u; }
        fs_stat(index, arg(base, 1u)); return 0u;
    }
    if syscall == WASI_FD_FILESTAT_SET_TIMES {
        if !fs_right(fd, 1u << 23u) { return 76u; }
        return fs_set_times(index, get_value(base + 1u), get_value(base + 2u), arg(base, 3u));
    }
    if syscall == WASI_FD_FILESTAT_SET_SIZE || syscall == WASI_FD_ALLOCATE {
        if !fs_right(fd, select(1u << 22u, 1u << 8u, syscall == WASI_FD_ALLOCATE)) { return 76u; }
        var length = get_value(base + 1u);
        if syscall == WASI_FD_ALLOCATE {
            let end = add64(length, get_value(base + 2u));
            if lt64(end, length) { return 61u; }
            length = end; length.x = max(length.x, read_heap(entry + 2u));
        }
        if length.y != 0u { return 27u; }
        if read_heap(fs_entry(index)) == 2u { return 31u; }
        return fs_resize(index, length.x);
    }
    if syscall == WASI_FD_SYNC || syscall == WASI_FD_DATASYNC {
        return select(76u, 0u, fs_right(fd, select(1u, 1u << 4u, syscall == WASI_FD_SYNC)));
    }
    if syscall == WASI_FD_ADVISE {
        if arg(base, 3u) > 5u { return 28u; }
        return select(76u, 0u, fs_right(fd, 1u << 7u));
    }
    return 58u;
}
