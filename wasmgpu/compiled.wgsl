// Boundaries preserve the interpreter's instruction accounting and virtual time.
fn compiled_tick(count: u32) {
    consume_fuel(count);
    let before = vm.compiled;
    vm.compiled += count;
    if vm.compiled < before { vm.compiled_high += 1u; }
    if program[12] != 0u {
        let delta = mul64(vec2u(count, 0u), vec2u(read_heap(config.fs_offset + 4u), 0u));
        let clock = add64(vec2u(read_heap(config.fs_offset + 1u), read_heap(config.fs_offset + 2u)), delta);
        write_heap(config.fs_offset + 1u, clock.x); write_heap(config.fs_offset + 2u, clock.y);
    }
}
fn compiled_indirect(type_id: u32, table: u32, index: u32) {
    if index >= table_size(table) { fail(3u); return; }
    let reference = read_heap(table_offset(table) + index);
    if reference == 0u { fail(9u); return; }
    if program[16u + (reference - 1u) * 8u + 4u] != type_id { fail(10u); return; }
    invoke(reference - 1u);
}
fn compiled_branch_table(offset: u32, count: u32, index: u32) {
    let location = program[5] + (offset + min(index, count - 1u)) * 3u;
    branch(program[location], program[location + 1u], program[location + 2u]);
}
fn compiled_load(address: u32, width: u32) -> vec2u {
    let word = config.memory_offset + address / 4u;
    let shift = (address & 3u) * 8u;
    var value = vec2u(read_heap(word) >> shift, 0u);
    if width == 1u { return value & vec2u(255u, 0u); }
    if width == 2u {
        if shift == 24u { value.x |= read_heap(word + 1u) << 8u; }
        return value & vec2u(65535u, 0u);
    }
    if shift != 0u { value.x |= read_heap(word + 1u) << (32u - shift); }
    if width == 8u {
        value.y = read_heap(word + 1u) >> shift;
        if shift != 0u { value.y |= read_heap(word + 2u) << (32u - shift); }
    }
    return value;
}
fn compiled_store(address: u32, width: u32, value: vec2u) {
    if (address & 3u) == 0u && width >= 4u {
        write_heap(config.memory_offset + address / 4u, value.x);
        if width == 8u { write_heap(config.memory_offset + address / 4u + 1u, value.y); }
    } else {
        for (var i = 0u; i < width; i += 1u) { write_byte(address + i, shr64(value, i * 8u).x); }
    }
}
// Dispatch only statically generated basic blocks. An unknown continuation
// belongs to a different compilation unit; the host schedules its native pipeline.
@compute @workgroup_size(64)
fn run(@builtin(global_invocation_id) id: vec3u) {
    lane = id.x;
    if lane >= config.count { return; }
    vm = states[lane];
    var remaining = config.quantum;
    loop {
        if vm.status != 0u || remaining == 0u { break; }
        let consumed = compiled_dispatch();
        if consumed == 0u { break; }
        // Quantum boundaries are basic-block boundaries (at most 31 extra ops).
        remaining -= min(remaining, consumed);
    }
    states[lane] = vm;
}
