struct Config {
    count: u32, stack_cap: u32, frame_cap: u32, memory_cap: u32,
    memory_offset: u32, table_offset: u32, table_cap: u32, data_flags: u32,
    element_flags: u32, quantum: u32, output_slots: u32, reserved: u32,
    fs_offset: u32, fs_files: u32, fs_fds: u32, fs_bytes: u32,
}
struct State {
    pc: u32, sp: u32, base: u32, function_id: u32,
    depth: u32, pages: u32, interpreted: u32, status: u32,
    trap: u32, fuel: u32, exit_code: u32, compiled: u32,
    fuel_high: u32, interpreted_high: u32, compiled_high: u32, padding: u32,
}
@group(0) @binding(0) var<storage, read> program: array<u32>;
@group(0) @binding(1) var<uniform> config: Config;
@group(0) @binding(2) var<storage, read_write> values: array<vec2u>;
@group(0) @binding(3) var<storage, read_write> heap: array<u32>;
@group(0) @binding(4) var<storage, read_write> frames: array<vec4u>;
@group(0) @binding(5) var<storage, read_write> states: array<State>;
@group(0) @binding(6) var<storage, read_write> output: array<vec2u>;
var<private> lane: u32;
var<private> vm: State;
fn fail(code: u32) { vm.trap = code; vm.status = 2u; }
fn fuel_available(amount: u32) -> bool { return vm.fuel_high != 0u || vm.fuel >= amount; }
fn consume_fuel(amount: u32) {
    if vm.fuel < amount { vm.fuel_high -= 1u; }
    vm.fuel -= amount;
}
fn get_value(index: u32) -> vec2u { return values[index * config.count + lane]; }
fn set_value(index: u32, value: vec2u) { values[index * config.count + lane] = value; }
fn push(value: vec2u) {
    if vm.sp >= config.stack_cap { fail(7u); return; }
    set_value(vm.sp, value); vm.sp += 1u;
}
fn pop() -> vec2u {
    if vm.sp == 0u { fail(11u); return vec2u(0u); }
    vm.sp -= 1u; return get_value(vm.sp);
}
fn read_heap(index: u32) -> u32 { return heap[index * config.count + lane]; }
fn write_heap(index: u32, value: u32) { heap[index * config.count + lane] = value; }
fn bounds(address: u32, size: u32) -> bool {
    // memory_cap is limited to 65535 pages by the host (u32 byte addressing).
    let length = vm.pages * 65536u;
    return address <= length && size <= length - address;
}
fn read_byte(address: u32) -> u32 {
    return (read_heap(config.memory_offset + address / 4u) >> ((address & 3u) * 8u)) & 255u;
}
fn write_byte(address: u32, value: u32) {
    let index = config.memory_offset + address / 4u; let shift = (address & 3u) * 8u;
    write_heap(index, (read_heap(index) & ~(255u << shift)) | ((value & 255u) << shift));
}
fn branch(destination_pc: u32, height: u32, arity: u32) {
    let metadata = 16u + vm.function_id * 8u;
    let destination = vm.base + program[metadata + 3u] + height;
    for (var i = 0u; i < arity; i += 1u) { set_value(destination + i, get_value(vm.sp - arity + i)); }
    vm.sp = destination + arity; vm.pc = destination_pc;
}
fn invoke(function_id: u32) {
    if function_id >= program[3] { fail(3u); return; }
    let metadata = 16u + function_id * 8u;
    let params = program[metadata + 1u]; let locals = program[metadata + 3u];
    let base = vm.sp - params;
    if program[metadata + 5u] != 0u {
        let syscall = program[metadata + 5u];
        let answer = wasi_dispatch(syscall, base);
        if vm.status == 5u { return; }
        vm.sp = base;
        if program[metadata + 2u] != 0u { push(vec2u(answer, 0u)); }
        return;
    }
    invoke_native(function_id);
}
fn invoke_native(function_id: u32) {
    let metadata = 16u + function_id * 8u;
    let params = program[metadata + 1u]; let locals = program[metadata + 3u];
    let base = vm.sp - params;
    if vm.depth >= config.frame_cap || base + locals > config.stack_cap { fail(7u); return; }
    frames[vm.depth * config.count + lane] = vec4u(vm.pc, vm.base, vm.function_id, base);
    vm.depth += 1u; vm.base = base; vm.function_id = function_id;
    for (var i = params; i < locals; i += 1u) { set_value(base + i, vec2u(0u)); }
    vm.sp = base + locals; vm.pc = program[metadata];
}
fn return_function() {
    let result_count = program[16u + vm.function_id * 8u + 2u];
    if vm.depth == 0u {
        for (var i = 0u; i < result_count; i += 1u) { output[i * config.count + lane] = get_value(vm.sp - result_count + i); }
        vm.status = 1u; return;
    }
    vm.depth -= 1u;
    let frame = frames[vm.depth * config.count + lane];
    for (var i = 0u; i < result_count; i += 1u) { set_value(frame.w + i, get_value(vm.sp - result_count + i)); }
    vm.sp = frame.w + result_count; vm.pc = frame.x; vm.base = frame.y; vm.function_id = frame.z;
}
fn memory_operation(op: u32, offset: u32, width: u32) {
    let store = op >= 0x36u;
    var value = vec2u(0u); if store { value = pop(); }
    let pointer = pop().x;
    let address = pointer + offset;
    if address < pointer || !bounds(address, width) { fail(2u); return; }
    if store {
        for (var i = 0u; i < width; i += 1u) { write_byte(address + i, shr64(value, i * 8u).x); }
    } else {
        for (var i = 0u; i < width; i += 1u) { value |= shl64(vec2u(read_byte(address + i), 0u), i * 8u); }
        let is_signed = op == 0x2cu || op == 0x2eu || op == 0x30u || op == 0x32u || op == 0x34u;
        if is_signed { value = sar64(shl64(value, 64u - width * 8u), 64u - width * 8u); }
        if op == 0x28u || op == 0x2au || (op >= 0x2cu && op <= 0x2fu) { value.y = 0u; }
        push(value);
    }
}
fn table_size(table: u32) -> u32 { return read_heap(program[program[4] + table * 4u]); }
fn table_offset(table: u32) -> u32 { return program[program[4] + table * 4u + 1u]; }
fn bulk_operation(op: u32, index: u32, other: u32) {
    if op == 9u { write_heap(config.data_flags + index, 1u); return; }
    if op == 13u { write_heap(config.element_flags + index, 1u); return; }
    if op == 16u { push(vec2u(table_size(index), 0u)); return; }
    if op == 15u {
        let delta = pop().x; let value = pop().x; let old = table_size(index);
        if delta > program[program[4] + index * 4u + 2u] - old { push(vec2u(0xffffffffu, 0u)); return; }
        if !fs_charge(delta) { return; }
        write_heap(program[program[4] + index * 4u], old + delta);
        for (var i = old; i < old + delta; i += 1u) { write_heap(table_offset(index) + i, value); }
        push(vec2u(old, 0u)); return;
    }
    let size = pop().x; let source = pop().x; let destination = pop().x;
    let table = op == 12u || op == 14u || op == 17u;
    if !fs_charge(size) { return; }
    var length = vm.pages * 65536u; var offset = 0u;
    if table { let id = select(index, other, op == 12u); length = table_size(id); offset = table_offset(id); }
    if destination > length || size > length - destination { fail(select(2u, 3u, table)); return; }
    if op == 8u || op == 12u {
        let info = program[select(1u, 2u, table)] + index * 2u;
        let data_offset = program[info];
        let dropped = read_heap(select(config.data_flags, config.element_flags, table) + index) != 0u;
        let data_length = select(program[info + 1u], 0u, dropped);
        if source > data_length || size > data_length - source { fail(select(2u, 3u, table)); return; }
        for (var i = 0u; i < size; i += 1u) {
            if table { write_heap(offset + destination + i, program[data_offset + source + i]); }
            else { let address = source + i; write_byte(destination + i, (program[data_offset + address / 4u] >> ((address & 3u) * 8u)) & 255u); }
        }
    } else if op == 11u || op == 17u {
        for (var i = 0u; i < size; i += 1u) {
            if table { write_heap(offset + destination + i, source); }
            else { write_byte(destination + i, source); }
        }
    } else {
        var source_offset = 0u;
        if table { length = table_size(other); source_offset = table_offset(other); }
        if source > length || size > length - source { fail(select(2u, 3u, table)); return; }
        for (var step = 0u; step < size; step += 1u) {
            let i = select(step, size - 1u - step, destination > source && (!table || index == other));
            if table { write_heap(offset + destination + i, read_heap(source_offset + source + i)); }
            else { write_byte(destination + i, read_byte(source + i)); }
        }
    }
}
fn interpret_step() {
        if !fuel_available(1u) { fail(8u); return; }
        consume_fuel(1u);
        vm.interpreted += 1u;
        if vm.interpreted == 0u { vm.interpreted_high += 1u; }
        if program[12] != 0u {
            let clock = add64(vec2u(read_heap(config.fs_offset + 1u), read_heap(config.fs_offset + 2u)), vec2u(read_heap(config.fs_offset + 4u), 0u));
            write_heap(config.fs_offset + 1u, clock.x); write_heap(config.fs_offset + 2u, clock.y);
        }
        let location = program[0] + vm.pc * 4u;
        let op = program[location]; let a = program[location + 1u];
        let b = program[location + 2u]; let c = program[location + 3u];
        vm.pc += 1u;
        switch op {
            case 0u: { fail(1u); }
            case 0x100u: { vm.pc = a; }
            case 0x101u: { if pop().x == 0u { vm.pc = a; } }
            case 0x102u: { branch(a, b, c); }
            case 0x103u: { if pop().x != 0u { branch(a, b, c); } }
            case 0x104u: {
                let destination_pc = program[0] + (a + min(pop().x, b - 1u)) * 4u;
                branch(program[destination_pc + 1u], program[destination_pc + 2u], program[destination_pc + 3u]);
            }
            case 0x0fu: { return_function(); }
            case 0x10u: { invoke(a); }
            case 0x11u: {
                let index = pop().x;
                if index >= table_size(b) { fail(3u); }
                else {
                    let reference = read_heap(table_offset(b) + index);
                    if reference == 0u { fail(9u); }
                    else if program[16u + (reference - 1u) * 8u + 4u] != a { fail(10u); }
                    else { invoke(reference - 1u); }
                }
            }
            case 0x1au: { let ignored = pop(); }
            case 0x1bu: { let condition = pop().x; let rhs = pop(); let lhs = pop(); push(select(rhs, lhs, condition != 0u)); }
            case 0x20u: { push(get_value(vm.base + a)); }
            case 0x21u: { let value = pop(); set_value(vm.base + a, value); }
            case 0x22u: { set_value(vm.base + a, get_value(vm.sp - 1u)); }
            case 0x23u: { push(vec2u(read_heap(a * 2u), read_heap(a * 2u + 1u))); }
            case 0x24u: { let value = pop(); write_heap(a * 2u, value.x); write_heap(a * 2u + 1u, value.y); }
            case 0x25u: {
                let index = pop().x;
                if index >= table_size(a) { fail(3u); } else { push(vec2u(read_heap(table_offset(a) + index), 0u)); }
            }
            case 0x26u: {
                let value = pop().x; let index = pop().x;
                if index >= table_size(a) { fail(3u); } else { write_heap(table_offset(a) + index, value); }
            }
            case 0x3fu: { push(vec2u(vm.pages, 0u)); }
            case 0x40u: {
                let delta = pop().x; let old = vm.pages;
                if delta > config.memory_cap - old { push(vec2u(0xffffffffu, 0u)); }
                else {
                    vm.pages += delta;
                    // The complete reserved arena was zero-initialized; pages never shrink.
                    push(vec2u(old, 0u));
                }
            }
            case 0x41u, 0x42u, 0x43u, 0x44u: { push(vec2u(a, b)); }
            default: {
                if op >= 0x28u && op <= 0x3eu { memory_operation(op, a, b); }
                else if op >= 0xfc08u && op <= 0xfc11u { bulk_operation(op - 0xfc00u, a, b); }
                else {
                    var rhs = vec2u(0u);
                    if a == 2u { rhs = pop(); }
                    let lhs = pop(); let answer = numeric(op, lhs, rhs);
                    if answer.trap != 0u { fail(answer.trap); } else { push(answer.value); }
                }
            }
        }
}
@compute @workgroup_size(64)
fn run(@builtin(global_invocation_id) id: vec3u) {
    lane = id.x;
    if lane >= config.count { return; }
    vm = states[lane];
    if vm.status != 0u { return; }
    for (var step = 0u; step < config.quantum; step += 1u) {
        interpret_step();
        if vm.status != 0u { break; }
    }
    states[lane] = vm;
}
