// The compiled tier outlines WASI into a shared GPU kernel. Pending arguments
// remain on the operand stack; output[0] holds a private continuation record
// until return_function writes final results. No syscall is handled by Python.
@compute @workgroup_size(64)
fn service(@builtin(global_invocation_id) id: vec3u) {
    lane = id.x;
    if lane >= config.count { return; }
    vm = states[lane];
    if vm.status != 5u { return; }
    let pending = output[lane];
    vm.status = 0u;
    let answer = wasi_dispatch(pending.x, pending.y);
    vm.sp = pending.y;
    if pending.x != WASI_PROC_EXIT { push(vec2u(answer, 0u)); }
    states[lane] = vm;
}
