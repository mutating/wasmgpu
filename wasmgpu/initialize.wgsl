struct InitialConfig { count: u32, words: u32, nonce: u32, first: u32 }
@group(0) @binding(0) var<storage, read> initial: array<u32>;
@group(0) @binding(1) var<storage, read_write> heap: array<u32>;
@group(0) @binding(2) var<uniform> config: InitialConfig;
@compute @workgroup_size(256)
fn initialize(@builtin(global_invocation_id) id: vec3u) {
    for (var i = id.x; i < config.words * config.count; i += 16776960u) {
        let word = i / config.count;
        var value = initial[word];
        if word == config.nonce { value = config.first + i % config.count; }
        heap[i] = value;
    }
}
