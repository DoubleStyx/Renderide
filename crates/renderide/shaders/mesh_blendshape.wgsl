// Weighted blendshape deltas (compute). `deltas` uses `BLENDSHAPE_OFFSET_GPU_STRIDE` (48) bytes per
// (shape_index, vertex): three vec4 chunks (position, normal, tangent deltas). This entry point
// consumes only `.pos.xyz` per shape weight.

struct Params {
    /// Vertices in the mesh (same for every chunk).
    vertex_count: u32,
    /// Blendshape count in **this** dispatch only.
    shape_count: u32,
    /// Same as `vertex_count`; stride inside the bound `deltas` subrange.
    vertices_per_shape: u32,
    /// Global shape index of `deltas[0]` (indexes into `weights`).
    weight_base: u32,
    /// `1` for the first chunk (seed from `base_pos`); `0` for later chunks (accumulate from `out_pos`).
    first_chunk: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct DeltaPacked {
    pos: vec4<f32>,
    _norm: vec4<f32>,
    _tang: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> base_pos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> deltas: array<DeltaPacked>;
@group(0) @binding(3) var<storage, read> weights: array<f32>;
@group(0) @binding(4) var<storage, read_write> out_pos: array<vec4<f32>>;

@compute @workgroup_size(64)
fn blendshape_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.vertex_count) {
        return;
    }
    var acc: vec3<f32>;
    if (params.first_chunk != 0u) {
        acc = base_pos[i].xyz;
    } else {
        acc = out_pos[i].xyz;
    }
    for (var s = 0u; s < params.shape_count; s = s + 1u) {
        let wi = weights[params.weight_base + s];
        if (wi != 0.0) {
            let di = deltas[s * params.vertices_per_shape + i];
            acc += wi * di.pos.xyz;
        }
    }
    out_pos[i] = vec4<f32>(acc, base_pos[i].w);
}
