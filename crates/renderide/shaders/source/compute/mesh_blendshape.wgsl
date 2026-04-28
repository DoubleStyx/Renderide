// Sparse weighted blendshape position deltas. Each entry is one influenced vertex for one frame;
// the encoder dispatches one scatter pass per selected Unity frame (optionally chunked by entry count).

struct Params {
    vertex_count: u32,
    sparse_base: u32,
    sparse_count: u32,
    /// Element offset into `out_pos` for this instance’s subrange (GPU skin cache arena).
    base_dst_e: u32,
    effective_weight: f32,
    _p1: u32,
    _p2: u32,
    _p3: u32,
}

struct SparseEntry {
    vertex_index: u32,
    dx: f32,
    dy: f32,
    dz: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> sparse: array<SparseEntry>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_pos: array<vec4<f32>>;

@compute @workgroup_size(64)
fn blendshape_scatter_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.sparse_count) {
        return;
    }
    let wi = params.effective_weight;
    if (wi == 0.0) {
        return;
    }
    let e = sparse[params.sparse_base + i];
    let vi = e.vertex_index;
    if (vi >= params.vertex_count) {
        return;
    }
    let d = vec3<f32>(e.dx, e.dy, e.dz);
    let oi = params.base_dst_e + vi;
    let p = out_pos[oi];
    out_pos[oi] = vec4<f32>(p.xyz + wi * d, p.w);
}
