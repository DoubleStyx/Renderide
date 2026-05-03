// Sparse weighted blendshape position, normal, and tangent deltas. Each entry is one influenced vertex for one frame;
// the encoder dispatches one scatter pass per selected Unity frame (optionally chunked by entry count).

const APPLY_NORMALS: u32 = 1u;
const APPLY_TANGENTS: u32 = 2u;

struct Params {
    vertex_count: u32,
    sparse_base: u32,
    sparse_count: u32,
    /// Element offset into `out_pos` for this instance's subrange (GPU skin cache arena).
    base_dst_pos_e: u32,
    base_dst_nrm_e: u32,
    base_dst_tan_e: u32,
    flags: u32,
    effective_weight: f32,
}

struct SparseEntry {
    vertex_index: u32,
    pos_dx: f32,
    pos_dy: f32,
    pos_dz: f32,
    nrm_dx: f32,
    nrm_dy: f32,
    nrm_dz: f32,
    tan_dx: f32,
    tan_dy: f32,
    tan_dz: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> sparse: array<SparseEntry>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_pos: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> out_nrm: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> out_tan: array<vec4<f32>>;

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
    let d = vec3<f32>(e.pos_dx, e.pos_dy, e.pos_dz);
    let oi = params.base_dst_pos_e + vi;
    let p = out_pos[oi];
    out_pos[oi] = vec4<f32>(p.xyz + wi * d, p.w);

    if ((params.flags & APPLY_NORMALS) != 0u) {
        let ni = params.base_dst_nrm_e + vi;
        let n = out_nrm[ni];
        let dn = vec3<f32>(e.nrm_dx, e.nrm_dy, e.nrm_dz);
        out_nrm[ni] = vec4<f32>(n.xyz + wi * dn, n.w);
    }

    if ((params.flags & APPLY_TANGENTS) != 0u) {
        let ti = params.base_dst_tan_e + vi;
        let t = out_tan[ti];
        let dt = vec3<f32>(e.tan_dx, e.tan_dy, e.tan_dz);
        out_tan[ti] = vec4<f32>(t.xyz + wi * dt, t.w);
    }
}
