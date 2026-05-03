// Linear blend skinning (compute). Bind buffers expected to match layout produced by mesh preprocess.
// Bone palette entries are world_bone * unity_bindpose (inverse bind matrix per bone), built on CPU each frame.
// Positions use M; normals use transpose(inverse(mat3(M))) per bone (inverse-transpose / cotangent rule).
//
// Source and destination buffers may be subranges of large arenas; [`SkinDispatchParams`] supplies element bases.

struct SkinDispatchParams {
    vertex_count: u32,
    base_src_pos_e: u32,
    base_src_nrm_e: u32,
    base_src_tan_e: u32,
    base_dst_pos_e: u32,
    base_dst_nrm_e: u32,
    base_dst_tan_e: u32,
    flags: u32,
}

const SKIN_TANGENTS: u32 = 1u;

@group(0) @binding(0) var<storage, read> bone_matrices: array<mat4x4<f32>>;
@group(0) @binding(1) var<storage, read> src_pos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> bone_idx: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> bone_weights: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> dst_pos: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> src_n: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> dst_n: array<vec4<f32>>;
@group(0) @binding(7) var<uniform> skin_dispatch: SkinDispatchParams;
@group(0) @binding(8) var<storage, read> src_t: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read_write> dst_t: array<vec4<f32>>;

fn mat3_linear(m: mat4x4<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(m[0].xyz, m[1].xyz, m[2].xyz);
}

/// Upper-3x3 inverse-transpose (cotangent rule for normals) of `m`.
///
/// For columns `c0, c1, c2`, the rows of `adj(m)` are `c1xc2`, `c2xc0`, `c0xc1`. `m^-1 = adj(m)/det`
/// stores those vectors as **rows**, so `m^-T` stores them as **columns** -- which is what
/// `mat3x3<f32>(...)` produces directly. Returns identity for singular linear parts to avoid NaNs
/// in the linear-blend skinning sum.
///
/// Replaces a previous `mat3_inverse` + outer `transpose` pair that ended up returning `m^-1`
/// instead of `m^-T`: the inner function constructed the adjugate rows as columns of the result,
/// so the outer `transpose` cancelled the wrong side. For pure-rotation bones that delivered
/// `R^T` to bind-pose normals where it should have been `R`, lighting the side facing **away**
/// from the light on every rotated skinned vertex.
fn inverse_transpose_3x3(m: mat3x3<f32>) -> mat3x3<f32> {
    let c0 = m[0];
    let c1 = m[1];
    let c2 = m[2];
    let det = dot(c0, cross(c1, c2));
    if (abs(det) < 1e-12) {
        return mat3x3<f32>(
            vec3<f32>(1.0, 0.0, 0.0),
            vec3<f32>(0.0, 1.0, 0.0),
            vec3<f32>(0.0, 0.0, 1.0),
        );
    }
    let inv_det = 1.0 / det;
    return mat3x3<f32>(
        cross(c1, c2) * inv_det,
        cross(c2, c0) * inv_det,
        cross(c0, c1) * inv_det,
    );
}

/// Upper 3x3 inverse-transpose of a 4x4 (cotangent rule for normals; handles non-uniform scale).
fn normal_matrix(m: mat4x4<f32>) -> mat3x3<f32> {
    return inverse_transpose_3x3(mat3_linear(m));
}

fn safe_normalize(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if (!(len_sq > 1e-12) || len_sq > 3.402823e38) {
        return fallback;
    }
    return v * inverseSqrt(len_sq);
}

@compute @workgroup_size(64)
fn skin_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= skin_dispatch.vertex_count) {
        return;
    }
    let src_pi = skin_dispatch.base_src_pos_e + i;
    let src_ni = skin_dispatch.base_src_nrm_e + i;
    let src_ti = skin_dispatch.base_src_tan_e + i;
    let dst_pi = skin_dispatch.base_dst_pos_e + i;
    let dst_ni = skin_dispatch.base_dst_nrm_e + i;
    let dst_ti = skin_dispatch.base_dst_tan_e + i;

    let p = src_pos[src_pi];
    let idx = bone_idx[i];
    let w = bone_weights[i];
    let p4 = vec4<f32>(p.xyz, 1.0);
    var acc = vec4<f32>(0.0);
    acc += w.x * (bone_matrices[idx.x] * p4);
    acc += w.y * (bone_matrices[idx.y] * p4);
    acc += w.z * (bone_matrices[idx.z] * p4);
    acc += w.w * (bone_matrices[idx.w] * p4);
    let ws = w.x + w.y + w.z + w.w;

    let nb = src_n[src_ni];
    let n_bind = vec3<f32>(nb.xyz);
    var acc_n = vec3<f32>(0.0);
    acc_n += w.x * (normal_matrix(bone_matrices[idx.x]) * n_bind);
    acc_n += w.y * (normal_matrix(bone_matrices[idx.y]) * n_bind);
    acc_n += w.z * (normal_matrix(bone_matrices[idx.z]) * n_bind);
    acc_n += w.w * (normal_matrix(bone_matrices[idx.w]) * n_bind);

    let tangent_enabled = (skin_dispatch.flags & SKIN_TANGENTS) != 0u;
    var tb = vec4<f32>(0.0);
    var t_bind = vec3<f32>(0.0);
    var acc_t = vec3<f32>(0.0);
    if (tangent_enabled) {
        tb = src_t[src_ti];
        t_bind = tb.xyz;
        acc_t += w.x * (mat3_linear(bone_matrices[idx.x]) * t_bind);
        acc_t += w.y * (mat3_linear(bone_matrices[idx.y]) * t_bind);
        acc_t += w.z * (mat3_linear(bone_matrices[idx.z]) * t_bind);
        acc_t += w.w * (mat3_linear(bone_matrices[idx.w]) * t_bind);
    }

    if (ws > 1e-6) {
        dst_pos[dst_pi] = vec4<f32>((acc / ws).xyz, p.w);
        let nn = safe_normalize(acc_n / ws, n_bind);
        dst_n[dst_ni] = vec4<f32>(nn, nb.w);
        if (tangent_enabled) {
            let tt = safe_normalize(acc_t / ws, t_bind);
            let sign = select(1.0, -1.0, tb.w < 0.0);
            dst_t[dst_ti] = vec4<f32>(tt, sign);
        }
    } else {
        dst_pos[dst_pi] = p;
        dst_n[dst_ni] = vec4<f32>(safe_normalize(n_bind, vec3<f32>(0.0, 0.0, 1.0)), nb.w);
        if (tangent_enabled) {
            let sign = select(1.0, -1.0, tb.w < 0.0);
            dst_t[dst_ti] = vec4<f32>(safe_normalize(t_bind, vec3<f32>(1.0, 0.0, 0.0)), sign);
        }
    }
}
