// Linear blend skinning (compute). Bind buffers expected to match layout produced by mesh preprocess.
// Bone palette entries are world_bone * unity_bindpose (inverse bind matrix per bone), built on CPU each frame.
// Positions, normals, and tangents use the upper linear part of M to match Unity's skinned mesh data.
//
// Source and destination buffers may be subranges of large arenas; [`SkinDispatchParams`] supplies element bases.

struct SkinDispatchParams {
    vertex_count: u32,
    base_bone_e: u32,
    base_src_pos_e: u32,
    base_src_nrm_e: u32,
    base_src_tan_e: u32,
    base_dst_pos_e: u32,
    base_dst_nrm_e: u32,
    base_dst_tan_e: u32,
    flags: u32,
    skin_weight_limit: u32,
    pad1: u32,
    pad2: u32,
}

struct BoneInfluence {
    index: u32,
    weight: f32,
}

struct SkinAccum {
    position: vec4<f32>,
    normal: vec4<f32>,
    tangent: vec3<f32>,
    bitangent: vec3<f32>,
    weight_sum: f32,
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
@group(0) @binding(10) var<storage, read> bone_influence_offsets: array<u32>;
@group(0) @binding(11) var<storage, read> bone_influences: array<BoneInfluence>;

fn mat3_linear(m: mat4x4<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(m[0].xyz, m[1].xyz, m[2].xyz);
}

fn safe_normalize(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if (!(len_sq > 1e-12) || len_sq > 3.402823e38) {
        return fallback;
    }
    return v * inverseSqrt(len_sq);
}

fn tangent_fallback_from_normal(n: vec3<f32>) -> vec3<f32> {
    let sign = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    return safe_normalize(
        vec3<f32>(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x),
        vec3<f32>(1.0, 0.0, 0.0),
    );
}

fn orthogonal_tangent(t: vec3<f32>, n: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let t_ortho = t - n * dot(t, n);
    let fallback_ortho = fallback - n * dot(fallback, n);
    return safe_normalize(
        t_ortho,
        safe_normalize(fallback_ortho, tangent_fallback_from_normal(n)),
    );
}

fn empty_skin_accum() -> SkinAccum {
    return SkinAccum(
        vec4<f32>(0.0),
        vec4<f32>(0.0),
        vec3<f32>(0.0),
        vec3<f32>(0.0),
        0.0,
    );
}

fn add_skin_influence(
    accum: SkinAccum,
    p4: vec4<f32>,
    nb: vec4<f32>,
    t_bind: vec3<f32>,
    b_bind: vec3<f32>,
    tangent_enabled: bool,
    bone_index: u32,
    weight: f32,
) -> SkinAccum {
    if (weight <= 0.0) {
        return accum;
    }
    let m = bone_matrices[bone_index];
    var next = accum;
    next.position += weight * (m * p4);
    next.normal += weight * (m * nb);
    if (tangent_enabled) {
        let linear = mat3_linear(m);
        next.tangent += weight * (linear * t_bind);
        next.bitangent += weight * (linear * b_bind);
    }
    next.weight_sum += weight;
    return next;
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
    let p4 = vec4<f32>(p.xyz, 1.0);

    let nb = src_n[src_ni];
    let n_bind = vec3<f32>(nb.xyz);

    let tangent_enabled = (skin_dispatch.flags & SKIN_TANGENTS) != 0u;
    var tb = vec4<f32>(0.0);
    var t_bind = vec3<f32>(0.0);
    var b_bind = vec3<f32>(0.0);
    if (tangent_enabled) {
        tb = src_t[src_ti];
        t_bind = tb.xyz;
        let bind_sign = select(1.0, -1.0, tb.w < 0.0);
        b_bind = cross(n_bind, t_bind) * bind_sign;
    }

    var accum = empty_skin_accum();
    if (skin_dispatch.skin_weight_limit == 0u) {
        let start = bone_influence_offsets[i];
        let end = bone_influence_offsets[i + 1u];
        var cursor = start;
        loop {
            if (cursor >= end) {
                break;
            }
            let influence = bone_influences[cursor];
            accum = add_skin_influence(
                accum,
                p4,
                nb,
                t_bind,
                b_bind,
                tangent_enabled,
                skin_dispatch.base_bone_e + influence.index,
                influence.weight,
            );
            cursor += 1u;
        }
    } else {
        let idx = bone_idx[i];
        let w = bone_weights[i];
        let limit = skin_dispatch.skin_weight_limit;
        if (limit >= 1u) {
            accum = add_skin_influence(
                accum,
                p4,
                nb,
                t_bind,
                b_bind,
                tangent_enabled,
                skin_dispatch.base_bone_e + idx.x,
                w.x,
            );
        }
        if (limit >= 2u) {
            accum = add_skin_influence(
                accum,
                p4,
                nb,
                t_bind,
                b_bind,
                tangent_enabled,
                skin_dispatch.base_bone_e + idx.y,
                w.y,
            );
        }
        if (limit >= 3u) {
            accum = add_skin_influence(
                accum,
                p4,
                nb,
                t_bind,
                b_bind,
                tangent_enabled,
                skin_dispatch.base_bone_e + idx.z,
                w.z,
            );
        }
        if (limit >= 4u) {
            accum = add_skin_influence(
                accum,
                p4,
                nb,
                t_bind,
                b_bind,
                tangent_enabled,
                skin_dispatch.base_bone_e + idx.w,
                w.w,
            );
        }
    }

    if (accum.weight_sum > 1e-6) {
        dst_pos[dst_pi] = vec4<f32>((accum.position / accum.weight_sum).xyz, p.w);
        let skinned_n = accum.normal / accum.weight_sum;
        dst_n[dst_ni] = skinned_n;
        if (tangent_enabled) {
            let n_fallback = safe_normalize(n_bind, vec3<f32>(0.0, 0.0, 1.0));
            let nn = safe_normalize(skinned_n.xyz, n_fallback);
            let tt = orthogonal_tangent(accum.tangent / accum.weight_sum, nn, t_bind);
            let bb = safe_normalize(accum.bitangent / accum.weight_sum, b_bind);
            let sign = select(1.0, -1.0, dot(cross(nn, tt), bb) < 0.0);
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
