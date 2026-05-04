//! World-mesh normal prepass for GTAO.
//!
//! Writes smooth vertex normals in the view-space convention consumed by `gtao_main`.

#import renderide::math as rmath
#import renderide::view_basis as vb

struct PerDrawUniforms {
    view_proj_left: mat4x4<f32>,
    view_proj_right: mat4x4<f32>,
    model: mat4x4<f32>,
    normal_matrix: mat3x3<f32>,
    _pad: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> instances: array<PerDrawUniforms>;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) view_n: vec3<f32>,
}

fn select_view_proj(draw: PerDrawUniforms, view_idx: u32) -> mat4x4<f32> {
    if (view_idx == 0u) {
        return draw.view_proj_left;
    }
    return draw.view_proj_right;
}

fn gtao_view_normal_from_world(world_n: vec3<f32>, vp: mat4x4<f32>) -> vec3<f32> {
    let basis = vb::from_view_projection(vp);
    return rmath::safe_normalize(vec3<f32>(
        dot(world_n, basis.x),
        dot(world_n, basis.y),
        -dot(world_n, basis.z),
    ), vec3<f32>(0.0, 0.0, -1.0));
}

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
) -> VertexOutput {
    let draw = instances[instance_index];
#ifdef MULTIVIEW
    let view_layer = view_idx;
#else
    let view_layer = 0u;
#endif
    let vp = select_view_proj(draw, view_layer);
    let world_p = draw.model * vec4<f32>(pos.xyz, 1.0);
    let world_n = rmath::safe_normalize(draw.normal_matrix * n.xyz, vec3<f32>(0.0, 1.0, 0.0));

    var out: VertexOutput;
    out.clip_pos = vp * world_p;
    out.view_n = gtao_view_normal_from_world(world_n, vp);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(rmath::safe_normalize(in.view_n, vec3<f32>(0.0, 0.0, -1.0)), 1.0);
}
