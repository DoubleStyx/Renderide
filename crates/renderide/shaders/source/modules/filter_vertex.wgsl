//! Shared vertex payload for screen-space filter materials.

#define_import_path renderide::filter_vertex

#import renderide::per_draw as pd

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) primary_uv: vec2<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) world_n: vec3<f32>,
    @location(3) @interpolate(flat) view_layer: u32,
    @location(4) view_n: vec3<f32>,
}

struct ViewBasis {
    x: vec3<f32>,
    y: vec3<f32>,
    z: vec3<f32>,
}

fn select_view_proj(d: pd::PerDrawUniforms, view_idx: u32) -> mat4x4<f32> {
    if (view_idx == 0u) {
        return d.view_proj_left;
    }
    return d.view_proj_right;
}

fn safe_normalize(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let len2 = dot(v, v);
    if (len2 <= 1e-12) {
        return fallback;
    }
    return v * inverseSqrt(len2);
}

fn projection_row_xyz(m: mat4x4<f32>, row: u32) -> vec3<f32> {
    return vec3<f32>(m[0u][row], m[1u][row], m[2u][row]);
}

fn view_basis_from_view_projection(vp: mat4x4<f32>) -> ViewBasis {
    let clip_x = projection_row_xyz(vp, 0u);
    let clip_y = projection_row_xyz(vp, 1u);
    let clip_w = projection_row_xyz(vp, 3u);

    let cross_fallback = safe_normalize(cross(clip_x, clip_y), vec3<f32>(0.0, 0.0, 1.0));
    let view_z = safe_normalize(-clip_w, cross_fallback);
    let view_x = safe_normalize(clip_x - view_z * dot(clip_x, view_z), vec3<f32>(1.0, 0.0, 0.0));
    let view_y_raw = clip_y - view_z * dot(clip_y, view_z);
    let view_y = safe_normalize(
        view_y_raw - view_x * dot(view_y_raw, view_x),
        safe_normalize(cross(view_z, view_x), vec3<f32>(0.0, 1.0, 0.0)),
    );

    return ViewBasis(view_x, view_y, view_z);
}

fn world_to_view_normal(world_n: vec3<f32>, vp: mat4x4<f32>) -> vec3<f32> {
    let basis = view_basis_from_view_projection(vp);
    return safe_normalize(vec3<f32>(
        dot(world_n, basis.x),
        dot(world_n, basis.y),
        dot(world_n, basis.z),
    ), vec3<f32>(0.0, 0.0, 1.0));
}

fn vertex_main(
    instance_index: u32,
    view_idx: u32,
    pos: vec4<f32>,
    n: vec4<f32>,
    primary_uv: vec2<f32>,
) -> VertexOutput {
    let d = pd::get_draw(instance_index);
    let world_p = d.model * vec4<f32>(pos.xyz, 1.0);
    let vp = select_view_proj(d, view_idx);
    let world_n = safe_normalize(d.normal_matrix * n.xyz, vec3<f32>(0.0, 1.0, 0.0));
    var out: VertexOutput;
    out.clip_pos = vp * world_p;
    out.primary_uv = primary_uv;
    out.world_pos = world_p.xyz;
    out.world_n = world_n;
    out.view_layer = view_idx;
    out.view_n = world_to_view_normal(world_n, vp);
    return out;
}
