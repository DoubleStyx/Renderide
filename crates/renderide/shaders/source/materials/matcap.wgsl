//! Matcap (`Shader "Matcap"`): tangent-space normal map, view-space normal matcap lookup.

// unity-shader-name: Matcap

#import renderide::globals as rg
#import renderide::per_draw as pd
#import renderide::normal_decode as nd
#import renderide::uv_utils as uvu

struct MatcapMaterial {
    _NormalMap_ST: vec4<f32>,
}

@group(1) @binding(0) var<uniform> mat: MatcapMaterial;
@group(1) @binding(1) var _MainTex: texture_2d<f32>;
@group(1) @binding(2) var _MainTex_sampler: sampler;
@group(1) @binding(3) var _NormalMap: texture_2d<f32>;
@group(1) @binding(4) var _NormalMap_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv_normal: vec2<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) world_t: vec3<f32>,
    @location(3) world_b: vec3<f32>,
    @location(4) view_x: vec3<f32>,
    @location(5) view_y: vec3<f32>,
}

struct ViewBasis {
    x: vec3<f32>,
    y: vec3<f32>,
}

fn safe_normalize(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if (len_sq <= 1e-12) {
        return fallback;
    }
    return v * inverseSqrt(len_sq);
}

fn view_projection_for_draw(d: pd::PerDrawUniforms, view_idx: u32) -> mat4x4<f32> {
#ifdef MULTIVIEW
    if (view_idx == 0u) {
        return d.view_proj_left;
    }
    return d.view_proj_right;
#else
    return d.view_proj_left;
#endif
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

    return ViewBasis(view_x, view_y);
}

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
    @location(4) tangent: vec4<f32>,
) -> VertexOutput {
    let d = pd::get_draw(instance_index);
#ifdef MULTIVIEW
    let view_layer = view_idx;
#else
    let view_layer = 0u;
#endif
    let world_p = d.model * vec4<f32>(pos.xyz, 1.0);
    let world_n = safe_normalize(d.normal_matrix * n.xyz, vec3<f32>(0.0, 1.0, 0.0));
    let tangent_world_raw = (d.model * vec4<f32>(tangent.xyz, 0.0)).xyz;
    let tangent_ortho = tangent_world_raw - world_n * dot(tangent_world_raw, world_n);
    let world_t = safe_normalize(tangent_ortho, vec3<f32>(1.0, 0.0, 0.0));
    let tangent_sign = select(1.0, -1.0, tangent.w < 0.0);
    let world_b = safe_normalize(cross(world_n, world_t) * tangent_sign, vec3<f32>(0.0, 0.0, 1.0));
    let vp = view_projection_for_draw(d, view_layer);
    let basis = view_basis_from_view_projection(vp);

    var out: VertexOutput;
    out.clip_pos = vp * world_p;
    out.uv_normal = uvu::apply_st(uv0, mat._NormalMap_ST);
    out.world_n = world_n;
    out.world_t = world_t;
    out.world_b = world_b;
    out.view_x = basis.x;
    out.view_y = basis.y;
    return out;
}

//#pass forward
@fragment
fn fs_main(
    @location(0) uv_normal: vec2<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) world_t: vec3<f32>,
    @location(3) world_b: vec3<f32>,
    @location(4) view_x: vec3<f32>,
    @location(5) view_y: vec3<f32>,
) -> @location(0) vec4<f32> {
    let normal_ts = nd::decode_ts_normal_with_placeholder(
        textureSample(_NormalMap, _NormalMap_sampler, uv_normal).xyz,
        1.0,
    );
    let tbn = mat3x3<f32>(
        safe_normalize(world_t, vec3<f32>(1.0, 0.0, 0.0)),
        safe_normalize(world_b, vec3<f32>(0.0, 0.0, 1.0)),
        safe_normalize(world_n, vec3<f32>(0.0, 1.0, 0.0)),
    );
    let n_world = safe_normalize(tbn * normal_ts, world_n);
    let n_view_xy = vec2<f32>(
        dot(safe_normalize(view_x, vec3<f32>(1.0, 0.0, 0.0)), n_world),
        dot(safe_normalize(view_y, vec3<f32>(0.0, 1.0, 0.0)), n_world),
    );
    let uv = n_view_xy * 0.5 + vec2<f32>(0.5);
    let col = textureSample(_MainTex, _MainTex_sampler, uvu::flip_v(uv));
    return rg::retain_globals_additive(col);
}
