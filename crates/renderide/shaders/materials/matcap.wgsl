//! Matcap (`Shader "Matcap"`): tangent-space normal map, view-space normal matcap lookup.


#import renderide::globals as rg
#import renderide::per_draw as pd
#import renderide::math as rmath
#import renderide::mesh::vertex as mv
#import renderide::normal_decode as nd
#import renderide::uv_utils as uvu
#import renderide::view_basis as vb

struct MatcapMaterial {
    _NORMALMAP: f32,
    _pad0: f32,
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
    let world_p = mv::world_position(d, pos);
    let world_n = rmath::safe_normalize(d.normal_matrix * n.xyz, vec3<f32>(0.0, 1.0, 0.0));
    let tangent_world_raw = mv::model_vector(d, tangent.xyz);
    let tangent_ortho = tangent_world_raw - world_n * dot(tangent_world_raw, world_n);
    let world_t = rmath::safe_normalize(tangent_ortho, vec3<f32>(1.0, 0.0, 0.0));
    let tangent_sign = select(1.0, -1.0, tangent.w < 0.0);
    let world_b = rmath::safe_normalize(cross(world_n, world_t) * tangent_sign, vec3<f32>(0.0, 0.0, 1.0));
    let vp = mv::select_view_proj(d, view_layer);
    let basis = vb::from_view_projection(vp);

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
    var normal_ts = vec3<f32>(0.0, 0.0, 1.0);
    if (uvu::kw_enabled(mat._NORMALMAP)) {
        normal_ts = nd::decode_ts_normal_with_placeholder_sample(
            textureSample(_NormalMap, _NormalMap_sampler, uv_normal),
            1.0,
        );
    }
    let tbn = mat3x3<f32>(
        rmath::safe_normalize(world_t, vec3<f32>(1.0, 0.0, 0.0)),
        rmath::safe_normalize(world_b, vec3<f32>(0.0, 0.0, 1.0)),
        rmath::safe_normalize(world_n, vec3<f32>(0.0, 1.0, 0.0)),
    );
    let n_world = rmath::safe_normalize(tbn * normal_ts, world_n);
    let n_view_xy = vec2<f32>(
        dot(rmath::safe_normalize(view_x, vec3<f32>(1.0, 0.0, 0.0)), n_world),
        dot(rmath::safe_normalize(view_y, vec3<f32>(0.0, 1.0, 0.0)), n_world),
    );
    let uv = n_view_xy * 0.5 + vec2<f32>(0.5);
    let col = textureSample(_MainTex, _MainTex_sampler, uv);
    return rg::retain_globals_additive(col);
}
