//! Grab-pass refraction filter (`Shader "Filters/Refract"`).


#import renderide::filter_math as fm
#import renderide::filter_vertex as fv
#import renderide::globals as rg
#import renderide::grab_pass as gp
#import renderide::normal_decode as nd
#import renderide::scene_depth_sample as sds
#import renderide::uv_utils as uvu

struct FiltersRefractMaterial {
    _NormalMap_ST: vec4<f32>,
    _RefractionStrength: f32,
    _DepthBias: f32,
    _DepthDivisor: f32,
    _NORMALMAP: f32,
}

@group(1) @binding(0) var<uniform> mat: FiltersRefractMaterial;
@group(1) @binding(1) var _NormalMap: texture_2d<f32>;
@group(1) @binding(2) var _NormalMap_sampler: sampler;

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
) -> fv::VertexOutput {
#ifdef MULTIVIEW
    return fv::vertex_main(instance_index, view_idx, pos, n, uv0);
#else
    return fv::vertex_main(instance_index, 0u, pos, n, uv0);
#endif
}

fn refract_offset(uv0: vec2<f32>, view_n: vec3<f32>, clip_recip_w: f32) -> vec2<f32> {
    var n = normalize(view_n);
    if (uvu::kw_enabled(mat._NORMALMAP)) {
        let ts = nd::decode_ts_normal_with_placeholder_sample(
            textureSample(_NormalMap, _NormalMap_sampler, uvu::apply_st(uv0, mat._NormalMap_ST)),
            1.0,
        );
        n = normalize(vec3<f32>(n.xy + ts.xy, n.z));
    }
    return n.xy * clip_recip_w * mat._RefractionStrength;
}

fn refracted_screen_uv(
    screen_uv: vec2<f32>,
    uv0: vec2<f32>,
    view_n: vec3<f32>,
    frag_pos: vec4<f32>,
    world_pos: vec3<f32>,
    view_layer: u32,
) -> vec2<f32> {
    let fade = sds::depth_fade(frag_pos, world_pos, view_layer, mat._DepthDivisor);
    let offset = refract_offset(uv0, view_n, frag_pos.w) * fade * fm::screen_vignette(screen_uv);
    let grab_uv = screen_uv - offset;
    let sampled_depth = sds::scene_linear_depth_at_uv(grab_uv, view_layer);
    let fragment_depth = sds::fragment_linear_depth(world_pos, view_layer);
    if (sampled_depth > fragment_depth + mat._DepthBias) {
        return screen_uv;
    }
    return grab_uv;
}

//#pass forward
@fragment
fn fs_main(
    @builtin(position) frag_pos: vec4<f32>,
    @location(0) uv0: vec2<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) world_n: vec3<f32>,
    @location(3) @interpolate(flat) view_layer: u32,
    @location(4) view_n: vec3<f32>,
) -> @location(0) vec4<f32> {
    let screen_uv = gp::frag_screen_uv(frag_pos);
    let color = gp::sample_scene_color(
        refracted_screen_uv(screen_uv, uv0, view_n, frag_pos, world_pos, view_layer),
        view_layer,
    );
    return rg::retain_globals_additive(color);
}
