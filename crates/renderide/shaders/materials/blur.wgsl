//! Grab-pass blur filter (`Shader "Filters/Blur"`).


#import renderide::filter_math as fm
#import renderide::filter_vertex as fv
#import renderide::globals as rg
#import renderide::grab_pass as gp
#import renderide::normal_decode as nd
#import renderide::scene_depth_sample as sds
#import renderide::uv_utils as uvu
#import renderide::pbs::normal as pnorm

struct FiltersBlurMaterial {
    _Spread: vec4<f32>,
    _SpreadTex_ST: vec4<f32>,
    _NormalMap_ST: vec4<f32>,
    _Iterations: f32,
    _RefractionStrength: f32,
    _DepthDivisor: f32,
    REFRACT: f32,
    REFRACT_NORMALMAP: f32,
    POISSON_DISC: f32,
}

@group(1) @binding(0) var<uniform> mat: FiltersBlurMaterial;
@group(1) @binding(1) var _SpreadTex: texture_2d<f32>;
@group(1) @binding(2) var _SpreadTex_sampler: sampler;
@group(1) @binding(3) var _NormalMap: texture_2d<f32>;
@group(1) @binding(4) var _NormalMap_sampler: sampler;

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
    @location(4) t: vec4<f32>,
) -> fv::VertexOutput {
#ifdef MULTIVIEW
    return fv::vertex_main(instance_index, view_idx, pos, n, t, uv0);
#else
    return fv::vertex_main(instance_index, 0u, pos, n, t, uv0);
#endif
}

fn refraction_enabled() -> bool {
    return uvu::kw_enabled(mat.REFRACT) || uvu::kw_enabled(mat.REFRACT_NORMALMAP);
}

fn refract_offset(uv0: vec2<f32>, view_n: vec3<f32>, clip_recip_w: f32) -> vec2<f32> {
    if (!refraction_enabled()) {
        return vec2<f32>(0.0);
    }
    var n = normalize(view_n);
    if (uvu::kw_enabled(mat.REFRACT_NORMALMAP)) {
        let ts = nd::decode_ts_normal_with_placeholder_sample(
            textureSample(_NormalMap, _NormalMap_sampler, uvu::apply_st(uv0, mat._NormalMap_ST)),
            1.0,
        );
        n = normalize(vec3<f32>(n.xy + ts.xy, n.z));
    }
    return n.xy * clip_recip_w * mat._RefractionStrength;
}

fn sample_blur(center_uv: vec2<f32>, spread: vec2<f32>, iterations: u32, view_layer: u32) -> vec4<f32> {
    var c = vec4<f32>(0.0);
    let use_poisson = uvu::kw_enabled(mat.POISSON_DISC);
    for (var i = 0u; i < 128u; i = i + 1u) {
        if (i >= iterations) {
            break;
        }
        let offset = select(
            fm::circular_blur_offset(i, iterations, spread),
            fm::poisson_blur_offset(i, spread),
            use_poisson,
        );
        c = c + gp::sample_scene_color(center_uv + offset, view_layer);
    }
    return c / max(f32(iterations), 1.0);
}

//#pass forward
@fragment
fn fs_main(
    @builtin(position) frag_pos: vec4<f32>,
    @location(0) uv0: vec2<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) world_n: vec3<f32>,
    @location(3) world_t: vec4<f32>,
    @location(4) @interpolate(flat) view_layer: u32,
    @location(5) view_n: vec3<f32>,
) -> @location(0) vec4<f32> {
    let screen_uv = gp::frag_screen_uv(frag_pos);
    let fade = sds::depth_fade(frag_pos, world_pos, view_layer, mat._DepthDivisor);
    let spread_tex = textureSample(_SpreadTex, _SpreadTex_sampler, uvu::apply_st(uv0, mat._SpreadTex_ST)).rg;
    let spread = mat._Spread.xy * spread_tex * fade;
    let center_uv = screen_uv - refract_offset(uv0, view_n, frag_pos.w) * fade;
    let iterations = u32(clamp(mat._Iterations, 1.0, 128.0));
    return rg::retain_globals_additive(sample_blur(center_uv, spread, iterations, view_layer));
}
