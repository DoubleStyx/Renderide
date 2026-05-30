//! Grab-pass blur filter (`Shader "Filters/Blur"`).
//!
//! Reads scene color via the grab pass and accumulates either a circular or Poisson-disc tap set
//! around the fragment's screen UV, optionally offset by view-space refraction (with normal-map
//! perturbation). The `SPREAD_TEX`, `REFRACT`/`REFRACT_NORMALMAP`, `RECTCLIP`, and `POISSON_DISC`
//! variant bits follow the material keyword groups.

//#render_queue Transparent+500
//#texture_default _SpreadTex white
//#texture_default _NormalMap bump
//#mat_default _DepthDivisor float 1.0
//#mat_default _Iterations float 4.0
//#mat_default _RefractionStrength float 0.01
//#mat_default _Spread vec4 0.1 0.1 0.0 0.0
//#mat_default _NormalMap_LodBias float 0.0
//#mat_default _SpreadTex_LodBias float 0.0

#import renderide::core::texture_sampling as ts
#import renderide::core::uv as uvu
#import renderide::post::filter_math as fm
#import renderide::post::poisson_blur as pb
#import renderide::post::filter_common as fc
#import renderide::post::filter_refraction as fr
#import renderide::frame::grab_pass as gp
#import renderide::frame::scene_depth_sample as sds
#import renderide::material::variant_bits as vb

struct FiltersBlurMaterial {
    _Spread: vec4<f32>,
    _SpreadTex_ST: vec4<f32>,
    _NormalMap_ST: vec4<f32>,
    _Rect: vec4<f32>,
    _Iterations: f32,
    _RefractionStrength: f32,
    _DepthDivisor: f32,
    _RenderideVariantBits: u32,
    _NormalMap_LodBias: f32,
    _SpreadTex_LodBias: f32,
    _pad0: vec2<f32>,
}

const BLUR_KW_POISSON_DISC: u32 = 1u << 0u;
const BLUR_KW_RECTCLIP: u32 = 1u << 1u;
const BLUR_KW_REFRACT: u32 = 1u << 2u;
const BLUR_KW_REFRACT_NORMALMAP: u32 = 1u << 3u;
const BLUR_KW_SPREAD_TEX: u32 = 1u << 4u;

@group(1) @binding(0) var<uniform> mat: FiltersBlurMaterial;
@group(1) @binding(1) var _SpreadTex: texture_2d<f32>;
@group(1) @binding(2) var _SpreadTex_sampler: sampler;
@group(1) @binding(3) var _NormalMap: texture_2d<f32>;
@group(1) @binding(4) var _NormalMap_sampler: sampler;

fn blur_kw(mask: u32) -> bool {
    return vb::enabled(mat._RenderideVariantBits, mask);
}

fn kw_POISSON_DISC() -> bool {
    return blur_kw(BLUR_KW_POISSON_DISC);
}

fn kw_RECTCLIP() -> bool {
    return blur_kw(BLUR_KW_RECTCLIP);
}

fn kw_REFRACT() -> bool {
    return blur_kw(BLUR_KW_REFRACT);
}

fn kw_REFRACT_NORMALMAP() -> bool {
    return blur_kw(BLUR_KW_REFRACT_NORMALMAP);
}

fn kw_SPREAD_TEX() -> bool {
    return blur_kw(BLUR_KW_SPREAD_TEX);
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
    @location(4) t: vec4<f32>,
) -> fr::VertexOutput {
#ifdef MULTIVIEW
    return fr::vertex_main(instance_index, view_idx, pos, n, t, uv0);
#else
    return fr::vertex_main(instance_index, 0u, pos, n, t, uv0);
#endif
}

fn refraction_enabled() -> bool {
    return kw_REFRACT() || kw_REFRACT_NORMALMAP();
}

fn refract_offset(uv0: vec2<f32>, view_n: vec3<f32>, view_t: vec4<f32>, clip_w: f32) -> vec2<f32> {
    return fr::normal_offset(
        refraction_enabled(),
        kw_REFRACT_NORMALMAP(),
        uv0,
        view_n,
        view_t,
        clip_w,
        mat._RefractionStrength,
        mat._NormalMap_ST,
        mat._NormalMap_LodBias,
        _NormalMap,
        _NormalMap_sampler,
    );
}

fn spread_modulation(uv0: vec2<f32>) -> vec2<f32> {
    if (!kw_SPREAD_TEX()) {
        return vec2<f32>(1.0);
    }
    return ts::sample_tex_2d(_SpreadTex, _SpreadTex_sampler, uvu::apply_st(uv0, mat._SpreadTex_ST), mat._SpreadTex_LodBias).rg;
}

fn sample_circular_blur(center_uv: vec2<f32>, spread: vec2<f32>, iterations: f32, view_layer: u32) -> vec4<f32> {
    var c = vec4<f32>(0.0);
    let clamped_iterations = clamp(iterations, 1.0, 128.0);
    for (var i = 0u; i < 128u; i = i + 1u) {
        if (f32(i) >= clamped_iterations) {
            break;
        }
        let angle = (f32(i) / clamped_iterations) * fm::TAU;
        let offset = vec2<f32>(-cos(angle), sin(angle)) * spread;
        c = c + gp::sample_scene_color(center_uv + offset, view_layer);
    }
    return c / clamped_iterations;
}

fn sample_blur(center_uv: vec2<f32>, spread: vec2<f32>, iterations: f32, view_layer: u32) -> vec4<f32> {
    if (kw_POISSON_DISC()) {
        return pb::sample_poisson_blur(center_uv, spread, iterations, view_layer);
    }
    return sample_circular_blur(center_uv, spread, iterations, view_layer);
}

//#pass type=forward name=forward_filter blend=material_filter
@fragment
fn fs_main(in: fr::VertexOutput) -> @location(0) vec4<f32> {
    fc::discard_rect_if_enabled(in.obj_xy, mat._Rect, kw_RECTCLIP());
    let screen_uv = fc::screen_uv(in.clip_pos);
    let center_uv = screen_uv - refract_offset(in.primary_uv, in.view_n, in.view_t, in.clip_w);
    let fade = sds::depth_fade_at_uv(center_uv, in.world_pos, in.view_layer, mat._DepthDivisor);
    let spread = mat._Spread.xy * spread_modulation(in.primary_uv) * fade;
    return fc::retain_globals(sample_blur(center_uv, spread, mat._Iterations, in.view_layer));
}
