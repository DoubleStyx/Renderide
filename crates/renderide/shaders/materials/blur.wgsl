//! Grab-pass blur filter (`Shader "Filters/Blur"`).
//!
//! Reads scene color via the grab pass and accumulates either a circular or Poisson-disc tap set
//! around the fragment's screen UV, optionally offset by view-space refraction (with normal-map
//! perturbation). The `SPREAD_TEX`, `REFRACT`/`REFRACT_NORMALMAP`, `RECTCLIP`, and `POISSON_DISC`
//! variant bits follow the material keyword groups.

//#texture_default _SpreadTex white
//#texture_default _NormalMap bump
//#mat_default _DepthDivisor float 1.0
//#mat_default _Iterations float 4.0
//#mat_default _RefractionStrength float 0.01
//#mat_default _Spread vec4 0.1 0.1 0.0 0.0

#import renderide::post::filter_math as fm
#import renderide::post::filter_vertex as fv
#import renderide::post::filter_common as fc
#import renderide::post::filter_refraction as fr
#import renderide::frame::grab_pass as gp
#import renderide::frame::scene_depth_sample as sds
#import renderide::core::uv as uvu
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

struct BlurVertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) primary_uv: vec2<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) view_n: vec3<f32>,
    @location(3) @interpolate(flat) view_layer: u32,
    @location(4) obj_xy: vec2<f32>,
    @location(5) view_t: vec4<f32>,
    @location(6) clip_w: f32,
}

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
) -> BlurVertexOutput {
#ifdef MULTIVIEW
    let layer = view_idx;
#else
    let layer = 0u;
#endif
    let inner = fv::vertex_main(instance_index, layer, pos, n, t, uv0);
    var out: BlurVertexOutput;
    out.clip_pos = inner.clip_pos;
    out.primary_uv = inner.primary_uv;
    out.world_pos = inner.world_pos;
    out.view_n = inner.view_n;
    out.view_layer = inner.view_layer;
    out.obj_xy = pos.xy;
    out.view_t = fr::view_tangent_for_draw(instance_index, layer, inner.world_t);
    out.clip_w = inner.clip_pos.w;
    return out;
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
        _NormalMap,
        _NormalMap_sampler,
    );
}

fn spread_modulation(uv0: vec2<f32>) -> vec2<f32> {
    if (!kw_SPREAD_TEX()) {
        return vec2<f32>(1.0);
    }
    return textureSample(_SpreadTex, _SpreadTex_sampler, uvu::apply_st(uv0, mat._SpreadTex_ST)).rg;
}

fn sample_blur(center_uv: vec2<f32>, spread: vec2<f32>, iterations: f32, view_layer: u32) -> vec4<f32> {
    var c = vec4<f32>(0.0);
    let use_poisson = kw_POISSON_DISC();
    let clamped_iterations = clamp(iterations, 1.0, 128.0);
    for (var i = 0u; i < 128u; i = i + 1u) {
        if (f32(i) >= clamped_iterations) {
            break;
        }
        let angle = (f32(i) / clamped_iterations) * fm::TAU;
        let offset = select(
            vec2<f32>(-cos(angle), sin(angle)) * spread,
            fm::poisson_blur_offset(i, spread),
            use_poisson,
        );
        c = c + gp::sample_scene_color(center_uv + offset, view_layer);
    }
    return c / clamped_iterations;
}

//#pass forward_filter
@fragment
fn fs_main(in: BlurVertexOutput) -> @location(0) vec4<f32> {
    fc::discard_rect_if_enabled(in.obj_xy, mat._Rect, kw_RECTCLIP());
    let screen_uv = fc::screen_uv(in.clip_pos);
    let center_uv = screen_uv - refract_offset(in.primary_uv, in.view_n, in.view_t, in.clip_w);
    let fade = sds::depth_fade_at_uv(center_uv, in.world_pos, in.view_layer, mat._DepthDivisor);
    let spread = mat._Spread.xy * spread_modulation(in.primary_uv) * fade;
    return fc::retain_globals(sample_blur(center_uv, spread, mat._Iterations, in.view_layer));
}
