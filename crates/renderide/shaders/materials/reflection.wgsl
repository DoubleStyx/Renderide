//! Unity `Shader "Reflection"`: samples a host-provided 2D reflection texture in screen space,
//! optionally distorted by a tangent-space normal map. **Not a grab pass** -- `_ReflectionTex` is a
//! regular `sampler2D`, populated by the host with whatever reflection RT (planar, cubemap-projected,
//! etc.) is available. Multi-view eye separation is handled by the renderer's per-eye render pass,
//! not the side-by-side `eyeIndex` texture-coordinate offset Unity needs for single-pass stereo.
//!
//! Froox variant bits populate `_RenderideVariantBits`; this shader decodes Reflection's
//! shader-specific keyword bits locally. `_OFFSET_TEXTURE` is reserved in the bit table so the
//! serialized layout stays stable, but the shader body never consults it.

//#texture_default _ReflectionTex white
//#texture_default _NormalMap bump
//#mat_default _Color vec4 1.0 1.0 1.0 1.0

#import renderide::frame::globals as rg
#import renderide::draw::per_draw as pd
#import renderide::material::alpha as ma
#import renderide::material::variant_bits as vb
#import renderide::mesh::vertex as mv
#import renderide::core::uv as uvu
#import renderide::core::normal_decode as nd

struct ReflectionMaterial {
    _Color: vec4<f32>,
    _NormalMap_ST: vec4<f32>,
    _Cutoff: f32,
    _Distort: f32,
    _RenderideVariantBits: u32,
    _pad0: u32,
}

const REFLECTION_KW_ALPHATEST: u32 = 1u << 0u;
const REFLECTION_KW_COLOR: u32 = 1u << 1u;
const REFLECTION_KW_MUL_ALPHA_INTENSITY: u32 = 1u << 2u;
const REFLECTION_KW_MUL_RGB_BY_ALPHA: u32 = 1u << 3u;
const REFLECTION_KW_NORMALMAP: u32 = 1u << 4u;
const REFLECTION_KW_OFFSET_TEXTURE: u32 = 1u << 5u;

@group(1) @binding(0) var<uniform> mat: ReflectionMaterial;
@group(1) @binding(1) var _ReflectionTex: texture_2d<f32>;
@group(1) @binding(2) var _ReflectionTex_sampler: sampler;
@group(1) @binding(3) var _NormalMap: texture_2d<f32>;
@group(1) @binding(4) var _NormalMap_sampler: sampler;

fn reflection_kw(mask: u32) -> bool {
    return vb::enabled(mat._RenderideVariantBits, mask);
}

fn kw_ALPHATEST() -> bool { return reflection_kw(REFLECTION_KW_ALPHATEST); }
fn kw_COLOR() -> bool { return reflection_kw(REFLECTION_KW_COLOR); }
fn kw_MUL_ALPHA_INTENSITY() -> bool { return reflection_kw(REFLECTION_KW_MUL_ALPHA_INTENSITY); }
fn kw_MUL_RGB_BY_ALPHA() -> bool { return reflection_kw(REFLECTION_KW_MUL_RGB_BY_ALPHA); }
fn kw_NORMALMAP() -> bool { return reflection_kw(REFLECTION_KW_NORMALMAP); }

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) screen_uv: vec3<f32>,
    @location(1) uv: vec2<f32>,
}

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) _n: vec4<f32>,
    @location(2) uv: vec2<f32>,
) -> VertexOutput {
    let d = pd::get_draw(instance_index);
    let world_p = mv::world_position(d, pos);
#ifdef MULTIVIEW
    let vp = mv::select_view_proj(d, view_idx);
#else
    let vp = mv::select_view_proj(d, 0u);
#endif
    let clip = vp * world_p;
    var out: VertexOutput;
    out.clip_pos = clip;
    // Equivalent of Unity's ComputeNonStereoScreenPos: ((clip.xy * vec2(1, -1) + clip.w) * 0.5, w)
    // packed into xy/z so the fragment can do uv/w to get [0..1] screen UV.
    out.screen_uv = vec3<f32>(
        (clip.x + clip.w) * 0.5,
        (clip.w - clip.y) * 0.5,
        clip.w,
    );
    out.uv = uvu::apply_st(uv, mat._NormalMap_ST);
    return out;
}

//#pass forward
@fragment
fn fs_main(
    @location(0) screen_uv: vec3<f32>,
    @location(1) uv: vec2<f32>,
) -> @location(0) vec4<f32> {
    var screen = screen_uv.xy / max(screen_uv.z, 1e-4);
    if (kw_NORMALMAP()) {
        let bump = nd::decode_ts_normal_with_placeholder_sample(
            textureSample(_NormalMap, _NormalMap_sampler, uv),
            1.0,
        );
        screen = screen + bump.xy * mat._Distort;
    }
    var col = textureSample(_ReflectionTex, _ReflectionTex_sampler, screen);
    if (kw_COLOR()) {
        col = col * mat._Color;
    }
    if (kw_ALPHATEST() && col.a < mat._Cutoff) {
        discard;
    }
    if (kw_MUL_RGB_BY_ALPHA()) {
        col = vec4<f32>(ma::apply_premultiply(col.rgb, col.a, true), col.a);
    }
    if (kw_MUL_ALPHA_INTENSITY()) {
        col.a = ma::alpha_intensity(col.a, col.rgb);
    }
    return rg::retain_globals_additive(col);
}
