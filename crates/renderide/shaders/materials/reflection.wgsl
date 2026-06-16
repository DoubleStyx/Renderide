//! Unity `Shader "Reflection"`: samples a host-provided 2D reflection texture in screen space,
//! optionally distorted by a tangent-space normal map. **Not a grab pass** -- `_ReflectionTex` is a
//! regular `sampler2D`, populated by the host with whatever reflection RT (planar, cubemap-projected,
//! etc.) is available. VR camera portals render left/right mirror eyes into side-by-side halves of
//! the same host render texture, matching Unity's non-single-pass fallback.
//!
//! Froox variant bits populate `_RenderideVariantBits`; this shader decodes Reflection's
//! shader-specific keyword bits locally. `_OFFSET_TEXTURE` is reserved in the bit table so the
//! serialized layout stays stable, but the shader body never consults it.

//#render_queue AlphaTest+200
//#texture_default _ReflectionTex white
//#texture_default _NormalMap bump
//#mat_default _Color vec4 1.0 1.0 1.0 1.0
//#mat_default _Cutoff float 0.5

#import renderide::billboard::vertex as bv
#import renderide::frame::globals as rg
#import renderide::draw::per_draw as pd
#import renderide::material::alpha as ma
#import renderide::material::variant_bits as vb
#import renderide::mesh::vertex as mv
#import renderide::core::texture_sampling as ts
#import renderide::core::uv as uvu
#import renderide::core::normal_decode as nd

struct ReflectionMaterial {
    _Color: vec4<f32>,
    _NormalMap_ST: vec4<f32>,
    _Cutoff: f32,
    _Distort: f32,
    _NormalMap_LodBias: f32,
    _RenderideVariantBits: u32,
    _pad0: vec2<u32>,
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
    @location(2) @interpolate(flat) view_layer: u32,
}

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
    @builtin(vertex_index) vertex_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv: vec2<f32>,
    @location(4) t: vec4<f32>,
    @location(5) uv1: vec2<f32>,
) -> VertexOutput {
#ifdef MULTIVIEW
    let view_layer = view_idx;
#else
    let view_layer = 0u;
#endif
    let d = pd::get_draw(instance_index);
    let vp = mv::select_view_proj(d, view_layer);

    var world_p: vec4<f32>;
    if (bv::kw_RENDER_BUFFER(mat._RenderideVariantBits)) {
        let render_billboard_vertex = bv::render_buffer_billboard_vertex(
            d, view_layer, pos, vertex_index, n, t, uv1,
        );
        world_p = render_billboard_vertex.world_pos;
    } else {
        world_p = mv::world_position(d, pos);
    }
    let clip = vp * world_p;
#ifdef MULTIVIEW
    let screen_y = (clip.y + clip.w) * 0.5;
#else
    let screen_y = (clip.w - clip.y) * 0.5;
#endif
    var out: VertexOutput;
    out.clip_pos = clip;
    // Unity ComputeNonStereoScreenPos with the projection sign matching the active target path.
    out.screen_uv = vec3<f32>(
        (clip.x + clip.w) * 0.5,
        screen_y,
        clip.w,
    );
    out.uv = uvu::apply_st(uv, mat._NormalMap_ST);
    out.view_layer = view_layer;
    return out;
}

fn projected_reflection_uv(screen_uv: vec3<f32>) -> vec2<f32> {
    return screen_uv.xy / max(screen_uv.z, 1e-4);
}

fn reflection_texture_uv(screen: vec2<f32>, view_layer: u32) -> vec2<f32> {
    var uv = screen;
#ifdef MULTIVIEW
    uv.x = uv.x * 0.5 + f32(view_layer) * 0.5;
#endif
    return uv;
}

//#pass type=forward blend=material_filter a2c=cutout
@fragment
fn fs_main(
    @location(0) screen_uv: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    var screen = projected_reflection_uv(screen_uv);
    if (kw_NORMALMAP()) {
        let bump = nd::decode_ts_normal_with_placeholder_sample(
            ts::sample_tex_2d(_NormalMap, _NormalMap_sampler, uv, mat._NormalMap_LodBias),
            1.0,
        );
        screen = screen + bump.xy * mat._Distort;
    }
    screen = reflection_texture_uv(screen, view_layer);
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
