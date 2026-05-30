//! Grab-pass refraction filter (`Shader "Filters/Refract"`).
//!
//! Froox variant bits populate `_RenderideVariantBits`; this shader decodes Refract's
//! shader-specific keyword bits locally.

//#render_queue Transparent+500
//#texture_default _NormalMap bump
//#mat_default _DepthBias float 0.01
//#mat_default _NormalMap_LodBias float 0.0
//#mat_default _RefractionStrength float 0.01

#import renderide::post::filter_common as fc
#import renderide::post::filter_refraction as fr
#import renderide::frame::grab_pass as gp
#import renderide::material::variant_bits as vb

struct FiltersRefractMaterial {
    _NormalMap_ST: vec4<f32>,
    _Rect: vec4<f32>,
    _RefractionStrength: f32,
    _DepthBias: f32,
    _DepthDivisor: f32,
    _RenderideVariantBits: u32,
    _NormalMap_LodBias: f32,
    _pad0: vec3<f32>,
}

const REFRACT_KW_NORMALMAP: u32 = 1u << 0u;
const REFRACT_KW_RECTCLIP: u32 = 1u << 1u;

@group(1) @binding(0) var<uniform> mat: FiltersRefractMaterial;
@group(1) @binding(1) var _NormalMap: texture_2d<f32>;
@group(1) @binding(2) var _NormalMap_sampler: sampler;

fn refract_kw(mask: u32) -> bool {
    return vb::enabled(mat._RenderideVariantBits, mask);
}

fn kw_RECTCLIP() -> bool {
    return refract_kw(REFRACT_KW_RECTCLIP);
}

fn kw_NORMALMAP() -> bool {
    return refract_kw(REFRACT_KW_NORMALMAP);
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

//#pass type=forward name=forward_filter blend=material_filter
@fragment
fn fs_main(in: fr::VertexOutput) -> @location(0) vec4<f32> {
    fc::discard_rect_if_enabled(in.obj_xy, mat._Rect, kw_RECTCLIP());
    let screen_uv = fc::screen_uv(in.clip_pos);
    let refracted_uv = fr::guarded_refracted_screen_uv(
        screen_uv,
        in.primary_uv,
        in.view_n,
        in.view_t,
        in.clip_pos,
        in.clip_w,
        in.world_pos,
        in.view_layer,
        kw_NORMALMAP(),
        mat._NormalMap_ST,
        mat._NormalMap_LodBias,
        _NormalMap,
        _NormalMap_sampler,
        mat._RefractionStrength,
        mat._DepthBias,
        mat._DepthDivisor,
    );
    let color = gp::sample_scene_color(refracted_uv, in.view_layer);
    return fc::retain_globals(color);
}
