//! Grab-pass refraction filter (`Shader "Filters/Refract"`).
//!
//! Froox variant bits populate `_RenderideVariantBits`; this shader decodes Refract's
//! shader-specific keyword bits locally.

//#texture_default _NormalMap bump
//#mat_default _DepthBias float 0.01
//#mat_default _RefractionStrength float 0.01

#import renderide::post::filter_vertex as fv
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
}

const REFRACT_KW_NORMALMAP: u32 = 1u << 0u;
const REFRACT_KW_RECTCLIP: u32 = 1u << 1u;

@group(1) @binding(0) var<uniform> mat: FiltersRefractMaterial;
@group(1) @binding(1) var _NormalMap: texture_2d<f32>;
@group(1) @binding(2) var _NormalMap_sampler: sampler;

struct RefractVertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) primary_uv: vec2<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) world_n: vec3<f32>,
    @location(3) @interpolate(flat) view_layer: u32,
    @location(4) view_n: vec3<f32>,
    @location(5) obj_xy: vec2<f32>,
    @location(6) view_t: vec4<f32>,
    @location(7) clip_w: f32,
}

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
) -> RefractVertexOutput {
#ifdef MULTIVIEW
    let layer = view_idx;
#else
    let layer = 0u;
#endif
    let inner = fv::vertex_main(instance_index, layer, pos, n, t, uv0);
    var out: RefractVertexOutput;
    out.clip_pos = inner.clip_pos;
    out.primary_uv = inner.primary_uv;
    out.world_pos = inner.world_pos;
    out.world_n = inner.world_n;
    out.view_layer = inner.view_layer;
    out.view_n = inner.view_n;
    out.obj_xy = pos.xy;
    out.view_t = fr::view_tangent_for_draw(instance_index, layer, inner.world_t);
    out.clip_w = inner.clip_pos.w;
    return out;
}

//#pass forward_filter
@fragment
fn fs_main(in: RefractVertexOutput) -> @location(0) vec4<f32> {
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
        _NormalMap,
        _NormalMap_sampler,
        mat._RefractionStrength,
        mat._DepthBias,
        mat._DepthDivisor,
    );
    let color = gp::sample_scene_color(refracted_uv, in.view_layer);
    return fc::retain_globals(color);
}
