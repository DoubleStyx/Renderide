//! Unity ProceduralSkybox asset (`Shader "ProceduralSky"`): analytic sky material with
//! Rayleigh+Mie scattering and three sun-disk modes (NONE / SIMPLE / HIGH_QUALITY).
//!
//! The renderer pipeline operates entirely in linear color space, so this shader implements
//! the linear branch only; the gamma-space branch and
//! `SKYBOX_COLOR_IN_TARGET_COLOR_SPACE` short-circuit are intentionally omitted.
//!
//! Froox variant bits populate `_RenderideVariantBits`; this shader decodes ProceduralSky's
//! shader-specific keyword bits locally. `UNITY_COLORSPACE_GAMMA` is reserved in the bit
//! table but never consulted because the renderer is linear-only. The sun-disk group has
//! no `_` placeholder, so the high-quality keyword is the zero-bit default.

//#mat_default _GroundColor vec4 0.369 0.349 0.341 1.0
//#mat_default _SkyTint vec4 0.5 0.5 0.5 1.0
//#mat_default _SunColor vec4 1.0 1.0 1.0 1.0
//#mat_default _SunDirection vec4 0.577 0.577 0.577 0.0

#import renderide::frame::globals as rg
#import renderide::draw::per_draw as pd
#import renderide::skybox::procedural as ps
#import renderide::mesh::vertex as mv
#import renderide::material::variant_bits as vb

struct ProceduralSkyboxMaterial {
    _SkyTint: vec4<f32>,
    _GroundColor: vec4<f32>,
    _SunColor: vec4<f32>,
    _SunDirection: vec4<f32>,
    _Exposure: f32,
    _SunSize: f32,
    _AtmosphereThickness: f32,
    _RenderideVariantBits: u32,
}

const PROCSKY_KW_SUNDISK_HIGH_QUALITY: u32 = 1u << 0u;
const PROCSKY_KW_SUNDISK_NONE: u32 = 1u << 1u;
const PROCSKY_KW_SUNDISK_SIMPLE: u32 = 1u << 2u;
const PROCSKY_KW_UNITY_COLORSPACE_GAMMA: u32 = 1u << 3u;
const PROCSKY_GROUP_SUNDISK: u32 =
    PROCSKY_KW_SUNDISK_HIGH_QUALITY | PROCSKY_KW_SUNDISK_NONE | PROCSKY_KW_SUNDISK_SIMPLE;

@group(1) @binding(0) var<uniform> mat: ProceduralSkyboxMaterial;

fn procsky_kw(mask: u32) -> bool {
    return vb::enabled(mat._RenderideVariantBits, mask);
}

fn kw_SUNDISK_NONE() -> bool {
    return procsky_kw(PROCSKY_KW_SUNDISK_NONE);
}

fn kw_SUNDISK_HIGH_QUALITY() -> bool {
    return (mat._RenderideVariantBits & PROCSKY_GROUP_SUNDISK) == 0u
        || procsky_kw(PROCSKY_KW_SUNDISK_HIGH_QUALITY);
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) ground_color: vec3<f32>,
    @location(1) sky_color: vec3<f32>,
    @location(2) sun_color: vec3<f32>,
    @location(3) fragment_ray: vec3<f32>,
    @location(4) sky_ground_factor: f32,
}

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
) -> VertexOutput {
    let d = pd::get_draw(instance_index);
    let world_p = mv::world_position(d, pos);
#ifdef MULTIVIEW
    let vp = mv::select_view_proj(d, view_idx);
#else
    let vp = mv::select_view_proj(d, 0u);
#endif

    var out: VertexOutput;
    out.clip_pos = vp * world_p;
    let terms = ps::visible_vertex_terms(procedural_sky_params(), mv::model_vector(d, pos.xyz));
    out.ground_color = terms.ground_color;
    out.sky_color = terms.sky_color;
    out.sun_color = terms.sun_color;
    out.fragment_ray = terms.fragment_ray;
    out.sky_ground_factor = terms.sky_ground_factor;
    return out;
}

//#pass forward
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let terms = ps::ProceduralSkyVisibleTerms(
        in.ground_color,
        in.sky_color,
        in.sun_color,
        in.fragment_ray,
        in.sky_ground_factor,
    );
    return rg::retain_globals_additive(vec4<f32>(
        ps::visible_fragment_color(procedural_sky_params(), terms),
        1.0,
    ));
}

fn procedural_sky_params() -> ps::ProceduralSkyParams {
    return ps::ProceduralSkyParams(
        mat._SkyTint.rgb,
        mat._GroundColor.rgb,
        mat._SunColor.rgb,
        mat._SunDirection.xyz,
        mat._Exposure,
        mat._SunSize,
        mat._AtmosphereThickness,
        procedural_sun_disk_mode(),
    );
}

fn procedural_sun_disk_mode() -> f32 {
    if (kw_SUNDISK_NONE()) {
        return 0.0;
    }
    if (kw_SUNDISK_HIGH_QUALITY()) {
        return 2.0;
    }
    return 1.0;
}
