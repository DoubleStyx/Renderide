//! Unity ProceduralSkybox asset (`Shader "ProceduralSky"`): analytic sky material with
//! Rayleigh+Mie scattering and three sun-disk modes (NONE / SIMPLE / HIGH_QUALITY).
//!
//! The renderer pipeline operates entirely in linear color space, so this shader implements
//! the linear branch only; the gamma-space branch and
//! `SKYBOX_COLOR_IN_TARGET_COLOR_SPACE` short-circuit are intentionally omitted.
//!
//! Froox variant bits populate `_RenderideVariantBits`; the shared ProceduralSkybox material
//! module decodes shader-specific keyword bits. `UNITY_COLORSPACE_GAMMA` is reserved in the
//! bit table but never consulted because the renderer is linear-only. The sun-disk group has
//! no `_` placeholder, so the high-quality keyword is the zero-bit default.

//#mat_default _GroundColor vec4 0.369 0.349 0.341 1.0
//#mat_default _SkyTint vec4 0.5 0.5 0.5 1.0
//#mat_default _SunColor vec4 1.0 1.0 1.0 1.0
//#mat_default _SunDirection vec4 0.577 0.577 0.577 0.0
//#mat_default _AtmosphereThickness float 1.0
//#mat_default _Exposure float 1.3
//#mat_default _SunSize float 0.04

#import renderide::frame::globals as rg
#import renderide::draw::per_draw as pd
#import renderide::skybox::procedural as ps
#import renderide::skybox::procedural_material as psmat
#import renderide::mesh::vertex as mv

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) ground_color: vec3<f32>,
    @location(1) sky_color: vec3<f32>,
    @location(2) sun_color: vec3<f32>,
    @location(3) ray: vec3<f32>,
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
    let ps_params = psmat::params();
    let scattering_params = ps::scattering_parameters(ps_params);
    let terms = ps::visible_vertex_terms(ps_params, scattering_params, mv::model_vector(d, pos.xyz));
    out.ground_color = terms.ground_color;
    out.sky_color = terms.sky_color;
    out.sun_color = terms.sun_color;
    out.ray = terms.ray;
    return out;
}

//#pass type=forward
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let terms = ps::ProceduralSkyVisibleTerms(
        in.ground_color,
        in.sky_color,
        in.sun_color,
        in.ray,
    );
    return rg::retain_globals_additive(vec4<f32>(
        ps::visible_fragment_color(psmat::params(), terms),
        1.0,
    ));
}
