//! Fullscreen ProceduralSkybox sky draw.
//!
//! The shared ProceduralSkybox material module owns the reflected `@group(1)` contract and
//! shader-specific keyword decoding for both this pass-side sky draw and the material root.

//#mat_default _SkyTint vec4 0.5 0.5 0.5 1.0
//#mat_default _GroundColor vec4 0.369 0.349 0.341 1.0
//#mat_default _SunColor vec4 1.0 1.0 1.0 1.0
//#mat_default _SunDirection vec4 0.577 0.577 0.577 0.0
//#mat_default _Exposure float 1.3
//#mat_default _SunSize float 0.04
//#mat_default _AtmosphereThickness float 1.0

#import renderide::frame::globals as rg
#import renderide::core::fullscreen as fs
#import renderide::skybox::procedural as ps
#import renderide::skybox::procedural_material as psmat
#import renderide::skybox::common as skybox

@group(2) @binding(0) var<uniform> view: skybox::SkyboxView;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) @interpolate(flat) view_layer: u32,
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
) -> VertexOutput {
    let clip = fs::fullscreen_clip_pos(vertex_index);
    var out: VertexOutput;
    out.clip_pos = clip;
#ifdef MULTIVIEW
    let view_layer = view_idx;
#else
    let view_layer = 0u;
#endif
    out.view_layer = view_layer;
    return out;
}

//#pass type=forward blend=off zwrite=off ztest=main
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let ndc = skybox::ndc_from_clip(in.clip_pos);
    let proj_params = select(rg::frame.proj_params_left, rg::frame.proj_params_right, in.view_layer != 0u);
    let view_ray = skybox::view_ray_from_ndc(
        ndc,
        proj_params,
        skybox::view_is_orthographic(view, in.view_layer),
    );
    let world_ray = skybox::world_ray_from_view_ray(view_ray, view, in.view_layer);
    let ps_params = psmat::params();
    let scattering_params = ps::scattering_parameters(ps_params);
    let terms = ps::visible_vertex_terms(ps_params, scattering_params, world_ray);
    return rg::retain_globals_additive(vec4<f32>(
        ps::visible_fragment_color(ps_params, terms),
        1.0,
    ));
}
