//! Fullscreen ProceduralSkybox sky draw.
//!
//! The shared ProceduralSkybox material module owns the reflected `@group(1)` contract and
//! shader-specific keyword decoding for both this pass-side sky draw and the material root.

#import renderide::frame::globals as rg
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
    let clip = skybox::fullscreen_clip_pos(vertex_index);
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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let ndc = in.clip_pos.xy * vec2<f32>(1.0, view.ndc_y_sign_pad.x);
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
