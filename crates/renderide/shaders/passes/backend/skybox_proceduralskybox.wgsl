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
    @location(0) ground_color: vec3<f32>,
    @location(1) sky_color: vec3<f32>,
    @location(2) sun_color: vec3<f32>,
    @location(3) fragment_ray: vec3<f32>,
    @location(4) sky_ground_factor: f32,
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
) -> VertexOutput {
    let clip = skybox::fullscreen_quad_clip_pos(vertex_index);
    var out: VertexOutput;
    out.clip_pos = clip;
#ifdef MULTIVIEW
    let view_layer = view_idx;
#else
    let view_layer = 0u;
#endif
    let ndc = vec2<f32>(clip.x, clip.y * view.ndc_y_sign_pad.x);
    let proj_params = select(rg::frame.proj_params_left, rg::frame.proj_params_right, view_layer != 0u);
    let view_ray = skybox::view_ray_from_ndc(
        ndc,
        proj_params,
        skybox::view_is_orthographic(view, view_layer),
    );
    let world_ray = skybox::world_ray_from_view_ray(view_ray, view, view_layer);
    let terms = ps::visible_vertex_terms(psmat::params(), world_ray);
    out.ground_color = terms.ground_color;
    out.sky_color = terms.sky_color;
    out.sun_color = terms.sun_color;
    out.fragment_ray = terms.fragment_ray;
    out.sky_ground_factor = terms.sky_ground_factor;
    return out;
}

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
        ps::visible_fragment_color(psmat::params(), terms),
        1.0,
    ));
}
