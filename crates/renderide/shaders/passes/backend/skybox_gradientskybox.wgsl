//! Fullscreen GradientSkybox sky draw.

#import renderide::frame::globals as rg
#import renderide::core::fullscreen as fs
#import renderide::skybox::common as skybox
#import renderide::skybox::gradient as skygrad

struct GradientSkyboxMaterial {
    _BaseColor: vec4<f32>,
    _Gradients: f32,
    _DirsSpread: array<vec4<f32>, 16>,
    _Color0: array<vec4<f32>, 16>,
    _Color1: array<vec4<f32>, 16>,
    _Params: array<vec4<f32>, 16>,
}

@group(1) @binding(0) var<uniform> mat: GradientSkyboxMaterial;
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
    out.view_layer = view_idx;
#else
    out.view_layer = 0u;
#endif
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
    return skygrad::gradient_sky_color(
        mat._BaseColor,
        mat._Gradients,
        mat._DirsSpread,
        mat._Color0,
        mat._Color1,
        mat._Params,
        world_ray,
    );
}
