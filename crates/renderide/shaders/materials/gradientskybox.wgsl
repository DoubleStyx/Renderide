//! Unity GradientSkybox (`Shader "GradientSkybox"`): sky gradient material.


//#mat_default _BaseColor vec4 1.0 1.0 1.0 1.0

#import renderide::draw::per_draw as pd
#import renderide::mesh::vertex as mv
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

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) ray: vec3<f32>,
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
    out.ray = mv::model_vector(d, pos.xyz);
    return out;
}

//#pass forward
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return skygrad::gradient_sky_color(
        mat._BaseColor,
        mat._Gradients,
        mat._DirsSpread,
        mat._Color0,
        mat._Color1,
        mat._Params,
        in.ray,
    );
}
