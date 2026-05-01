//! Unity GradientSkybox (`Shader "GradientSkybox"`): sky gradient material.


#import renderide::globals as rg
#import renderide::per_draw as pd
#import renderide::mesh::vertex as mv

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

fn gradient_color(ray_in: vec3<f32>) -> vec4<f32> {
    let ray = normalize(ray_in);
    var col = mat._BaseColor.rgb;
    let count = min(u32(max(mat._Gradients, 0.0)), 16u);
    for (var i = 0u; i < count; i = i + 1u) {
        let dirs_spread = mat._DirsSpread[i];
        let params = mat._Params[i];
        var r = 0.5 - dot(ray, dirs_spread.xyz) * 0.5;
        r = r / dirs_spread.w;
        if (r <= 1.0) {
            r = pow(r, params.y);
            r = clamp((r - params.z) / (params.w - params.z), 0.0, 1.0);
            let c = mix(mat._Color0[i], mat._Color1[i], r);
            if (params.x == 0.0) {
                col = col * (1.0 - c.a) + c.rgb * c.a;
            } else {
                col = col + c.rgb * c.a;
            }
        }
    }
    return rg::retain_globals_additive(vec4<f32>(col, 1.0));
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
    return gradient_color(in.ray);
}
