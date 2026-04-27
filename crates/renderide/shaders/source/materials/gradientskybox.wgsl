//! Unity GradientSkybox (`Shader "GradientSkybox"`): sky gradient material.

// unity-shader-name: GradientSkybox

#import renderide::globals as rg
#import renderide::per_draw as pd

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
        let spread = max(abs(dirs_spread.w), 0.000001);
        let expv = max(params.y, 0.000001);
        let denom = max(abs(params.w - params.z), 0.000001);
        var r = (0.5 - dot(ray, normalize(dirs_spread.xyz)) * 0.5) / spread;
        if (r <= 1.0) {
            r = pow(max(r, 0.0), expv);
            r = clamp((r - params.z) / denom, 0.0, 1.0);
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
    let world_p = d.model * vec4<f32>(pos.xyz, 1.0);
#ifdef MULTIVIEW
    var vp: mat4x4<f32>;
    if (view_idx == 0u) {
        vp = d.view_proj_left;
    } else {
        vp = d.view_proj_right;
    }
#else
    let vp = d.view_proj_left;
#endif
    var out: VertexOutput;
    out.clip_pos = vp * world_p;
    out.ray = (d.model * vec4<f32>(pos.xyz, 0.0)).xyz;
    return out;
}

//#pass forward
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return gradient_color(in.ray);
}
