//! Fullscreen GradientSkybox sky draw.

#import renderide::globals as rg
#import renderide::skybox_common as skybox

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
    @location(0) ndc: vec2<f32>,
    @location(1) @interpolate(flat) view_layer: u32,
}

fn gradient_color(ray: vec3<f32>) -> vec4<f32> {
    var col = mat._BaseColor.rgb;
    let count = min(u32(max(mat._Gradients, 0.0)), 16u);
    for (var i = 0u; i < count; i = i + 1u) {
        let dirs_spread = mat._DirsSpread[i];
        let params = mat._Params[i];
        let spread = max(abs(dirs_spread.w), 0.000001);
        let expv = max(params.y, 0.000001);
        let fromv = params.z;
        let tov = params.w;
        let denom = max(abs(tov - fromv), 0.000001);
        var r = (0.5 - dot(ray, normalize(dirs_spread.xyz)) * 0.5) / spread;
        if (r <= 1.0) {
            r = pow(max(r, 0.0), expv);
            r = clamp((r - fromv) / denom, 0.0, 1.0);
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
    @builtin(vertex_index) vertex_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
) -> VertexOutput {
    let clip = skybox::fullscreen_clip_pos(vertex_index);
    var out: VertexOutput;
    out.clip_pos = clip;
    out.ndc = clip.xy;
#ifdef MULTIVIEW
    out.view_layer = view_idx;
#else
    out.view_layer = 0u;
#endif
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let proj_params = select(rg::frame.proj_params_left, rg::frame.proj_params_right, in.view_layer != 0u);
    let view_ray = skybox::view_ray_from_ndc(in.ndc, proj_params);
    let world_ray = skybox::world_ray_from_view_ray(view_ray, view, in.view_layer);
    return gradient_color(world_ray);
}
