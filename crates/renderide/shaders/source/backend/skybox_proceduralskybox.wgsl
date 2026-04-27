//! Fullscreen ProceduralSkybox sky draw.

#import renderide::globals as rg
#import renderide::skybox_common as skybox
#import renderide::uv_utils as uvu

struct ProceduralSkyboxMaterial {
    _SkyTint: vec4<f32>,
    _GroundColor: vec4<f32>,
    _SunColor: vec4<f32>,
    _SunDirection: vec4<f32>,
    _Exposure: f32,
    _SunSize: f32,
    _AtmosphereThickness: f32,
    _SUNDISK_NONE: f32,
    _SUNDISK_SIMPLE: f32,
    _SUNDISK_HIGH_QUALITY: f32,
}

@group(1) @binding(0) var<uniform> mat: ProceduralSkyboxMaterial;
@group(2) @binding(0) var<uniform> view: skybox::SkyboxView;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
    @location(1) @interpolate(flat) view_layer: u32,
}

fn procedural_color(ray: vec3<f32>) -> vec4<f32> {
    let y = ray.y;
    let horizon = pow(1.0 - clamp(abs(y), 0.0, 1.0), 2.0);
    let sky_amount = smoothstep(-0.02, 0.08, y);
    let atmosphere = max(mat._AtmosphereThickness, 0.0);
    let scatter = vec3<f32>(0.20, 0.36, 0.75) * (0.25 + atmosphere * 0.25) * max(y, 0.0);
    let sky = mat._SkyTint.rgb * (0.35 + 0.65 * max(y, 0.0)) + scatter;
    let ground = mat._GroundColor.rgb * (0.55 + 0.45 * horizon);
    var col = mix(ground, sky, sky_amount);
    col = col + mat._SkyTint.rgb * horizon * 0.18;

    if (!uvu::kw_enabled(mat._SUNDISK_NONE)) {
        let sun_dir = normalize(mat._SunDirection.xyz);
        let sun_dot = max(dot(ray, sun_dir), 0.0);
        let size = clamp(mat._SunSize, 0.0001, 1.0);
        let exponent = mix(4096.0, 48.0, size);
        var sun = pow(sun_dot, exponent);
        if (uvu::kw_enabled(mat._SUNDISK_HIGH_QUALITY)) {
            sun = sun + pow(sun_dot, max(exponent * 0.18, 4.0)) * 0.18;
        }
        col = col + mat._SunColor.rgb * sun;
    }

    return rg::retain_globals_additive(vec4<f32>(max(col * mat._Exposure, vec3<f32>(0.0)), 1.0));
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
    return procedural_color(world_ray);
}
