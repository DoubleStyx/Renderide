//! Unity ProceduralSkybox asset (`Shader "ProceduralSky"`): analytic sky material.

// unity-shader-name: ProceduralSkybox

#import renderide::globals as rg
#import renderide::per_draw as pd
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

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) ray: vec3<f32>,
}

fn procedural_color(ray_in: vec3<f32>) -> vec4<f32> {
    let ray = normalize(ray_in);
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
    return procedural_color(in.ray);
}
