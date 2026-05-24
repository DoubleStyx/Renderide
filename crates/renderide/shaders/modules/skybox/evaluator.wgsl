//! Parameter-driven skybox sampling used by generated cubemap compute passes.

#define_import_path renderide::skybox::evaluator

#import renderide::ibl::ggx_prefilter as ggx
#import renderide::skybox::procedural as ps

const MAX_GRADIENTS: u32 = 16u;

struct SkyboxEvaluatorParams {
    sample_size: u32,
    mode: u32,
    gradient_count: u32,
    _pad: u32,
    color_a: vec4<f32>,
    color_b: vec4<f32>,
    direction: vec4<f32>,
    scalars: vec4<f32>,
    dirs_spread: array<vec4<f32>, 16>,
    gradient_color_a: array<vec4<f32>, 16>,
    gradient_color_b: array<vec4<f32>, 16>,
    gradient_params: array<vec4<f32>, 16>,
}

/// Cubemap face direction at integer face/texel coordinates `(face, x, y)` for an `n`-edge face.
fn cube_dir(face: u32, x: u32, y: u32, n: u32) -> vec3<f32> {
    return ggx::cube_dir(face, x, y, n);
}

fn sample_procedural(params: SkyboxEvaluatorParams, ray: vec3<f32>) -> vec3<f32> {
    let sky_params = ps::ProceduralSkyParams(
        params.color_a.rgb,
        params.color_b.rgb,
        params.gradient_color_a[0].rgb,
        params.direction.xyz,
        params.scalars.x,
        params.scalars.y,
        params.scalars.z,
        params.scalars.w,
    );
    return ps::sample(sky_params, ray);
}

fn sample_gradient(params: SkyboxEvaluatorParams, ray: vec3<f32>) -> vec3<f32> {
    var color = params.color_a.rgb;
    let count = min(params.gradient_count, MAX_GRADIENTS);
    for (var i = 0u; i < count; i = i + 1u) {
        let dirs_spread = params.dirs_spread[i];
        let gradient_params = params.gradient_params[i];
        var r = 0.5 - dot(ray, dirs_spread.xyz) * 0.5;
        r = r / dirs_spread.w;
        if (r <= 1.0) {
            r = pow(r, gradient_params.y);
            r = clamp((r - gradient_params.z) / (gradient_params.w - gradient_params.z), 0.0, 1.0);
            let c = mix(params.gradient_color_a[i], params.gradient_color_b[i], r);
            if (gradient_params.x == 0.0) {
                color = color * (1.0 - c.a) + c.rgb * c.a;
            } else {
                color = color + c.rgb * c.a;
            }
        }
    }
    return max(color, vec3<f32>(0.0));
}

fn sample_sky(params: SkyboxEvaluatorParams, ray: vec3<f32>) -> vec3<f32> {
    if (params.mode == 2u) {
        return sample_gradient(params, ray);
    }
    return sample_procedural(params, ray);
}
