//! Analytic skybox sampling shared by generated environment and SH2 compute passes.

#define_import_path renderide::skybox_evaluator

const MAX_GRADIENTS: u32 = 16u;

struct SkyboxEvaluatorParams {
    sample_size: u32,
    mode: u32,
    gradient_count: u32,
    color_a: vec4<f32>,
    color_b: vec4<f32>,
    direction: vec4<f32>,
    scalars: vec4<f32>,
    dirs_spread: array<vec4<f32>, 16>,
    gradient_color_a: array<vec4<f32>, 16>,
    gradient_color_b: array<vec4<f32>, 16>,
    gradient_params: array<vec4<f32>, 16>,
}

fn cube_dir(face: u32, x: u32, y: u32, n: u32) -> vec3<f32> {
    let u = (f32(x) + 0.5) / f32(n);
    let v = (f32(y) + 0.5) / f32(n);
    if (face == 0u) { return normalize(vec3<f32>(1.0, v * -2.0 + 1.0, u * -2.0 + 1.0)); }
    if (face == 1u) { return normalize(vec3<f32>(-1.0, v * -2.0 + 1.0, u * 2.0 - 1.0)); }
    if (face == 2u) { return normalize(vec3<f32>(u * 2.0 - 1.0, 1.0, v * 2.0 - 1.0)); }
    if (face == 3u) { return normalize(vec3<f32>(u * 2.0 - 1.0, -1.0, v * -2.0 + 1.0)); }
    if (face == 4u) { return normalize(vec3<f32>(u * 2.0 - 1.0, v * -2.0 + 1.0, 1.0)); }
    return normalize(vec3<f32>(u * -2.0 + 1.0, v * -2.0 + 1.0, -1.0));
}

fn sample_procedural(params: SkyboxEvaluatorParams, ray: vec3<f32>) -> vec3<f32> {
    let y = ray.y;
    let horizon = pow(1.0 - clamp(abs(y), 0.0, 1.0), 2.0);
    let sky_amount = smoothstep(-0.02, 0.08, y);
    let atmosphere = max(params.scalars.z, 0.0);
    let scatter = vec3<f32>(0.20, 0.36, 0.75) * (0.25 + atmosphere * 0.25) * max(y, 0.0);
    let sky = params.color_a.rgb * (0.35 + 0.65 * max(y, 0.0)) + scatter;
    let ground = params.color_b.rgb * (0.55 + 0.45 * horizon);
    var col = mix(ground, sky, sky_amount);
    col = col + params.color_a.rgb * horizon * 0.18;

    if (params.scalars.w > 0.5) {
        let sun_dir = normalize(params.direction.xyz + vec3<f32>(0.0, 0.00001, 0.0));
        let sun_dot = max(dot(ray, sun_dir), 0.0);
        let size = clamp(params.scalars.y, 0.0001, 1.0);
        let exponent = mix(4096.0, 48.0, size);
        var sun = pow(sun_dot, exponent);
        if (params.scalars.w > 1.5) {
            sun = sun + pow(sun_dot, max(exponent * 0.18, 4.0)) * 0.18;
        }
        col = col + params.gradient_color_a[0].rgb * sun;
    }

    return max(col * max(params.scalars.x, 0.0), vec3<f32>(0.0));
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
