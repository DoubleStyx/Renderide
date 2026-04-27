struct Params {
    sample_size: u32,
    mode: u32,
    gradient_count: u32,
    _pad0: u32,
    color0: vec4<f32>,
    color1: vec4<f32>,
    direction: vec4<f32>,
    scalars: vec4<f32>,
    dirs_spread: array<vec4<f32>, 16>,
    gradient_color0: array<vec4<f32>, 16>,
    gradient_color1: array<vec4<f32>, 16>,
    gradient_params: array<vec4<f32>, 16>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> output_coeffs: array<vec4<f32>, 9>;

const WORKGROUP_SIZE: u32 = 64u;
const COEFFS: u32 = 9u;
const SH_C0: f32 = 0.2820947917;
const SH_C1: f32 = 0.4886025119;
const SH_C2: f32 = 1.0925484306;
const SH_C3: f32 = 0.3153915652;
const SH_C4: f32 = 0.5462742153;

var<workgroup> partial: array<vec4<f32>, 576>;

fn area_element(x: f32, y: f32) -> f32 {
    return atan2(x * y, sqrt(x * x + y * y + 1.0));
}

fn texel_solid_angle(x: u32, y: u32, n: u32) -> f32 {
    let inv = 1.0 / f32(n);
    let x0 = (f32(x) * inv) * 2.0 - 1.0;
    let y0 = (f32(y) * inv) * 2.0 - 1.0;
    let x1 = (f32(x + 1u) * inv) * 2.0 - 1.0;
    let y1 = (f32(y + 1u) * inv) * 2.0 - 1.0;
    return abs(area_element(x0, y0) - area_element(x0, y1) - area_element(x1, y0) + area_element(x1, y1));
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

fn sample_procedural(dir: vec3<f32>) -> vec3<f32> {
    let sky = max(params.color0.rgb, vec3<f32>(0.0));
    let ground = max(params.color1.rgb, vec3<f32>(0.0));
    let exposure = max(params.scalars.x, 0.0);
    let sun_size = max(params.scalars.y, 0.001);
    let sun_dir = normalize(params.direction.xyz + vec3<f32>(0.0, 0.00001, 0.0));
    let t = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    let sun = pow(max(dot(dir, sun_dir), 0.0), 1.0 / sun_size) * params.gradient_color0[0].rgb;
    return (mix(ground, sky, t) + sun) * exposure;
}

fn sample_gradient(dir: vec3<f32>) -> vec3<f32> {
    var color = max(params.color0.rgb, vec3<f32>(0.0));
    let count = min(params.gradient_count, 16u);
    for (var i = 0u; i < count; i = i + 1u) {
        let row = params.dirs_spread[i];
        let axis = normalize(row.xyz + vec3<f32>(0.0, 0.00001, 0.0));
        let spread = clamp(abs(row.w), 0.0001, 2.0);
        let d = clamp(dot(dir, axis) * 0.5 + 0.5, 0.0, 1.0);
        let shaped = pow(d, 1.0 / spread);
        let grad = mix(params.gradient_color0[i].rgb, params.gradient_color1[i].rgb, shaped);
        let amount = clamp(params.gradient_params[i].x, 0.0, 1.0);
        color = mix(color, grad, amount);
    }
    return color;
}

fn sample_sky(dir: vec3<f32>) -> vec3<f32> {
    if (params.mode == 2u) {
        return sample_gradient(dir);
    }
    return sample_procedural(dir);
}

fn add_coeffs(base: u32, c: vec3<f32>, dir: vec3<f32>, weight: f32) {
    partial[base + 0u] = partial[base + 0u] + vec4<f32>(c * (SH_C0 * weight), 0.0);
    partial[base + 1u] = partial[base + 1u] + vec4<f32>(c * (SH_C1 * dir.y * weight), 0.0);
    partial[base + 2u] = partial[base + 2u] + vec4<f32>(c * (SH_C1 * dir.z * weight), 0.0);
    partial[base + 3u] = partial[base + 3u] + vec4<f32>(c * (SH_C1 * dir.x * weight), 0.0);
    partial[base + 4u] = partial[base + 4u] + vec4<f32>(c * (SH_C2 * dir.x * dir.y * weight), 0.0);
    partial[base + 5u] = partial[base + 5u] + vec4<f32>(c * (SH_C2 * dir.y * dir.z * weight), 0.0);
    partial[base + 6u] = partial[base + 6u] + vec4<f32>(c * (SH_C3 * (3.0 * dir.z * dir.z - 1.0) * weight), 0.0);
    partial[base + 7u] = partial[base + 7u] + vec4<f32>(c * (SH_C2 * dir.x * dir.z * weight), 0.0);
    partial[base + 8u] = partial[base + 8u] + vec4<f32>(c * (SH_C4 * (dir.x * dir.x - dir.y * dir.y) * weight), 0.0);
}

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let n = max(params.sample_size, 1u);
    let face_size = n * n;
    let total = face_size * 6u;
    let base = local_id.x * COEFFS;
    for (var c = 0u; c < COEFFS; c = c + 1u) {
        partial[base + c] = vec4<f32>(0.0);
    }
    var i = local_id.x;
    while (i < total) {
        let face = i / face_size;
        let rem = i - face * face_size;
        let y = rem / n;
        let x = rem - y * n;
        let dir = cube_dir(face, x, y, n);
        add_coeffs(base, sample_sky(dir), dir, texel_solid_angle(x, y, n));
        i = i + WORKGROUP_SIZE;
    }
    workgroupBarrier();
    if (local_id.x == 0u) {
        for (var coeff = 0u; coeff < COEFFS; coeff = coeff + 1u) {
            var sum = vec4<f32>(0.0);
            for (var lane = 0u; lane < WORKGROUP_SIZE; lane = lane + 1u) {
                sum = sum + partial[lane * COEFFS + coeff];
            }
            output_coeffs[coeff] = sum;
        }
    }
}
