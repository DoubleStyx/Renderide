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
@group(0) @binding(1) var source_tex: texture_2d<f32>;
@group(0) @binding(2) var source_sampler: sampler;
@group(0) @binding(3) var<storage, read_write> output_coeffs: array<vec4<f32>, 9>;

const WORKGROUP_SIZE: u32 = 64u;
const COEFFS: u32 = 9u;
const PI: f32 = 3.14159265359;
const TAU: f32 = 6.28318530718;
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

fn positive_fmod(v: vec2<f32>, wrap: vec2<f32>) -> vec2<f32> {
    var r = v - trunc(v / wrap) * wrap;
    r = r + wrap;
    return r - trunc(r / wrap) * wrap;
}

fn projection360_dir_to_uv(view_dir: vec3<f32>) -> vec2<f32> {
    var angle = vec2<f32>(
        atan2(view_dir.x, view_dir.z),
        acos(clamp(dot(view_dir, vec3<f32>(0.0, 1.0, 0.0)), -1.0, 1.0)) - PI * 0.5,
    );
    angle = angle + params.color0.xy * 0.5 + params.color0.zw;
    angle = positive_fmod(angle, vec2<f32>(TAU, PI));
    return angle / max(abs(params.color0.xy), vec2<f32>(0.000001));
}

fn projection360_main_tex_uv(uv: vec2<f32>) -> vec2<f32> {
    let uv_st = uv * params.color1.xy + params.color1.zw;
    if (params.scalars.x > 0.5) {
        return uv_st;
    }
    return vec2<f32>(uv_st.x, 1.0 - uv_st.y);
}

fn projection360_equirect_uv(world_dir: vec3<f32>) -> vec2<f32> {
    let uv = clamp(
        projection360_dir_to_uv(-world_dir),
        vec2<f32>(0.0),
        vec2<f32>(1.0),
    );
    return projection360_main_tex_uv(uv);
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
        let color = textureSampleLevel(source_tex, source_sampler, projection360_equirect_uv(dir), 0.0).rgb;
        add_coeffs(base, color, dir, texel_solid_angle(x, y, n));
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
