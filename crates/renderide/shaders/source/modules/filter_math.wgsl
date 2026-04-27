//! Color and coordinate helpers shared by grab-pass filter materials.

#define_import_path renderide::filter_math

const TAU: f32 = 6.28318530718;

fn safe_div_vec2(value: vec2<f32>, denom: vec2<f32>) -> vec2<f32> {
    return value / max(abs(denom), vec2<f32>(1e-6));
}

fn rgb_to_hsv_no_clip(rgb: vec3<f32>) -> vec3<f32> {
    let k = vec4<f32>(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    let p = select(vec4<f32>(rgb.bg, k.wz), vec4<f32>(rgb.gb, k.xy), rgb.b < rgb.g);
    let q = select(vec4<f32>(p.xyw, rgb.r), vec4<f32>(rgb.r, p.yzx), p.x < rgb.r);
    let d = q.x - min(q.w, q.y);
    let e = 1e-10;
    return vec3<f32>(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

fn hsv_to_rgb(hsv: vec3<f32>) -> vec3<f32> {
    let p = abs(fract(hsv.xxx + vec3<f32>(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
    return hsv.z * mix(vec3<f32>(1.0), clamp(p - vec3<f32>(1.0), vec3<f32>(0.0), vec3<f32>(1.0)), hsv.y);
}

fn luma_bt601(rgb: vec3<f32>) -> f32 {
    return dot(rgb, vec3<f32>(0.299, 0.587, 0.114));
}

fn circular_blur_offset(sample_index: u32, sample_count: u32, spread: vec2<f32>) -> vec2<f32> {
    let angle = (f32(sample_index) / max(f32(sample_count), 1.0)) * TAU;
    return vec2<f32>(-cos(angle), sin(angle)) * spread;
}

fn screen_vignette(uv: vec2<f32>) -> f32 {
    let pos = clamp((1.0 - abs(uv * 2.0 - 1.0)) * 32.0, vec2<f32>(0.0), vec2<f32>(1.0));
    return pos.x * pos.y;
}
