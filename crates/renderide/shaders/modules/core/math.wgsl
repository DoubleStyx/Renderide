//! Shared scalar/vector math helpers used by material shader modules.

#define_import_path renderide::core::math

const EPSILON: f32 = 1e-6;

fn saturate(v: f32) -> f32 {
    return clamp(v, 0.0, 1.0);
}

fn luminance_rgb(rgb: vec3<f32>) -> f32 {
    return (rgb.r + rgb.g + rgb.b) * 0.33333334;
}

fn safe_normalize(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if (len_sq <= 1e-12) {
        return fallback;
    }
    return v * inverseSqrt(len_sq);
}

fn safe_lerp_factor(start_value: f32, end_value: f32, value: f32) -> f32 {
    let denom = end_value - start_value;
    if (abs(denom) < EPSILON) {
        return select(0.0, 1.0, value <= start_value);
    }
    return saturate((value - start_value) / denom);
}

fn safe_linear_factor(start_value: f32, end_value: f32, value: f32) -> f32 {
    let denom = end_value - start_value;
    if (abs(denom) < EPSILON) {
        return select(0.0, 1.0, value >= end_value);
    }
    return saturate((value - start_value) / denom);
}

fn rotate2(v: vec2<f32>, angle: f32) -> vec2<f32> {
    let s = sin(angle);
    let c = cos(angle);
    return vec2<f32>(c * v.x - s * v.y, s * v.x + c * v.y);
}

fn rect_has_area(rect: vec4<f32>) -> bool {
    let size = rect.zw - rect.xy;
    return abs(size.x * size.y) > EPSILON;
}

fn outside_rect(p: vec2<f32>, rect: vec4<f32>) -> bool {
    let min_v = rect.xy;
    let max_v = rect.zw;
    return p.x < min_v.x || p.x > max_v.x || p.y < min_v.y || p.y > max_v.y;
}

fn inside_rect_mask(p: vec2<f32>, rect: vec4<f32>) -> f32 {
    return select(1.0, 0.0, outside_rect(p, rect));
}
