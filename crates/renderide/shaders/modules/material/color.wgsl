//! Shared color-channel math for material shader families.

#define_import_path renderide::material::color

fn rgba_weight_sum(weights: vec4<f32>) -> f32 {
    return weights.r + weights.g + weights.b + weights.a;
}

fn normalized_rgba_weights(weights: vec4<f32>, min_sum: f32) -> vec4<f32> {
    let sum = max(rgba_weight_sum(weights), min_sum);
    return weights / sum;
}

fn blend4_vec4(a: vec4<f32>, b: vec4<f32>, c: vec4<f32>, d: vec4<f32>, weights: vec4<f32>) -> vec4<f32> {
    return a * weights.r + b * weights.g + c * weights.b + d * weights.a;
}
