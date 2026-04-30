//! Shared layer-weight math for splat and color-mask PBS materials.

#define_import_path renderide::pbs::splat

#import renderide::material::color as color

fn normalize_weights(weights: vec4<f32>) -> vec4<f32> {
    return color::normalized_rgba_weights(weights, 1e-4);
}

fn color_mask_weights(mask: vec4<f32>) -> vec4<f32> {
    return color::normalized_rgba_weights(mask, 1e-5);
}

fn height_blended_weights(weights: vec4<f32>, heights: vec4<f32>, transition_range: f32) -> vec4<f32> {
    let weighted_heights = heights * weights;
    let max_height = max(max(weighted_heights.x, weighted_heights.y), max(weighted_heights.z, weighted_heights.w));
    let band = max(transition_range, 1e-4);
    let shifted = (weighted_heights - vec4<f32>(max_height) + vec4<f32>(band)) / band;
    return normalize_weights(clamp(shifted, vec4<f32>(0.0), vec4<f32>(1.0)));
}
