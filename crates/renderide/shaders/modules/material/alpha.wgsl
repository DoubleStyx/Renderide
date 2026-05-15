//! Shared alpha and blend helpers for material shaders.

#define_import_path renderide::material::alpha

#import renderide::core::math as rmath

fn apply_premultiply(color: vec3<f32>, alpha: f32, enabled: bool) -> vec3<f32> {
    return select(color, color * alpha, enabled);
}

fn mask_luminance(mask_sample: vec4<f32>) -> f32 {
    return mask_sample.a * rmath::luminance_rgb(mask_sample.rgb);
}

fn alpha_intensity(alpha: f32, rgb: vec3<f32>) -> f32 {
    return alpha * rmath::luminance_rgb(rgb);
}

fn alpha_intensity_squared(alpha: f32, rgb: vec3<f32>) -> f32 {
    let lum = rmath::luminance_rgb(rgb);
    return alpha * lum * lum;
}
