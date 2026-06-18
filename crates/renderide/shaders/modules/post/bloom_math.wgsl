//! Pure bloom threshold, firefly reduction, and composite math.

#define_import_path renderide::post::bloom_math

/// Rec. 709 luminance in linear space.
fn rec709_luminance(v: vec3<f32>) -> f32 {
    return dot(v, vec3<f32>(0.2126, 0.7152, 0.0722));
}

/// Karis firefly reduction weight, applied per 13-tap downsample group.
fn karis_average(color: vec3<f32>) -> f32 {
    let luma = rec709_luminance(color) / 4.0;
    return 1.0 / (1.0 + luma);
}

/// Unity-style quadratic soft-knee threshold.
fn soft_threshold(color: vec3<f32>, threshold_precomputations: vec4<f32>) -> vec3<f32> {
    let brightness = max(color.r, max(color.g, color.b));
    var softness = brightness - threshold_precomputations.y;
    softness = clamp(softness, 0.0, threshold_precomputations.z);
    softness = softness * softness * threshold_precomputations.w;
    var contribution = max(brightness - threshold_precomputations.x, softness);
    contribution /= max(brightness, 1e-5);
    return color * contribution;
}

fn positive_bloom_source(color: vec3<f32>) -> vec3<f32> {
    return max(color, vec3<f32>(0.0));
}

fn bloom_source(color: vec3<f32>, threshold_precomputations: vec4<f32>) -> vec3<f32> {
    var source = positive_bloom_source(color);
    if (threshold_precomputations.x > 0.0) {
        source = soft_threshold(source, threshold_precomputations);
    }
    return source;
}

fn composite(
    scene_rgb: vec3<f32>,
    bloom_rgb: vec3<f32>,
    threshold_precomputations: vec4<f32>,
    intensity: f32,
    energy_conserving: f32,
) -> vec3<f32> {
    let t = clamp(intensity, 0.0, 1.0);
    let energy = scene_rgb + t * (bloom_rgb - bloom_source(scene_rgb, threshold_precomputations));
    let additive = scene_rgb + intensity * bloom_rgb;
    return mix(additive, energy, energy_conserving);
}
