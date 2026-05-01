//! Shared helpers for the Resonite Unity toon shaders.

#define_import_path renderide::toon::brdf

#import renderide::pbs::sampling as psamp

/// Stepped wrapped-Lambert diffuse matching the Unity ToonBRDF diffuse cadence.
fn diffuse(n: vec3<f32>, l: vec3<f32>, transmission: f32) -> f32 {
    let nl = dot(n, l);
    let denom = (1.0 + transmission) * (1.0 + transmission);
    let wrapped = clamp((nl + transmission) / max(denom, 1e-4), 0.0, 1.0);
    return min(round(wrapped * 2.0) / 2.0 + transmission, 1.0);
}

/// Stepped normalized Blinn-Phong specular, used as an analytical replacement for Unity's LUT.
fn specular(n: vec3<f32>, l: vec3<f32>, v: vec3<f32>, smoothness: f32, specular_highlights: f32) -> f32 {
    if (specular_highlights < 0.5) {
        return 0.0;
    }
    let nl = max(dot(n, l), 0.0);
    let r = reflect(-v, n);
    let rl = max(dot(r, l), 0.0);
    let rough = psamp::roughness_from_smoothness(smoothness);
    let shininess = (1.0 - rough) * (1.0 - rough) * 256.0 + 1.0;
    let raw = pow(rl, shininess) * (shininess + 8.0) / (8.0 * 3.14159265);
    let steps = max((1.0 - smoothness) * 4.0, 0.01);
    let stepped = round(raw * steps) / steps;
    return stepped * nl;
}

/// View-dependent stylization rim from the Unity ToonBRDF Fresnel implementation.
fn fresnel(
    diff_color: vec3<f32>,
    view_dir: vec3<f32>,
    n: vec3<f32>,
    enabled: f32,
    diffuse_contribution: f32,
    power: f32,
    strength: f32,
    tint: vec3<f32>,
) -> vec3<f32> {
    if (enabled < 0.5) {
        return vec3<f32>(0.0);
    }
    let rim = 1.0 - clamp(dot(normalize(view_dir), n), 0.0, 1.0);
    let fresnel_color = mix(vec3<f32>(0.5), diff_color, diffuse_contribution);
    let fresnel_power = pow(rim, max(20.0 - power * 20.0, 1e-4));
    let fresnel = fresnel_color * fresnel_power;
    return (strength * 5.0) * fresnel * tint;
}
