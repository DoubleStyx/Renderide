//! Shared helpers for the Resonite Unity toon shaders.

#define_import_path renderide::material::toon_brdf

#import renderide::pbs::sampling as psamp

/// Stepped wrapped-Lambert diffuse matching the Unity ToonBRDF diffuse cadence.
fn diffuse(n: vec3<f32>, l: vec3<f32>, transmission: f32) -> f32 {
    let nl = dot(n, l);
    let denom = (1.0 + transmission) * (1.0 + transmission);
    let wrapped = clamp((nl + transmission) / max(denom, 1e-4), 0.0, 1.0);
    return min(round(wrapped * 2.0) / 2.0 + transmission, 1.0);
}

/// Unity Standard SpecularSetup diffuse reflectivity remainder.
fn one_minus_reflectivity(spec_color: vec3<f32>) -> f32 {
    return 1.0 - max(max(spec_color.r, spec_color.g), spec_color.b);
}

/// Unity `EnergyConservationBetweenDiffuseAndSpecular` diffuse reduction.
fn energy_conserved_diffuse(base_color: vec3<f32>, spec_color: vec3<f32>) -> vec3<f32> {
    return base_color * clamp(one_minus_reflectivity(spec_color), 0.0, 1.0);
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

/// Direct toon-light contribution matching Unity ToonBRDF's `(diffuse + specular) * steppedDiffuse`.
fn direct_light(
    diff_color: vec3<f32>,
    spec_color: vec3<f32>,
    n: vec3<f32>,
    l: vec3<f32>,
    v: vec3<f32>,
    smoothness: f32,
    transmission: f32,
    specular_highlights: f32,
    radiance: vec3<f32>,
) -> vec3<f32> {
    let diffuse_step = diffuse(n, l, transmission);
    let specular_step = specular(n, l, v, smoothness, specular_highlights);
    return radiance * (diff_color + spec_color * specular_step) * diffuse_step;
}

/// Indirect toon contribution matching Unity's diffuse SH plus glossy specular blend.
fn indirect_light(
    diff_color: vec3<f32>,
    spec_color: vec3<f32>,
    one_minus_reflectivity_value: f32,
    smoothness: f32,
    n: vec3<f32>,
    v: vec3<f32>,
    ambient: vec3<f32>,
    specular_radiance: vec3<f32>,
) -> vec3<f32> {
    let nv = clamp(dot(n, v), 0.0, 1.0);
    let fresnel_term = pow(1.0 - nv, 4.0);
    let grazing_term = clamp(smoothness + (1.0 - one_minus_reflectivity_value), 0.0, 1.0);
    let specular_tint = mix(spec_color, vec3<f32>(grazing_term), fresnel_term);
    return ambient * diff_color + specular_radiance * specular_tint;
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
