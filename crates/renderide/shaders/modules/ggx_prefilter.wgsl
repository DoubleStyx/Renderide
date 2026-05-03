//! GGX importance sampling helpers shared by IBL prefiltering and analytic skybox bake passes.
//!
//! Provides the math used to convolve a radiance environment into a roughness-keyed mip pyramid
//! via Karis-style split-sum + Filament solid-angle source-mip selection.
//!
//! Roughness convention: every helper takes **perceptual roughness** `r in [0, 1]`. The runtime
//! sampling side ([`renderide::pbs::brdf::skybox_specular_lod`]) uses the parabolic
//! `lod = max_lod * r * (2 - r)`. The prefilter side inverts it via [`lod_to_perceptual_roughness`]
//! so mip *i* is filtered for the roughness that the runtime LOD lookup will land on.

#define_import_path renderide::ggx_prefilter

const PI: f32 = 3.14159265358979323846;

/// Cubemap face direction at integer face/texel coordinates `(face, x, y)` for an `n`-edge face.
///
/// Convention matches [`renderide::skybox_evaluator::cube_dir`].
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

/// Van der Corput radical inverse for low-discrepancy 2D sampling.
fn radical_inverse_vdc(bits_in: u32) -> f32 {
    var bits = bits_in;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return f32(bits) * 2.3283064365386963e-10;
}

/// Hammersley low-discrepancy 2D sample `i / n` paired with the radical inverse of `i`.
fn hammersley(i: u32, n: u32) -> vec2<f32> {
    return vec2<f32>(f32(i) / f32(max(n, 1u)), radical_inverse_vdc(i));
}

/// Builds an orthonormal basis from `n` and rotates a tangent-space vector into world space.
fn tangent_to_world(local_dir: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let up = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(n.z) < 0.999);
    let tangent = normalize(cross(up, n));
    let bitangent = cross(n, tangent);
    return normalize(tangent * local_dir.x + bitangent * local_dir.y + n * local_dir.z);
}

/// GGX importance sample of the half vector for perceptual roughness `r` around normal `n`.
fn importance_sample_ggx(xi: vec2<f32>, r: f32, n: vec3<f32>) -> vec3<f32> {
    let alpha = max(r * r, 0.0001);
    let alpha_sq = alpha * alpha;
    let phi = 2.0 * PI * xi.x;
    let cos_theta = sqrt((1.0 - xi.y) / max(1.0 + (alpha_sq - 1.0) * xi.y, 0.000001));
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    let h = vec3<f32>(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
    return tangent_to_world(h, n);
}

/// GGX/Trowbridge-Reitz NDF for `n_dot_h` and perceptual roughness `r`.
fn d_ggx(n_dot_h: f32, r: f32) -> f32 {
    let alpha = max(r * r, 0.0001);
    let alpha_sq = alpha * alpha;
    let f = (n_dot_h * alpha_sq - n_dot_h) * n_dot_h + 1.0;
    return alpha_sq / max(PI * f * f, 1e-7);
}

/// PDF of the GGX importance sampling distribution, assuming `v == n` (Filament convention).
fn ggx_sample_pdf(n_dot_h: f32, r: f32) -> f32 {
    let d = d_ggx(n_dot_h, r);
    return max(d * n_dot_h * 0.25, 1e-7);
}

/// Source mip level whose texel solid angle ~= the importance sample's solid angle (Filament).
///
/// `omega_p = 4PI / (6 * src_face^2)` is the per-base-texel solid angle and `omega_s = 1 / (n*pdf)`
/// is the importance sample's. `lod = log4(K * omega_s / omega_p)` with `K = 4` (box filter
/// constant) folds into `0.5 * log2(omega_s / omega_p) + 1.0`.
fn solid_angle_lod(pdf: f32, sample_count: u32, src_face_size: u32) -> f32 {
    let omega_p = 4.0 * PI / (6.0 * f32(max(src_face_size, 1u)) * f32(max(src_face_size, 1u)));
    let omega_s = 1.0 / (f32(max(sample_count, 1u)) * max(pdf, 1e-7));
    return 0.5 * log2(max(omega_s / omega_p, 1e-7)) + 1.0;
}

/// Inverse of the runtime parabolic LOD lookup. Given `t = mip / max_mip in [0, 1]`, returns the
/// perceptual roughness that the runtime `lod = max_lod * r * (2 - r)` would map back to mip *i*.
fn lod_to_perceptual_roughness(t: f32) -> f32 {
    return 1.0 - sqrt(max(1.0 - clamp(t, 0.0, 1.0), 0.0));
}
