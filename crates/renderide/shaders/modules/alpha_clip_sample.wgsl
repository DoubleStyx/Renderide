//! Base-mip sampling for **alpha test** and **mask clip** (`discard` vs `_Cutoff`).
//!
//! Implicit `textureSample` LOD follows screen derivatives; coverage used for clip can drift at
//! distance. Use [`texture_alpha_base_mip`] / [`mask_luminance_mul_base_mip`] / [`texture_rgba_base_mip`]
//! only for **discard** decisions; keep `textureSample` for RGB where mip filtering is desired.
//!
//! Import with `#import renderide::alpha_clip_sample as acs`.

#define_import_path renderide::alpha_clip_sample

/// LOD index for coverage-only comparisons (stable silhouette vs implicit mips).
const CLIP_COVERAGE_LOD: f32 = 0.0;

/// Alpha channel of `tex` at [`CLIP_COVERAGE_LOD`] (for `_ALPHATEST` / tint x alpha).
fn texture_alpha_base_mip(tex: texture_2d<f32>, samp: sampler, uv: vec2<f32>) -> f32 {
    return textureSampleLevel(tex, samp, uv, CLIP_COVERAGE_LOD).a;
}

/// Full sample at [`CLIP_COVERAGE_LOD`] (e.g. Fresnel / overlay clip parity with tinted RGBA).
fn texture_rgba_base_mip(tex: texture_2d<f32>, samp: sampler, uv: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(tex, samp, uv, CLIP_COVERAGE_LOD);
}

/// Unity-style mask clip: `(r+g+b)/3 * a` at [`CLIP_COVERAGE_LOD`].
fn mask_luminance_mul_base_mip(tex: texture_2d<f32>, samp: sampler, uv: vec2<f32>) -> f32 {
    let mask = textureSampleLevel(tex, samp, uv, CLIP_COVERAGE_LOD);
    return (mask.r + mask.g + mask.b) * 0.33333334 * mask.a;
}
