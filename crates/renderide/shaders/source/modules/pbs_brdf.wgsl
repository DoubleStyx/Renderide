//! Filament-style analytic Cook–Torrance BRDF and clustered direct-light terms for PBS materials
//! (metallic / specular workflows).
//!
//! Math reference: Filament `surface_brdf.fs`, `surface_material.fs`, `surface_shading_*.fs`. Specifically
//! - D: GGX/Trowbridge–Reitz, Karis-style numerically stable form (`d_ggx`).
//! - V: height-correlated Smith–GGX visibility (`v_smith_ggx_correlated`); already folds in the
//!   `1/(4·NoL·NoV)` denominator, so the assembled specular is `D · V · F` (no extra divide).
//! - F: Schlick with `f90 = saturate(50·dot(f0, 1/3))` so dielectrics fade to zero at grazing.
//! - Diffuse: Lambert (`1/π`); diffuse reflectance is pre-multiplied by `(1 − metallic)` (or by
//!   `one_minus_reflectivity` in the specular workflow) — there is no extra `(1 − F)` discount on
//!   the *direct* term, which is the IBL split-sum convention rather than the analytic one.
//!
//! Public entry contract: callers pass **perceptual roughness** (`= 1 − smoothness`, clamped to
//! `[0.045, 1.0]`). Squaring to linear `α` happens once inside this module, matching Unity BiRP's
//! `BRDF1_Unity_PBS` convention so material shaders stay unchanged.
//!
//! Import with `#import renderide::pbs::brdf`. Depends on [`renderide::globals`] for [`GpuLight`].

#import renderide::globals as rg

#define_import_path renderide::pbs::brdf

/// Lower bound on linear roughness `α`. Below this the GGX lobe becomes a near-delta that produces
/// fp16 sparkles and division-by-near-zero artefacts; matches Filament `MIN_ROUGHNESS` for desktop.
const MIN_ALPHA: f32 = 0.002025;

/// `(1 − x)^5` — used by Schlick Fresnel.
fn pow5(x: f32) -> f32 {
    let x2 = x * x;
    return x2 * x2 * x;
}

/// GGX/Trowbridge–Reitz NDF in Karis's numerically stable form.
///
/// Returns `α² / (π · ((NoH²)(α²−1)+1)²)`, rearranged through `k = α / (1 − NoH² + (NoH·α)²)` so
/// the squaring stays well-conditioned at very small `α`. `roughness` is **linear** (`α`).
fn d_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = n_dot_h * roughness;
    let k = roughness / max(1.0 - n_dot_h * n_dot_h + a * a, 1e-7);
    return min(k * k * (1.0 / 3.14159265), 65504.0);
}

/// Height-correlated Smith–GGX visibility (Heitz 2014). Returns `0.5 / (λV + λL)`, which already
/// folds in the `1/(4·NoL·NoV)` denominator of Cook–Torrance. `roughness` is **linear** (`α`).
fn v_smith_ggx_correlated(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let a2 = roughness * roughness;
    let lv = n_dot_l * sqrt((n_dot_v - a2 * n_dot_v) * n_dot_v + a2);
    let ll = n_dot_v * sqrt((n_dot_l - a2 * n_dot_l) * n_dot_l + a2);
    return 0.5 / max(lv + ll, 1e-7);
}

/// Schlick approximation of the Fresnel term.
///
/// `f90` lets dielectrics with very low `f0` smoothly fade to zero at grazing instead of always
/// snapping to white. Filament computes it as `saturate(50·dot(f0, 1/3))`, encoding "if the
/// material has any meaningful base reflectance, it should reach ~1 at grazing."
fn f_schlick(f0: vec3<f32>, f90: f32, v_dot_h: f32) -> vec3<f32> {
    return f0 + (vec3<f32>(f90) - f0) * pow5(1.0 - v_dot_h);
}

/// Filament's `f90` from `f0`. `50·(1/3) ≈ 16.67`; saturated so very dark dielectrics don't go to white.
fn f90_from_f0(f0: vec3<f32>) -> f32 {
    return clamp(dot(f0, vec3<f32>(50.0 / 3.0)), 0.0, 1.0);
}

/// Lambertian diffuse normalization (`1/π`).
fn fd_lambert() -> f32 {
    return 1.0 / 3.14159265;
}

/// Normalized windowed inverse-square distance attenuation for punctual lights.
/// `(saturate(1 − t⁴))² / max(t², ε²)` evaluated in `t = dist/range` space so the entire falloff
/// shape stretches with the light's range slider rather than clipping a world-space inverse-square
/// curve. Matches Unity BiRP's LUT-style behaviour where the range slider only changes how far the
/// light reaches, not its peak brightness; the Karis/Lagarde quartic window keeps the boundary at
/// `dist == range` smooth and exactly zero. The `ε = 0.01` floor (relative to range) caps the
/// near-light singularity at a range-independent peak. Intensity is applied by the call site.
fn distance_attenuation(dist: f32, range: f32) -> f32 {
    if (range <= 0.0) {
        return 0.0;
    }
    let t = dist / range;
    let t2 = max(t * t, 0.0001);
    let window_inner = clamp(1.0 - t2 * t2, 0.0, 1.0);
    let window = window_inner * window_inner;
    return window / t2;
}

/// Result of evaluating one punctual light at a surface point.
struct LightSample {
    /// Direction from the surface toward the light source (unit length when `attenuation > 0`).
    l: vec3<f32>,
    /// Combined intensity, distance, and spot attenuation (already includes `light.intensity`).
    attenuation: f32,
}

/// Resolves the per-light-type direction and attenuation. Single source of truth for point /
/// directional / spot dispatch shared by all four direct-radiance functions in this module.
fn eval_light(light: rg::GpuLight, world_pos: vec3<f32>) -> LightSample {
    let light_pos = light.position.xyz;
    let light_dir = light.direction.xyz;
    var out: LightSample;
    if light.light_type == 0u {
        let to_light = light_pos - world_pos;
        let dist = length(to_light);
        out.l = normalize(to_light);
        out.attenuation = light.intensity * distance_attenuation(dist, light.range);
    } else if light.light_type == 1u {
        let dir_len_sq = dot(light_dir, light_dir);
        out.l = select(vec3<f32>(0.0, 0.0, 1.0), normalize(-light_dir), dir_len_sq > 1e-16);
        out.attenuation = light.intensity;
    } else {
        let to_light = light_pos - world_pos;
        let dist = length(to_light);
        out.l = normalize(to_light);
        let spot_cos = dot(-out.l, normalize(light_dir));
        let inner_cos = min(light.spot_cos_half_angle + 0.1, 1.0);
        let spot_atten = smoothstep(light.spot_cos_half_angle, inner_cos, spot_cos);
        out.attenuation = light.intensity * spot_atten * distance_attenuation(dist, light.range);
    }
    return out;
}

/// Filament-style direct radiance for the metallic workflow.
///
/// `roughness` is perceptual (caller passes `1 − smoothness`, clamped to `[0.045, 1.0]`). `f0` is
/// the dielectric-↔-metal blend (`mix(0.04, base_color, metallic)`). Diffuse is pre-discounted by
/// `(1 − metallic)` only — the `(1 − F)` term is intentionally absent for the analytic direct lobe.
fn direct_radiance_metallic(
    light: rg::GpuLight,
    world_pos: vec3<f32>,
    n: vec3<f32>,
    v: vec3<f32>,
    roughness: f32,
    metallic: f32,
    base_color: vec3<f32>,
    f0: vec3<f32>,
) -> vec3<f32> {
    let ls = eval_light(light, world_pos);
    let n_dot_l = max(dot(n, ls.l), 0.0);
    if n_dot_l <= 0.0 {
        return vec3<f32>(0.0);
    }
    let h = normalize(v + ls.l);
    let n_dot_v = max(dot(n, v), 1e-4);
    let n_dot_h = clamp(dot(n, h), 0.0, 1.0);
    let v_dot_h = clamp(dot(v, h), 0.0, 1.0);

    let alpha = max(roughness * roughness, MIN_ALPHA);
    let f90 = f90_from_f0(f0);
    let f = f_schlick(f0, f90, v_dot_h);
    let d = d_ggx(n_dot_h, alpha);
    let vis = v_smith_ggx_correlated(n_dot_v, n_dot_l, alpha);
    let fr = (d * vis) * f;

    let diffuse_color = base_color * (1.0 - metallic);
    let fd = diffuse_color * fd_lambert();

    let radiance = light.color.xyz * ls.attenuation * n_dot_l;
    return (fd + fr) * radiance;
}

/// Filament-style direct radiance for the specular (Unity Standard SpecularSetup) workflow.
///
/// `roughness` is perceptual. `f0` is the tinted specular color from the host (already encodes the
/// dielectric/metal split chosen by the artist). `one_minus_reflectivity` is the diffuse-energy
/// discount derived from `f0`'s peak channel (Unity `EnergyConservationBetweenDiffuseAndSpecular`).
/// As in the metallic path, no extra `(1 − F)` is applied to direct diffuse.
fn direct_radiance_specular(
    light: rg::GpuLight,
    world_pos: vec3<f32>,
    n: vec3<f32>,
    v: vec3<f32>,
    roughness: f32,
    base_color: vec3<f32>,
    f0: vec3<f32>,
    one_minus_reflectivity: f32,
) -> vec3<f32> {
    let ls = eval_light(light, world_pos);
    let n_dot_l = max(dot(n, ls.l), 0.0);
    if n_dot_l <= 0.0 {
        return vec3<f32>(0.0);
    }
    let h = normalize(v + ls.l);
    let n_dot_v = max(dot(n, v), 1e-4);
    let n_dot_h = clamp(dot(n, h), 0.0, 1.0);
    let v_dot_h = clamp(dot(v, h), 0.0, 1.0);

    let alpha = max(roughness * roughness, MIN_ALPHA);
    let f90 = f90_from_f0(f0);
    let f = f_schlick(f0, f90, v_dot_h);
    let d = d_ggx(n_dot_h, alpha);
    let vis = v_smith_ggx_correlated(n_dot_v, n_dot_l, alpha);
    let fr = (d * vis) * f;

    let diffuse_color = base_color * one_minus_reflectivity;
    let fd = diffuse_color * fd_lambert();

    let radiance = light.color.xyz * ls.attenuation * n_dot_l;
    return (fd + fr) * radiance;
}

/// Lambertian direct radiance only (specular highlights disabled), metallic path. Diffuse is
/// pre-discounted by `(1 − metallic)` so disabling specular on a metal still produces a near-black
/// surface (correct: a perfect metal has no diffuse channel).
fn diffuse_only_metallic(
    light: rg::GpuLight,
    world_pos: vec3<f32>,
    n: vec3<f32>,
    base_color: vec3<f32>,
    metallic: f32,
) -> vec3<f32> {
    let ls = eval_light(light, world_pos);
    let n_dot_l = max(dot(n, ls.l), 0.0);
    let diffuse_color = base_color * (1.0 - metallic);
    return diffuse_color * fd_lambert() * light.color.xyz * ls.attenuation * n_dot_l;
}

/// Lambertian direct radiance only, specular workflow (diffuse pre-discounted by `one_minus_reflectivity`).
fn diffuse_only_specular(
    light: rg::GpuLight,
    world_pos: vec3<f32>,
    n: vec3<f32>,
    base_color: vec3<f32>,
    one_minus_reflectivity: f32,
) -> vec3<f32> {
    let ls = eval_light(light, world_pos);
    let n_dot_l = max(dot(n, ls.l), 0.0);
    return base_color * one_minus_reflectivity * fd_lambert() * light.color.xyz * ls.attenuation * n_dot_l;
}
