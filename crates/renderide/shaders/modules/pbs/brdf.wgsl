//! Analytic Cook-Torrance BRDF and clustered direct-light terms for PBS materials (metallic /
//! specular workflows).
//!
//! - D: GGX/Trowbridge-Reitz, Karis-style numerically stable form (`d_ggx`).
//! - V: height-correlated Smith-GGX visibility (`v_smith_ggx_correlated`); already folds in the
//!   `1/(4*NoL*NoV)` denominator, so the assembled specular is `D * V * F` (no extra divide).
//! - F: Schlick with `f90 = saturate(50*dot(f0, 1/3))`.
//! - Diffuse: Disney/Burley rough diffuse; diffuse reflectance is pre-multiplied by
//!   `(1 - metallic)` (or by `one_minus_reflectivity` in the specular workflow) and the direct lobe
//!   is scaled by normalized Fresnel transmission so diffuse falls away as specular reflection rises
//!   at grazing angles.
//!
//! Public entry contract: callers pass **perceptual roughness** (`= 1 - smoothness`, clamped to
//! `[0.0, 1.0]`). Direct GGX paths apply Unity BiRP's linear-roughness floor only when evaluating
//! analytic specular, leaving reflection-probe LOD free to reach mip 0 for mirror-smooth surfaces.
//!
//! Import with `#import renderide::pbs::brdf`. Depends on frame type layouts for [`GpuLight`].

#import renderide::frame::globals as rg
#import renderide::frame::types as ft
#import renderide::lighting::birp as bl

#define_import_path renderide::pbs::brdf

/// Lower bound on direct linear roughness `alpha` used by Unity BiRP's `BRDF1_Unity_PBS`.
///
/// Below this the GGX lobe becomes a near-delta that produces fp16 sparkles and
/// division-by-near-zero artefacts. Keep this floor out of indirect reflection-probe LOD selection.
const MIN_ALPHA: f32 = 0.002;
/// Default dielectric reflectance slider value for the Standard PBS path.
const DEFAULT_DIELECTRIC_REFLECTANCE: f32 = 0.5;
/// Default dielectric F0 for an air-to-1.5-IOR interface.
const DEFAULT_DIELECTRIC_F0: f32 = 0.16 * DEFAULT_DIELECTRIC_REFLECTANCE * DEFAULT_DIELECTRIC_REFLECTANCE;
/// Pi.
const PI: f32 = 3.14159265359;

/// Variance scale for [`filter_perceptual_roughness`].
const SPECULAR_AA_VARIANCE: f32 = 0.25;

/// Maximum kernel-roughness widening for [`filter_perceptual_roughness`]. Caps the filter so very
/// high curvature doesn't drive the entire surface to a fully-rough lobe.
const SPECULAR_AA_THRESHOLD: f32 = 0.18;

/// Tokuyoshi & Kaplanyan 2019 "Improved Geometric Specular Antialiasing".
///
/// Widens the GGX lobe by the screen-space variance of the surface normal so that sub-pixel
/// normal jitter does not alias into the specular highlight. MSAA can only multisample geometric
/// coverage; the fragment shader still runs once per pixel, so a narrow specular lobe evaluated
/// at the pixel centre will sparkle on curved metals regardless of MSAA tier. This filter widens
/// `alpha` per pixel based on `dpdx`/`dpdy` of the world normal, producing a softer pre-filtered lobe
/// where the normal is changing fast.
///
/// `perceptual_roughness` is `1 - smoothness` (this module's standard input), and the returned
/// value is also perceptual -- call sites can drop-in replace their existing `roughness` and the
/// downstream BRDF squares to `alpha` once as before.
///
/// Fragment-only (uses derivatives). Call once before the cluster light loop so the derivatives
/// evaluate at uniform control flow and the widened roughness is shared across all light samples.
fn filter_perceptual_roughness(perceptual_roughness: f32, world_n: vec3<f32>) -> f32 {
    let du = dpdx(world_n);
    let dv = dpdy(world_n);
    let variance = SPECULAR_AA_VARIANCE * (dot(du, du) + dot(dv, dv));
    let clamped = clamp(perceptual_roughness, 0.0, 1.0);
    let alpha = clamped * clamped;
    let kernel = min(2.0 * variance, SPECULAR_AA_THRESHOLD);
    let alpha2 = clamp(alpha * alpha + kernel, 0.0, 1.0);
    return sqrt(sqrt(alpha2));
}

/// Linear GGX roughness for direct-light BRDF evaluation.
fn direct_alpha_from_perceptual_roughness(perceptual_roughness: f32) -> f32 {
    let clamped = clamp(perceptual_roughness, 0.0, 1.0);
    return max(clamped * clamped, MIN_ALPHA);
}

/// Perceptual roughness after applying the direct-light GGX floor.
fn direct_perceptual_roughness(perceptual_roughness: f32) -> f32 {
    return sqrt(direct_alpha_from_perceptual_roughness(perceptual_roughness));
}

/// `(1 - x)^5` -- used by Schlick Fresnel.
fn pow5(x: f32) -> f32 {
    let x2 = x * x;
    return x2 * x2 * x;
}

/// Maximum component of a linear RGB reflectance value.
fn max_component(v: vec3<f32>) -> f32 {
    return max(max(v.r, v.g), v.b);
}

/// GGX/Trowbridge-Reitz NDF in Karis's numerically stable form.
///
/// Returns `alpha^2 / (PI * ((NoH^2)(alpha^2-1)+1)^2)`, rearranged through `k = alpha / (1 - NoH^2 + (NoH*alpha)^2)` so
/// the squaring stays well-conditioned at very small `alpha`. `roughness` is **linear** (`alpha`).
fn d_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = n_dot_h * roughness;
    let k = roughness / max(1.0 - n_dot_h * n_dot_h + a * a, 1e-7);
    return min(k * k * (1.0 / 3.14159265), 65504.0);
}

/// Height-correlated Smith-GGX visibility (Heitz 2014). Returns `0.5 / (lambdaV + lambdaL)`, which already
/// folds in the `1/(4*NoL*NoV)` denominator of Cook-Torrance. `roughness` is **linear** (`alpha`).
fn v_smith_ggx_correlated(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let a2 = roughness * roughness;
    let lv = n_dot_l * sqrt((n_dot_v - a2 * n_dot_v) * n_dot_v + a2);
    let ll = n_dot_v * sqrt((n_dot_l - a2 * n_dot_l) * n_dot_l + a2);
    return 0.5 / max(lv + ll, 1e-7);
}

/// Schlick approximation of the Fresnel term.
///
/// `f90` lets dielectrics with very low `f0` smoothly fade to zero at grazing instead of always
/// snapping to white. The `50*dot(f0, 1/3)` scale means a material with meaningful base
/// reflectance reaches approximately 1 at grazing.
fn f_schlick(f0: vec3<f32>, f90: f32, v_dot_h: f32) -> vec3<f32> {
    return f0 + (vec3<f32>(f90) - f0) * pow5(1.0 - v_dot_h);
}

/// Scalar Schlick approximation used by Disney/Burley rough diffuse.
fn f_schlick_scalar(f0: f32, f90: f32, cos_theta: f32) -> f32 {
    return f0 + (f90 - f0) * pow5(1.0 - clamp(cos_theta, 0.0, 1.0));
}

/// Derives `f90` from `f0`. `50*(1/3) ~= 16.67`; saturated so very dark dielectrics don't go to white.
fn f90_from_f0(f0: vec3<f32>) -> f32 {
    return clamp(dot(f0, vec3<f32>(50.0 / 3.0)), 0.0, 1.0);
}

/// Scalar transmission for the direct diffuse lobe derived from the current Fresnel term.
///
/// The normalization keeps the material's authored normal-incidence diffuse response stable because
/// the metallic and specular workflows already encode the F0 energy split in their diffuse color.
fn direct_diffuse_fresnel_transmission(f: vec3<f32>, f0: vec3<f32>) -> f32 {
    let f_peak = max_component(clamp(f, vec3<f32>(0.0), vec3<f32>(1.0)));
    let f0_peak = max_component(clamp(f0, vec3<f32>(0.0), vec3<f32>(1.0)));
    return clamp((1.0 - f_peak) / max(1.0 - f0_peak, 1e-4), 0.0, 1.0);
}

/// Samples the frame-global DFG LUT with manual bilinear filtering.
fn sample_ibl_dfg_lut(perceptual_roughness: f32, n_dot_v: f32) -> vec2<f32> {
    let dims_u = textureDimensions(rg::ibl_dfg_lut);
    let dims = vec2<f32>(f32(dims_u.x), f32(dims_u.y));
    let max_xy = vec2<i32>(i32(dims_u.x) - 1, i32(dims_u.y) - 1);
    let uv = vec2<f32>(
        clamp(n_dot_v, 0.0, 1.0),
        clamp(perceptual_roughness, 0.0, 1.0),
    );
    let xy = uv * dims - vec2<f32>(0.5);
    let base = floor(xy);
    let base_i = vec2<i32>(base);
    let f = xy - base;
    let p00 = clamp(base_i, vec2<i32>(0), max_xy);
    let p10 = clamp(base_i + vec2<i32>(1, 0), vec2<i32>(0), max_xy);
    let p01 = clamp(base_i + vec2<i32>(0, 1), vec2<i32>(0), max_xy);
    let p11 = clamp(base_i + vec2<i32>(1, 1), vec2<i32>(0), max_xy);
    let a = textureLoad(rg::ibl_dfg_lut, p00, 0).rg;
    let b = textureLoad(rg::ibl_dfg_lut, p10, 0).rg;
    let c = textureLoad(rg::ibl_dfg_lut, p01, 0).rg;
    let d = textureLoad(rg::ibl_dfg_lut, p11, 0).rg;
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

/// Split-sum specular energy for the frame-global DFG LUT.
fn specular_energy_from_dfg(dfg: vec2<f32>, f0: vec3<f32>) -> vec3<f32> {
    let clamped_f0 = clamp(f0, vec3<f32>(0.0), vec3<f32>(1.0));
    let f90 = vec3<f32>(f90_from_f0(clamped_f0));
    return clamped_f0 * (dfg.y - dfg.x) + f90 * dfg.x;
}

/// Multiple-scattering compensation for the direct microfacet lobe.
fn energy_compensation_from_dfg(dfg: vec2<f32>, f0: vec3<f32>) -> vec3<f32> {
    let clamped_f0 = clamp(f0, vec3<f32>(0.0), vec3<f32>(1.0));
    return vec3<f32>(1.0) + clamped_f0 * (vec3<f32>(1.0 / max(dfg.y, 1e-4)) - vec3<f32>(1.0));
}

/// Split-sum specular energy for the frame-global DFG LUT.
fn indirect_specular_energy_from_dfg(dfg: vec2<f32>, f0: vec3<f32>, enabled: bool) -> vec3<f32> {
    if (!enabled) {
        return vec3<f32>(0.0);
    }
    return specular_energy_from_dfg(dfg, f0);
}

/// Simple material-AO to specular-AO remap for indirect reflections.
fn specular_ao_lagarde(n_dot_v: f32, visibility: f32, perceptual_roughness: f32) -> f32 {
    let no_v = clamp(n_dot_v, 0.0, 1.0);
    let ao = clamp(visibility, 0.0, 1.0);
    let linear_roughness = clamp(perceptual_roughness, 0.0, 1.0) * clamp(perceptual_roughness, 0.0, 1.0);
    let exponent = exp2(-16.0 * linear_roughness - 1.0);
    return clamp(pow(no_v + ao, exponent) - 1.0 + ao, 0.0, 1.0);
}

/// Multi-bounce diffuse visibility for material AO.
fn indirect_diffuse_visibility(visibility: f32, diffuse_color: vec3<f32>) -> vec3<f32> {
    return multi_bounce_visibility(visibility, diffuse_color);
}

/// Multi-bounce specular visibility for material AO and the given F0.
fn indirect_specular_visibility(
    n_dot_v: f32,
    visibility: f32,
    perceptual_roughness: f32,
    f0: vec3<f32>,
) -> vec3<f32> {
    let single_bounce = specular_ao_lagarde(n_dot_v, visibility, perceptual_roughness);
    return multi_bounce_visibility(single_bounce, f0);
}

/// Jimenez-style multi-bounce visibility fit for colored diffuse and specular reflectance.
fn multi_bounce_visibility(visibility: f32, albedo: vec3<f32>) -> vec3<f32> {
    let ao = clamp(visibility, 0.0, 1.0);
    let clamped_albedo = clamp(albedo, vec3<f32>(0.0), vec3<f32>(1.0));
    let a = 2.0404 * clamped_albedo - vec3<f32>(0.3324);
    let b = -4.7951 * clamped_albedo + vec3<f32>(0.6417);
    let c = 2.7552 * clamped_albedo + vec3<f32>(0.6903);
    return max(vec3<f32>(ao), ((ao * a + b) * ao + c) * ao);
}

/// Indirect-diffuse scale paired with the split-sum specular energy.
fn indirect_diffuse_energy_scale(specular_energy: vec3<f32>, enabled: bool) -> vec3<f32> {
    if (!enabled) {
        return vec3<f32>(1.0);
    }
    return max(vec3<f32>(0.0), vec3<f32>(1.0) - specular_energy);
}

/// Indirect diffuse term for Unity Standard metallic materials.
fn indirect_diffuse_metallic(
    ambient: vec3<f32>,
    base_color: vec3<f32>,
    metallic: f32,
    specular_energy: vec3<f32>,
    occlusion: f32,
    glossy_reflections_enabled: bool,
) -> vec3<f32> {
    let energy_scale = indirect_diffuse_energy_scale(specular_energy, glossy_reflections_enabled);
    let diffuse_color = base_color * (1.0 - clamp(metallic, 0.0, 1.0));
    let visibility = indirect_diffuse_visibility(occlusion, diffuse_color);
    return ambient * diffuse_color * energy_scale * visibility;
}

/// Indirect diffuse term for Unity Standard specular materials.
fn indirect_diffuse_specular(
    ambient: vec3<f32>,
    base_color: vec3<f32>,
    one_minus_reflectivity: f32,
    specular_energy: vec3<f32>,
    occlusion: f32,
    glossy_reflections_enabled: bool,
) -> vec3<f32> {
    let energy_scale = indirect_diffuse_energy_scale(specular_energy, glossy_reflections_enabled);
    let diffuse_color = base_color * clamp(one_minus_reflectivity, 0.0, 1.0);
    let visibility = indirect_diffuse_visibility(occlusion, diffuse_color);
    return ambient * diffuse_color * energy_scale * visibility;
}

/// Unity Standard metallic workflow diffuse reflectivity remainder.
fn metallic_one_minus_reflectivity(metallic: f32) -> f32 {
    let one_minus_dielectric_spec = 1.0 - DEFAULT_DIELECTRIC_F0;
    return one_minus_dielectric_spec * (1.0 - clamp(metallic, 0.0, 1.0));
}

/// Unity Standard premultiplied transparency output alpha.
fn unity_premultiplied_alpha(alpha: f32, one_minus_reflectivity: f32) -> f32 {
    let diffuse_alpha = clamp(alpha, 0.0, 1.0);
    let diffuse_visibility = clamp(one_minus_reflectivity, 0.0, 1.0);
    return 1.0 - diffuse_visibility + diffuse_alpha * diffuse_visibility;
}

/// Unity Standard metallic workflow F0 tint.
fn metallic_f0(base_color: vec3<f32>, metallic: f32) -> vec3<f32> {
    return mix(vec3<f32>(DEFAULT_DIELECTRIC_F0), base_color, metallic);
}

/// Unity Standard SpecularSetup F0 tint.
fn specular_f0(specular_color: vec3<f32>) -> vec3<f32> {
    return clamp(specular_color, vec3<f32>(0.0), vec3<f32>(1.0));
}

/// Unity's diffuse-energy discount for SpecularSetup materials.
fn specular_one_minus_reflectivity(f0: vec3<f32>) -> f32 {
    return 1.0 - max(max(f0.r, f0.g), f0.b);
}

/// Disney/Burley rough diffuse normalization with grazing retro-reflection.
fn fd_burley(n_dot_v: f32, n_dot_l: f32, l_dot_h: f32, perceptual_roughness: f32) -> f32 {
    let roughness = clamp(perceptual_roughness, 0.0, 1.0);
    let loh = clamp(l_dot_h, 0.0, 1.0);
    let fd90 = 0.5 + 2.0 * roughness * loh * loh;
    let light_scatter = f_schlick_scalar(1.0, fd90, n_dot_l);
    let view_scatter = f_schlick_scalar(1.0, fd90, n_dot_v);
    return light_scatter * view_scatter * (1.0 / PI);
}

/// Unity BiRP-style distance attenuation for punctual lights.
///
/// Preserved as the `renderide::pbs::brdf` entry point for material shaders that already call it.
fn distance_attenuation(dist: f32, range: f32) -> f32 {
    return bl::distance_attenuation(dist, range);
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
fn eval_light(light: ft::GpuLight, world_pos: vec3<f32>) -> LightSample {
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
        out.attenuation = bl::direct_light_intensity(light.intensity);
    } else {
        let to_light = light_pos - world_pos;
        let dist = length(to_light);
        out.l = normalize(to_light);
        let spot_atten = bl::spot_angle_attenuation(light, out.l);
        out.attenuation = light.intensity * spot_atten * distance_attenuation(dist, light.range);
    }
    return out;
}

/// Signed direct radiance carried by one light sample before BRDF multiplication.
fn signed_light_radiance(light: ft::GpuLight, attenuation: f32, n_dot_l: f32) -> vec3<f32> {
    return light.color.xyz * attenuation * n_dot_l;
}

/// Direct radiance for the metallic workflow.
///
/// `specular_roughness` and `diffuse_roughness` are perceptual. The split keeps geometric specular
/// AA from widening the rough diffuse response. `f0` is the dielectric-<->-metal blend
/// (`mix(0.04, base_color, metallic)`). Diffuse is pre-discounted by `(1 - metallic)` and then
/// scaled by normalized Fresnel transmission for the analytic direct lobe.
fn direct_radiance_metallic(
    light: ft::GpuLight,
    world_pos: vec3<f32>,
    n: vec3<f32>,
    v: vec3<f32>,
    specular_roughness: f32,
    diffuse_roughness: f32,
    metallic: f32,
    base_color: vec3<f32>,
    f0: vec3<f32>,
    energy_compensation: vec3<f32>,
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
    let l_dot_h = clamp(dot(ls.l, h), 0.0, 1.0);

    let alpha = direct_alpha_from_perceptual_roughness(specular_roughness);
    let f90 = f90_from_f0(f0);
    let f = f_schlick(f0, f90, v_dot_h);
    let d = d_ggx(n_dot_h, alpha);
    let vis = v_smith_ggx_correlated(n_dot_v, n_dot_l, alpha);
    let fr = (d * vis) * f * energy_compensation;

    let diffuse_color = base_color * (1.0 - metallic);
    let fd = diffuse_color * direct_diffuse_fresnel_transmission(f, f0) * fd_burley(n_dot_v, n_dot_l, l_dot_h, diffuse_roughness);

    let radiance = signed_light_radiance(light, ls.attenuation, n_dot_l);
    return (fd + fr) * radiance;
}

/// Direct radiance for the specular (Unity Standard SpecularSetup) workflow.
///
/// `specular_roughness` and `diffuse_roughness` are perceptual. The split keeps geometric specular
/// AA from widening the rough diffuse response. `f0` is the tinted specular color from the host
/// (already encodes the dielectric/metal split chosen by the artist). `one_minus_reflectivity` is
/// the diffuse-energy discount derived from `f0`'s peak channel (Unity
/// `EnergyConservationBetweenDiffuseAndSpecular`). The direct diffuse lobe is scaled by normalized
/// Fresnel transmission after that F0 split.
fn direct_radiance_specular(
    light: ft::GpuLight,
    world_pos: vec3<f32>,
    n: vec3<f32>,
    v: vec3<f32>,
    specular_roughness: f32,
    diffuse_roughness: f32,
    base_color: vec3<f32>,
    f0: vec3<f32>,
    one_minus_reflectivity: f32,
    energy_compensation: vec3<f32>,
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
    let l_dot_h = clamp(dot(ls.l, h), 0.0, 1.0);

    let alpha = direct_alpha_from_perceptual_roughness(specular_roughness);
    let f90 = f90_from_f0(f0);
    let f = f_schlick(f0, f90, v_dot_h);
    let d = d_ggx(n_dot_h, alpha);
    let vis = v_smith_ggx_correlated(n_dot_v, n_dot_l, alpha);
    let fr = (d * vis) * f * energy_compensation;

    let diffuse_color = base_color * one_minus_reflectivity;
    let fd = diffuse_color * direct_diffuse_fresnel_transmission(f, f0) * fd_burley(n_dot_v, n_dot_l, l_dot_h, diffuse_roughness);

    let radiance = signed_light_radiance(light, ls.attenuation, n_dot_l);
    return (fd + fr) * radiance;
}

/// Rough diffuse direct radiance only (specular highlights disabled), metallic path. Diffuse is
/// pre-discounted by `(1 - metallic)` so disabling specular on a metal still produces a near-black
/// surface (correct: a perfect metal has no diffuse channel).
fn diffuse_only_metallic(
    light: ft::GpuLight,
    world_pos: vec3<f32>,
    n: vec3<f32>,
    v: vec3<f32>,
    roughness: f32,
    base_color: vec3<f32>,
    metallic: f32,
) -> vec3<f32> {
    let ls = eval_light(light, world_pos);
    let n_dot_l = max(dot(n, ls.l), 0.0);
    if n_dot_l <= 0.0 {
        return vec3<f32>(0.0);
    }
    let h = normalize(v + ls.l);
    let n_dot_v = max(dot(n, v), 1e-4);
    let l_dot_h = clamp(dot(ls.l, h), 0.0, 1.0);
    let diffuse_color = base_color * (1.0 - metallic);
    return diffuse_color * fd_burley(n_dot_v, n_dot_l, l_dot_h, roughness) * signed_light_radiance(light, ls.attenuation, n_dot_l);
}

/// Rough diffuse direct radiance only, specular workflow.
fn diffuse_only_specular(
    light: ft::GpuLight,
    world_pos: vec3<f32>,
    n: vec3<f32>,
    v: vec3<f32>,
    roughness: f32,
    base_color: vec3<f32>,
    one_minus_reflectivity: f32,
) -> vec3<f32> {
    let ls = eval_light(light, world_pos);
    let n_dot_l = max(dot(n, ls.l), 0.0);
    if n_dot_l <= 0.0 {
        return vec3<f32>(0.0);
    }
    let h = normalize(v + ls.l);
    let n_dot_v = max(dot(n, v), 1e-4);
    let l_dot_h = clamp(dot(ls.l, h), 0.0, 1.0);
    return base_color * one_minus_reflectivity * fd_burley(n_dot_v, n_dot_l, l_dot_h, roughness) * signed_light_radiance(light, ls.attenuation, n_dot_l);
}
