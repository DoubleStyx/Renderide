//! Direct + indirect lighting for the Xiexe Toon 2.0 BRDF.
//!
//! Layers XSToon 2.0 stylization on top of PBSMetallic's energy budget.
//!
//! - Direct + indirect specular: GGX/Trowbridge-Reitz D, height-correlated Smith-GGX V, and
//!   Schlick Fresnel with `brdf::metallic_f0(diffuse_color, metallic)` (the same F0 PBSMetallic
//!   uses), DFG LUT energy compensation for the direct lobe, and multi-bounce specular AO for the
//!   probe radiance.
//! - Direct diffuse: Disney/Burley rough diffuse with the toon shadow ramp as a 3-channel
//!   multiplicative tint replacing `NdotL * att`, then normalized Fresnel transmission as an energy
//!   envelope. Colored / banded ramps drive the toon stylization.
//! - Indirect diffuse / specular: PBSMetallic's `(1 - indirect_specular_energy)` split so the
//!   indirect-light budget is shared between the SH probe and the spec lobe. Colored
//!   `_OcclusionColor` modulates indirect diffuse only and is lifted through the PBS multi-bounce
//!   AO fit.
//! - XSToon 2.0 stylization preserved on top: toon ramp diffuse, matcap (`_MATCAP`
//!   keyword), rim / shadow rim, subsurface scattering, outline lighting, the
//!   indirect-spec ramp-shadow blend, the `_ReflectivityMask.r` additive reflection weight, and
//!   the `col += max(directSpec_sum, rim)` composition step.
//!
//! `_SpecularIntensity` and `_SpecularAlbedoTint` are artist controls layered on
//! top of the PBS direct-spec lobe.

#define_import_path renderide::xiexe::toon2::lighting

#import renderide::xiexe::toon2::base as xb
#import renderide::xiexe::toon2::variant_bits as xvb
#import renderide::frame::globals as rg
#import renderide::frame::types as ft
#import renderide::pbs::cluster as pcls
#import renderide::pbs::brdf as brdf
#import renderide::lighting::birp as bl
#import renderide::lighting::reflection_probes as rprobe

/// SH-probe sample used for xiexe's uncoloured indirect-diffuse term.
fn indirect_diffuse(s: xb::SurfaceData, world_pos: vec3<f32>, view_layer: u32) -> vec3<f32> {
    return rprobe::indirect_diffuse(world_pos, s.normal, view_layer, true);
}

/// Scalar AO weight used by the indirect-specular Lagarde occlusion term.
fn occlusion_scalar(s: xb::SurfaceData) -> f32 {
    return clamp(xb::grayscale(s.occlusion), 0.0, 1.0);
}

/// Colored indirect-diffuse visibility that preserves XSToon's `_OcclusionColor` tint while using
/// PBS multi-bounce AO to avoid over-dark indirect lighting.
fn indirect_diffuse_visibility(s: xb::SurfaceData) -> vec3<f32> {
    let colored_occlusion = clamp(s.occlusion, vec3<f32>(0.0), vec3<f32>(1.0));
    let scalar_occlusion = occlusion_scalar(s);
    if (scalar_occlusion <= 1e-4) {
        return colored_occlusion;
    }

    let visibility = brdf::indirect_diffuse_visibility(scalar_occlusion, s.albedo.rgb);
    return min(vec3<f32>(1.0), colored_occlusion * visibility / vec3<f32>(scalar_occlusion));
}

/// Reflection tint used by `_RimCubemapTint`. Falls back to white when no specular probe is bound so
/// the tint slider does not collapse the rim light to black.
fn environment_tint(s: xb::SurfaceData, view_dir: vec3<f32>, world_pos: vec3<f32>, view_layer: u32) -> vec3<f32> {
    if (!rprobe::has_indirect_specular(view_layer, true)) {
        return vec3<f32>(1.0);
    }
    let indirect_roughness = brdf::filter_perceptual_roughness(s.roughness, s.normal);
    return rprobe::raw_indirect_specular_with_horizon(world_pos, s.normal, s.raw_normal, view_dir, indirect_roughness, true, view_layer);
}

/// `UNITY_SPECCUBE_LOD_STEPS` on PC/console.
const SPECCUBE_LOD_STEPS: f32 = 6.0;

/// Resolves a single frame light into a `LightSample` (direction toward the light,
/// color, attenuation, directional flag).
fn sample_light(light: ft::GpuLight, world_pos: vec3<f32>) -> xb::LightSample {
    if (light.light_type == 1u) {
        let dir_len_sq = dot(light.direction.xyz, light.direction.xyz);
        return xb::LightSample(
            select(vec3<f32>(0.0, 0.0, 1.0), normalize(-light.direction.xyz), dir_len_sq > 1e-16),
            light.color.xyz,
            bl::direct_light_intensity(light.intensity),
            true,
        );
    }

    let to_light = light.position.xyz - world_pos;
    let dist = length(to_light);
    let l = xb::safe_normalize(to_light, vec3<f32>(0.0, 1.0, 0.0));
    var attenuation = bl::punctual_attenuation(light.intensity, dist, light.range);
    if (light.light_type == 2u) {
        attenuation = attenuation * bl::spot_angle_attenuation(light, l);
    }
    return xb::LightSample(l, light.color.xyz, attenuation, false);
}

/// Toon ramp lookup. The half-Lambert remap (`NdotL * 0.5 + 0.5`) maps to the U axis;
/// the ramp-mask sample maps to the V axis. `_ShadowSharpness` sharpens the attenuation before it
/// multiplies half-Lambert.
fn ramp_for_ndl(ndl: f32, attenuation: f32, ramp_mask: f32) -> vec3<f32> {
    let att_sharp = mix(attenuation, round(attenuation), clamp(xb::mat._ShadowSharpness, 0.0, 1.0));
    let x = clamp((ndl * 0.5 + 0.5) * att_sharp, 0.0, 1.0);
    return textureSample(xb::_Ramp, xb::_Ramp_sampler, vec2<f32>(x, clamp(ramp_mask, 0.0, 1.0))).rgb;
}

/// XSToon-style remap used by `_SpecularArea` before it is passed to the PBS GGX path as
/// perceptual roughness.
fn remap_specular_area(area: f32) -> f32 {
    let remapped = max(0.01, area);
    return remapped * (1.7 - 0.7 * remapped);
}

/// Direct-specular inputs derived once per fragment for the primary lobe.
struct DirectSpecularTerms {
    /// Primary lobe F0, identical to PBSMetallic's `metallic_f0(base, metallic)`.
    specular_reflectance: vec3<f32>,
    /// Primary lobe perceptual roughness, derived from `_SpecularArea` via
    /// `roughness = 1 - remap_specular_area(_SpecularArea)`.
    roughness: f32,
    /// Primary lobe roughness widened by geometric specular antialiasing.
    aa_roughness: f32,
    /// Multiple-scattering energy compensation sampled from the frame DFG LUT.
    energy_compensation: vec3<f32>,
}

/// Resolves the primary direct-specular terms shared by every clustered light.
///
/// F0 matches PBSMetallic's `metallic_f0` (dielectric = 0.04, metallic = base color)
/// so a default Xiexe material with `_SpecularIntensity = 1`, `_SpecularAlbedoTint = 0`,
/// and a white ramp lights identically to a matched PBSMetallic ball. The `_Reflectivity`
/// scalar is intentionally not fed into F0 here; `_Reflectivity` is a leftover dial that does
/// not participate in the BRDF, and only `_ReflectivityMask.r` gates the additive indirect-spec
/// blend.
fn primary_direct_specular_terms(s: xb::SurfaceData, view_dir: vec3<f32>) -> DirectSpecularTerms {
    let specular_reflectance = brdf::metallic_f0(s.diffuse_color, s.metallic);
    let roughness = clamp(1.0 - remap_specular_area(xb::mat._SpecularArea), 0.0, 1.0);
    let aa_roughness = brdf::filter_perceptual_roughness(roughness, s.normal);
    let n_dot_v = clamp(dot(s.normal, view_dir), 0.0, 1.0);
    let direct_roughness = brdf::direct_perceptual_roughness(roughness);
    let dfg = brdf::sample_ibl_dfg_lut(direct_roughness, n_dot_v);
    let energy_compensation = brdf::energy_compensation_from_dfg(dfg, specular_reflectance);
    return DirectSpecularTerms(specular_reflectance, roughness, aa_roughness, energy_compensation);
}

/// GGX direct-specular lobe evaluated against an arbitrary normal.
fn direct_specular_ggx(
    normal: vec3<f32>,
    s: xb::SurfaceData,
    light: xb::LightSample,
    view_dir: vec3<f32>,
    perceptual_roughness: f32,
    specular_reflectance: vec3<f32>,
    energy_compensation: vec3<f32>,
    intensity: f32,
    albedo_tint: f32,
) -> vec3<f32> {
    if (intensity <= 1e-4) {
        return vec3<f32>(0.0);
    }

    if (light.attenuation <= 1e-4) {
        return vec3<f32>(0.0);
    }

    let direct_lobe = brdf::eval_direct_specular_lobe(
        normal,
        light.direction,
        view_dir,
        perceptual_roughness,
        specular_reflectance,
        energy_compensation,
    );
    if (direct_lobe.n_dot_l <= 1e-4) {
        return vec3<f32>(0.0);
    }

    let radiance = light.color * light.attenuation * direct_lobe.n_dot_l;
    var specular = direct_lobe.specular_brdf;
    specular = specular * radiance * intensity;
    specular = specular * mix(vec3<f32>(1.0), s.diffuse_color, clamp(albedo_tint, 0.0, 1.0));
    return specular;
}

/// Primary specular lobe driven by `_SpecularArea` and `_SpecularIntensity`.
fn direct_specular(
    s: xb::SurfaceData,
    light: xb::LightSample,
    view_dir: vec3<f32>,
    terms: DirectSpecularTerms,
) -> vec3<f32> {
    return direct_specular_ggx(
        s.normal,
        s,
        light,
        view_dir,
        terms.aa_roughness,
        terms.specular_reflectance,
        terms.energy_compensation,
        max(0.0, xb::mat._SpecularIntensity),
        xb::mat._SpecularAlbedoTint,
    );
}

/// Fresnel transmission envelope for XSToon's ramped direct diffuse term.
fn direct_diffuse_fresnel_transmission(
    normal: vec3<f32>,
    light_direction: vec3<f32>,
    view_dir: vec3<f32>,
    specular_reflectance: vec3<f32>,
) -> f32 {
    let h = xb::safe_normalize(light_direction + view_dir, normal);
    let ldh = xb::saturate(dot(light_direction, h));
    let f = brdf::f_schlick(
        specular_reflectance,
        brdf::f90_from_f0(specular_reflectance),
        ldh,
    );
    return brdf::direct_diffuse_fresnel_transmission(f, specular_reflectance);
}

/// Rough diffuse BRDF scalar for XSToon's ramped direct diffuse term.
fn direct_diffuse_brdf(
    normal: vec3<f32>,
    light_direction: vec3<f32>,
    view_dir: vec3<f32>,
    perceptual_roughness: f32,
) -> f32 {
    let h = xb::safe_normalize(light_direction + view_dir, normal);
    return brdf::fd_burley(
        xb::saturate(dot(normal, view_dir)),
        xb::saturate(dot(normal, light_direction)),
        xb::saturate(dot(light_direction, h)),
        perceptual_roughness,
    );
}

/// Rim contribution from the dominant light plus ambient probe lighting.
fn rim_light(
    s: xb::SurfaceData,
    light: xb::LightSample,
    view_dir: vec3<f32>,
    ambient: vec3<f32>,
    env_map: vec3<f32>,
) -> vec3<f32> {
    let ndl = xb::saturate(dot(s.normal, light.direction));
    let vdn = abs(dot(view_dir, s.normal));
    let sharp = max(xb::mat._RimSharpness, 0.001);
    var rim = xb::saturate(1.0 - vdn) * pow(ndl, max(xb::mat._RimThreshold, 0.0));
    rim = smoothstep(xb::mat._RimRange - sharp, xb::mat._RimRange + sharp, rim);

    var col = rim * xb::mat._RimIntensity * (light.color + ambient);
    col = col * mix(vec3<f32>(1.0), vec3<f32>(light.attenuation) + ambient, clamp(xb::mat._RimAttenEffect, 0.0, 1.0));
    col = col * xb::mat._RimColor.rgb;
    col = col * mix(vec3<f32>(1.0), s.diffuse_color, clamp(xb::mat._RimAlbedoTint, 0.0, 1.0));
    col = col * mix(vec3<f32>(1.0), env_map, clamp(xb::mat._RimCubemapTint, 0.0, 1.0));
    return col;
}

/// Shadow-rim multiplier from the dominant light plus a small ambient lift.
fn shadow_rim(
    s: xb::SurfaceData,
    view_dir: vec3<f32>,
    light: xb::LightSample,
    ambient: vec3<f32>,
) -> vec3<f32> {
    let ndl = xb::saturate(dot(s.normal, light.direction));
    let vdn = abs(dot(view_dir, s.normal));
    let sharp = max(xb::mat._ShadowRimSharpness, 0.001);
    var rim = xb::saturate(1.0 - vdn) * pow(xb::saturate(1.0 - ndl), max(xb::mat._ShadowRimThreshold * 2.0, 0.0));
    rim = smoothstep(xb::mat._ShadowRimRange - sharp, xb::mat._ShadowRimRange + sharp, rim);

    let tint = xb::mat._ShadowRim.rgb * mix(vec3<f32>(1.0), s.diffuse_color, clamp(xb::mat._ShadowRimAlbedoTint, 0.0, 1.0)) + ambient * 0.1;
    return mix(vec3<f32>(1.0), tint, rim);
}

/// Stylized subsurface scattering behavior, including the all-zero `_SSColor`
/// early-out, distortion-by-normal half-vector, `VdotH^_SSPower` intensity, and `_SSColor *
/// (VdotH + indirectDiffuse) * attenuation * _SSScale * thickness * lightCol * albedo`
/// final tint. When the `THICKNESS_MAP` keyword is off, `s.thickness` defaults to `1.0`
/// in `surface::sample_surface_for_layout` so the math is identical to the gated material path.
fn subsurface(
    s: xb::SurfaceData,
    light: xb::LightSample,
    view_dir: vec3<f32>,
    ambient: vec3<f32>,
) -> vec3<f32> {
    if (dot(xb::mat._SSColor.rgb, xb::mat._SSColor.rgb) <= 1e-8) {
        return vec3<f32>(0.0);
    }

    let raw_ndl = dot(s.normal, light.direction);
    let ndl = xb::saturate(raw_ndl);
    if (ndl <= 1e-4 || light.attenuation <= 1e-4) {
        return vec3<f32>(0.0);
    }

    let attenuation = xb::saturate(light.attenuation * (raw_ndl * 0.5 + 0.5));
    let h = xb::safe_normalize(light.direction + s.normal * xb::mat._SSDistortion, s.normal);
    let vdh = pow(xb::saturate(dot(view_dir, -h)), max(xb::mat._SSPower, 0.001));
    let scatter = xb::mat._SSColor.rgb * (vdh + ambient) * attenuation * xb::mat._SSScale * s.thickness;
    return max(vec3<f32>(0.0), light.color * scatter * s.albedo.rgb) * ndl * light.attenuation;
}

/// View-space matcap UV. Projects `n` onto the camera's right and up basis vectors and remaps to
/// `[0, 1]`.
fn matcap_uv(view_dir: vec3<f32>, n: vec3<f32>) -> vec2<f32> {
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let view_up = xb::safe_normalize(up - view_dir * dot(view_dir, up), vec3<f32>(0.0, 1.0, 0.0));
    let view_right = xb::safe_normalize(cross(view_dir, view_up), vec3<f32>(1.0, 0.0, 0.0));
    return vec2<f32>(dot(view_right, n), dot(view_up, n)) * 0.5 + vec2<f32>(0.5);
}

/// Samples the indirect-reflection contribution for a selected XSToon keyword layout.
fn indirect_reflection_branch_for_layout(
    s: xb::SurfaceData,
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    world_pos: vec3<f32>,
    view_layer: u32,
    perceptual_roughness: f32,
    specular_reflectance: vec3<f32>,
    ambient: vec3<f32>,
    dominant_light_col_atten: vec3<f32>,
    keyword_layout: u32,
) -> vec3<f32> {
    if (xvb::matcap_enabled_for_layout(keyword_layout)) {
        let stereo_view_dir = rg::stereo_center_view_dir_for_world_pos(world_pos, view_layer);
        let uv = matcap_uv(stereo_view_dir, normal);
        let lod = clamp((1.0 - clamp(perceptual_roughness, 0.0, 1.0)) * SPECCUBE_LOD_STEPS, 0.0, SPECCUBE_LOD_STEPS);
        var spec = textureSampleLevel(xb::_Matcap, xb::_Matcap_sampler, uv, lod).rgb * xb::mat._MatcapTint.rgb;
        spec = spec * (ambient + dominant_light_col_atten * 0.5);
        return spec;
    }

    let roughness = brdf::filter_perceptual_roughness(clamp(perceptual_roughness, 0.0, 1.0), normal);
    let n_dot_v = clamp(dot(normal, view_dir), 0.0, 1.0);
    let indirect_enabled = rprobe::has_indirect_specular(view_layer, xvb::reflection_uses_pbr_for_layout(keyword_layout));
    let dfg = brdf::sample_ibl_dfg_lut(roughness, n_dot_v);
    let specular_energy = brdf::indirect_specular_energy_from_dfg(dfg, specular_reflectance, indirect_enabled);
    let specular_visibility =
        brdf::indirect_specular_visibility(n_dot_v, occlusion_scalar(s), roughness, specular_reflectance);
    let spec = rprobe::indirect_specular_with_energy(
        world_pos,
        normal,
        s.raw_normal,
        view_dir,
        roughness,
        specular_energy * specular_visibility,
        1.0,
        indirect_enabled,
        view_layer,
    );
    return spec;
}

/// Indirect-specular contribution for a selected XSToon keyword layout.
fn indirect_specular_for_layout(
    s: xb::SurfaceData,
    view_dir: vec3<f32>,
    world_pos: vec3<f32>,
    view_layer: u32,
    ambient: vec3<f32>,
    dominant_light_col_atten: vec3<f32>,
    dominant_ramp: vec3<f32>,
    keyword_layout: u32,
) -> vec3<f32> {
    let specular_reflectance = brdf::metallic_f0(s.diffuse_color, s.metallic);

    var spec = indirect_reflection_branch_for_layout(
        s,
        s.normal,
        view_dir,
        world_pos,
        view_layer,
        s.roughness,
        specular_reflectance,
        ambient,
        dominant_light_col_atten,
        keyword_layout,
    );

    if (!xvb::matcap_enabled_for_layout(keyword_layout)) {
        let roughness = clamp(s.roughness, 0.0, 1.0);
        spec = mix(spec, spec * dominant_ramp, roughness);
    }

    return spec;
}

/// Base-pass emission contribution for a selected XSToon keyword layout.
fn emission_color_for_layout(s: xb::SurfaceData, base_pass: bool, keyword_layout: u32) -> vec3<f32> {
    if (!base_pass || !xvb::emission_map_enabled_for_layout(keyword_layout)) {
        return vec3<f32>(0.0);
    }
    return s.emission * xb::mat._EmissionColor.rgb;
}

/// Forward-pass clustered light walk for a selected XSToon keyword layout.
fn clustered_toon_lighting_for_layout(
    frag_xy: vec2<f32>,
    s: xb::SurfaceData,
    world_pos: vec3<f32>,
    view_layer: u32,
    include_directional: bool,
    include_local: bool,
    base_pass: bool,
    keyword_layout: u32,
) -> vec3<f32> {
    let view_dir = rg::view_dir_for_world_pos(world_pos, view_layer);
    let ambient = indirect_diffuse(s, world_pos, view_layer);
    let env = environment_tint(s, view_dir, world_pos, view_layer);
    let primary_specular_terms = primary_direct_specular_terms(s, view_dir);

    // Indirect diffuse / specular share a single energy budget: whatever the spec
    // probe lobe takes, the diffuse term must give up.
    let indirect_specular_reflectance = brdf::metallic_f0(s.diffuse_color, s.metallic);
    let n_dot_v = clamp(dot(s.normal, view_dir), 0.0, 1.0);
    let indirect_specular_enabled =
        rprobe::has_indirect_specular(view_layer, xvb::reflection_uses_pbr_for_layout(keyword_layout));
    let indirect_roughness = brdf::filter_perceptual_roughness(s.roughness, s.normal);
    let indirect_dfg = brdf::sample_ibl_dfg_lut(indirect_roughness, n_dot_v);
    let indirect_specular_energy = brdf::indirect_specular_energy_from_dfg(
        indirect_dfg,
        indirect_specular_reflectance,
        indirect_specular_enabled,
    );
    let indirect_diffuse_energy_scale =
        brdf::indirect_diffuse_energy_scale(indirect_specular_energy, indirect_specular_enabled);

    let cluster_id = pcls::cluster_id_from_frag(
        frag_xy,
        world_pos,
        rg::frame.view_space_z_coeffs,
        rg::frame.view_space_z_coeffs_right,
        view_layer,
        rg::frame.viewport_width,
        rg::frame.viewport_height,
        rg::frame.cluster_count_x,
        rg::frame.cluster_count_y,
        rg::frame.cluster_count_z,
        rg::frame.near_clip,
        rg::frame.far_clip,
    );
    let count = pcls::cluster_light_count_at(cluster_id);
    let i_max = count;

    var direct_diffuse = vec3<f32>(0.0);
    var direct_spec = vec3<f32>(0.0);
    var sss = vec3<f32>(0.0);

    var dominant_light = xb::LightSample(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0), 0.0, true);
    var dominant_light_col_atten = vec3<f32>(0.0);
    var dominant_ramp = vec3<f32>(0.0);
    var dominant_weight = -1.0;

    for (var i = 0u; i < i_max; i++) {
        let li = pcls::cluster_light_index_at(cluster_id, i);
        if (li >= rg::frame.light_count) {
            continue;
        }

        let light = sample_light(rg::lights[li], world_pos);
        if ((light.is_directional && !include_directional) || (!light.is_directional && !include_local)) {
            continue;
        }

        let ndl = dot(s.normal, light.direction);
        let ramp = ramp_for_ndl(ndl, light.attenuation, s.ramp_mask);
        let light_col_atten = light.color * light.attenuation;
        // Rough diffuse times the boosted `light.attenuation` times the toon ramp, wrapped by the
        // PBS direct diffuse Fresnel-transmission envelope. The toon ramp is the 3-channel
        // stylized replacement for `NdL` and bakes
        // attenuation into its `U` axis to compress the curve for distant punctual lights.
        // `s.albedo` is already metallic-discounted in `surface::sample_surface_for_layout`.
        let diffuse_transmission = direct_diffuse_fresnel_transmission(
            s.normal,
            light.direction,
            view_dir,
            primary_specular_terms.specular_reflectance,
        );
        let diffuse_brdf = direct_diffuse_brdf(s.normal, light.direction, view_dir, s.roughness);
        direct_diffuse = direct_diffuse
            + s.albedo.rgb * diffuse_transmission * diffuse_brdf * light.color * light.attenuation * ramp;
        direct_spec = direct_spec + direct_specular(s, light, view_dir, primary_specular_terms);
        sss = sss + subsurface(s, light, view_dir, ambient);

        let weight = xb::grayscale(light_col_atten * vec3<f32>(xb::saturate(ndl)));
        if (weight > dominant_weight) {
            dominant_weight = weight;
            dominant_light = light;
            dominant_light_col_atten = light_col_atten;
            dominant_ramp = ramp;
        }
    }

    // Diffuse = sum_lights(direct) + albedo * ambient * energy_scale * colored_occlusion.
    // The `energy_scale` is `(1 - indirect_specular_energy)` so the indirect-light budget is
    // split between the diffuse and specular probe responses. Colored `_OcclusionColor` is the
    // XSToon stylization layered on top, and it only modulates indirect diffuse here (matching
    // PBSMetallic's AO behavior; direct diffuse stays unattenuated).
    var diffuse = direct_diffuse;
    if (base_pass) {
        diffuse = diffuse + s.albedo.rgb * ambient * indirect_diffuse_energy_scale * indirect_diffuse_visibility(s);
    }

    // Shadow rim multiplies diffuse before any specular accumulation.
    var col = diffuse;
    if (base_pass && dominant_weight > 0.0) {
        col = col * shadow_rim(s, view_dir, dominant_light, ambient);
    }

    // Additive reflection blend gated by `_ReflectivityMask.r`. Reflection blend-mode
    // multiplicative / subtractive branches are intentionally absent here.
    if (base_pass) {
        let reflection = indirect_specular_for_layout(
            s,
            view_dir,
            world_pos,
            view_layer,
            ambient,
            dominant_light_col_atten,
            dominant_ramp,
            keyword_layout,
        );
        col = col + reflection * clamp(s.reflectivity_mask, 0.0, 1.0);
    }

    // Direct specular and rim share a `max` composition so a saturated highlight does not stack on
    // top of a saturated rim.
    var spec_or_rim = direct_spec;
    if (base_pass && dominant_weight > 0.0) {
        let rim = rim_light(s, dominant_light, view_dir, ambient, env);
        spec_or_rim = max(spec_or_rim, rim);
    }
    col = col + spec_or_rim;

    col = col + sss;

    if (base_pass) {
        col = col + emission_color_for_layout(s, base_pass, keyword_layout);
    }

    return max(col, vec3<f32>(0.0));
}

/// Outline-pass clustered light walk for the "Lit" outline mode.
///
/// Returns `dominantDirectLight + indirectDiffuse` for lit outline mode.
/// The outline-pass fragment multiplies this result by `_OutlineColor` (and optionally
/// `_OutlineAlbedoTint * albedo`) in `outline.wgsl`.
fn clustered_outline_lighting(
    frag_xy: vec2<f32>,
    s: xb::SurfaceData,
    world_pos: vec3<f32>,
    view_layer: u32,
) -> vec3<f32> {
    let ambient = indirect_diffuse(s, world_pos, view_layer);
    let cluster_id = pcls::cluster_id_from_frag(
        frag_xy,
        world_pos,
        rg::frame.view_space_z_coeffs,
        rg::frame.view_space_z_coeffs_right,
        view_layer,
        rg::frame.viewport_width,
        rg::frame.viewport_height,
        rg::frame.cluster_count_x,
        rg::frame.cluster_count_y,
        rg::frame.cluster_count_z,
        rg::frame.near_clip,
        rg::frame.far_clip,
    );
    let count = pcls::cluster_light_count_at(cluster_id);
    let i_max = count;

    var dominant_direct = vec3<f32>(0.0);
    var dominant_weight = -1.0;
    for (var i = 0u; i < i_max; i++) {
        let li = pcls::cluster_light_index_at(cluster_id, i);
        if (li >= rg::frame.light_count) {
            continue;
        }

        let light = sample_light(rg::lights[li], world_pos);
        let ndl = xb::saturate(dot(s.normal, light.direction));
        let direct = xb::saturate(light.attenuation * ndl) * light.color;
        let weight = xb::grayscale(direct);
        if (weight > dominant_weight) {
            dominant_weight = weight;
            dominant_direct = direct;
        }
    }

    return dominant_direct + ambient;
}
