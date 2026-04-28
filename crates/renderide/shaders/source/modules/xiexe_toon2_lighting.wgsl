//! Direct + indirect lighting for the Xiexe Toon 2.0 BRDF.
//!
//! Keeps the XSToon2 ramp-diffuse surface model and clustered forward light walk, but
//! ports the broken lighting math to the newer XSToon3-style accumulator flow:
//! corrected Schlick/GGX specular, blend-mode-aware reflections, clear-coat layering,
//! emission scaling, and dominant-light-driven rim / shadow-rim / outline modulation.

#define_import_path renderide::xiexe::toon2::lighting

#import renderide::xiexe::toon2::base as xb
#import renderide::globals as rg
#import renderide::pbs::cluster as pcls
#import renderide::pbs::brdf as brdf
#import renderide::birp::light as bl
#import renderide::sh2_ambient as shamb
#import renderide::uv_utils as uvu

/// SH-probe sample used for xiexe's uncoloured indirect-diffuse term.
fn indirect_diffuse(s: xb::SurfaceData) -> vec3<f32> {
    return shamb::ambient_probe(s.normal);
}

/// Scalar AO weight used when modern XSToon3 paths expect a single occlusion factor.
fn occlusion_scalar(s: xb::SurfaceData) -> f32 {
    return clamp(xb::grayscale(s.occlusion), 0.0, 1.0);
}

/// Skybox tint used by `_RimCubemapTint`. Falls back to white when no specular skybox is bound so
/// the tint slider does not collapse the rim light to black.
fn environment_tint(s: xb::SurfaceData, view_dir: vec3<f32>) -> vec3<f32> {
    if (rg::frame.skybox_specular.y <= 0.5) {
        return vec3<f32>(1.0);
    }
    return brdf::indirect_specular(s.normal, view_dir, s.roughness, vec3<f32>(1.0), 1.0, true);
}

/// `UNITY_SPECCUBE_LOD_STEPS` on PC/console.
const SPECCUBE_LOD_STEPS: f32 = 6.0;

/// Resolves a single `rg::GpuLight` into a `LightSample` (direction toward the light,
/// color, attenuation, directional flag).
fn sample_light(light: rg::GpuLight, world_pos: vec3<f32>) -> xb::LightSample {
    if (light.light_type == 1u) {
        let dir_len_sq = dot(light.direction.xyz, light.direction.xyz);
        return xb::LightSample(
            select(vec3<f32>(0.0, 0.0, 1.0), normalize(-light.direction.xyz), dir_len_sq > 1e-16),
            light.color.xyz,
            light.intensity,
            true,
        );
    }

    let to_light = light.position.xyz - world_pos;
    let dist = length(to_light);
    let l = xb::safe_normalize(to_light, vec3<f32>(0.0, 1.0, 0.0));
    var attenuation = bl::punctual_attenuation(light.intensity, dist, light.range);
    if (light.light_type == 2u) {
        let spot_cos = dot(-l, xb::safe_normalize(light.direction.xyz, vec3<f32>(0.0, -1.0, 0.0)));
        let inner_cos = min(light.spot_cos_half_angle + 0.1, 1.0);
        attenuation = attenuation * smoothstep(light.spot_cos_half_angle, inner_cos, spot_cos);
    }
    return xb::LightSample(l, light.color.xyz, attenuation, false);
}

/// Toon ramp lookup. The half-Lambert remap (`NdotL · 0.5 + 0.5`) maps to the U axis;
/// the ramp-mask sample maps to the V axis. `_ShadowSharpness` sharpens the
/// attenuation before it multiplies half-Lambert so the banding stays on the shadow
/// transition rather than on the ramp itself.
fn ramp_for_ndl(ndl: f32, attenuation: f32, ramp_mask: f32) -> vec3<f32> {
    let att_sharp = mix(attenuation, round(attenuation), clamp(xb::mat._ShadowSharpness, 0.0, 1.0));
    let x = clamp((ndl * 0.5 + 0.5) * att_sharp, 0.0, 1.0);
    return textureSample(xb::_Ramp, xb::_Ramp_sampler, vec2<f32>(x, clamp(ramp_mask, 0.0, 1.0))).rgb;
}

/// XSToon3-style remap used by `_SpecularArea` and clear-coat roughness inputs before they are
/// squared into the GGX linear roughness parameter.
fn remap_specular_area(area: f32) -> f32 {
    let remapped = max(0.01, area);
    return remapped * (1.7 - 0.7 * remapped);
}

/// Corrected Schlick/GGX direct-specular lobe from XSToon3, evaluated against an arbitrary normal.
///
/// `area` is the XSToon roughness-style control before the `1.7 - 0.7x` remap; `intensity` is
/// the already-multiplied specular weight (for example `_SpecularIntensity * specularMap.r`).
fn direct_specular_from_normal(
    normal: vec3<f32>,
    s: xb::SurfaceData,
    light: xb::LightSample,
    view_dir: vec3<f32>,
    area: f32,
    intensity: f32,
    albedo_tint: f32,
) -> vec3<f32> {
    if (intensity <= 1e-4) {
        return vec3<f32>(0.0);
    }

    let ndl = xb::saturate(dot(normal, light.direction));
    if (ndl <= 1e-4 || light.attenuation <= 1e-4) {
        return vec3<f32>(0.0);
    }

    let h = xb::safe_normalize(light.direction + view_dir, normal);
    let ndh = xb::saturate(dot(normal, h));
    let ndv = max(abs(dot(view_dir, normal)), 1e-4);
    let ldh = xb::saturate(dot(light.direction, h));
    let roughness = remap_specular_area(area);
    let alpha = max(roughness * roughness, 0.0045);

    let d_term = brdf::d_ggx(ndh, alpha);
    let v_term = brdf::v_smith_ggx_correlated(ndv, ndl, alpha);
    let f_term = 1.0 - xb::pow5(1.0 - ldh);

    var specular = max(vec3<f32>(0.0), vec3<f32>(d_term * v_term * f_term));
    specular = specular * light.color * intensity * ndl * light.attenuation;
    specular = specular * mix(vec3<f32>(1.0), s.diffuse_color, clamp(albedo_tint, 0.0, 1.0));
    return specular;
}

/// Primary specular lobe driven by `_SpecularArea`, `_SpecularIntensity`, and `_SpecularMap.rgb`.
fn direct_specular(
    s: xb::SurfaceData,
    light: xb::LightSample,
    view_dir: vec3<f32>,
) -> vec3<f32> {
    let intensity = max(0.0, xb::mat._SpecularIntensity * s.specular_mask.r);
    let area = max(0.0, xb::mat._SpecularArea * s.specular_mask.b);
    let albedo_tint = clamp(xb::mat._SpecularAlbedoTint * s.specular_mask.g, 0.0, 1.0);
    return direct_specular_from_normal(s.normal, s, light, view_dir, area, intensity, albedo_tint);
}

/// Secondary clear-coat direct-specular lobe driven by the metallic-gloss map's `g/b` channels.
fn clearcoat_direct_specular(
    s: xb::SurfaceData,
    light: xb::LightSample,
    view_dir: vec3<f32>,
) -> vec3<f32> {
    if (!xb::clearcoat_enabled()) {
        return vec3<f32>(0.0);
    }

    let area = 1.0 - s.clearcoat_smoothness;
    return direct_specular_from_normal(s.raw_normal, s, light, view_dir, area, s.clearcoat_strength, 0.0);
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

/// Stylised subsurface scattering from XSToon3, preserving the XSToon2 property set.
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
/// `[0, 1]`, matching Unity's `matcapSample`.
fn matcap_uv(view_dir: vec3<f32>, n: vec3<f32>) -> vec2<f32> {
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let view_up = xb::safe_normalize(up - view_dir * dot(view_dir, up), vec3<f32>(0.0, 1.0, 0.0));
    let view_right = xb::safe_normalize(cross(view_dir, view_up), vec3<f32>(1.0, 0.0, 0.0));
    return vec2<f32>(dot(view_right, n), dot(view_up, n)) * 0.5 + vec2<f32>(0.5);
}

/// Reflection blend-weight shared by additive / multiplicative / subtractive indirect-specular modes.
fn reflection_blend_weight(s: xb::SurfaceData) -> f32 {
    return clamp(s.reflectivity * s.reflectivity_mask, 0.0, 1.0);
}

/// True when `_ReflectionBlendMode` selects the multiplicative branch.
fn reflection_is_multiplicative() -> bool {
    return abs(xb::mat._ReflectionBlendMode - 1.0) < 0.5;
}

/// Samples one indirect-reflection branch using the current reflection mode while preserving the
/// existing XSToon2 bindings. Mode `1` ("baked cubemap") falls back to the renderer skybox source
/// because the XSToon2 contract has no dedicated cubemap binding.
fn indirect_reflection_branch(
    s: xb::SurfaceData,
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    perceptual_roughness: f32,
    fresnel: vec3<f32>,
    ambient: vec3<f32>,
    dominant_light_col_atten: vec3<f32>,
    reflectivity_scale: f32,
) -> vec3<f32> {
    if (xb::reflection_disabled()) {
        return vec3<f32>(0.0);
    }

    let reflectivity_mask = clamp(s.reflectivity_mask, 0.0, 1.0);
    if (reflectivity_mask <= 1e-4) {
        return vec3<f32>(0.0);
    }

    if (xb::matcap_enabled()) {
        let uv = uvu::flip_v(matcap_uv(view_dir, normal));
        let lod = clamp((1.0 - clamp(perceptual_roughness, 0.0, 1.0)) * SPECCUBE_LOD_STEPS, 0.0, SPECCUBE_LOD_STEPS);
        var spec = textureSampleLevel(xb::_Matcap, xb::_Matcap_sampler, uv, lod).rgb * xb::mat._MatcapTint.rgb;
        if (!reflection_is_multiplicative()) {
            spec = spec * (ambient + dominant_light_col_atten * 0.5);
        }
        return spec * max(reflectivity_scale, 0.0) * reflectivity_mask;
    }

    var spec = brdf::indirect_specular(
        normal,
        view_dir,
        clamp(perceptual_roughness, 0.0, 1.0),
        fresnel,
        occlusion_scalar(s),
        xb::reflection_uses_pbr(),
    ) * reflectivity_mask;

    if (abs(xb::mat._ReflectionMode - 1.0) < 0.5 && !reflection_is_multiplicative()) {
        spec = spec * (ambient + dominant_light_col_atten * 0.5);
    }
    return spec;
}

/// Indirect-specular contribution including the clear-coat lobe.
fn indirect_specular(
    s: xb::SurfaceData,
    view_dir: vec3<f32>,
    ambient: vec3<f32>,
    dominant_light_col_atten: vec3<f32>,
) -> vec3<f32> {
    let reflectivity = clamp(s.reflectivity, 0.0, 1.0);
    let fresnel = vec3<f32>(0.16 * reflectivity * reflectivity * (1.0 - s.metallic)) + s.diffuse_color * s.metallic;

    var spec = indirect_reflection_branch(
        s,
        s.normal,
        view_dir,
        s.roughness,
        fresnel,
        ambient,
        dominant_light_col_atten,
        reflectivity,
    );

    if (xb::clearcoat_enabled()) {
        let clearcoat_f0 = vec3<f32>(0.16 * s.clearcoat_strength * s.clearcoat_strength * s.clearcoat_smoothness);
        spec = spec + indirect_reflection_branch(
            s,
            s.raw_normal,
            view_dir,
            1.0 - s.clearcoat_smoothness,
            clearcoat_f0,
            ambient,
            dominant_light_col_atten,
            s.clearcoat_strength,
        );
    }

    return spec;
}

/// Applies XSToon3's additive / multiplicative / subtractive reflection blend modes to the
/// accumulated diffuse surface color.
fn apply_reflection_blend(surface: vec3<f32>, reflection: vec3<f32>, weight: f32) -> vec3<f32> {
    let clamped_weight = clamp(weight, 0.0, 1.0);
    if (clamped_weight <= 1e-4) {
        return surface;
    }

    if (reflection_is_multiplicative()) {
        return mix(surface, surface * reflection, clamped_weight);
    }
    if (abs(xb::mat._ReflectionBlendMode - 2.0) < 0.5) {
        return surface - reflection * clamped_weight;
    }
    return surface + reflection * clamped_weight;
}

/// Approximates XSToon3's scene-brightness measurement from the dominant light and SH ambient.
fn environment_brightness(ambient: vec3<f32>, dominant_light_col_atten: vec3<f32>) -> f32 {
    return (xb::grayscale(ambient) + xb::grayscale(dominant_light_col_atten)) * 0.5;
}

/// Base-pass emission contribution, including `_EmissionToDiffuse` and `_ScaleWithLight`.
fn emission_color(
    s: xb::SurfaceData,
    ambient: vec3<f32>,
    dominant_light_col_atten: vec3<f32>,
    base_pass: bool,
) -> vec3<f32> {
    if (!base_pass || !xb::emission_map_enabled()) {
        return vec3<f32>(0.0);
    }

    var emission = mix(s.emission, s.emission * s.diffuse_color, clamp(xb::mat._EmissionToDiffuse, 0.0, 1.0));
    emission = emission * xb::mat._EmissionColor.rgb;

    if (xb::scale_with_light_enabled()) {
        let sensitivity = clamp(xb::mat._ScaleWithLightSensitivity, 0.0, 1.0);
        let scale = xb::saturate(smoothstep(1.0 - sensitivity, 1.0 + sensitivity, 1.0 - environment_brightness(ambient, dominant_light_col_atten)));
        emission = emission * scale;
    }

    return emission;
}

/// Forward-pass clustered light walk.
///
/// Composition follows the XSToon3 accumulator order while preserving XSToon2's ramp-only diffuse
/// model and existing material properties:
///   `surface = diffuse`
///   `surface *= occlusionColor`
///   `surface = applyReflectionBlend(surface, indirectSpecular, reflectivityWeight)`
///   `surface += directSpecular * occlusion`
///   `surface += rim`
///   `surface += subsurface`
///   `surface *= shadowRim`
///   `surface += emission`
fn clustered_toon_lighting(
    frag_xy: vec2<f32>,
    s: xb::SurfaceData,
    world_pos: vec3<f32>,
    view_layer: u32,
    include_directional: bool,
    include_local: bool,
    base_pass: bool,
) -> vec3<f32> {
    let view_dir = xb::safe_normalize(rg::frame.camera_world_pos.xyz - world_pos, vec3<f32>(0.0, 0.0, 1.0));
    let ambient = indirect_diffuse(s);
    let env = environment_tint(s, view_dir);
    let direct_specular_occlusion = occlusion_scalar(s);

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
    let i_max = min(count, pcls::MAX_LIGHTS_PER_TILE);

    var direct_diffuse = vec3<f32>(0.0);
    var direct_spec = vec3<f32>(0.0);
    var sss = vec3<f32>(0.0);

    var dominant_light = xb::LightSample(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0), 0.0, true);
    var dominant_light_col_atten = vec3<f32>(0.0);
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
        direct_diffuse = direct_diffuse + s.albedo.rgb * ramp * light_col_atten;
        direct_spec = direct_spec + direct_specular(s, light, view_dir);
        direct_spec = direct_spec + clearcoat_direct_specular(s, light, view_dir);
        sss = sss + subsurface(s, light, view_dir, ambient);

        let weight = xb::grayscale(light_col_atten * vec3<f32>(xb::saturate(dot(s.normal, light.direction))));
        if (weight > dominant_weight) {
            dominant_weight = weight;
            dominant_light = light;
            dominant_light_col_atten = light_col_atten;
        }
    }

    var surface = direct_diffuse;
    if (base_pass) {
        surface = surface + s.albedo.rgb * ambient;
    }
    surface = surface * s.occlusion;

    if (base_pass) {
        let reflection = indirect_specular(s, view_dir, ambient, dominant_light_col_atten);
        surface = apply_reflection_blend(surface, reflection, reflection_blend_weight(s));
    }

    surface = surface + direct_spec * direct_specular_occlusion;
    surface = surface + sss;

    if (base_pass) {
        if (dominant_weight > 0.0) {
            surface = surface + rim_light(s, dominant_light, view_dir, ambient, env);
            surface = surface * shadow_rim(s, view_dir, dominant_light, ambient);
        }

        surface = surface + emission_color(s, ambient, dominant_light_col_atten, base_pass);
    }

    return max(surface, vec3<f32>(0.0));
}

/// Outline-pass clustered light walk for the "Lit" outline mode.
///
/// Uses the dominant direct-light term plus SH ambient rather than summing every light, matching
/// the newer XSToon3 "main light + ambient" outline response while preserving the existing shell
/// extrusion pass and property aliases.
fn clustered_outline_lighting(
    frag_xy: vec2<f32>,
    s: xb::SurfaceData,
    world_pos: vec3<f32>,
    view_layer: u32,
) -> vec3<f32> {
    let ambient = indirect_diffuse(s);
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
    let i_max = min(count, pcls::MAX_LIGHTS_PER_TILE);

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
