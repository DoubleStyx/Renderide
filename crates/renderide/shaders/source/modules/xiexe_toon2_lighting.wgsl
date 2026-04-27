//! Direct + indirect lighting for the Xiexe Toon 2.0 BRDF.
//!
//! Holds the cluster light walk used by both the forward (`clustered_toon_lighting`) and
//! outline (`clustered_outline_lighting`) paths, plus the per-light stylised terms:
//! ramp-driven half-Lambert diffuse, GGX direct specular, rim, shadow rim, subsurface,
//! and matcap/PBR indirect specular.
//!
//! Math follows the upstream Xiexe Toon lighting include files rather than re-using the
//! project's Filament BRDF, so a few
//! upstream quirks are intentionally preserved verbatim:
//!
//! - `_ShadowSharpness` snap is on the shadow attenuation, not the half-Lambert remap
//!   (`XSFrag.cginc:13–15`, `XSLightingFunctions.cginc:325–340`).
//! - `XSGGXTerm` treats its second argument as **linear `α`**; xiexe always passes
//!   `roughness²` so effectively `α = perceptual²`. The visibility term meanwhile uses
//!   the height-correlated Smith with `α = perceptual` directly. Different `α`'s for
//!   `D` and `V` is upstream behaviour.
//! - `F_Schlick` uses Lazarev's exp2 form *with the upstream `(- a · VoH) - (b · VoH)`
//!   bracketing typo* — the inner `· VoH` got dropped from the first term, so the
//!   exponent is `−12.53789·VoH` rather than the canonical Lazarev quadratic. xiexe
//!   passes `f0 = 0` so the term collapses to that single exp2.
//! - Direct specular accumulates `attenuation²` (`XSLightingFunctions.cginc:207` and
//!   `212`) — the second multiply on line 212 stacks with the one already inside `smooth`.
//! - `_SpecularMap` channel multipliers (`.r`/`.g`/`.b`) are commented out upstream;
//!   the texture is bound but unread. Preserved here.
//! - `calcReflectionBlending` (`XSLightingFunctions.cginc:409–417`) collapses to plain
//!   additive — the `_ReflectionBlendMode` branches are commented out.
//! - `calcEmission` returns `i.emissionMap` directly (`XSLightingFunctions.cginc:388–407`);
//!   the `_EmissionToDiffuse` and `_ScaleWithLight` blends are commented out.
//!
//! Approximations needed because we don't have Unity's per-frame specular probe state:
//! - PBR indirect specular (Unity `unity_SpecCube`) → ambient-tinted metallic blend.
//!   Matcap mode follows the reference exactly except for `_LightColor0`, which we
//!   approximate as the dominant light's `color · attenuation` tracked across the
//!   cluster walk.

#define_import_path renderide::xiexe::toon2::lighting

#import renderide::xiexe::toon2::base as xb
#import renderide::xiexe::toon2::surface as xsurf
#import renderide::globals as rg
#import renderide::pbs::cluster as pcls
#import renderide::birp::light as bl
#import renderide::sh2_ambient as shamb

/// SH-probe sample used for xiexe's uncoloured indirect-diffuse term.
fn indirect_diffuse(s: xb::SurfaceData) -> vec3<f32> {
    return shamb::ambient_probe(s.normal);
}

/// `UNITY_SPECCUBE_LOD_STEPS` on PC/console. Matcap LOD selection in
/// `XSLightingFunctions.cginc:227` uses `(1 − smoothness) · UNITY_SPECCUBE_LOD_STEPS`.
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
/// **attenuation** before it multiplies half-Lambert — matching `XSFrag.cginc:13–15`
/// and ensuring banding only appears at shadow boundaries (where `attenuation < 1`),
/// never on the diffuse ramp itself.
fn ramp_for_ndl(ndl: f32, attenuation: f32, ramp_mask: f32) -> vec3<f32> {
    let att_sharp = mix(attenuation, round(attenuation), clamp(xb::mat._ShadowSharpness, 0.0, 1.0));
    let x = clamp((ndl * 0.5 + 0.5) * att_sharp, 0.0, 1.0);
    return textureSample(xb::_Ramp, xb::_Ramp_sampler, vec2<f32>(x, clamp(ramp_mask, 0.0, 1.0))).rgb;
}

/// `XSGGXTerm` from `XSLightingFunctions.cginc:14–20`: `α² / (π · ((NdotH²)(α²−1)+1)²)`,
/// rearranged through the `(NdotH·α² − NdotH)·NdotH + 1` denominator. Caller passes
/// linear `α` directly. `direct_specular` constructs `α = roughness · roughness` to
/// match xiexe's `XSGGXTerm(NdotH, roughness²)` convention.
fn xs_ggx_term(n_dot_h: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let denom = (n_dot_h * a2 - n_dot_h) * n_dot_h + 1.0;
    return (1.0 / 3.14159265) * a2 / max(denom * denom, 1e-7);
}

/// Unity's `SmithJointGGXVisibilityTerm` (height-correlated Smith), ported verbatim from
/// `UnityStandardBRDF.cginc`. `roughness` is **perceptual** (= `1 − smoothness`,
/// remapped); xiexe passes the same value here that it squares into `xs_ggx_term`'s `α`.
fn smith_joint_ggx_visibility(n_dot_l: f32, n_dot_v: f32, roughness: f32) -> f32 {
    let a = roughness;
    let lambda_v = n_dot_l * (n_dot_v * (1.0 - a) + a);
    let lambda_l = n_dot_v * (n_dot_l * (1.0 - a) + a);
    return 0.5 / max(lambda_v + lambda_l, 1e-5);
}

/// xiexe's `F_Schlick` — Lazarev exp2 form with the upstream input-space typo
/// (`exp2((-5.55473 · VoH) - (6.98316 · VoH))` simplifies to `exp2(-12.53789 · VoH)`,
/// the inner `· VoH` having been dropped from the first term in upstream code). Kept
/// verbatim for parity. `direct_specular` always passes `f0 = 0`, so the helper returns
/// just the exponent term (the full Schlick is `f0 + (1-f0) · exp2(...)`).
fn xs_fresnel_zero(voh: f32) -> f32 {
    return exp2((-5.55473 - 6.98316) * voh);
}

/// One light's GGX direct-specular contribution. Mirrors `calcDirectSpecular`
/// (`XSLightingFunctions.cginc:180–216`):
/// - `_SpecularArea` is `max(0.01, _SpecularArea)` then remapped via
///   `smoothness *= 1.7 − 0.7·smoothness`. The `_SpecularMap.b` multiplier is commented
///   out upstream and is therefore **not** applied here.
/// - `α = roughness² = perceptual²` for the GGX `D` term; the `V` term gets perceptual
///   `α` (different convention is intentional upstream).
/// - `F` is the typo-form Schlick with `f0 = 0` (so just the exp2).
/// - The reflection scalar `V · D · π · sndl · F · attenuation · _SpecularIntensity` is
///   then multiplied **again** by `attenuation · lightCol`, stacking `attenuation²` —
///   upstream behaviour, line 212 follows the inner multiply on line 207.
/// - Albedo tint is `lerp(spec, spec · diffuseColor, _SpecularAlbedoTint)`, with the
///   `_SpecularMap.g` multiplier commented out upstream.
fn direct_specular(
    s: xb::SurfaceData,
    light: xb::LightSample,
    view_dir: vec3<f32>,
    ndl: f32,
) -> vec3<f32> {
    let h = xb::safe_normalize(light.direction + view_dir, s.normal);
    let ndh = xb::saturate(dot(s.normal, h));
    let ndv = max(abs(dot(view_dir, s.normal)), 1e-4);
    let ldh = xb::saturate(dot(light.direction, h));
    let sndl = xb::saturate(ndl);

    var smoothness = max(0.01, xb::mat._SpecularArea);
    smoothness = smoothness * (1.7 - 0.7 * smoothness);
    let roughness = 1.0 - smoothness;
    let alpha = roughness * roughness;

    let v_term = smith_joint_ggx_visibility(sndl, ndv, roughness);
    let f_term = xs_fresnel_zero(ldh);
    let d_term = xs_ggx_term(ndh, alpha);
    let reflection = v_term * d_term * 3.14159265;
    let smooth_scalar = max(0.0, reflection * sndl) * f_term * light.attenuation * xb::mat._SpecularIntensity;

    var specular = vec3<f32>(smooth_scalar) * light.attenuation * light.color;
    let tinted = specular * s.diffuse_color;
    specular = mix(specular, tinted, clamp(xb::mat._SpecularAlbedoTint, 0.0, 1.0));
    return specular;
}

/// Rim contribution. Matches `calcRimLight` in `XSLightingFunctions.cginc:162–169`:
/// `rim = saturate(1 − VdotN) · pow(saturate(NdotL), _RimThreshold)` smoothstepped
/// against `_RimRange ± _RimSharpness`, scaled by `_RimIntensity · (lightCol +
/// indirectDiffuse)`, attenuation-modulated by `lerp(1, attenuation + indirectDiffuse,
/// _RimAttenEffect)`, and tinted by `_RimColor · lerp(1, diffuseColor, _RimAlbedoTint) ·
/// lerp(1, envMap, _RimCubemapTint)`.
fn rim_light(
    s: xb::SurfaceData,
    light: xb::LightSample,
    view_dir: vec3<f32>,
    ndl: f32,
    env_map: vec3<f32>,
) -> vec3<f32> {
    let vdn = abs(dot(view_dir, s.normal));
    let sharp = max(xb::mat._RimSharpness, 0.001);
    var rim = xb::saturate(1.0 - vdn) * pow(xb::saturate(ndl), max(xb::mat._RimThreshold, 0.0));
    rim = smoothstep(xb::mat._RimRange - sharp, xb::mat._RimRange + sharp, rim);
    let indirect = indirect_diffuse(s);
    var col = rim * xb::mat._RimIntensity * (light.color * light.attenuation + indirect);
    col = col * mix(vec3<f32>(1.0), vec3<f32>(light.attenuation) + indirect, clamp(xb::mat._RimAttenEffect, 0.0, 1.0));
    col = col * xb::mat._RimColor.rgb;
    col = col * mix(vec3<f32>(1.0), s.diffuse_color, clamp(xb::mat._RimAlbedoTint, 0.0, 1.0));
    col = col * mix(vec3<f32>(1.0), env_map, clamp(xb::mat._RimCubemapTint, 0.0, 1.0));
    return col;
}

/// Shadow-rim multiplier in `[0, 1]`. Matches `calcShadowRim` in
/// `XSLightingFunctions.cginc:171–178`. The fragment colour multiplies by this on the
/// direct-diffuse term only.
fn shadow_rim(s: xb::SurfaceData, view_dir: vec3<f32>, ndl: f32) -> vec3<f32> {
    let vdn = abs(dot(view_dir, s.normal));
    let sharp = max(xb::mat._ShadowRimSharpness, 0.001);
    var rim = xb::saturate(1.0 - vdn) * pow(xb::saturate(1.0 - ndl), max(xb::mat._ShadowRimThreshold * 2.0, 0.0));
    rim = smoothstep(xb::mat._ShadowRimRange - sharp, xb::mat._ShadowRimRange + sharp, rim);
    let indirect = indirect_diffuse(s);
    let tint = xb::mat._ShadowRim.rgb * mix(vec3<f32>(1.0), s.diffuse_color, clamp(xb::mat._ShadowRimAlbedoTint, 0.0, 1.0)) + indirect * 0.1;
    return mix(vec3<f32>(1.0), tint, rim);
}

/// Stylised subsurface scattering. Reproduces the original xiexe construction:
/// `H = normalize(L + N · _SSDistortion)`, `vdh = pow(saturate(dot(V, -H)), _SSPower)`,
/// modulated by attenuation, half-Lambert, thickness, and `_SSColor · _SSScale · albedo`.
/// Uses the frame SH2 probe for the additive term, matching
/// `calcSubsurfaceScattering` in `XSLightingFunctions.cginc:362–386`.
fn subsurface(
    s: xb::SurfaceData,
    light: xb::LightSample,
    view_dir: vec3<f32>,
    ndl: f32,
) -> vec3<f32> {
    if (dot(xb::mat._SSColor.rgb, xb::mat._SSColor.rgb) <= 1e-8) {
        return vec3<f32>(0.0);
    }
    let attenuation = xb::saturate(light.attenuation * (ndl * 0.5 + 0.5));
    let h = xb::safe_normalize(light.direction + s.normal * xb::mat._SSDistortion, s.normal);
    let vdh = pow(xb::saturate(dot(view_dir, -h)), max(xb::mat._SSPower, 0.001));
    let scatter = xb::mat._SSColor.rgb * (vdh + indirect_diffuse(s)) * attenuation * xb::mat._SSScale * s.thickness;
    return max(vec3<f32>(0.0), light.color * scatter * s.albedo.rgb);
}

/// View-space matcap UV. Projects `n` onto the camera's right and up basis vectors
/// (derived from `view_dir` and world up) and remaps to `[0, 1]`. Matches Unity's
/// `matcapSample` in `XSHelperFunctions.cginc:134–140`.
fn matcap_uv(view_dir: vec3<f32>, n: vec3<f32>) -> vec2<f32> {
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let view_up = xb::safe_normalize(up - view_dir * dot(view_dir, up), vec3<f32>(0.0, 1.0, 0.0));
    let view_right = xb::safe_normalize(cross(view_dir, view_up), vec3<f32>(1.0, 0.0, 0.0));
    return vec2<f32>(dot(view_right, n), dot(view_up, n)) * 0.5 + vec2<f32>(0.5);
}

/// Indirect-specular contribution. Two upstream branches:
///
/// - `MATCAP` keyword — sample `_Matcap` at the view-space normal projection, weighted
///   by `_MatcapTint`, modulated by `(indirectDiffuse + dominantLight · 0.5)`.
///   Matches `XSLightingFunctions.cginc:222–233` exactly except `_LightColor0` is
///   approximated as the dominant light's `color · attenuation` tracked across the
///   cluster walk (we don't have Unity's "main directional" handle).
///
/// - PBR fallback — upstream samples `unity_SpecCube` probes (lines 237–266); we don't
///   have specular probes, so fall back to an ambient-tinted metallic blend that preserves
///   `lerp(indirectSpecular, metallicColor, pow(vdn, 0.05))` shape with diffuse SH standing
///   in for the probe sample. Approximation, not strict parity.
///
/// Final `lerp(spec, spec · ramp, metallicSmoothness.w)` darkens the result by the toon
/// ramp proportional to perceptual roughness (`metallicSmoothness.w` is `(1 − gloss) ·
/// (1.7 − 0.7·(1 − gloss))` upstream, which is exactly `s.roughness`).
fn indirect_specular(
    s: xb::SurfaceData,
    view_dir: vec3<f32>,
    dominant_ramp: vec3<f32>,
    dominant_light_col_atten: vec3<f32>,
) -> vec3<f32> {
    var spec = vec3<f32>(0.0);
    if (xb::matcap_enabled()) {
        let uv = matcap_uv(view_dir, s.normal);
        spec = textureSampleLevel(xb::_Matcap, xb::_Matcap_sampler, uv, (1.0 - s.smoothness) * SPECCUBE_LOD_STEPS).rgb * xb::mat._MatcapTint.rgb;
        spec = spec * (indirect_diffuse(s) + dominant_light_col_atten * 0.5);
    } else {
        // Probe approximation — see header note. Diffuse SH is the specular-probe stand-in;
        // the `lerp(probe, metallicColor, pow(vdn, 0.05))` shape is preserved.
        let vdn = max(abs(dot(view_dir, s.normal)), 1e-4);
        let probe = indirect_diffuse(s);
        let metallic_color = probe * mix(vec3<f32>(0.05), s.diffuse_color, s.metallic);
        spec = mix(probe, metallic_color, pow(vdn, 0.05));
    }

    spec = mix(spec, spec * dominant_ramp, s.roughness);
    return spec;
}

/// Forward-pass clustered light walk.
///
/// Final composition follows `BRDF_XSLighting` (`XSLighting.cginc:57–72`):
///   `col = diffuse · shadowRim`
///   `col += indirectSpecular`
///   `col += max(directSpecular, rimLight)`
///   `col += subsurface`
///   `col *= occlusion` (when AO is enabled)
///   `col += emission` (added *after* occlusion so emissives aren't AO-darkened)
///
/// `diffuse` is `albedo · (Σ_lights ramp · att · lightCol + indirectDiffuse)` — the
/// `albedo · indirectDiffuse` is added once at the end of the loop on the base pass.
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

    // Env-map sample is reused only by `rim_light`'s `_RimCubemapTint` blend. Without a
    // probe pipeline we substitute a neutral-toned env value so `_RimCubemapTint` reads
    // as a soft tint toward white rather than a hard black band.
    let env = vec3<f32>(1.0);

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
    var rim = vec3<f32>(0.0);
    var sss = vec3<f32>(0.0);
    var strongest_shadow = vec3<f32>(1.0);

    // Track the dominant punctual contribution for `indirect_specular`'s matcap
    // modulator and ramp-darkening multiplier — the upstream `_LightColor0` and
    // `calcRamp(d.ndl)` references are scoped to a single base-pass directional light.
    var dominant_weight = -1.0;
    var dominant_light_col_atten = vec3<f32>(0.0);
    var dominant_ramp = vec3<f32>(1.0);

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
        // `calcDiffuse` (`XSLightingFunctions.cginc:351–358`): per-light diffuse is
        // `albedo · ramp · att · lightCol`. `att` is already baked into `ramp`'s X
        // coordinate, so the explicit `·att` here stacks (matching upstream).
        direct_diffuse = direct_diffuse + s.albedo.rgb * ramp * light_col_atten;
        direct_spec = direct_spec + direct_specular(s, light, view_dir, ndl);
        rim = rim + rim_light(s, light, view_dir, ndl, env);
        sss = sss + subsurface(s, light, view_dir, ndl);
        strongest_shadow = min(strongest_shadow, shadow_rim(s, view_dir, ndl));

        let weight = dot(light_col_atten, vec3<f32>(0.2125, 0.7154, 0.0721));
        if (weight > dominant_weight) {
            dominant_weight = weight;
            dominant_light_col_atten = light_col_atten;
            dominant_ramp = ramp;
        }
    }

    var diffuse = direct_diffuse;
    if (base_pass) {
        // `i.albedo * indirectDiffuse` from `calcDiffuse` — added once on the base pass.
        diffuse = diffuse + s.albedo.rgb * indirect_diffuse(s);
    }

    var col = diffuse * strongest_shadow;
    col = col + indirect_specular(s, view_dir, dominant_ramp, dominant_light_col_atten);
    col = col + max(direct_spec, rim);
    col = col + sss;
    col = col * s.occlusion;
    if (base_pass) {
        col = col + s.emission;
    }
    return max(col, vec3<f32>(0.0));
}

/// Outline-pass clustered light walk for the "Lit" outline mode.
///
/// Reproduces the reference outline lighting from `XSLightingFunctions.cginc:307–310`:
///   `outlineColor = ol · saturate(att · NdotL) · lightCol + indirectDiffuse · ol`
/// where `ol = _OutlineColor (· diffuse if _OutlineAlbedoTint)`. Returns the *light
/// modulator* (without `ol`); the caller multiplies by `ol`. The "indirect diffuse"
/// approximation is the same SH2 indirect-diffuse sample used elsewhere in this module.
fn clustered_outline_lighting(
    frag_xy: vec2<f32>,
    s: xb::SurfaceData,
    world_pos: vec3<f32>,
    view_layer: u32,
) -> vec3<f32> {
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

    var direct = vec3<f32>(0.0);
    for (var i = 0u; i < i_max; i++) {
        let li = pcls::cluster_light_index_at(cluster_id, i);
        if (li >= rg::frame.light_count) {
            continue;
        }
        let light = sample_light(rg::lights[li], world_pos);
        let ndl = xb::saturate(dot(s.normal, light.direction));
        direct = direct + xb::saturate(light.attenuation * ndl) * light.color;
    }
    return direct + indirect_diffuse(s);
}
