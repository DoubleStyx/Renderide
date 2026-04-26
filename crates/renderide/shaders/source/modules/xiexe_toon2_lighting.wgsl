//! Direct + indirect lighting for the Xiexe Toon 2.0 BRDF.
//!
//! Holds the cluster light walk used by both the forward (`clustered_toon_lighting`) and
//! outline (`clustered_outline_lighting`) paths, plus the per-light stylised terms:
//! ramp-driven half-Lambert diffuse, GGX direct specular, rim, shadow rim, subsurface,
//! and matcap/PBR indirect specular.
//!
//! Shadow-sharpness parity fix vs `references_external/.../XSFrag.cginc:13–15`: the
//! `lerp(att, round(att), _ShadowSharpness)` snap is applied to the punctual / shadow
//! attenuation **before** it multiplies the half-Lambert ramp, not to the half-Lambert
//! itself. This sharpens shadow boundaries without producing diffuse-ramp banding on
//! lit-side surfaces.

#define_import_path renderide::xiexe::toon2::lighting

#import renderide::xiexe::toon2::base as xb
#import renderide::xiexe::toon2::surface as xsurf
#import renderide::globals as rg
#import renderide::pbs::cluster as pcls

/// Normalized windowed inverse-square distance attenuation for punctual lights.
/// `intensity * (saturate(1 − t⁴))² / max(t², ε²)` evaluated in `t = dist/range` space so the
/// falloff shape stretches with the light's range slider rather than clipping a world-space
/// inverse-square curve. Matches Unity BiRP's LUT-style behaviour where the range slider only
/// changes how far the light reaches, not its peak brightness; the Karis/Lagarde quartic window
/// keeps the boundary at `dist == range` smooth and exactly zero. The `ε = 0.01` floor (relative
/// to range) caps the near-light singularity at a range-independent peak.
fn punctual_attenuation(intensity: f32, dist: f32, range: f32) -> f32 {
    if (range <= 0.0) {
        return 0.0;
    }
    let t = dist / range;
    let t2 = max(t * t, 0.0001);
    let window_inner = clamp(1.0 - t2 * t2, 0.0, 1.0);
    let window = window_inner * window_inner;
    return intensity * window / t2;
}

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
    var attenuation = punctual_attenuation(light.intensity, dist, light.range);
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

/// GGX/Trowbridge–Reitz NDF in Karis's stable form. `alpha` is **linear** roughness
/// (`α`); the xiexe convention is `α = perceptual_roughness²`, matching
/// `XSGGXTerm(NdotH, roughness²)` from `XSLightingFunctions.cginc:14–20`.
fn ggx_distribution(n_dot_h: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(3.14159265 * denom * denom, 1e-7);
}

/// Schlick-style Smith visibility approximation (Karis k = (r+1)²/8). `roughness` is
/// perceptual.
fn smith_visibility(n_dot_l: f32, n_dot_v: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = r * r * 0.125;
    let gl = n_dot_l / max(n_dot_l * (1.0 - k) + k, 1e-4);
    let gv = n_dot_v / max(n_dot_v * (1.0 - k) + k, 1e-4);
    return gl * gv;
}

/// Lazarev's exp2-form Schlick Fresnel for `f0 = 0` (xiexe always passes a zero `f0` to
/// `F_Schlick`, so this collapses to just the exponent term).
fn fresnel_schlick_scalar(voh: f32) -> f32 {
    return exp2((-5.55473 * voh - 6.98316) * voh);
}

/// One light's GGX direct-specular contribution. Uses the xiexe-specific specular-area
/// remap (`smoothness *= 1.7 − 0.7·smoothness`) and `α = roughness²` so highlights match
/// the Unity reference's sharpness profile.
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
    var smoothness = clamp(xb::mat._SpecularArea * s.specular_mask.b, 0.01, 1.0);
    smoothness = smoothness * (1.7 - 0.7 * smoothness);
    let roughness = clamp(1.0 - smoothness, 0.045, 1.0);
    let d = ggx_distribution(ndh, roughness * roughness);
    let v = smith_visibility(xb::saturate(ndl), ndv, roughness);
    let f = fresnel_schlick_scalar(ldh);
    var spec = max(0.0, v * d * 3.14159265 * xb::saturate(ndl)) * f;
    spec = spec * xb::mat._SpecularIntensity * s.specular_mask.r * light.attenuation;

    var out_spec = vec3<f32>(spec) * light.color;
    out_spec = mix(out_spec, out_spec * s.diffuse_color, clamp(xb::mat._SpecularAlbedoTint * s.specular_mask.g, 0.0, 1.0));
    return out_spec;
}

/// Rim contribution. Matches `calcRimLight` in `XSLightingFunctions.cginc:162–169`:
/// `rim = saturate(1 − VdotN) · pow(saturate(NdotL), _RimThreshold)` smoothstepped
/// against `_RimRange ± _RimSharpness`, then tinted by light, ambient, albedo, and env.
fn rim_light(
    s: xb::SurfaceData,
    light: xb::LightSample,
    view_dir: vec3<f32>,
    ndl: f32,
    ambient: vec3<f32>,
    env_map: vec3<f32>,
) -> vec3<f32> {
    let vdn = abs(dot(view_dir, s.normal));
    let sharp = max(xb::mat._RimSharpness, 0.001);
    var rim = xb::saturate(1.0 - vdn) * pow(xb::saturate(ndl), max(xb::mat._RimThreshold, 0.0));
    rim = smoothstep(xb::mat._RimRange - sharp, xb::mat._RimRange + sharp, rim);
    var col = rim * xb::mat._RimIntensity * (light.color * light.attenuation + ambient);
    col = col * mix(vec3<f32>(1.0), vec3<f32>(light.attenuation) + ambient, clamp(xb::mat._RimAttenEffect, 0.0, 1.0));
    col = col * xb::mat._RimColor.rgb;
    col = col * mix(vec3<f32>(1.0), s.diffuse_color, clamp(xb::mat._RimAlbedoTint, 0.0, 1.0));
    col = col * mix(vec3<f32>(1.0), env_map, clamp(xb::mat._RimCubemapTint, 0.0, 1.0));
    return col;
}

/// Shadow-rim contribution. Matches `calcShadowRim` in `XSLightingFunctions.cginc:171–178`.
/// Returns a multiplier (mostly `1`, dipping toward the tint on shadowed silhouettes).
fn shadow_rim(s: xb::SurfaceData, view_dir: vec3<f32>, ndl: f32, ambient: vec3<f32>) -> vec3<f32> {
    let vdn = abs(dot(view_dir, s.normal));
    let sharp = max(xb::mat._ShadowRimSharpness, 0.001);
    var rim = xb::saturate(1.0 - vdn) * pow(xb::saturate(1.0 - ndl), max(xb::mat._ShadowRimThreshold * 2.0, 0.0));
    rim = smoothstep(xb::mat._ShadowRimRange - sharp, xb::mat._ShadowRimRange + sharp, rim);
    let tint = xb::mat._ShadowRim.rgb * mix(vec3<f32>(1.0), s.diffuse_color, clamp(xb::mat._ShadowRimAlbedoTint, 0.0, 1.0)) + ambient * 0.1;
    return mix(vec3<f32>(1.0), tint, rim);
}

/// Stylised subsurface scattering. Reproduces the original xiexe construction:
/// `H = normalize(L + N · _SSDistortion)`, `vdh = pow(saturate(dot(V, -H)), _SSPower)`,
/// modulated by attenuation, half-Lambert, thickness, and `_SSColor · _SSScale · albedo`.
fn subsurface(
    s: xb::SurfaceData,
    light: xb::LightSample,
    view_dir: vec3<f32>,
    ndl: f32,
    ambient: vec3<f32>,
) -> vec3<f32> {
    if (dot(xb::mat._SSColor.rgb, xb::mat._SSColor.rgb) <= 1e-8) {
        return vec3<f32>(0.0);
    }
    let attenuation = xb::saturate(light.attenuation * (ndl * 0.5 + 0.5));
    let h = xb::safe_normalize(light.direction + s.normal * xb::mat._SSDistortion, s.normal);
    let vdh = pow(xb::saturate(dot(view_dir, -h)), max(xb::mat._SSPower, 0.001));
    let scatter = xb::mat._SSColor.rgb * (vdh + ambient) * attenuation * xb::mat._SSScale * s.thickness;
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

/// Indirect-specular contribution: matcap when enabled, otherwise an ambient-tinted
/// metallic blend. Reflection blend modes (`additive` / `subtractive` / `multiply`)
/// follow `_ReflectionBlendMode`.
fn indirect_specular(s: xb::SurfaceData, view_dir: vec3<f32>, ramp_shadow: vec3<f32>, ambient: vec3<f32>) -> vec3<f32> {
    var spec = vec3<f32>(0.0);
    if (xb::matcap_enabled()) {
        let uv = matcap_uv(view_dir, s.normal);
        spec = textureSampleLevel(xb::_Matcap, xb::_Matcap_sampler, uv, (1.0 - s.smoothness) * 6.0).rgb * xb::mat._MatcapTint.rgb;
        spec = spec * (ambient + vec3<f32>(0.5));
    } else if (xb::mat._ReflectionMode < 0.5) {
        let metallic_color = mix(vec3<f32>(0.05), s.diffuse_color, s.metallic);
        spec = ambient * metallic_color * (1.0 - s.roughness);
    }

    spec = spec * s.reflectivity;
    spec = mix(spec, spec * ramp_shadow, s.roughness);

    if (xb::mat._ReflectionBlendMode > 0.5 && xb::mat._ReflectionBlendMode < 1.5) {
        return spec - vec3<f32>(1.0);
    }
    if (xb::mat._ReflectionBlendMode > 1.5 && xb::mat._ReflectionBlendMode < 2.5) {
        return -spec;
    }
    return spec;
}

/// Forward-pass clustered light walk. Iterates the cluster's light list, accumulates
/// direct diffuse (via the toon ramp), specular, rim, SSS, and tracks the strongest
/// shadow-rim multiplier. On `base_pass` adds ambient + emission + indirect specular.
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
    let ambient = vec3<f32>(0.03) * s.diffuse_color;
    let env = indirect_specular(s, view_dir, vec3<f32>(1.0), ambient);

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
    let count = rg::cluster_light_counts[cluster_id];
    let i_max = min(count, pcls::MAX_LIGHTS_PER_TILE);

    var lit = vec3<f32>(0.0);
    var spec = vec3<f32>(0.0);
    var rim = vec3<f32>(0.0);
    var sss = vec3<f32>(0.0);
    var strongest_shadow = vec3<f32>(1.0);

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
        let light_col = light.color * light.attenuation;
        lit = lit + s.albedo.rgb * ramp * light_col;
        spec = spec + direct_specular(s, light, view_dir, ndl);
        rim = rim + rim_light(s, light, view_dir, ndl, ambient, env);
        sss = sss + subsurface(s, light, view_dir, ndl, ambient);
        strongest_shadow = min(strongest_shadow, shadow_rim(s, view_dir, ndl, ambient));
    }

    if (base_pass) {
        lit = lit + ambient * s.albedo.rgb + s.emission;
        lit = lit + indirect_specular(s, view_dir, strongest_shadow, ambient);
    }

    var color = lit * strongest_shadow + max(spec, rim) + sss;
    color = color * s.occlusion;
    return max(color, vec3<f32>(0.0));
}

/// Outline-pass clustered light walk for the "Lit" outline mode.
///
/// Reproduces the reference outline lighting from `XSLightingFunctions.cginc:307–310`:
///   `outlineColor = ol · saturate(att · NdotL) · lightCol + indirectDiffuse · ol`
/// where `ol = _OutlineColor (· diffuse if _OutlineAlbedoTint)`. Returns the *light
/// modulator* (without `ol`); the caller multiplies by `ol`. The "indirect diffuse"
/// approximation is the ambient term used elsewhere in this module — there's no
/// spherical-harmonic probe pipeline yet, so a single ambient constant stands in.
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
    let count = rg::cluster_light_counts[cluster_id];
    let i_max = min(count, pcls::MAX_LIGHTS_PER_TILE);

    let ambient = vec3<f32>(0.03);
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
    return direct + ambient;
}
