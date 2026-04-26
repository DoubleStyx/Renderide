//! Unity surface shader `Shader "ToonStandard"`: Xiexe-style stylized toon BRDF with stepped
//! Blinn-Phong specular, wrapped/stepped diffuse, optional Fresnel rim.
//!
//! The Unity reference relies on the `unity_NHxRoughness` 2D LUT for specular response. This port
//! computes the specular analytically (normalized Blinn-Phong with smoothness-driven exponent) so
//! it doesn't need a built-in LUT bound. Stepping cadences match the Unity reference
//! (`max((1-smoothness)*4, 0.01)` for specular, two diffuse bands plus transmission).

// unity-shader-name: ToonStandard

#import renderide::globals as rg
#import renderide::per_draw as pd
#import renderide::pbs::brdf as brdf
#import renderide::pbs::normal as pnorm
#import renderide::pbs::cluster as pcls
#import renderide::uv_utils as uvu
#import renderide::normal_decode as nd

struct ToonStandardMaterial {
    _Color: vec4<f32>,
    _SpecColor: vec4<f32>,
    _EmissionColor: vec4<f32>,
    _FresnelTint: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _BumpScale: f32,
    _Glossiness: f32,
    _Transmission: f32,
    _Fresnel: f32,
    _FresnelStrength: f32,
    _FresnelPower: f32,
    _FresnelDiffCont: f32,
    _Cutoff: f32,
    _SpecularHighlights: f32,
    _GlossyReflections: f32,
}

@group(1) @binding(0) var<uniform> mat: ToonStandardMaterial;
@group(1) @binding(1) var _MainTex: texture_2d<f32>;
@group(1) @binding(2) var _MainTex_sampler: sampler;
@group(1) @binding(3) var _SpecGlossMap: texture_2d<f32>;
@group(1) @binding(4) var _SpecGlossMap_sampler: sampler;
@group(1) @binding(5) var _BumpMap: texture_2d<f32>;
@group(1) @binding(6) var _BumpMap_sampler: sampler;
@group(1) @binding(7) var _EmissionMap: texture_2d<f32>;
@group(1) @binding(8) var _EmissionMap_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) @interpolate(flat) view_layer: u32,
}

fn sample_normal_world(uv_main: vec2<f32>, world_n: vec3<f32>) -> vec3<f32> {
    let n = normalize(world_n);
    let tbn = pnorm::orthonormal_tbn(n);
    let ts_n = nd::decode_ts_normal_with_placeholder(
        textureSample(_BumpMap, _BumpMap_sampler, uv_main).xyz,
        mat._BumpScale,
    );
    return normalize(tbn * ts_n);
}

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
) -> VertexOutput {
    let d = pd::get_draw(instance_index);
    let world_p = d.model * vec4<f32>(pos.xyz, 1.0);
    let wn = normalize(d.normal_matrix * n.xyz);
#ifdef MULTIVIEW
    var vp: mat4x4<f32>;
    if (view_idx == 0u) {
        vp = d.view_proj_left;
    } else {
        vp = d.view_proj_right;
    }
#else
    let vp = d.view_proj_left;
#endif
    var out: VertexOutput;
    out.clip_pos = vp * world_p;
    out.world_pos = world_p.xyz;
    out.world_n = wn;
    out.uv0 = uv0;
#ifdef MULTIVIEW
    out.view_layer = view_idx;
#else
    out.view_layer = 0u;
#endif
    return out;
}

/// Stepped wrapped-Lambert diffuse (matches the Unity ToonBRDF_Diffuse cadence).
fn toon_diffuse(n: vec3<f32>, l: vec3<f32>) -> f32 {
    let nl = dot(n, l);
    let t = mat._Transmission;
    let denom = (1.0 + t) * (1.0 + t);
    let wrapped = clamp((nl + t) / max(denom, 1e-4), 0.0, 1.0);
    return min(round(wrapped * 2.0) / 2.0 + t, 1.0);
}

/// Stepped normalized Blinn-Phong specular (analytical replacement for the unity_NHxRoughness LUT).
fn toon_specular(n: vec3<f32>, l: vec3<f32>, v: vec3<f32>, smoothness: f32) -> f32 {
    if (mat._SpecularHighlights < 0.5) {
        return 0.0;
    }
    let nl = max(dot(n, l), 0.0);
    let r = reflect(-v, n);
    let rl = max(dot(r, l), 0.0);
    let rough = clamp(1.0 - smoothness, 0.045, 1.0);
    let shininess = (1.0 - rough) * (1.0 - rough) * 256.0 + 1.0;
    let raw = pow(rl, shininess) * (shininess + 8.0) / (8.0 * 3.14159265);
    let steps = max((1.0 - smoothness) * 4.0, 0.01);
    let stepped = round(raw * steps) / steps;
    return stepped * nl;
}

/// View-dependent stylization rim from the Unity ToonBRDF_Fresnel implementation.
fn toon_fresnel(diff_color: vec3<f32>, view_dir: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    if (mat._Fresnel < 0.5) {
        return vec3<f32>(0.0);
    }
    let rim = 1.0 - clamp(dot(normalize(view_dir), n), 0.0, 1.0);
    let fresnel_color = mix(vec3<f32>(0.5), diff_color, mat._FresnelDiffCont);
    let fresnel_power = pow(rim, max(20.0 - mat._FresnelPower * 20.0, 1e-4));
    let fresnel = fresnel_color * fresnel_power;
    return (mat._FresnelStrength * 5.0) * fresnel * mat._FresnelTint.rgb;
}

fn shade(
    frag_xy: vec2<f32>,
    world_pos: vec3<f32>,
    world_n: vec3<f32>,
    uv0: vec2<f32>,
    view_layer: u32,
    include_directional: bool,
    include_local: bool,
) -> vec4<f32> {
    let uv_main = uvu::apply_st(uv0, mat._MainTex_ST);
    let albedo_s = textureSample(_MainTex, _MainTex_sampler, uv_main);
    let c = albedo_s * mat._Color;
    let base_color = c.rgb;

    let spec_s = textureSample(_SpecGlossMap, _SpecGlossMap_sampler, uv_main);
    let spec_color = spec_s.rgb * mat._SpecColor.rgb;
    let smoothness = clamp(spec_s.a * mat._Glossiness, 0.0, 1.0);

    let emission = textureSample(_EmissionMap, _EmissionMap_sampler, uv_main).rgb * mat._EmissionColor.rgb;

    let n = sample_normal_world(uv_main, world_n);
    let cam = rg::frame.camera_world_pos.xyz;
    let v = normalize(cam - world_pos);

    let cluster_id = pcls::cluster_id_from_frag(
        frag_xy, world_pos, rg::frame.view_space_z_coeffs, rg::frame.view_space_z_coeffs_right,
        view_layer, rg::frame.viewport_width, rg::frame.viewport_height,
        rg::frame.cluster_count_x, rg::frame.cluster_count_y, rg::frame.cluster_count_z,
        rg::frame.near_clip, rg::frame.far_clip,
    );
    let count = rg::cluster_light_counts[cluster_id];
    let i_max = min(count, pcls::MAX_LIGHTS_PER_TILE);
    var lo = vec3<f32>(0.0);
    for (var i = 0u; i < i_max; i++) {
        let li = pcls::cluster_light_index_at(cluster_id, i);
        if (li >= rg::frame.light_count) {
            continue;
        }
        let light = rg::lights[li];
        let is_directional = light.light_type == 1u;
        if ((is_directional && !include_directional) || (!is_directional && !include_local)) {
            continue;
        }
        var l: vec3<f32>;
        var attenuation: f32;
        if (light.light_type == 1u) {
            let dir_len_sq = dot(light.direction, light.direction);
            l = select(vec3<f32>(0.0, 0.0, 1.0), normalize(-light.direction), dir_len_sq > 1e-16);
            attenuation = light.intensity;
        } else {
            let to_light = light.position - world_pos;
            let dist = length(to_light);
            l = normalize(to_light);
            attenuation = light.intensity * brdf::distance_attenuation(dist, light.range);
            if (light.light_type == 2u) {
                let spot_cos = dot(-l, normalize(light.direction));
                let inner = min(light.spot_cos_half_angle + 0.1, 1.0);
                attenuation = attenuation * smoothstep(light.spot_cos_half_angle, inner, spot_cos);
            }
        }
        let diff_step = toon_diffuse(n, l);
        let spec_step = toon_specular(n, l, v, smoothness);
        let radiance = light.color * attenuation;
        lo = lo + radiance * (base_color * diff_step + spec_color * spec_step);
    }

    let fresnel = toon_fresnel(base_color, v, n);
    let extra = select(vec3<f32>(0.0), emission + fresnel, include_directional);
    return vec4<f32>(lo + extra, c.a);
}

//#pass forward
@fragment
fn fs_forward_base(
    @builtin(position) frag_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    return shade(frag_pos.xy, world_pos, world_n, uv0, view_layer, true, true);
}
