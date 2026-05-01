//! Stylized toon water (`Shader "ToonWater"`) with grab-pass refraction, voronoi-driven wave
//! displacement, and depth-based visibility/crests.
//!
//! Adapted from the Unity reference's `StandardToonWater` surface shader to fit the project's
//! self-contained clustered-toon pattern (see `toonstandard.wgsl`). Notable changes:
//! - `_Time.x` / `_SinTime.w` replaced with host-driven `_AnimationOffset` (renderer does not
//!   expose engine time uniforms to materials directly).
//! - Scene depth sampled via [`renderide::scene_depth_sample`]; reconstructed view-space depth
//!   replaces the Unity `_CameraDepthTexture` + `_InverseView` unprojection.
//! - Refracted scene color sampled via [`renderide::grab_pass`].
//! - Planar-reflection compositing (`_ReflectionTex`) is gated by `_PlanarReflection` and is a
//!   straight overlay until a planar-reflection pass is wired into the render graph.


#import renderide::globals as rg
#import renderide::per_draw as pd
#import renderide::mesh::vertex as mv
#import renderide::pbs::brdf as brdf
#import renderide::pbs::cluster as pcls
#import renderide::pbs::sampling as psamp
#import renderide::grab_pass as gp
#import renderide::scene_depth_sample as sds
#import renderide::uv_utils as uvu
#import renderide::voronoi as vor

struct ToonWaterMaterial {
    _Color: vec4<f32>,
    _SpecColor: vec4<f32>,
    _EmissionColor: vec4<f32>,
    _FresnelTint: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _Cutoff: f32,
    _Glossiness: f32,
    _Transmission: f32,
    _BumpScale: f32,
    _WaveHeight: f32,
    _WaveScale: f32,
    _WaveCrest: f32,
    _AnimationOffset: f32,
    _Fresnel: f32,
    _FresnelStrength: f32,
    _FresnelPower: f32,
    _FresnelDiffCont: f32,
    _SeparateVoronoi: f32,
    _SpecularHighlights: f32,
    _PlanarReflection: f32,
    _SmoothnessTextureChannel: f32,
    _MainTex_StorageVInverted: f32,
    _pad0: vec3<f32>,
}

@group(1) @binding(0) var<uniform> mat: ToonWaterMaterial;
@group(1) @binding(1)  var _MainTex: texture_2d<f32>;
@group(1) @binding(2)  var _MainTex_sampler: sampler;
@group(1) @binding(3)  var _SpecGlossMap: texture_2d<f32>;
@group(1) @binding(4)  var _SpecGlossMap_sampler: sampler;
@group(1) @binding(5)  var _BumpMap: texture_2d<f32>;
@group(1) @binding(6)  var _BumpMap_sampler: sampler;
@group(1) @binding(7)  var _EmissionMap: texture_2d<f32>;
@group(1) @binding(8)  var _EmissionMap_sampler: sampler;
@group(1) @binding(9)  var _VoronoiTex: texture_2d<f32>;
@group(1) @binding(10) var _VoronoiTex_sampler: sampler;
@group(1) @binding(11) var _NoiseTex: texture_2d<f32>;
@group(1) @binding(12) var _NoiseTex_sampler: sampler;
@group(1) @binding(13) var _ReflectionTex: texture_2d<f32>;
@group(1) @binding(14) var _ReflectionTex_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) world_t: vec4<f32>,
    @location(3) uv0: vec2<f32>,
    @location(4) uv1: vec2<f32>,
    @location(5) @interpolate(flat) view_layer: u32,
}

fn voronoi_sample_at(uv: vec2<f32>) -> f32 {
    if (mat._SeparateVoronoi > 0.5) {
        let scale = max(10.0 - 10.0 * mat._WaveScale, 1e-4);
        return vor::voronoi_min_dist(uv * scale, mat._AnimationOffset);
    }
    return textureSampleLevel(_VoronoiTex, _VoronoiTex_sampler, uv, 0.0).r;
}

fn voronoi_sample_at_fragment(uv: vec2<f32>) -> f32 {
    if (mat._SeparateVoronoi > 0.5) {
        let scale = max(10.0 - 10.0 * mat._WaveScale, 1e-4);
        return vor::voronoi_min_dist(uv * scale, mat._AnimationOffset);
    }
    return textureSample(_VoronoiTex, _VoronoiTex_sampler, uv).r;
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
    @location(3) color: vec4<f32>,
    @location(4) t: vec4<f32>,
    @location(5) uv1: vec2<f32>,
) -> VertexOutput {
    let d = pd::get_draw(instance_index);

    let voronoi_v = voronoi_sample_at(select(uv0, uv1, mat._SeparateVoronoi > 0.5));
    var displaced_pos = pos;
    displaced_pos.y = displaced_pos.y + voronoi_v * mat._WaveHeight;

    let world_p = mv::world_position(d, displaced_pos);
    let wn = mv::world_normal(d, n);
    let wt = mv::world_tangent(d, t);
#ifdef MULTIVIEW
    let vp = mv::select_view_proj(d, view_idx);
    let layer = view_idx;
#else
    let vp = mv::select_view_proj(d, 0u);
    let layer = 0u;
#endif

    var out: VertexOutput;
    out.clip_pos = vp * world_p;
    out.world_pos = world_p.xyz;
    out.world_n = wn;
    out.world_t = wt;
    out.uv0 = uv0;
    out.uv1 = uv1;
    out.view_layer = layer;
    return out;
}

fn refract_screen_uv(uv_in: vec2<f32>) -> vec2<f32> {
    var uv = uv_in;
    let phase = 0.25 * mat._WaveHeight * sin(mat._AnimationOffset) * sin(uv.y * 50.0);
    uv.x = uv.x + 0.1 * sin((1.0 - uv.x - uv.y) * phase);
    return clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0));
}

fn toon_diffuse(n: vec3<f32>, l: vec3<f32>) -> f32 {
    let nl = dot(n, l);
    let t = mat._Transmission;
    let denom = (1.0 + t) * (1.0 + t);
    let wrapped = clamp((nl + t) / max(denom, 1e-4), 0.0, 1.0);
    return min(round(wrapped * 2.0) / 2.0 + t, 1.0);
}

fn toon_specular(n: vec3<f32>, l: vec3<f32>, v: vec3<f32>, smoothness: f32) -> f32 {
    if (mat._SpecularHighlights < 0.5) {
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

//#pass forward
@fragment
fn fs_main(
    @builtin(position) frag_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) world_t: vec4<f32>,
    @location(3) uv0: vec2<f32>,
    @location(4) uv1: vec2<f32>,
    @location(5) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    let uv_main = uvu::apply_st_for_storage(uv0, mat._MainTex_ST, mat._MainTex_StorageVInverted);

    let voronoi_f = voronoi_sample_at_fragment(uv0);

    let screen_uv = gp::frag_screen_uv(frag_pos);
    let refracted_uv = refract_screen_uv(screen_uv);
    let grab_color = gp::sample_scene_color(refracted_uv, view_layer).rgb;

    let scene_eye_z = sds::scene_linear_depth_at_uv(refracted_uv, view_layer);
    let frag_eye_z = sds::fragment_linear_depth(world_pos, view_layer);
    let depth_diff = max(scene_eye_z - frag_eye_z, 0.0);

    let noise = textureSample(_NoiseTex, _NoiseTex_sampler, uv0 * 1.5 + vec2<f32>(mat._AnimationOffset)).r;
    var crest = max(pow(voronoi_f * 1.5, 10.0) * (mat._WaveHeight * 10.0) - noise * 20.0, 0.0);
    crest = crest + max((depth_diff * 1.5) * (mat._WaveHeight * 1000.0) - noise * (100.0 * mat._WaveHeight), 0.0);
    crest = min(step(0.9, crest), 1.0) * mat._WaveCrest;

    let visibility = max(pow(mat._Transmission, depth_diff) - (1.0 - mat._Transmission), 0.0);
    let final_water = mix(mat._Color.rgb, grab_color * mat._Color.rgb, visibility);

    let albedo = final_water + vec3<f32>(crest);
    let spec_s = textureSample(_SpecGlossMap, _SpecGlossMap_sampler, uv_main);
    let spec_color = spec_s.rgb * mat._SpecColor.rgb;
    let smoothness = clamp(spec_s.a * mat._Glossiness, 0.0, 1.0);

    let n = psamp::sample_world_normal(_BumpMap, _BumpMap_sampler, uv_main, 0.0, mat._BumpScale, world_n, world_t);
    let cam = rg::camera_world_pos_for_view(view_layer);
    let v = normalize(cam - world_pos);

    let cluster_id = pcls::cluster_id_from_frag(
        frag_pos.xy, world_pos, rg::frame.view_space_z_coeffs, rg::frame.view_space_z_coeffs_right,
        view_layer, rg::frame.viewport_width, rg::frame.viewport_height,
        rg::frame.cluster_count_x, rg::frame.cluster_count_y, rg::frame.cluster_count_z,
        rg::frame.near_clip, rg::frame.far_clip,
    );
    let count = pcls::cluster_light_count_at(cluster_id);
    let i_max = min(count, pcls::MAX_LIGHTS_PER_TILE);
    var lo = vec3<f32>(0.0);
    for (var i = 0u; i < i_max; i++) {
        let li = pcls::cluster_light_index_at(cluster_id, i);
        if (li >= rg::frame.light_count) {
            continue;
        }
        let light = rg::lights[li];
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
        lo = lo + radiance * (albedo * diff_step + spec_color * spec_step);
    }

    let emission = textureSample(_EmissionMap, _EmissionMap_sampler, uv_main).rgb * mat._EmissionColor.rgb;
    var color = lo + emission + toon_fresnel(albedo, v, n);

    if (mat._PlanarReflection > 0.5) {
        let refl = textureSample(_ReflectionTex, _ReflectionTex_sampler, refracted_uv).rgb;
        color = color + refl * (1.0 - smoothness);
    }

    return vec4<f32>(color, mat._Color.a);
}
