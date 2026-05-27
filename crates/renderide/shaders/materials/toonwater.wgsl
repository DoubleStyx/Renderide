//! Stylized toon water (`Shader "ToonWater"`) with grab-pass refraction, voronoi-driven wave
//! displacement, and depth-based visibility/crests.
//!
//! Implemented as a self-contained clustered-toon material (see `toonstandard.wgsl`). Notable
//! compatibility behavior:
//! - `_Time.x`, `_Time.y`, and `_SinTime.w` are derived from the renderer frame time.
//! - Scene depth sampled via [`renderide::frame::scene_depth_sample`]; reconstructed world height
//!   replaces the Unity `_CameraDepthTexture` + `_InverseView` unprojection.
//! - Refracted scene color sampled via [`renderide::frame::grab_pass`].
//! - Planar-reflection radiance (`_ReflectionTex`) is gated by `_PlanarReflection` and routed
//!   through the toon indirect-specular blend.


//#texture_default _MainTex white
//#texture_default _SpecGlossMap white
//#texture_default _BumpMap bump
//#texture_default _EmissionMap white
//#texture_default _VoronoiTex empty
//#texture_default _NoiseTex empty
//#texture_default _ReflectionTex black
//#mat_default _SpecularHighlights float 1.0
//#mat_default _SmoothnessTextureChannel float 0.0
//#mat_default _BumpScale float 1.0
//#mat_default _Color vec4 1.0 1.0 1.0 1.0
//#mat_default _Fresnel float 1.0
//#mat_default _FresnelTint vec4 1.0 1.0 1.0 1.0
//#mat_default _PlanarReflection float 1.0
//#mat_default _Glossiness float 0.5
//#mat_default _FresnelDiffCont float 0.5
//#mat_default _FresnelPower float 0.5
//#mat_default _FresnelStrength float 0.2
//#mat_default _WaveCrest float 0.5
//#mat_default _WaveScale float 0.5

#import renderide::lighting::birp as bl
#import renderide::frame::globals as rg
#import renderide::draw::per_draw as pd
#import renderide::mesh::vertex as mv
#import renderide::pbs::brdf as brdf
#import renderide::pbs::cluster as pcls
#import renderide::pbs::sampling as psamp
#import renderide::frame::grab_pass as gp
#import renderide::frame::scene_depth_sample as sds
#import renderide::material::toon_brdf as tbrdf
#import renderide::core::uv as uvu
#import renderide::material::voronoi as vor
#import renderide::core::texture_sampling as ts
#import renderide::lighting::reflection_probes as rprobe
#import renderide::lighting::light_cookies as cookies

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
    _MainTex_LodBias: f32,
    _SpecGlossMap_LodBias: f32,
    _BumpMap_LodBias: f32,
    _EmissionMap_LodBias: f32,
    _VoronoiTex_LodBias: f32,
    _NoiseTex_LodBias: f32,
    _ReflectionTex_LodBias: f32,
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
    @location(5) object_y: f32,
    @location(6) @interpolate(flat) view_layer: u32,
}

fn unity_time_x() -> f32 {
    return rg::frame.frame_time.x * 0.05;
}

fn unity_time_y() -> f32 {
    return rg::frame.frame_time.x;
}

fn unity_sin_time_w() -> f32 {
    return sin(rg::frame.frame_time.x);
}

fn voronoi_sample_at(uv: vec2<f32>) -> f32 {
    if (mat._SeparateVoronoi > 0.5) {
        let scale = max(10.0 - 10.0 * mat._WaveScale, 1e-4);
        return vor::voronoi_min_dist(uv * scale, unity_time_y());
    }
    return textureSampleLevel(_VoronoiTex, _VoronoiTex_sampler, uv, 0.0).r;
}

fn voronoi_sample_at_fragment(uv: vec2<f32>) -> f32 {
    if (mat._SeparateVoronoi > 0.5) {
        let scale = max(10.0 - 10.0 * mat._WaveScale, 1e-4);
        return vor::voronoi_min_dist(uv * scale, unity_time_y());
    }
    return ts::sample_tex_2d(_VoronoiTex, _VoronoiTex_sampler, uv, mat._VoronoiTex_LodBias).r;
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
    let layer = mv::packed_view_layer(instance_index, view_idx);
#else
    let vp = mv::select_view_proj(d, 0u);
    let layer = mv::packed_view_layer(instance_index, 0u);
#endif

    var out: VertexOutput;
    out.clip_pos = vp * world_p;
    out.world_pos = world_p.xyz;
    out.world_n = wn;
    out.world_t = wt;
    out.uv0 = uv0;
    out.uv1 = uv1;
    out.object_y = displaced_pos.y;
    out.view_layer = layer;
    return out;
}

fn refract_screen_uv(uv_in: vec2<f32>) -> vec2<f32> {
    var uv = uv_in;
    let phase = (1.0 - uv.x) * 0.25 * mat._WaveHeight * unity_sin_time_w() * sin(uv.y * 50.0);
    uv.x = uv.x + 0.1 * sin(phase);
    return uv;
}

//#pass type=forward zwrite=off
@fragment
fn fs_main(
    @builtin(position) frag_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) world_t: vec4<f32>,
    @location(3) uv0: vec2<f32>,
    @location(4) uv1: vec2<f32>,
    @location(5) object_y: f32,
    @location(6) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    let uv_main = uvu::apply_st(uv0, mat._MainTex_ST);

    let voronoi_f = voronoi_sample_at_fragment(uv0);

    let screen_uv = gp::frag_screen_uv(frag_pos);
    let refracted_uv = refract_screen_uv(screen_uv);
    let grab_color = gp::sample_scene_color(refracted_uv, view_layer).rgb;

    let refracted_world_y = sds::scene_world_y_at_uv(refracted_uv, view_layer) + object_y;

    let noise = ts::sample_tex_2d(_NoiseTex, _NoiseTex_sampler, uv0 * 1.5 + vec2<f32>(unity_time_x()), mat._NoiseTex_LodBias).r;
    var crest = max(pow(voronoi_f * 1.5, 10.0) * (mat._WaveHeight * 10.0) - noise * 20.0, 0.0);
    crest = crest + max((refracted_world_y * 1.5) * (mat._WaveHeight * 1000.0) - noise * (100.0 * mat._WaveHeight), 0.0);
    crest = min(step(0.9, crest), 1.0) * mat._WaveCrest;

    let visibility = max(pow(mat._Transmission, max(-refracted_world_y, 0.0)) - (1.0 - mat._Transmission), 0.0);
    let final_water = mix(mat._Color.rgb, grab_color * mat._Color.rgb, visibility);

    let albedo = final_water + vec3<f32>(crest);
    let spec_s = ts::sample_tex_2d(_SpecGlossMap, _SpecGlossMap_sampler, uv_main, mat._SpecGlossMap_LodBias);
    let spec_color = spec_s.rgb * mat._SpecColor.rgb;
    let one_minus_reflectivity = tbrdf::one_minus_reflectivity(spec_color);
    let diff_color = tbrdf::energy_conserved_diffuse(albedo, spec_color);
    let smoothness = clamp(spec_s.a * mat._Glossiness, 0.0, 1.0);

    let n = psamp::sample_world_normal(_BumpMap, _BumpMap_sampler, uv_main, mat._BumpMap_LodBias, mat._BumpScale, world_n, world_t);
    let v = rg::view_dir_for_world_pos(world_pos, view_layer);

    let cluster_id = pcls::cluster_id_from_frag(
        frag_pos.xy, world_pos, rg::frame.view_space_z_coeffs, rg::frame.view_space_z_coeffs_right,
        view_layer, rg::frame.viewport_width, rg::frame.viewport_height,
        rg::frame.cluster_count_x, rg::frame.cluster_count_y, rg::frame.cluster_count_z,
        rg::frame.near_clip, rg::frame.far_clip,
    );
    let count = pcls::cluster_light_count_at(cluster_id);
    let i_max = count;
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
            attenuation = bl::direct_light_scale();
        } else {
            let to_light = light.position - world_pos;
            let dist = length(to_light);
            l = normalize(to_light);
            attenuation = brdf::distance_attenuation(dist, light.range);
            if (light.light_type == 2u) {
                attenuation = attenuation * bl::spot_angle_attenuation(light, l);
            }
            attenuation = attenuation * cookies::multiplier(light, world_pos);
        }
        let radiance = bl::light_radiance(light) * attenuation;
        lo = lo + tbrdf::direct_light(
            diff_color,
            spec_color,
            n,
            l,
            v,
            smoothness,
            mat._Transmission,
            mat._SpecularHighlights,
            radiance,
        );
    }

    let emission = ts::sample_tex_2d(_EmissionMap, _EmissionMap_sampler, uv_main, mat._EmissionMap_LodBias).rgb * mat._EmissionColor.rgb;
    let ambient = rprobe::indirect_diffuse(world_pos, n, view_layer, true);
    let planar_specular = select(
        vec3<f32>(0.0),
        ts::sample_tex_2d(_ReflectionTex, _ReflectionTex_sampler, screen_uv, mat._ReflectionTex_LodBias).rgb,
        mat._PlanarReflection > 0.5,
    );
    let indirect = tbrdf::indirect_light(
        diff_color,
        spec_color,
        one_minus_reflectivity,
        smoothness,
        n,
        v,
        ambient,
        planar_specular,
    );

    let color = lo + indirect + emission + tbrdf::fresnel(
        diff_color,
        v,
        n,
        mat._Fresnel,
        mat._FresnelDiffCont,
        mat._FresnelPower,
        mat._FresnelStrength,
        mat._FresnelTint.rgb,
    );

    return vec4<f32>(color, mat._Color.a);
}
