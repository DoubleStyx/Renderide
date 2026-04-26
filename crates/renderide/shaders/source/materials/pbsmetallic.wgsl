//! Unity Standard metallic PBS (`Shader "PBSMetallic"`): forward base + forward additive.
//!
//! This mirrors the built-in Standard metallic forward passes within the renderer's forward path:
//! `FORWARD` writes ambient/emission plus directional lighting, and `FORWARD_DELTA` additively
//! accumulates local lights. Unity's ShadowCaster/Deferred/Meta passes are not declared here because
//! this render path has one forward color target, not shadow-map, G-buffer, or lightmapping targets.

// unity-shader-name: PBSMetallic

#import renderide::globals as rg
#import renderide::per_draw as pd
#import renderide::pbs::brdf as brdf
#import renderide::pbs::normal as pnorm
#import renderide::pbs::cluster as pcls
#import renderide::alpha_clip_sample as acs
#import renderide::uv_utils as uvu
#import renderide::normal_decode as nd
#import renderide::texture_sampling as ts

struct PbsMetallicMaterial {
    _Color: vec4<f32>,
    _EmissionColor: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _DetailAlbedoMap_ST: vec4<f32>,
    _Cutoff: f32,
    _Glossiness: f32,
    _SmoothnessTextureChannel: f32,
    _Metallic: f32,
    _BumpScale: f32,
    _Parallax: f32,
    _DetailNormalMapScale: f32,
    _NORMALMAP: f32,
    _ALPHATEST_ON: f32,
    _ALPHABLEND_ON: f32,
    _ALPHAPREMULTIPLY_ON: f32,
    _EMISSION: f32,
    _METALLICGLOSSMAP: f32,
    _DETAIL_MULX2: f32,
    _SMOOTHNESS_TEXTURE_ALBEDO_CHANNEL_A: f32,
    _SPECULARHIGHLIGHTS_OFF: f32,
    _GLOSSYREFLECTIONS_OFF: f32,
    _PARALLAXMAP: f32,
    _MainTex_LodBias: f32,
    _MetallicGlossMap_LodBias: f32,
    _BumpMap_LodBias: f32,
    _ParallaxMap_LodBias: f32,
    _OcclusionMap_LodBias: f32,
    _EmissionMap_LodBias: f32,
    _DetailMask_LodBias: f32,
    _DetailAlbedoMap_LodBias: f32,
    _DetailNormalMap_LodBias: f32,
}

@group(1) @binding(0)  var<uniform> mat: PbsMetallicMaterial;
@group(1) @binding(1)  var _MainTex: texture_2d<f32>;
@group(1) @binding(2)  var _MainTex_sampler: sampler;
@group(1) @binding(3)  var _MetallicGlossMap: texture_2d<f32>;
@group(1) @binding(4)  var _MetallicGlossMap_sampler: sampler;
@group(1) @binding(5)  var _BumpMap: texture_2d<f32>;
@group(1) @binding(6)  var _BumpMap_sampler: sampler;
@group(1) @binding(7)  var _ParallaxMap: texture_2d<f32>;
@group(1) @binding(8)  var _ParallaxMap_sampler: sampler;
@group(1) @binding(9)  var _OcclusionMap: texture_2d<f32>;
@group(1) @binding(10) var _OcclusionMap_sampler: sampler;
@group(1) @binding(11) var _EmissionMap: texture_2d<f32>;
@group(1) @binding(12) var _EmissionMap_sampler: sampler;
@group(1) @binding(13) var _DetailMask: texture_2d<f32>;
@group(1) @binding(14) var _DetailMask_sampler: sampler;
@group(1) @binding(15) var _DetailAlbedoMap: texture_2d<f32>;
@group(1) @binding(16) var _DetailAlbedoMap_sampler: sampler;
@group(1) @binding(17) var _DetailNormalMap: texture_2d<f32>;
@group(1) @binding(18) var _DetailNormalMap_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) uv1: vec2<f32>,
    @location(4) @interpolate(flat) view_layer: u32,
}

struct SurfaceData {
    base_color: vec3<f32>,
    alpha: f32,
    metallic: f32,
    roughness: f32,
    occlusion: f32,
    normal: vec3<f32>,
    emission: vec3<f32>,
}

fn kw(v: f32) -> bool {
    return v > 0.5;
}

fn alpha_test_enabled() -> bool {
    return kw(mat._ALPHATEST_ON);
}

fn alpha_premultiply_enabled() -> bool {
    return kw(mat._ALPHAPREMULTIPLY_ON);
}

fn specular_highlights_enabled() -> bool {
    return !kw(mat._SPECULARHIGHLIGHTS_OFF);
}

fn glossy_reflections_enabled() -> bool {
    return !kw(mat._GLOSSYREFLECTIONS_OFF);
}

fn metallic_gloss_map_enabled() -> bool {
    return kw(mat._METALLICGLOSSMAP);
}

fn smoothness_from_albedo_alpha() -> bool {
    return mat._SmoothnessTextureChannel > 0.5 || kw(mat._SMOOTHNESS_TEXTURE_ALBEDO_CHANNEL_A);
}

fn uv_with_parallax(uv: vec2<f32>, world_pos: vec3<f32>) -> vec2<f32> {
    if (!kw(mat._PARALLAXMAP)) {
        return uv;
    }
    let h = ts::sample_tex_2d(_ParallaxMap, _ParallaxMap_sampler, uv, mat._ParallaxMap_LodBias).r;
    let view_dir = normalize(rg::frame.camera_world_pos.xyz - world_pos);
    let view_xy = view_dir.xy / max(abs(view_dir.z), 0.25);
    return uv + (h - 0.5) * mat._Parallax * view_xy;
}

fn sample_normal_world(
    uv_main: vec2<f32>,
    uv_detail: vec2<f32>,
    world_n: vec3<f32>,
    detail_mask: f32,
) -> vec3<f32> {
    var n = normalize(world_n);
    if (!kw(mat._NORMALMAP)) {
        return n;
    }

    let tbn = pnorm::orthonormal_tbn(n);
    var ts_n = nd::decode_ts_normal_with_placeholder_sample(
        ts::sample_tex_2d(_BumpMap, _BumpMap_sampler, uv_main, mat._BumpMap_LodBias),
        mat._BumpScale,
    );

    if (kw(mat._DETAIL_MULX2) && detail_mask > 0.001) {
        let ts_detail = nd::decode_ts_normal_with_placeholder_sample(
            ts::sample_tex_2d(_DetailNormalMap, _DetailNormalMap_sampler, uv_detail, mat._DetailNormalMap_LodBias),
            mat._DetailNormalMapScale,
        );
        ts_n = normalize(vec3<f32>(ts_n.xy + ts_detail.xy * detail_mask, ts_n.z));
    }

    n = normalize(tbn * ts_n);
    return n;
}

fn sample_surface(uv0: vec2<f32>, uv1: vec2<f32>, world_pos: vec3<f32>, world_n: vec3<f32>) -> SurfaceData {
    let uv_base = uvu::apply_st(uv0, mat._MainTex_ST);
    let uv_main = uv_with_parallax(uv_base, world_pos);
    let uv_detail = uvu::apply_st(uv0, mat._DetailAlbedoMap_ST);

    let albedo_sample = ts::sample_tex_2d(_MainTex, _MainTex_sampler, uv_main, mat._MainTex_LodBias);
    let base_alpha = mat._Color.a * albedo_sample.a;
    let clip_alpha = mat._Color.a * acs::texture_alpha_base_mip(_MainTex, _MainTex_sampler, uv_main);
    if (alpha_test_enabled() && clip_alpha <= mat._Cutoff) {
        discard;
    }

    var base_color = mat._Color.rgb * albedo_sample.rgb;

    let mg = ts::sample_tex_2d(_MetallicGlossMap, _MetallicGlossMap_sampler, uv_main, mat._MetallicGlossMap_LodBias);
    var metallic = mat._Metallic;
    var smoothness = mat._Glossiness;
    if (metallic_gloss_map_enabled()) {
        metallic = mg.r;
        smoothness = mg.a;
    }
    if (smoothness_from_albedo_alpha()) {
        smoothness = albedo_sample.a;
    }
    metallic = clamp(metallic, 0.0, 1.0);
    let roughness = clamp(1.0 - smoothness, 0.045, 1.0);

    let occlusion = ts::sample_tex_2d(_OcclusionMap, _OcclusionMap_sampler, uv_main, mat._OcclusionMap_LodBias).r;

    var detail_mask = 0.0;
    if (kw(mat._DETAIL_MULX2)) {
        detail_mask = ts::sample_tex_2d(_DetailMask, _DetailMask_sampler, uv_main, mat._DetailMask_LodBias).a;
        let detail = ts::sample_tex_2d(_DetailAlbedoMap, _DetailAlbedoMap_sampler, uv_detail, mat._DetailAlbedoMap_LodBias).rgb;
        base_color = base_color * mix(vec3<f32>(1.0), detail * 2.0, detail_mask);
    }

    let n = sample_normal_world(uv_main, uv_detail, world_n, detail_mask);

    let emission_color = mat._EmissionColor.rgb;
    var emission = vec3<f32>(0.0);
    if (dot(emission_color, emission_color) > 1e-8) {
        emission = ts::sample_tex_2d(_EmissionMap, _EmissionMap_sampler, uv_main, mat._EmissionMap_LodBias).rgb * emission_color;
    }

    return SurfaceData(base_color, base_alpha, metallic, roughness, occlusion, n, emission);
}

fn clustered_direct_lighting(
    frag_xy: vec2<f32>,
    world_pos: vec3<f32>,
    view_layer: u32,
    s: SurfaceData,
    include_directional: bool,
    include_local: bool,
) -> vec3<f32> {
    let cam = rg::frame.camera_world_pos.xyz;
    let v = normalize(cam - world_pos);
    let f0 = mix(vec3<f32>(0.04), s.base_color, s.metallic);

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
        if (specular_highlights_enabled()) {
            lo = lo + brdf::direct_radiance_metallic(
                light,
                world_pos,
                s.normal,
                v,
                s.roughness,
                s.metallic,
                s.base_color,
                f0,
            );
        } else {
            lo = lo + brdf::diffuse_only_metallic(light, world_pos, s.normal, s.base_color, s.metallic);
        }
    }

    return lo;
}

fn apply_premultiply(color: vec3<f32>, alpha: f32) -> vec3<f32> {
    return select(color, color * alpha, alpha_premultiply_enabled());
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
    out.uv1 = uv0;
#ifdef MULTIVIEW
    out.view_layer = view_idx;
#else
    out.view_layer = 0u;
#endif
    return out;
}

//#pass forward
@fragment
fn fs_forward_base(
    @builtin(position) frag_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) uv1: vec2<f32>,
    @location(4) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    let s = sample_surface(uv0, uv1, world_pos, world_n);
    let ambient = select(vec3<f32>(0.0), vec3<f32>(0.03), glossy_reflections_enabled());
    let direct = clustered_direct_lighting(frag_pos.xy, world_pos, view_layer, s, true, true);
    let color = (ambient * s.base_color * s.occlusion) + direct + s.emission;
    return vec4<f32>(apply_premultiply(color, s.alpha), s.alpha);
}
