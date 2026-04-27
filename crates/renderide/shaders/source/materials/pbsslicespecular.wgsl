//! Unity surface shader `Shader "PBSSliceSpecular"`: SpecularSetup lighting with plane-based slicing.
//!
//! Sibling of [`pbsslice`](super::pbsslice); same `_Slicers[8]` plane evaluation and edge blending,
//! but reads tinted f0 + smoothness from `_SpecularColor` / `_SpecularMap` instead of
//! `_Metallic` / `_MetallicMap`.

// unity-shader-name: PBSSliceSpecular

#import renderide::globals as rg
#import renderide::sh2_ambient as shamb
#import renderide::per_draw as pd
#import renderide::pbs::brdf as brdf
#import renderide::pbs::normal as pnorm
#import renderide::pbs::cluster as pcls
#import renderide::uv_utils as uvu
#import renderide::normal_decode as nd

struct PBSSliceSpecularMaterial {
    _Color: vec4<f32>,
    _SpecularColor: vec4<f32>,
    _EdgeColor: vec4<f32>,
    _EmissionColor: vec4<f32>,
    _EdgeEmissionColor: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _MainTex_StorageVInverted: f32,
    _DetailAlbedoMap_ST: vec4<f32>,
    _DetailNormalMap_ST: vec4<f32>,
    _EdgeTransitionStart: f32,
    _EdgeTransitionEnd: f32,
    _NormalScale: f32,
    _DetailNormalMapScale: f32,
    _AlphaClip: f32,
    _WORLD_SPACE: f32,
    _OBJECT_SPACE: f32,
    _ALPHACLIP: f32,
    _ALBEDOTEX: f32,
    _DETAIL_ALBEDOTEX: f32,
    _NORMALMAP: f32,
    _DETAIL_NORMALMAP: f32,
    _EMISSIONTEX: f32,
    _SPECULARMAP: f32,
    _OCCLUSION: f32,
    _pad0: f32,
    _Slicers: array<vec4<f32>, 8>,
}

@group(1) @binding(0)  var<uniform> mat: PBSSliceSpecularMaterial;
@group(1) @binding(1)  var _MainTex: texture_2d<f32>;
@group(1) @binding(2)  var _MainTex_sampler: sampler;
@group(1) @binding(3)  var _NormalMap: texture_2d<f32>;
@group(1) @binding(4)  var _NormalMap_sampler: sampler;
@group(1) @binding(5)  var _EmissionMap: texture_2d<f32>;
@group(1) @binding(6)  var _EmissionMap_sampler: sampler;
@group(1) @binding(7)  var _OcclusionMap: texture_2d<f32>;
@group(1) @binding(8)  var _OcclusionMap_sampler: sampler;
@group(1) @binding(9)  var _SpecularMap: texture_2d<f32>;
@group(1) @binding(10) var _SpecularMap_sampler: sampler;
@group(1) @binding(11) var _DetailAlbedoMap: texture_2d<f32>;
@group(1) @binding(12) var _DetailAlbedoMap_sampler: sampler;
@group(1) @binding(13) var _DetailNormalMap: texture_2d<f32>;
@group(1) @binding(14) var _DetailNormalMap_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) object_pos: vec3<f32>,
    @location(2) world_n: vec3<f32>,
    @location(3) uv0: vec2<f32>,
    @location(4) @interpolate(flat) view_layer: u32,
}

fn plane_distance(p: vec3<f32>, normal: vec3<f32>, offset: f32) -> f32 {
    return dot(p, normal) + offset;
}

fn safe_lerp_factor(a: f32, b: f32, value: f32) -> f32 {
    let denom = b - a;
    if (abs(denom) < 1e-6) {
        return select(0.0, 1.0, value <= a);
    }
    return clamp((value - a) / denom, 0.0, 1.0);
}

fn slice_position(world_pos: vec3<f32>, object_pos: vec3<f32>) -> vec3<f32> {
    let use_world = uvu::kw_enabled(mat._WORLD_SPACE) || (!uvu::kw_enabled(mat._OBJECT_SPACE));
    return select(object_pos, world_pos, use_world);
}

fn blend_detail_normal(base_ts: vec3<f32>, detail_ts: vec3<f32>) -> vec3<f32> {
    return normalize(vec3<f32>(base_ts.xy + detail_ts.xy, base_ts.z * detail_ts.z));
}

fn sample_albedo_color(uv_main: vec2<f32>, edge_lerp: f32) -> vec4<f32> {
    let tint = mix(mat._Color, mat._EdgeColor, edge_lerp);
    if (uvu::kw_enabled(mat._ALBEDOTEX)) {
        return textureSample(_MainTex, _MainTex_sampler, uv_main) * tint;
    }
    return tint;
}

fn sample_normal_world(
    uv_main: vec2<f32>,
    uv_detail: vec2<f32>,
    world_n: vec3<f32>,
    front_facing: bool,
) -> vec3<f32> {
    var n = normalize(world_n);
    let use_normal_map = uvu::kw_enabled(mat._NORMALMAP) || uvu::kw_enabled(mat._DETAIL_NORMALMAP);
    if (use_normal_map) {
        let tbn = pnorm::orthonormal_tbn(n);
        var ts = nd::decode_ts_normal_with_placeholder(
            textureSample(_NormalMap, _NormalMap_sampler, uv_main).xyz,
            mat._NormalScale,
        );
        if (uvu::kw_enabled(mat._DETAIL_NORMALMAP)) {
            let detail = nd::decode_ts_normal_with_placeholder(
                textureSample(_DetailNormalMap, _DetailNormalMap_sampler, uv_detail).xyz,
                mat._DetailNormalMapScale,
            );
            ts = blend_detail_normal(ts, detail);
        }
        if (!front_facing) {
            ts = vec3<f32>(ts.x, ts.y, -ts.z);
        }
        return normalize(tbn * ts);
    }
    if (!front_facing) {
        n = -n;
    }
    return n;
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
    out.object_pos = pos.xyz;
    out.world_n = wn;
    out.uv0 = uv0;
#ifdef MULTIVIEW
    out.view_layer = view_idx;
#else
    out.view_layer = 0u;
#endif
    return out;
}

//#pass forward
@fragment
fn fs_main(
    @builtin(position) frag_pos: vec4<f32>,
    @builtin(front_facing) front_facing: bool,
    @location(0) world_pos: vec3<f32>,
    @location(1) object_pos: vec3<f32>,
    @location(2) world_n: vec3<f32>,
    @location(3) uv0: vec2<f32>,
    @location(4) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    let uv_main = uvu::apply_st_for_storage(uv0, mat._MainTex_ST, mat._MainTex_StorageVInverted);
    let uv_detail_albedo = uvu::apply_st(uv0, mat._DetailAlbedoMap_ST);
    let uv_detail_normal = uvu::apply_st(uv0, mat._DetailNormalMap_ST);

    let slice_p = slice_position(world_pos, object_pos);
    var min_distance: f32 = 60000.0;
    for (var si: i32 = 0; si < 8; si = si + 1) {
        let slicer = mat._Slicers[si];
        if (all(slicer.xyz == vec3<f32>(0.0))) {
            break;
        }
        min_distance = min(min_distance, plane_distance(slice_p, slicer.xyz, slicer.w));
    }
    if (min_distance < 0.0) {
        discard;
    }
    let edge_lerp = 1.0 - safe_lerp_factor(mat._EdgeTransitionStart, mat._EdgeTransitionEnd, min_distance);

    var c = sample_albedo_color(uv_main, edge_lerp);
    if (uvu::kw_enabled(mat._DETAIL_ALBEDOTEX)) {
        let detail = textureSample(_DetailAlbedoMap, _DetailAlbedoMap_sampler, uv_detail_albedo).rgb * 2.0;
        c = vec4<f32>(c.rgb * detail, c.a);
    }

    if (uvu::kw_enabled(mat._ALPHACLIP) && c.a <= mat._AlphaClip) {
        discard;
    }

    let base_color = c.rgb;
    let alpha = c.a;
    let n = sample_normal_world(uv_main, uv_detail_normal, world_n, front_facing);

    var occlusion: f32 = 1.0;
    if (uvu::kw_enabled(mat._OCCLUSION)) {
        occlusion = textureSample(_OcclusionMap, _OcclusionMap_sampler, uv_main).r;
    }

    var spec = mat._SpecularColor;
    if (uvu::kw_enabled(mat._SPECULARMAP)) {
        spec = textureSample(_SpecularMap, _SpecularMap_sampler, uv_main);
    }
    let f0 = clamp(spec.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    let smoothness = clamp(spec.a, 0.0, 1.0);
    let roughness = clamp(1.0 - smoothness, 0.045, 1.0);
    let one_minus_reflectivity = 1.0 - max(max(f0.r, f0.g), f0.b);

    var emission = mat._EmissionColor.rgb;
    if (uvu::kw_enabled(mat._EMISSIONTEX)) {
        emission = emission * textureSample(_EmissionMap, _EmissionMap_sampler, uv_main).rgb;
    }
    let edge_emission = mix(emission, mat._EdgeEmissionColor.rgb, edge_lerp);

    let cam = rg::frame.camera_world_pos.xyz;
    let v = normalize(cam - world_pos);

    let aa_roughness = brdf::filter_perceptual_roughness(roughness, n);

    let cluster_id = pcls::cluster_id_from_frag(
        frag_pos.xy,
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
    for (var i: u32 = 0u; i < i_max; i = i + 1u) {
        let li = pcls::cluster_light_index_at(cluster_id, i);
        if (li >= rg::frame.light_count) {
            continue;
        }
        let light = rg::lights[li];
        lo = lo + brdf::direct_radiance_specular(
            light,
            world_pos,
            n,
            v,
            aa_roughness,
            base_color,
            f0,
            one_minus_reflectivity,
        );
    }

    let amb = shamb::ambient_probe(n);
    let color = amb * base_color * occlusion + lo + edge_emission;
    return vec4<f32>(color, alpha);
}
