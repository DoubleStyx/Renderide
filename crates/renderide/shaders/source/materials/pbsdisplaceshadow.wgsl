//! Unity surface shader `Shader "PBSDisplaceShadow"`: metallic Standard lighting with optional
//! vertex-stage displacement modes:
//!
//! * `VERTEX_OFFSET`: scalar displacement along vertex normal (`_VertexOffsetMap.r`).
//! * `UV_OFFSET`: shifts UV per-vertex by `_UVOffsetMap.rg` × magnitude (vertex-stage UV warp).
//! * `OBJECT_POS_OFFSET` / `VERTEX_POS_OFFSET`: 3-axis position offset from `_PositionOffsetMap`
//!   in object or world space (`_PositionOffsetMagnitude.xyz`).
//!
//! All three sample textures from `vs_main` via `textureSampleLevel(..., 0.0)` — this is the
//! first WGSL material in the renderer to exercise vertex-stage texture fetch.

// unity-shader-name: PBSDisplaceShadow

#import renderide::globals as rg
#import renderide::sh2_ambient as shamb
#import renderide::per_draw as pd
#import renderide::pbs::brdf as brdf
#import renderide::pbs::displace as pdisp
#import renderide::pbs::normal as pnorm
#import renderide::pbs::cluster as pcls
#import renderide::alpha_clip_sample as acs
#import renderide::uv_utils as uvu
#import renderide::normal_decode as nd

struct PbsDisplaceShadowMaterial {
    _Color: vec4<f32>,
    _EmissionColor: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _MainTex_StorageVInverted: f32,
    _VertexOffsetMap_ST: vec4<f32>,
    _PositionOffsetMagnitude: vec4<f32>,
    _NormalScale: f32,
    _Glossiness: f32,
    _Metallic: f32,
    _AlphaClip: f32,
    _VertexOffsetMagnitude: f32,
    _VertexOffsetBias: f32,
    _UVOffsetMagnitude: f32,
    _UVOffsetBias: f32,
    _ALPHACLIP: f32,
    _ALBEDOTEX: f32,
    _EMISSIONTEX: f32,
    _NORMALMAP: f32,
    _METALLICMAP: f32,
    _OCCLUSION: f32,
    VERTEX_OFFSET: f32,
    UV_OFFSET: f32,
    OBJECT_POS_OFFSET: f32,
    VERTEX_POS_OFFSET: f32,
}

@group(1) @binding(0)  var<uniform> mat: PbsDisplaceShadowMaterial;
@group(1) @binding(1)  var _MainTex: texture_2d<f32>;
@group(1) @binding(2)  var _MainTex_sampler: sampler;
@group(1) @binding(3)  var _NormalMap: texture_2d<f32>;
@group(1) @binding(4)  var _NormalMap_sampler: sampler;
@group(1) @binding(5)  var _EmissionMap: texture_2d<f32>;
@group(1) @binding(6)  var _EmissionMap_sampler: sampler;
@group(1) @binding(7)  var _OcclusionMap: texture_2d<f32>;
@group(1) @binding(8)  var _OcclusionMap_sampler: sampler;
@group(1) @binding(9)  var _MetallicMap: texture_2d<f32>;
@group(1) @binding(10) var _MetallicMap_sampler: sampler;
@group(1) @binding(11) var _VertexOffsetMap: texture_2d<f32>;
@group(1) @binding(12) var _VertexOffsetMap_sampler: sampler;
@group(1) @binding(13) var _UVOffsetMap: texture_2d<f32>;
@group(1) @binding(14) var _UVOffsetMap_sampler: sampler;
@group(1) @binding(15) var _PositionOffsetMap: texture_2d<f32>;
@group(1) @binding(16) var _PositionOffsetMap_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) @interpolate(flat) view_layer: u32,
}

fn sample_normal_world(uv_main: vec2<f32>, world_n: vec3<f32>) -> vec3<f32> {
    let n = normalize(world_n);
    if (!uvu::kw_enabled(mat._NORMALMAP)) {
        return n;
    }
    let tbn = pnorm::orthonormal_tbn(n);
    let ts_n = nd::decode_ts_normal_with_placeholder(
        textureSample(_NormalMap, _NormalMap_sampler, uv_main).xyz,
        mat._NormalScale,
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
    let displaced_uv = pdisp::apply_vertex_offsets(
        pos.xyz,
        n.xyz,
        uv0,
        uvu::kw_enabled(mat.VERTEX_OFFSET),
        uvu::kw_enabled(mat.UV_OFFSET),
        uvu::kw_enabled(mat.OBJECT_POS_OFFSET),
        uvu::kw_enabled(mat.VERTEX_POS_OFFSET),
        mat._VertexOffsetMap_ST,
        mat._PositionOffsetMagnitude.xyz,
        mat._VertexOffsetMagnitude,
        mat._VertexOffsetBias,
        mat._UVOffsetMagnitude,
        mat._UVOffsetBias,
        _VertexOffsetMap,
        _VertexOffsetMap_sampler,
        _UVOffsetMap,
        _UVOffsetMap_sampler,
        _PositionOffsetMap,
        _PositionOffsetMap_sampler,
    );
    let displaced = displaced_uv.position;
    let uv = displaced_uv.uv;

    let world_p = d.model * vec4<f32>(displaced, 1.0);
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
    out.uv0 = uv;
#ifdef MULTIVIEW
    out.view_layer = view_idx;
#else
    out.view_layer = 0u;
#endif
    return out;
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
    let uv_main = uvu::apply_st_for_storage(uv0, mat._MainTex_ST, mat._MainTex_StorageVInverted);

    var c = mat._Color;
    if (uvu::kw_enabled(mat._ALBEDOTEX)) {
        c = c * textureSample(_MainTex, _MainTex_sampler, uv_main);
    }
    let clip_alpha = mat._Color.a * acs::texture_alpha_base_mip(_MainTex, _MainTex_sampler, uv_main);
    if (uvu::kw_enabled(mat._ALPHACLIP) && clip_alpha <= mat._AlphaClip) {
        discard;
    }

    var metallic = mat._Metallic;
    var smoothness = mat._Glossiness;
    if (uvu::kw_enabled(mat._METALLICMAP)) {
        let m = textureSample(_MetallicMap, _MetallicMap_sampler, uv_main);
        metallic = m.r;
        smoothness = m.a;
    }
    metallic = clamp(metallic, 0.0, 1.0);
    let roughness = brdf::perceptual_roughness_from_smoothness(smoothness);

    var occlusion = 1.0;
    if (uvu::kw_enabled(mat._OCCLUSION)) {
        occlusion = textureSample(_OcclusionMap, _OcclusionMap_sampler, uv_main).r;
    }

    var emission = mat._EmissionColor.rgb;
    if (uvu::kw_enabled(mat._EMISSIONTEX)) {
        emission = emission * textureSample(_EmissionMap, _EmissionMap_sampler, uv_main).rgb;
    }

    let n = sample_normal_world(uv_main, world_n);
    let base_color = c.rgb;
    let f0 = brdf::metallic_f0(base_color, metallic);
    let cam = rg::frame.camera_world_pos.xyz;
    let v = normalize(cam - world_pos);

    let aa_roughness = brdf::filter_perceptual_roughness(roughness, n);

    let cluster_id = pcls::cluster_id_from_frag(
        frag_xy, world_pos, rg::frame.view_space_z_coeffs, rg::frame.view_space_z_coeffs_right,
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
        let is_directional = light.light_type == 1u;
        if ((is_directional && !include_directional) || (!is_directional && !include_local)) {
            continue;
        }
        lo = lo + brdf::direct_radiance_metallic(
            light, world_pos, n, v, aa_roughness, metallic, base_color, f0,
        );
    }
    let ambient_probe = select(vec3<f32>(0.0), shamb::ambient_probe(n), include_directional);
    let ambient = brdf::indirect_diffuse_metallic(
        ambient_probe,
        base_color,
        metallic,
        occlusion,
    );
    let indirect_specular = select(
        vec3<f32>(0.0),
        brdf::indirect_specular(n, v, aa_roughness, f0, occlusion, true),
        include_directional,
    );
    let extra = select(vec3<f32>(0.0), emission, include_directional);
    return vec4<f32>(ambient + indirect_specular + lo + extra, c.a);
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
