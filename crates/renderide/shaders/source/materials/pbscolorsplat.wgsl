//! Unity surface shader `Shader "PBSColorSplat"`: metallic Standard lighting with up to four
//! albedo/normal/metallic/emission layers blended by an RGBA splat-weight texture.
//!
//! `_ColorMap` chooses the per-layer weights (R=layer 0 .. A=layer 3); when `_HEIGHTMAP` is enabled
//! the weights are first multiplied by `_PackedHeightMap` and re-derived via a Voronoi-style
//! "max height" softmax (`_HeightTransitionRange` controls the band width). Per-layer normal maps
//! and metallic/gloss maps are packed two layers per texture (`_PackedNormalMap01`/`_PackedNormalMap23`,
//! `_MetallicGloss01`/`_MetallicGloss23`); per-layer emission can come from four separate maps
//! (`_EMISSIONTEX`) or one packed map (`_PACKED_EMISSIONTEX`).

// unity-shader-name: PBSColorSplat

#import renderide::globals as rg
#import renderide::per_draw as pd
#import renderide::pbs::brdf as brdf
#import renderide::pbs::normal as pnorm
#import renderide::pbs::cluster as pcls
#import renderide::uv_utils as uvu

struct PbsColorSplatMaterial {
    _Color: vec4<f32>,
    _Color1: vec4<f32>,
    _Color2: vec4<f32>,
    _Color3: vec4<f32>,
    _EmissionColor: vec4<f32>,
    _EmissionColor1: vec4<f32>,
    _EmissionColor2: vec4<f32>,
    _EmissionColor3: vec4<f32>,
    _Albedo_ST: vec4<f32>,
    _ColorMap_ST: vec4<f32>,
    _HeightTransitionRange: f32,
    _NormalScale: f32,
    _NormalScale1: f32,
    _NormalScale2: f32,
    _NormalScale3: f32,
    _Glossiness: f32,
    _Glossiness1: f32,
    _Glossiness2: f32,
    _Glossiness3: f32,
    _Metallic: f32,
    _Metallic1: f32,
    _Metallic2: f32,
    _Metallic3: f32,
    _HEIGHTMAP: f32,
    _PACKED_NORMALMAP: f32,
    _EMISSIONTEX: f32,
    _PACKED_EMISSIONTEX: f32,
    _METALLICMAP: f32,
}

@group(1) @binding(0)  var<uniform> mat: PbsColorSplatMaterial;
@group(1) @binding(1)  var _ColorMap: texture_2d<f32>;
@group(1) @binding(2)  var _ColorMap_sampler: sampler;
@group(1) @binding(3)  var _PackedHeightMap: texture_2d<f32>;
@group(1) @binding(4)  var _PackedHeightMap_sampler: sampler;
@group(1) @binding(5)  var _Albedo: texture_2d<f32>;
@group(1) @binding(6)  var _Albedo_sampler: sampler;
@group(1) @binding(7)  var _Albedo1: texture_2d<f32>;
@group(1) @binding(8)  var _Albedo1_sampler: sampler;
@group(1) @binding(9)  var _Albedo2: texture_2d<f32>;
@group(1) @binding(10) var _Albedo2_sampler: sampler;
@group(1) @binding(11) var _Albedo3: texture_2d<f32>;
@group(1) @binding(12) var _Albedo3_sampler: sampler;
@group(1) @binding(13) var _PackedNormalMap01: texture_2d<f32>;
@group(1) @binding(14) var _PackedNormalMap01_sampler: sampler;
@group(1) @binding(15) var _PackedNormalMap23: texture_2d<f32>;
@group(1) @binding(16) var _PackedNormalMap23_sampler: sampler;
@group(1) @binding(17) var _EmissionMap: texture_2d<f32>;
@group(1) @binding(18) var _EmissionMap_sampler: sampler;
@group(1) @binding(19) var _EmissionMap1: texture_2d<f32>;
@group(1) @binding(20) var _EmissionMap1_sampler: sampler;
@group(1) @binding(21) var _EmissionMap2: texture_2d<f32>;
@group(1) @binding(22) var _EmissionMap2_sampler: sampler;
@group(1) @binding(23) var _EmissionMap3: texture_2d<f32>;
@group(1) @binding(24) var _EmissionMap3_sampler: sampler;
@group(1) @binding(25) var _PackedEmissionMap: texture_2d<f32>;
@group(1) @binding(26) var _PackedEmissionMap_sampler: sampler;
@group(1) @binding(27) var _MetallicGloss01: texture_2d<f32>;
@group(1) @binding(28) var _MetallicGloss01_sampler: sampler;
@group(1) @binding(29) var _MetallicGloss23: texture_2d<f32>;
@group(1) @binding(30) var _MetallicGloss23_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) @interpolate(flat) view_layer: u32,
}

struct SurfaceData {
    base_color: vec3<f32>,
    alpha: f32,
    metallic: f32,
    roughness: f32,
    normal: vec3<f32>,
    emission: vec3<f32>,
}

fn splat_weights(uv_albedo: vec2<f32>, uv_color: vec2<f32>) -> vec4<f32> {
    var w = textureSample(_ColorMap, _ColorMap_sampler, uv_color);
    if (uvu::kw_enabled(mat._HEIGHTMAP)) {
        let heights = textureSample(_PackedHeightMap, _PackedHeightMap_sampler, uv_albedo) * w;
        let max_height = max(max(heights.x, heights.y), max(heights.z, heights.w));
        let band = max(mat._HeightTransitionRange, 1e-4);
        let shifted = (heights - vec4<f32>(max_height) + vec4<f32>(band)) / band;
        w = clamp(shifted, vec4<f32>(0.0), vec4<f32>(1.0));
    }
    let total = max(w.x + w.y + w.z + w.w, 1e-4);
    return w / total;
}

fn unpack_normal_xy(xy: vec2<f32>, scale: f32) -> vec3<f32> {
    let scaled = (xy * 2.0 - 1.0) * scale;
    let z = sqrt(max(1.0 - dot(scaled, scaled), 0.0));
    return vec3<f32>(scaled, z);
}

fn sample_normal_world(uv_albedo: vec2<f32>, world_n: vec3<f32>, weights: vec4<f32>) -> vec3<f32> {
    let n = normalize(world_n);
    if (!uvu::kw_enabled(mat._PACKED_NORMALMAP)) {
        return n;
    }
    let n01 = textureSample(_PackedNormalMap01, _PackedNormalMap01_sampler, uv_albedo);
    let n23 = textureSample(_PackedNormalMap23, _PackedNormalMap23_sampler, uv_albedo);
    let n0 = unpack_normal_xy(n01.xy, mat._NormalScale);
    let n1 = unpack_normal_xy(n01.zw, mat._NormalScale1);
    let n2 = unpack_normal_xy(n23.xy, mat._NormalScale2);
    let n3 = unpack_normal_xy(n23.zw, mat._NormalScale3);
    let blended = n0 * weights.x + n1 * weights.y + n2 * weights.z + n3 * weights.w;
    let tbn = pnorm::orthonormal_tbn(n);
    return normalize(tbn * normalize(blended));
}

fn sample_metallic_gloss(uv_albedo: vec2<f32>, weights: vec4<f32>) -> vec2<f32> {
    var m0 = vec2<f32>(mat._Metallic, mat._Glossiness);
    var m1 = vec2<f32>(mat._Metallic1, mat._Glossiness1);
    var m2 = vec2<f32>(mat._Metallic2, mat._Glossiness2);
    var m3 = vec2<f32>(mat._Metallic3, mat._Glossiness3);
    if (uvu::kw_enabled(mat._METALLICMAP)) {
        let s01 = textureSample(_MetallicGloss01, _MetallicGloss01_sampler, uv_albedo);
        let s23 = textureSample(_MetallicGloss23, _MetallicGloss23_sampler, uv_albedo);
        m0 = s01.xy * m0;
        m1 = s01.zw * m1;
        m2 = s23.xy * m2;
        m3 = s23.zw * m3;
    }
    return m0 * weights.x + m1 * weights.y + m2 * weights.z + m3 * weights.w;
}

fn sample_emission(uv_albedo: vec2<f32>, weights: vec4<f32>) -> vec3<f32> {
    var e0 = mat._EmissionColor;
    var e1 = mat._EmissionColor1;
    var e2 = mat._EmissionColor2;
    var e3 = mat._EmissionColor3;
    if (uvu::kw_enabled(mat._EMISSIONTEX)) {
        e0 = e0 * textureSample(_EmissionMap, _EmissionMap_sampler, uv_albedo);
        e1 = e1 * textureSample(_EmissionMap1, _EmissionMap1_sampler, uv_albedo);
        e2 = e2 * textureSample(_EmissionMap2, _EmissionMap2_sampler, uv_albedo);
        e3 = e3 * textureSample(_EmissionMap3, _EmissionMap3_sampler, uv_albedo);
    } else if (uvu::kw_enabled(mat._PACKED_EMISSIONTEX)) {
        let packed = textureSample(_PackedEmissionMap, _PackedEmissionMap_sampler, uv_albedo);
        e0 = e0 * packed.x;
        e1 = e1 * packed.y;
        e2 = e2 * packed.z;
        e3 = e3 * packed.w;
    }
    let blended = e0 * weights.x + e1 * weights.y + e2 * weights.z + e3 * weights.w;
    return blended.rgb;
}

fn sample_surface(uv0: vec2<f32>, world_n: vec3<f32>) -> SurfaceData {
    let uv_albedo = uvu::apply_st(uv0, mat._Albedo_ST);
    let uv_color = uvu::apply_st(uv0, mat._ColorMap_ST);

    let weights = splat_weights(uv_albedo, uv_color);

    let c0 = textureSample(_Albedo, _Albedo_sampler, uv_albedo) * mat._Color;
    let c1 = textureSample(_Albedo1, _Albedo1_sampler, uv_albedo) * mat._Color1;
    let c2 = textureSample(_Albedo2, _Albedo2_sampler, uv_albedo) * mat._Color2;
    let c3 = textureSample(_Albedo3, _Albedo3_sampler, uv_albedo) * mat._Color3;
    let c = c0 * weights.x + c1 * weights.y + c2 * weights.z + c3 * weights.w;

    let mg = sample_metallic_gloss(uv_albedo, weights);
    let metallic = clamp(mg.x, 0.0, 1.0);
    let smoothness = clamp(mg.y, 0.0, 1.0);
    let roughness = clamp(1.0 - smoothness, 0.045, 1.0);

    return SurfaceData(
        c.rgb,
        c.a,
        metallic,
        roughness,
        sample_normal_world(uv_albedo, world_n, weights),
        sample_emission(uv_albedo, weights),
    );
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
        lo = lo + brdf::direct_radiance_metallic(
            light, world_pos, s.normal, v, s.roughness, s.metallic, s.base_color, f0,
        );
    }
    return lo;
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

//#pass forward
@fragment
fn fs_forward_base(
    @builtin(position) frag_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    let s = sample_surface(uv0, world_n);
    let direct = clustered_direct_lighting(frag_pos.xy, world_pos, view_layer, s, true, true);
    let ambient = vec3<f32>(0.03) * s.base_color;
    return vec4<f32>(ambient + direct + s.emission, s.alpha);
}
