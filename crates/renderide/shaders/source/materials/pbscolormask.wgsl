//! Unity surface shader `Shader "ColorMask"` (asset: `PBSColorMask.shader`): metallic Standard
//! lighting with four base colors blended by an RGBA mask texture.
//!
//! Each mask channel selects one of `_Color`/`_Color1`/`_Color2`/`_Color3`; the result is
//! normalized by the channel sum and optionally multiplied by `_MainTex`. Per-channel emission
//! follows the same pattern. Mirrors the keyword surface (`_ALBEDOTEX`, `_EMISSIONTEX`,
//! `_NORMALMAP`, `_METALLICMAP`, `_OCCLUSION`, `_MULTI_VALUES`).

// unity-shader-name: PBSColorMask
// unity-shader-name: ColorMask

#import renderide::globals as rg
#import renderide::per_draw as pd
#import renderide::pbs::brdf as brdf
#import renderide::pbs::cluster as pcls
#import renderide::uv_utils as uvu
#import renderide::normal_decode as nd

/// Material uniforms for `PBSColorMask`. Field names match the Unity property block.
struct PbsColorMaskMaterial {
    /// Color slot 0, selected by `_ColorMask.r`.
    _Color: vec4<f32>,
    /// Color slot 1, selected by `_ColorMask.g`.
    _Color1: vec4<f32>,
    /// Color slot 2, selected by `_ColorMask.b`.
    _Color2: vec4<f32>,
    /// Color slot 3, selected by `_ColorMask.a`.
    _Color3: vec4<f32>,
    /// Emission slot 0.
    _EmissionColor: vec4<f32>,
    /// Emission slot 1.
    _EmissionColor1: vec4<f32>,
    /// Emission slot 2.
    _EmissionColor2: vec4<f32>,
    /// Emission slot 3.
    _EmissionColor3: vec4<f32>,
    /// Albedo `_ST` (xy = scale, zw = offset).
    _MainTex_ST: vec4<f32>,
    /// Tangent-space normal scale.
    _NormalScale: f32,
    /// Smoothness fallback when `_METALLICMAP` is disabled.
    _Glossiness: f32,
    /// Metallic fallback when `_METALLICMAP` is disabled.
    _Metallic: f32,
    /// Keyword: enable albedo texture multiply.
    _ALBEDOTEX: f32,
    /// Keyword: enable emission texture multiply.
    _EMISSIONTEX: f32,
    /// Keyword: enable normal map sampling.
    _NORMALMAP: f32,
    /// Keyword: read metallic + smoothness from `_MetallicMap` (R=metallic, A=smoothness).
    _METALLICMAP: f32,
    /// Keyword: read occlusion from `_OcclusionMap.r`.
    _OCCLUSION: f32,
    /// Keyword: when set with `_METALLICMAP`, multiply map values by `_Metallic`/`_Glossiness`.
    _MULTI_VALUES: f32,
}

@group(1) @binding(0)  var<uniform> mat: PbsColorMaskMaterial;
@group(1) @binding(1)  var _MainTex: texture_2d<f32>;
@group(1) @binding(2)  var _MainTex_sampler: sampler;
@group(1) @binding(3)  var _ColorMask: texture_2d<f32>;
@group(1) @binding(4)  var _ColorMask_sampler: sampler;
@group(1) @binding(5)  var _NormalMap: texture_2d<f32>;
@group(1) @binding(6)  var _NormalMap_sampler: sampler;
@group(1) @binding(7)  var _EmissionMap: texture_2d<f32>;
@group(1) @binding(8)  var _EmissionMap_sampler: sampler;
@group(1) @binding(9)  var _OcclusionMap: texture_2d<f32>;
@group(1) @binding(10) var _OcclusionMap_sampler: sampler;
@group(1) @binding(11) var _MetallicMap: texture_2d<f32>;
@group(1) @binding(12) var _MetallicMap_sampler: sampler;

/// Interpolated vertex output forwarded to both forward-base and forward-add fragments.
struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) @interpolate(flat) view_layer: u32,
}

/// Resolved per-fragment shading inputs for the metallic Cook–Torrance path.
struct SurfaceData {
    base_color: vec3<f32>,
    alpha: f32,
    metallic: f32,
    roughness: f32,
    occlusion: f32,
    normal: vec3<f32>,
    emission: vec3<f32>,
}

/// Sample the normal map (when enabled) and transform the tangent-space normal to world space.
fn sample_normal_world(uv_main: vec2<f32>, world_n: vec3<f32>) -> vec3<f32> {
    let tbn = brdf::orthonormal_tbn(normalize(world_n));
    var ts_n = vec3<f32>(0.0, 0.0, 1.0);
    if (uvu::kw_enabled(mat._NORMALMAP)) {
        ts_n = nd::decode_ts_normal_with_placeholder(
            textureSample(_NormalMap, _NormalMap_sampler, uv_main).xyz,
            mat._NormalScale,
        );
    }
    return normalize(tbn * ts_n);
}

/// Resolve the [`SurfaceData`] for a fragment, mirroring Unity's `surf` for `PBSColorMask`.
fn sample_surface(uv0: vec2<f32>, world_n: vec3<f32>) -> SurfaceData {
    let uv_main = uvu::apply_st(uv0, mat._MainTex_ST);

    let mask = textureSample(_ColorMask, _ColorMask_sampler, uv_main);
    let weight_inv = max(mask.r + mask.g + mask.b + mask.a, 1e-5);
    let weight = clamp(1.0 / weight_inv, 0.0, 1.0);

    var c =
        mat._Color * mask.r
        + mat._Color1 * mask.g
        + mat._Color2 * mask.b
        + mat._Color3 * mask.a;
    c = c * weight;
    if (uvu::kw_enabled(mat._ALBEDOTEX)) {
        c = c * textureSample(_MainTex, _MainTex_sampler, uv_main);
    }

    var metallic = mat._Metallic;
    var smoothness = mat._Glossiness;
    if (uvu::kw_enabled(mat._METALLICMAP)) {
        var m = textureSample(_MetallicMap, _MetallicMap_sampler, uv_main);
        if (uvu::kw_enabled(mat._MULTI_VALUES)) {
            m.r = m.r * mat._Metallic;
            m.a = m.a * mat._Glossiness;
        }
        metallic = m.r;
        smoothness = m.a;
    }
    metallic = clamp(metallic, 0.0, 1.0);
    let roughness = clamp(1.0 - smoothness, 0.045, 1.0);

    var occlusion = 1.0;
    if (uvu::kw_enabled(mat._OCCLUSION)) {
        occlusion = textureSample(_OcclusionMap, _OcclusionMap_sampler, uv_main).r;
    }

    var emission =
        mat._EmissionColor * mask.r
        + mat._EmissionColor1 * mask.g
        + mat._EmissionColor2 * mask.b
        + mat._EmissionColor3 * mask.a;
    emission = emission * weight;
    if (uvu::kw_enabled(mat._EMISSIONTEX)) {
        emission = emission * textureSample(_EmissionMap, _EmissionMap_sampler, uv_main);
    }

    return SurfaceData(
        c.rgb,
        c.a,
        metallic,
        roughness,
        occlusion,
        sample_normal_world(uv_main, world_n),
        emission.rgb,
    );
}

/// Iterate the cluster's lights and accumulate Cook–Torrance radiance, gated by directional/local.
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
            light,
            world_pos,
            s.normal,
            v,
            s.roughness,
            s.metallic,
            s.base_color,
            f0,
        );
    }
    return lo * s.occlusion;
}

/// Vertex stage: forward world position, world-space normal, and primary UV.
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

/// Forward-base pass: ambient + directional lighting + emission.
//#material forward
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
    let ambient = vec3<f32>(0.03) * s.base_color * s.occlusion;
    return vec4<f32>(ambient + direct + s.emission, s.alpha);
}

/// Forward-add pass: additive accumulation of local (point/spot) lights.
