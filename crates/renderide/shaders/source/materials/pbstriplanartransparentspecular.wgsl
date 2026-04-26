//! Unity surface shader `Shader "PBSTriplanarTransparentSpecular"`: Standard SpecularSetup with triplanar
//! projection sampled from world or object space.
//!
//! Mirrors [`pbstriplanar`](super::pbstriplanar) for the SpecularSetup workflow: per-axis sampling
//! of `_MainTex`, `_SpecularMap`, `_EmissionMap`, `_NormalMap`, `_OcclusionMap` blended by
//! `pow(abs(world_normal), _TriBlendPower)` with Reoriented Normal Mapping per plane.

// unity-shader-name: PBSTriplanarTransparentSpecular

#import renderide::globals as rg
#import renderide::per_draw as pd
#import renderide::pbs::brdf as brdf
#import renderide::pbs::cluster as pcls
#import renderide::uv_utils as uvu
#import renderide::normal_decode as nd

/// Material uniforms for `PBSTriplanarSpecular`.
struct PbsTriplanarTransparentSpecularMaterial {
    /// Tint color (`Color`).
    _Color: vec4<f32>,
    /// Emission color (`EmissionColor`).
    _EmissionColor: vec4<f32>,
    /// Tinted specular color when `_SPECULARMAP` is disabled (RGB = f0, A = smoothness).
    _SpecularColor: vec4<f32>,
    /// Albedo `_ST` applied to all three projected planes.
    _MainTex_ST: vec4<f32>,
    /// Tangent-space normal scale.
    _NormalScale: f32,
    /// Triplanar blend exponent — higher values produce sharper transitions between planes.
    _TriBlendPower: f32,
    /// Keyword: project from world space (mutually exclusive with `_OBJECTSPACE`).
    _WORLDSPACE: f32,
    /// Keyword: project from object space (mutually exclusive with `_WORLDSPACE`).
    _OBJECTSPACE: f32,
    /// Keyword: enable albedo texture sampling (otherwise tint-only).
    _ALBEDOTEX: f32,
    /// Keyword: enable emission texture sampling.
    _EMISSIONTEX: f32,
    /// Keyword: enable normal map sampling with RNM blending across planes.
    _NORMALMAP: f32,
    /// Keyword: read tinted f0 + smoothness from `_SpecularMap`.
    _SPECULARMAP: f32,
    /// Keyword: read occlusion from `_OcclusionMap.g` (matches Unity's reference).
    _OCCLUSION: f32,
}

@group(1) @binding(0)  var<uniform> mat: PbsTriplanarTransparentSpecularMaterial;
@group(1) @binding(1)  var _MainTex: texture_2d<f32>;
@group(1) @binding(2)  var _MainTex_sampler: sampler;
@group(1) @binding(3)  var _NormalMap: texture_2d<f32>;
@group(1) @binding(4)  var _NormalMap_sampler: sampler;
@group(1) @binding(5)  var _SpecularMap: texture_2d<f32>;
@group(1) @binding(6)  var _SpecularMap_sampler: sampler;
@group(1) @binding(7)  var _EmissionMap: texture_2d<f32>;
@group(1) @binding(8)  var _EmissionMap_sampler: sampler;
@group(1) @binding(9)  var _OcclusionMap: texture_2d<f32>;
@group(1) @binding(10) var _OcclusionMap_sampler: sampler;

/// Interpolated vertex output forwarded to both forward-base and forward-add fragments.
struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) proj_pos: vec3<f32>,
    @location(3) @interpolate(flat) view_layer: u32,
}

/// Resolved per-fragment shading inputs for the SpecularSetup path.
struct SurfaceData {
    base_color: vec3<f32>,
    alpha: f32,
    f0: vec3<f32>,
    roughness: f32,
    one_minus_reflectivity: f32,
    occlusion: f32,
    normal: vec3<f32>,
    emission: vec3<f32>,
}

/// Reoriented Normal Mapping blend (see [`pbstriplanar::blend_rnm`] for context).
fn blend_rnm(n1_in: vec3<f32>, n2_in: vec3<f32>) -> vec3<f32> {
    let n1 = vec3<f32>(n1_in.x, n1_in.y, n1_in.z + 1.0);
    let n2 = vec3<f32>(-n2_in.x, -n2_in.y, n2_in.z);
    return n1 * dot(n1, n2) / max(n1.z, 1e-4) - n2;
}

/// Apply `_MainTex_ST` to a planar projection coordinate (with WebGPU V-flip).
fn apply_main_st(uv: vec2<f32>) -> vec2<f32> {
    let tiled = uv * mat._MainTex_ST.xy + mat._MainTex_ST.zw;
    return vec2<f32>(tiled.x, 1.0 - tiled.y);
}

/// Triplanar weights from a world-space normal.
fn triplanar_weights(world_n: vec3<f32>) -> vec3<f32> {
    let blend_power = max(mat._TriBlendPower, 0.0001);
    let raw = pow(abs(world_n), vec3<f32>(blend_power));
    let sum = max(raw.x + raw.y + raw.z, 1e-4);
    return raw / sum;
}

/// Per-plane projected UVs along with the per-axis sign used to correct mirrored projections.
struct PlanarUvs {
    uv_x: vec2<f32>,
    uv_y: vec2<f32>,
    uv_z: vec2<f32>,
    axis_sign: vec3<f32>,
}

/// Construct the three planar UVs from a projection-space position and the world normal.
fn build_planar_uvs(proj_pos: vec3<f32>, world_n: vec3<f32>) -> PlanarUvs {
    var uvs: PlanarUvs;
    uvs.uv_x = apply_main_st(proj_pos.zy);
    uvs.uv_y = apply_main_st(proj_pos.xz);
    uvs.uv_z = apply_main_st(proj_pos.xy);
    let axis_sign = vec3<f32>(
        select(-1.0, 1.0, world_n.x >= 0.0),
        select(-1.0, 1.0, world_n.y >= 0.0),
        select(-1.0, 1.0, world_n.z >= 0.0),
    );
    uvs.uv_x.x = uvs.uv_x.x * axis_sign.x;
    uvs.uv_y.x = uvs.uv_y.x * axis_sign.y;
    uvs.uv_z.x = uvs.uv_z.x * -axis_sign.z;
    uvs.axis_sign = axis_sign;
    return uvs;
}

/// Sample a 2D texture three times along the planar UVs and blend by triplanar weights.
fn triplanar_rgba(
    tex: texture_2d<f32>,
    samp: sampler,
    uvs: PlanarUvs,
    weights: vec3<f32>,
) -> vec4<f32> {
    let cx = textureSample(tex, samp, uvs.uv_x);
    let cy = textureSample(tex, samp, uvs.uv_y);
    let cz = textureSample(tex, samp, uvs.uv_z);
    return cx * weights.x + cy * weights.y + cz * weights.z;
}

/// Build a triplanar world-space normal via Reoriented Normal Mapping when `_NORMALMAP` is on,
/// otherwise return the renormalized geometric normal.
fn sample_normal_world(uvs: PlanarUvs, world_n: vec3<f32>, weights: vec3<f32>) -> vec3<f32> {
    let n_geo = normalize(world_n);
    if (!uvu::kw_enabled(mat._NORMALMAP)) {
        return n_geo;
    }

    var t_x = nd::decode_ts_normal_with_placeholder(
        textureSample(_NormalMap, _NormalMap_sampler, uvs.uv_x).xyz,
        mat._NormalScale,
    );
    var t_y = nd::decode_ts_normal_with_placeholder(
        textureSample(_NormalMap, _NormalMap_sampler, uvs.uv_y).xyz,
        mat._NormalScale,
    );
    var t_z = nd::decode_ts_normal_with_placeholder(
        textureSample(_NormalMap, _NormalMap_sampler, uvs.uv_z).xyz,
        mat._NormalScale,
    );

    t_x.x = t_x.x * uvs.axis_sign.x;
    t_y.x = t_y.x * uvs.axis_sign.y;
    t_z.x = t_z.x * -uvs.axis_sign.z;

    let abs_n = abs(n_geo);
    let n_x_base = vec3<f32>(n_geo.z, n_geo.y, abs_n.x);
    let n_y_base = vec3<f32>(n_geo.x, n_geo.z, abs_n.y);
    let n_z_base = vec3<f32>(n_geo.x, n_geo.y, abs_n.z);

    var blended_x = blend_rnm(n_x_base, t_x);
    var blended_y = blend_rnm(n_y_base, t_y);
    var blended_z = blend_rnm(n_z_base, t_z);

    blended_x.z = blended_x.z * uvs.axis_sign.x;
    blended_y.z = blended_y.z * uvs.axis_sign.y;
    blended_z.z = blended_z.z * uvs.axis_sign.z;

    let world_x = vec3<f32>(blended_x.z, blended_x.y, blended_x.x);
    let world_y = vec3<f32>(blended_y.x, blended_y.z, blended_y.y);
    let world_z = vec3<f32>(blended_z.x, blended_z.y, blended_z.z);

    return normalize(world_x * weights.x + world_y * weights.y + world_z * weights.z);
}

/// Resolve the [`SurfaceData`] for a fragment, mirroring Unity's triplanar `surf` for `PBSTriplanarSpecular`.
fn sample_surface(world_pos: vec3<f32>, world_n: vec3<f32>, proj_pos: vec3<f32>) -> SurfaceData {
    let uvs = build_planar_uvs(proj_pos, world_n);
    let weights = triplanar_weights(world_n);

    var c = mat._Color;
    if (uvu::kw_enabled(mat._ALBEDOTEX)) {
        c = c * triplanar_rgba(_MainTex, _MainTex_sampler, uvs, weights);
    }

    var spec = mat._SpecularColor;
    if (uvu::kw_enabled(mat._SPECULARMAP)) {
        spec = triplanar_rgba(_SpecularMap, _SpecularMap_sampler, uvs, weights);
    }
    let f0 = clamp(spec.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    let smoothness = clamp(spec.a, 0.0, 1.0);
    let roughness = clamp(1.0 - smoothness, 0.045, 1.0);
    let one_minus_reflectivity = 1.0 - max(max(f0.r, f0.g), f0.b);

    var occlusion = 1.0;
    if (uvu::kw_enabled(mat._OCCLUSION)) {
        let occ = triplanar_rgba(_OcclusionMap, _OcclusionMap_sampler, uvs, weights);
        occlusion = occ.g;
    }

    var emission = mat._EmissionColor;
    if (uvu::kw_enabled(mat._EMISSIONTEX)) {
        emission = emission * triplanar_rgba(_EmissionMap, _EmissionMap_sampler, uvs, weights);
    }

    return SurfaceData(
        c.rgb,
        c.a,
        f0,
        roughness,
        one_minus_reflectivity,
        occlusion,
        sample_normal_world(uvs, world_n, weights),
        emission.rgb,
    );
}

/// Iterate the cluster's lights and accumulate Cook–Torrance specular radiance, gated by directional/local.
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
        lo = lo + brdf::direct_radiance_specular(
            light,
            world_pos,
            s.normal,
            v,
            s.roughness,
            s.base_color,
            s.f0,
            s.one_minus_reflectivity,
        );
    }
    return lo * s.occlusion;
}

/// Vertex stage: forward world position, world-space normal, and projection-space position.
@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
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
    out.proj_pos = select(world_p.xyz, pos.xyz, uvu::kw_enabled(mat._OBJECTSPACE));
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
    @location(2) proj_pos: vec3<f32>,
    @location(3) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    let s = sample_surface(world_pos, world_n, proj_pos);
    let direct = clustered_direct_lighting(frag_pos.xy, world_pos, view_layer, s, true, true);
    let ambient = vec3<f32>(0.03) * s.base_color * s.occlusion;
    return vec4<f32>(ambient + direct + s.emission, s.alpha);
}

/// Forward-add pass: additive accumulation of local (point/spot) lights.
