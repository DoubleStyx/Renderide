//! Unity surface shader `Shader "PBSDualSidedTransparentSpecular"`: Standard SpecularSetup with
//! two-sided normals, authored for transparent draws.
//!
//! Functionally identical to [`pbsdualsidedspecular`](super::pbsdualsidedspecular) at the WGSL
//! level — Unity's `alpha`/`Queue=Transparent` directives only change default render queue and
//! blend factors, which this renderer drives from host `_SrcBlend`/`_DstBlend`/`_ZWrite` material
//! properties. The shader is split into its own embedded stem
//! (`pbsdualsidedtransparentspecular_default`) so host shader-asset routing maps
//! `PBSDualSidedTransparentSpecular` to the matching pipeline cache key.

// unity-shader-name: PBSDualSidedTransparentSpecular

#import renderide::globals as rg
#import renderide::per_draw as pd
#import renderide::pbs::brdf as brdf
#import renderide::pbs::normal as pnorm
#import renderide::pbs::cluster as pcls
#import renderide::alpha_clip_sample as acs
#import renderide::uv_utils as uvu
#import renderide::normal_decode as nd

/// Material uniforms for `PBSDualSidedTransparentSpecular`.
struct PbsDualSidedTransparentSpecularMaterial {
    /// Tint color (`Color`).
    _Color: vec4<f32>,
    /// Emission color (`EmissionColor`).
    _EmissionColor: vec4<f32>,
    /// Tinted specular color when `_SPECULARMAP` is disabled (RGB = f0, A = smoothness).
    _SpecularColor: vec4<f32>,
    /// Albedo `_ST` (xy = scale, zw = offset).
    _MainTex_ST: vec4<f32>,
    /// Tangent-space normal scale.
    _NormalScale: f32,
    /// Alpha-clip threshold; applied only when `_ALPHACLIP` is enabled.
    _AlphaClip: f32,
    /// Keyword: enable alpha clipping against `_AlphaClip`.
    _ALPHACLIP: f32,
    /// Keyword: enable albedo texture sampling.
    _ALBEDOTEX: f32,
    /// Keyword: enable emission texture sampling.
    _EMISSIONTEX: f32,
    /// Keyword: enable normal map sampling.
    _NORMALMAP: f32,
    /// Keyword: read tinted f0 + smoothness from `_SpecularMap`.
    _SPECULARMAP: f32,
    /// Keyword: read occlusion from `_OcclusionMap.r`.
    _OCCLUSION: f32,
    /// Keyword: multiply albedo by vertex color.
    VCOLOR_ALBEDO: f32,
    /// Keyword: multiply emission by vertex color.
    VCOLOR_EMIT: f32,
    /// Keyword: multiply specular RGBA by vertex color.
    VCOLOR_SPECULAR: f32,
}

@group(1) @binding(0)  var<uniform> mat: PbsDualSidedTransparentSpecularMaterial;
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

/// Interpolated vertex output forwarded to both forward-base and forward-add fragments.
struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) color: vec4<f32>,
    @location(4) @interpolate(flat) view_layer: u32,
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

/// Sample tangent-space normal, place it in world space, and flip Z for back-faces (two-sided).
fn sample_normal_world(uv_main: vec2<f32>, world_n: vec3<f32>, front_facing: bool) -> vec3<f32> {
    let tbn = pnorm::orthonormal_tbn(normalize(world_n));
    var ts_n = vec3<f32>(0.0, 0.0, 1.0);
    if (uvu::kw_enabled(mat._NORMALMAP)) {
        ts_n = nd::decode_ts_normal_with_placeholder(
            textureSample(_NormalMap, _NormalMap_sampler, uv_main).xyz,
            mat._NormalScale,
        );
    }
    if (!front_facing) {
        ts_n.z = -ts_n.z;
    }
    return normalize(tbn * ts_n);
}

/// Resolve the [`SurfaceData`] for a fragment, mirroring Unity's `surf` for `PBSDualSidedTransparentSpecular`.
fn sample_surface(
    uv0: vec2<f32>,
    world_n: vec3<f32>,
    front_facing: bool,
    vertex_color: vec4<f32>,
) -> SurfaceData {
    let uv_main = uvu::apply_st(uv0, mat._MainTex_ST);

    var albedo = mat._Color;
    if (uvu::kw_enabled(mat._ALBEDOTEX)) {
        albedo = albedo * textureSample(_MainTex, _MainTex_sampler, uv_main);
    }
    if (uvu::kw_enabled(mat.VCOLOR_ALBEDO)) {
        albedo = albedo * vertex_color;
    }
    let vertex_alpha = select(1.0, vertex_color.a, uvu::kw_enabled(mat.VCOLOR_ALBEDO));
    let clip_alpha = select(
        albedo.a,
        mat._Color.a
            * vertex_alpha
            * acs::texture_alpha_base_mip(_MainTex, _MainTex_sampler, uv_main),
        uvu::kw_enabled(mat._ALBEDOTEX),
    );
    if (uvu::kw_enabled(mat._ALPHACLIP) && clip_alpha <= mat._AlphaClip) {
        discard;
    }

    var spec = mat._SpecularColor;
    if (uvu::kw_enabled(mat._SPECULARMAP)) {
        spec = textureSample(_SpecularMap, _SpecularMap_sampler, uv_main);
    }
    if (uvu::kw_enabled(mat.VCOLOR_SPECULAR)) {
        spec = spec * vertex_color;
    }
    let f0 = clamp(spec.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    let smoothness = clamp(spec.a, 0.0, 1.0);
    let roughness = clamp(1.0 - smoothness, 0.045, 1.0);
    let one_minus_reflectivity = 1.0 - max(max(f0.r, f0.g), f0.b);

    var occlusion = 1.0;
    if (uvu::kw_enabled(mat._OCCLUSION)) {
        occlusion = textureSample(_OcclusionMap, _OcclusionMap_sampler, uv_main).r;
    }

    let emission_color = mat._EmissionColor.rgb;
    var emission = vec3<f32>(0.0);
    if (dot(emission_color, emission_color) > 1e-8) {
        emission = emission_color;
        if (uvu::kw_enabled(mat._EMISSIONTEX)) {
            emission = emission * textureSample(_EmissionMap, _EmissionMap_sampler, uv_main).rgb;
        }
    }
    if (uvu::kw_enabled(mat.VCOLOR_EMIT)) {
        emission = emission * vertex_color.rgb;
    }

    return SurfaceData(
        albedo.rgb,
        albedo.a,
        f0,
        roughness,
        one_minus_reflectivity,
        occlusion,
        sample_normal_world(uv_main, world_n, front_facing),
        emission,
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

    let aa_roughness = brdf::filter_perceptual_roughness(s.roughness, s.normal);

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
            aa_roughness,
            s.base_color,
            s.f0,
            s.one_minus_reflectivity,
        );
    }
    return lo;
}

/// Vertex stage: forward world position, world-space normal, primary UV, and vertex color.
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
    out.color = color;
#ifdef MULTIVIEW
    out.view_layer = view_idx;
#else
    out.view_layer = 0u;
#endif
    return out;
}

/// Forward-base pass: ambient + directional lighting + emission.
//#pass forward
@fragment
fn fs_forward_base(
    @builtin(position) frag_pos: vec4<f32>,
    @builtin(front_facing) front_facing: bool,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) color: vec4<f32>,
    @location(4) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    let s = sample_surface(uv0, world_n, front_facing, color);
    let direct = clustered_direct_lighting(frag_pos.xy, world_pos, view_layer, s, true, true);
    let ambient = vec3<f32>(0.03) * s.base_color * s.occlusion;
    return vec4<f32>(ambient + direct + s.emission, s.alpha);
}

/// Forward-add pass: additive accumulation of local (point/spot) lights.
