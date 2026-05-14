//! Unity surface shader `Shader "ColorMask"` (asset: `PBSColorMask.shader`): metallic Standard
//! lighting with four base colors blended by an RGBA mask texture.
//!
//! Each mask channel selects one of `_Color`/`_Color1`/`_Color2`/`_Color3`; the result is
//! normalized by the channel sum and optionally multiplied by `_MainTex`. Per-channel emission
//! follows the same pattern. The Unity asset declares the keyword surface
//! (`_ALBEDOTEX`, `_EMISSIONTEX`, `_NORMALMAP`, `_METALLICMAP`, `_OCCLUSION`); a sixth keyword
//! `_MULTI_VALUES` is present in the Unity property block but its pragma in the source asset is
//! misspelled (`#pragma mutli_compile`) and is silently dropped by Froox's pragma parser, so it
//! never participates in this shader's variant set.


//#texture_default _MainTex white
//#texture_default _ColorMask black
//#texture_default _NormalMap bump
//#texture_default _EmissionMap black
//#texture_default _OcclusionMap white
//#texture_default _MetallicMap black

#import renderide::material::variant_bits as vb
#import renderide::mesh::vertex as mv
#import renderide::pbs::normal as pnorm
#import renderide::pbs::lighting as plight
#import renderide::pbs::surface as psurf
#import renderide::core::uv as uvu
#import renderide::core::normal_decode as nd

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
    /// Color mask `_ST` (xy = scale, zw = offset).
    _ColorMask_ST: vec4<f32>,
    /// Tangent-space normal scale.
    _NormalScale: f32,
    /// Smoothness fallback when `_METALLICMAP` is disabled.
    _Glossiness: f32,
    /// Metallic fallback when `_METALLICMAP` is disabled.
    _Metallic: f32,
    /// Renderer-reserved Froox variant bits (sorted UniqueKeywords).
    _RenderideVariantBits: u32,
}

const PBSCOLORMASK_KW_ALBEDOTEX: u32 = 1u << 0u;
const PBSCOLORMASK_KW_EMISSIONTEX: u32 = 1u << 1u;
const PBSCOLORMASK_KW_METALLICMAP: u32 = 1u << 2u;
const PBSCOLORMASK_KW_NORMALMAP: u32 = 1u << 3u;
const PBSCOLORMASK_KW_OCCLUSION: u32 = 1u << 4u;

fn pbscolormask_kw(mask: u32) -> bool {
    return vb::enabled(mat._RenderideVariantBits, mask);
}

fn kw_ALBEDOTEX() -> bool {
    return pbscolormask_kw(PBSCOLORMASK_KW_ALBEDOTEX);
}

fn kw_EMISSIONTEX() -> bool {
    return pbscolormask_kw(PBSCOLORMASK_KW_EMISSIONTEX);
}

fn kw_METALLICMAP() -> bool {
    return pbscolormask_kw(PBSCOLORMASK_KW_METALLICMAP);
}

fn kw_NORMALMAP() -> bool {
    return pbscolormask_kw(PBSCOLORMASK_KW_NORMALMAP);
}

fn kw_OCCLUSION() -> bool {
    return pbscolormask_kw(PBSCOLORMASK_KW_OCCLUSION);
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

/// Resolved per-fragment shading inputs for the metallic Cook-Torrance path.
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
fn sample_normal_world(uv_main: vec2<f32>, world_n: vec3<f32>, world_t: vec4<f32>) -> vec3<f32> {
    let tbn = pnorm::orthonormal_tbn(normalize(world_n), normalize(world_t));
    var ts_n = vec3<f32>(0.0, 0.0, 1.0);
    if (kw_NORMALMAP()) {
        ts_n = nd::decode_ts_normal_with_placeholder_sample(
            textureSample(_NormalMap, _NormalMap_sampler, uv_main),
            mat._NormalScale,
        );
    }
    return normalize(tbn * ts_n);
}

/// Resolve the [`SurfaceData`] for a fragment, mirroring Unity's `surf` for `PBSColorMask`.
fn sample_surface(uv0: vec2<f32>, world_n: vec3<f32>, world_t: vec4<f32>) -> SurfaceData {
    let uv_main = uvu::apply_st(uv0, mat._MainTex_ST);
    let uv_mask =
        uvu::apply_st(uv0, mat._ColorMask_ST);

    let mask = textureSample(_ColorMask, _ColorMask_sampler, uv_mask);
    let weight_inv = max(mask.r + mask.g + mask.b + mask.a, 1e-5);
    let weight = clamp(1.0 / weight_inv, 0.0, 1.0);

    var c =
        mat._Color * mask.r
        + mat._Color1 * mask.g
        + mat._Color2 * mask.b
        + mat._Color3 * mask.a;
    c = c * weight;
    if (kw_ALBEDOTEX()) {
        c = c * textureSample(_MainTex, _MainTex_sampler, uv_main);
    }

    var metallic = mat._Metallic;
    var smoothness = mat._Glossiness;
    if (kw_METALLICMAP()) {
        let m = textureSample(_MetallicMap, _MetallicMap_sampler, uv_main);
        metallic = m.r;
        smoothness = m.a;
    }
    metallic = clamp(metallic, 0.0, 1.0);
    let roughness = clamp(1.0 - smoothness, 0.0, 1.0);

    var occlusion = 1.0;
    if (kw_OCCLUSION()) {
        occlusion = textureSample(_OcclusionMap, _OcclusionMap_sampler, uv_main).r;
    }

    var emission =
        mat._EmissionColor * mask.r
        + mat._EmissionColor1 * mask.g
        + mat._EmissionColor2 * mask.b
        + mat._EmissionColor3 * mask.a;
    emission = emission * weight;
    if (kw_EMISSIONTEX()) {
        emission = emission * textureSample(_EmissionMap, _EmissionMap_sampler, uv_main);
    }

    return SurfaceData(
        c.rgb,
        c.a,
        metallic,
        roughness,
        occlusion,
        sample_normal_world(uv_main, world_n, world_t),
        emission.rgb,
    );
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
    @location(4) t: vec4<f32>,
) -> mv::WorldVertexOutput {
#ifdef MULTIVIEW
    return mv::world_vertex_main(instance_index, view_idx, pos, n, t, uv0);
#else
    return mv::world_vertex_main(instance_index, 0u, pos, n, t, uv0);
#endif
}

/// Forward-base pass: ambient + directional lighting + emission.
//#pass forward
@fragment
fn fs_forward_base(
    @builtin(position) frag_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) world_t: vec4<f32>,
    @location(3) uv0: vec2<f32>,
    @location(4) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    let s = sample_surface(uv0, world_n, world_t);
    let surface = psurf::metallic(
        s.base_color,
        s.alpha,
        s.metallic,
        s.roughness,
        s.occlusion,
        s.normal,
        s.emission,
    );
    return vec4<f32>(
        plight::shade_metallic_clustered(
            frag_pos.xy,
            world_pos,
            view_layer,
            surface,
            plight::default_lighting_options(),
        ),
        s.alpha,
    );
}
