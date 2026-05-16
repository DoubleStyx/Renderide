//! Unity surface shader `Shader "ColorMaskSpecular"` (asset: `PBSColorMaskSpecular.shader`):
//! Standard SpecularSetup with four base colors blended by an RGBA mask texture.
//!
//! Each mask channel selects one of `_Color`/`_Color1`/`_Color2`/`_Color3`; the result is
//! normalized by the channel sum and optionally multiplied by `_MainTex`. Per-channel emission
//! follows the same pattern. Mirrors the keyword surface (`_ALBEDOTEX`, `_EMISSIONTEX`,
//! `_NORMALMAP`, `_SPECULARMAP`, `_OCCLUSION`, `_MULTI_VALUES`).


//#texture_default _MainTex white
//#texture_default _ColorMask black
//#texture_default _NormalMap bump
//#texture_default _EmissionMap black
//#texture_default _OcclusionMap white
//#texture_default _SpecularMap white
//#mat_default _Color vec4 1.0 0.0 0.0 1.0
//#mat_default _Color1 vec4 0.0 1.0 0.0 1.0
//#mat_default _Color2 vec4 0.0 0.0 1.0 1.0
//#mat_default _Color3 vec4 1.0 1.0 1.0 1.0
//#mat_default _NormalScale float 1.0
//#mat_default _SpecularColor vec4 1.0 1.0 1.0 0.5

#import renderide::material::variant_bits as vb
#import renderide::material::color as mcolor
#import renderide::mesh::vertex as mv
#import renderide::pbs::normal as pnorm
#import renderide::pbs::lighting as plight
#import renderide::pbs::sampling as psamp
#import renderide::pbs::splat as splat
#import renderide::pbs::surface as psurf
#import renderide::core::uv as uvu
#import renderide::core::normal_decode as nd

/// Material uniforms for `PBSColorMaskSpecular`.
struct PbsColorMaskSpecularMaterial {
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
    /// Tinted specular color when `_SPECULARMAP` is disabled (RGB = f0, A = smoothness).
    _SpecularColor: vec4<f32>,
    /// Albedo `_ST` (xy = scale, zw = offset).
    _MainTex_ST: vec4<f32>,
    /// Color mask `_ST` (xy = scale, zw = offset).
    _ColorMask_ST: vec4<f32>,
    /// Tangent-space normal scale.
    _NormalScale: f32,
    /// Renderer-reserved Froox variant bits (sorted UniqueKeywords).
    _RenderideVariantBits: u32,
}

const PBSCOLORMASKSPECULAR_KW_ALBEDOTEX: u32 = 1u << 0u;
const PBSCOLORMASKSPECULAR_KW_EMISSIONTEX: u32 = 1u << 1u;
const PBSCOLORMASKSPECULAR_KW_MULTI_VALUES: u32 = 1u << 2u;
const PBSCOLORMASKSPECULAR_KW_NORMALMAP: u32 = 1u << 3u;
const PBSCOLORMASKSPECULAR_KW_OCCLUSION: u32 = 1u << 4u;
const PBSCOLORMASKSPECULAR_KW_SPECULARMAP: u32 = 1u << 5u;

fn pbscolormaskspecular_kw(mask: u32) -> bool {
    return vb::enabled(mat._RenderideVariantBits, mask);
}

fn kw_ALBEDOTEX() -> bool {
    return pbscolormaskspecular_kw(PBSCOLORMASKSPECULAR_KW_ALBEDOTEX);
}

fn kw_EMISSIONTEX() -> bool {
    return pbscolormaskspecular_kw(PBSCOLORMASKSPECULAR_KW_EMISSIONTEX);
}

fn kw_MULTI_VALUES() -> bool {
    return pbscolormaskspecular_kw(PBSCOLORMASKSPECULAR_KW_MULTI_VALUES);
}

fn kw_NORMALMAP() -> bool {
    return pbscolormaskspecular_kw(PBSCOLORMASKSPECULAR_KW_NORMALMAP);
}

fn kw_OCCLUSION() -> bool {
    return pbscolormaskspecular_kw(PBSCOLORMASKSPECULAR_KW_OCCLUSION);
}

fn kw_SPECULARMAP() -> bool {
    return pbscolormaskspecular_kw(PBSCOLORMASKSPECULAR_KW_SPECULARMAP);
}

@group(1) @binding(0)  var<uniform> mat: PbsColorMaskSpecularMaterial;
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
@group(1) @binding(11) var _SpecularMap: texture_2d<f32>;
@group(1) @binding(12) var _SpecularMap_sampler: sampler;

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

/// Sample the normal map (when enabled) and transform the tangent-space normal to world space.
fn sample_normal_world(uv_main: vec2<f32>, world_n: vec3<f32>, world_t: vec4<f32>) -> vec3<f32> {
    let tbn = pnorm::orthonormal_tbn(world_n, world_t);
    var ts_n = vec3<f32>(0.0, 0.0, 1.0);
    if (kw_NORMALMAP()) {
        ts_n = nd::decode_ts_normal_with_placeholder_sample(
            textureSample(_NormalMap, _NormalMap_sampler, uv_main),
            mat._NormalScale,
        );
    }
    return normalize(tbn * ts_n);
}

/// Resolve the [`SurfaceData`] for a fragment, mirroring Unity's `surf` for `ColorMaskSpecular`.
fn sample_surface(uv0: vec2<f32>, world_n: vec3<f32>, world_t: vec4<f32>) -> SurfaceData {
    let uv_main = uvu::apply_st(uv0, mat._MainTex_ST);
    let uv_mask =
        uvu::apply_st(uv0, mat._ColorMask_ST);

    let mask = textureSample(_ColorMask, _ColorMask_sampler, uv_mask);
    let weights = splat::color_mask_weights(mask);

    var c = mcolor::blend4_vec4(mat._Color, mat._Color1, mat._Color2, mat._Color3, weights);
    if (kw_ALBEDOTEX()) {
        c = c * textureSample(_MainTex, _MainTex_sampler, uv_main);
    }

    var spec = mat._SpecularColor;
    if (kw_SPECULARMAP()) {
        spec = textureSample(_SpecularMap, _SpecularMap_sampler, uv_main);
        if (kw_MULTI_VALUES()) {
            spec = spec * mat._SpecularColor;
        }
    }
    let f0 = clamp(spec.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    let smoothness = clamp(spec.a, 0.0, 1.0);
    let roughness = psamp::roughness_from_smoothness(smoothness);
    let one_minus_reflectivity = 1.0 - max(max(f0.r, f0.g), f0.b);

    var occlusion = 1.0;
    if (kw_OCCLUSION()) {
        occlusion = textureSample(_OcclusionMap, _OcclusionMap_sampler, uv_main).r;
    }

    var emission = mcolor::blend4_vec4(
        mat._EmissionColor,
        mat._EmissionColor1,
        mat._EmissionColor2,
        mat._EmissionColor3,
        weights,
    );
    if (kw_EMISSIONTEX()) {
        emission = emission * textureSample(_EmissionMap, _EmissionMap_sampler, uv_main);
    }

    return SurfaceData(
        c.rgb,
        c.a,
        f0,
        roughness,
        one_minus_reflectivity,
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
    let surface = psurf::specular_with_geometric_normal(
        s.base_color,
        s.alpha,
        s.f0,
        s.roughness,
        s.occlusion,
        s.normal,
        world_n,
        s.emission,
    );
    return vec4<f32>(
        plight::shade_specular_clustered(
            frag_pos.xy,
            world_pos,
            view_layer,
            surface,
            plight::default_lighting_options(),
        ),
        s.alpha,
    );
}
