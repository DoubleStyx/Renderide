//! Shared types, bindings, constants, and generic helpers for the Xiexe Toon 2.0 family.
//!
//! This is the dependency floor for `renderide::xiexe::toon2::{surface, alpha, lighting,
//! outline}` and the `renderide::xiexe::toon2` aggregator. It owns the material struct,
//! every `@group(1)` texture/sampler binding, the public-facing structs returned across
//! the vertex/fragment boundary, and the math primitives used by every other xiexe-toon
//! module. Behavioural code lives in the sibling modules.

#define_import_path renderide::xiexe::toon2::base

#import renderide::core::math as rmath
#import renderide::mesh::vertex as mv
#import renderide::draw::types as dt

/// Alpha-handling mode tag passed by each dispatcher. Selects one of the seven branches
/// implemented in `xiexe_toon2_alpha::apply_alpha`.
const ALPHA_OPAQUE: u32 = 0u;
/// Hard-clip cutout against `_Cutoff`.
const ALPHA_CUTOUT: u32 = 1u;
/// Alpha-to-coverage with Bayer screen-space dither.
const ALPHA_A2C: u32 = 2u;
/// Bayer dither + `_CutoutMask`-driven coverage blend.
const ALPHA_A2C_MASKED: u32 = 3u;
/// Bayer dither with optional `_FadeDither` distance falloff.
const ALPHA_DITHERED: u32 = 4u;
/// Standard alpha-blend fade (no clip).
const ALPHA_FADE: u32 = 5u;
/// Standard alpha-blend transparent (RGB pre-multiplied by alpha by the caller).
const ALPHA_TRANSPARENT: u32 = 6u;

/// 8x8 Bayer matrix used by the dithered/A2C alpha modes (values in [1, 64]).
const BAYER_GRID: array<f32, 64> = array<f32, 64>(
    1.0, 49.0, 13.0, 61.0,  4.0, 52.0, 16.0, 64.0,
    33.0, 17.0, 45.0, 29.0, 36.0, 20.0, 48.0, 32.0,
    9.0, 57.0,  5.0, 53.0, 12.0, 60.0,  8.0, 56.0,
    41.0, 25.0, 37.0, 21.0, 44.0, 28.0, 40.0, 24.0,
    3.0, 51.0, 15.0, 63.0,  2.0, 50.0, 14.0, 62.0,
    35.0, 19.0, 47.0, 31.0, 34.0, 18.0, 46.0, 30.0,
    11.0, 59.0,  7.0, 55.0, 10.0, 58.0,  6.0, 54.0,
    43.0, 27.0, 39.0, 23.0, 42.0, 26.0, 38.0, 22.0
);

/// Per-material uniform block at `@group(1) @binding(0)`. Field names mirror the original
/// Unity property identifiers so naga reflection produces stable interned property IDs.
struct XiexeToon2Material {
    /// Tint multiplied into the albedo sample.
    _Color: vec4<f32>,
    /// Emissive tint multiplied into the emission sample.
    _EmissionColor: vec4<f32>,
    /// Rim-light tint.
    _RimColor: vec4<f32>,
    /// Shadow-rim tint applied as a multiplicative shadow accent.
    _ShadowRim: vec4<f32>,
    /// Tint that ambient-occlusion contributes when AO is below 1.
    _OcclusionColor: vec4<f32>,
    /// Outline base color (opaque outline pass).
    _OutlineColor: vec4<f32>,
    /// Subsurface-scattering tint.
    _SSColor: vec4<f32>,
    /// Matcap reflection tint.
    _MatcapTint: vec4<f32>,

    /// `_MainTex` UV scale/offset.
    _MainTex_ST: vec4<f32>,
    /// `_BumpMap` UV scale/offset.
    _BumpMap_ST: vec4<f32>,
    /// `_MetallicGlossMap` UV scale/offset.
    _MetallicGlossMap_ST: vec4<f32>,
    /// `_EmissionMap` UV scale/offset.
    _EmissionMap_ST: vec4<f32>,
    /// `_OcclusionMap` UV scale/offset.
    _OcclusionMap_ST: vec4<f32>,
    /// `_ThicknessMap` UV scale/offset.
    _ThicknessMap_ST: vec4<f32>,
    /// `_CutoutMask` UV scale/offset.
    _CutoutMask_ST: vec4<f32>,
    /// `_ReflectivityMask` UV scale/offset.
    _ReflectivityMask_ST: vec4<f32>,

    /// Alpha cutoff threshold for the cutout / masked-coverage paths.
    _Cutoff: f32,
    /// Saturation lerp; `1` keeps full color, `0` desaturates albedo to luminance.
    _Saturation: f32,
    /// Tangent-space normal-map intensity.
    _BumpScale: f32,
    /// Metallic factor multiplied into `_MetallicGlossMap.r`.
    _Metallic: f32,
    /// Smoothness factor multiplied into `_MetallicGlossMap.a`.
    _Glossiness: f32,
    /// Reflectivity weight applied to indirect specular.
    _Reflectivity: f32,

    /// Lerp factor that tints rim by albedo.
    _RimAlbedoTint: f32,
    /// Lerp factor that tints rim by the env-map sample.
    _RimCubemapTint: f32,
    /// Lerp factor that gates rim by light attenuation + ambient.
    _RimAttenEffect: f32,
    /// Rim brightness multiplier.
    _RimIntensity: f32,
    /// Centre of the rim smoothstep window (NdotV-derived value).
    _RimRange: f32,
    /// Power applied to NdotL when modulating rim by lit-side.
    _RimThreshold: f32,
    /// Smoothstep half-width around `_RimRange`.
    _RimSharpness: f32,

    /// Specular highlight intensity multiplier.
    _SpecularIntensity: f32,
    /// Specular-area smoothness override; remapped to roughness in `direct_specular`.
    _SpecularArea: f32,
    /// Lerp factor that tints specular highlight by albedo.
    _SpecularAlbedoTint: f32,

    /// Sharpens the shadow attenuation transition. `0` = smooth, `1` = hard step.
    _ShadowSharpness: f32,
    /// Centre of the shadow-rim smoothstep window.
    _ShadowRimRange: f32,
    /// Power applied to (1 - NdotL) when modulating shadow-rim by shadowed-side.
    _ShadowRimThreshold: f32,
    /// Smoothstep half-width around `_ShadowRimRange`.
    _ShadowRimSharpness: f32,
    /// Lerp factor that tints shadow rim by albedo.
    _ShadowRimAlbedoTint: f32,

    /// Lerp factor that tints the outline by albedo.
    _OutlineAlbedoTint: f32,
    /// Outline lighting mode (`Lit = 0`, `Emissive = 1`). Aliased name used by
    /// `XSToon2.0 Outlined.shader`. Treated as one of three aliases that all map onto the
    /// same `Lit/Emissive` enum.
    _OutlineLighting: f32,
    /// Outline lighting mode alias.
    _OutlineEmissive: f32,
    /// Outline lighting mode alias used by `XSToon2.0.shader` (preserves the original
    /// upstream property name including its "ues" suffix typo).
    _OutlineEmissiveues: f32,
    /// Outline width in centimetres of object space, distance-faded to one cm at one metre.
    _OutlineWidth: f32,

    /// Subsurface distortion of the half-vector by the surface normal.
    _SSDistortion: f32,
    /// Subsurface power applied to back-lit transmission.
    _SSPower: f32,
    /// Subsurface scale multiplied into the final SSS contribution.
    _SSScale: f32,

    /// Enables the `_FadeDither` distance fade in the dithered alpha mode.
    _FadeDither: f32,
    /// Distance at which the fade-dither starts to take effect.
    _FadeDitherDistance: f32,

    /// Enables vertex-color-tinted albedo (Unity `VERTEX_COLOR_ALBEDO`).
    _VertexColorAlbedo: f32,

    /// UV set selector (0 = UV0, 1 = UV1) for albedo.
    _UVSetAlbedo: f32,
    /// UV set selector for the base normal map.
    _UVSetNormal: f32,
    /// UV set selector for the metallic/gloss map.
    _UVSetMetallic: f32,
    /// UV set selector for the reflectivity mask.
    _UVSetReflectivity: f32,
    /// UV set selector for the thickness map.
    _UVSetThickness: f32,
    /// UV set selector for the occlusion map.
    _UVSetOcclusion: f32,
    /// UV set selector for the emission map.
    _UVSetEmission: f32,

    /// Renderer-reserved variant bitfield decoded by `renderide::xiexe::toon2::variant_bits`.
    /// Each bit corresponds to one entry of the alphabetically-sorted `UniqueKeywords` list
    /// built from the Unity shader's `#pragma multi_compile` groups.
    _RenderideVariantBits: u32,
}

/// Per-material uniform binding consumed by every xiexe-toon submodule.
@group(1) @binding(0) var<uniform> mat: XiexeToon2Material;
/// Albedo / opacity texture.
@group(1) @binding(1) var _MainTex: texture_2d<f32>;
/// Albedo / opacity sampler.
@group(1) @binding(2) var _MainTex_sampler: sampler;
/// Tangent-space normal map.
@group(1) @binding(3) var _BumpMap: texture_2d<f32>;
/// Tangent-space normal-map sampler.
@group(1) @binding(4) var _BumpMap_sampler: sampler;
/// Metallic in `.r`, smoothness in `.a` (Unity convention).
@group(1) @binding(5) var _MetallicGlossMap: texture_2d<f32>;
/// Metallic / smoothness sampler.
@group(1) @binding(6) var _MetallicGlossMap_sampler: sampler;
/// Emission color texture.
@group(1) @binding(7) var _EmissionMap: texture_2d<f32>;
/// Emission sampler.
@group(1) @binding(8) var _EmissionMap_sampler: sampler;
/// `_RampSelectionMask.r` selects which row of `_Ramp` to sample.
@group(1) @binding(9) var _RampSelectionMask: texture_2d<f32>;
/// Ramp-selection sampler.
@group(1) @binding(10) var _RampSelectionMask_sampler: sampler;
/// 1-D toon shadow ramp (X = NdotL+atten, Y = ramp-mask row).
@group(1) @binding(11) var _Ramp: texture_2d<f32>;
/// Toon-ramp sampler.
@group(1) @binding(12) var _Ramp_sampler: sampler;
/// Ambient-occlusion mask.
@group(1) @binding(13) var _OcclusionMap: texture_2d<f32>;
/// Occlusion sampler.
@group(1) @binding(14) var _OcclusionMap_sampler: sampler;
/// Per-vertex outline-width mask sampled in the outline vertex stage.
@group(1) @binding(15) var _OutlineMask: texture_2d<f32>;
/// Outline-mask sampler.
@group(1) @binding(16) var _OutlineMask_sampler: sampler;
/// Subsurface thickness map.
@group(1) @binding(17) var _ThicknessMap: texture_2d<f32>;
/// Thickness sampler.
@group(1) @binding(18) var _ThicknessMap_sampler: sampler;
/// Cutout coverage mask used by the masked-A2C and dithered modes.
@group(1) @binding(19) var _CutoutMask: texture_2d<f32>;
/// Cutout-mask sampler.
@group(1) @binding(20) var _CutoutMask_sampler: sampler;
/// Matcap reflection texture.
@group(1) @binding(21) var _Matcap: texture_2d<f32>;
/// Matcap sampler.
@group(1) @binding(22) var _Matcap_sampler: sampler;
/// Reflectivity mask gating indirect specular contribution.
@group(1) @binding(23) var _ReflectivityMask: texture_2d<f32>;
/// Reflectivity-mask sampler.
@group(1) @binding(24) var _ReflectivityMask_sampler: sampler;

/// Vertex-to-fragment payload carried by every xiexe-toon variant. The two-UV-set layout
/// mirrors Unity `VertexOutput`/`g2f` so material-property UV selectors keep working.
struct VertexOutput {
    /// Clip-space position for rasterisation.
    @builtin(position) clip_pos: vec4<f32>,
    /// World-space position used by lighting and view-direction math.
    @location(0) world_pos: vec3<f32>,
    /// World-space geometric normal (pre-perturbation).
    @location(1) world_n: vec3<f32>,
    /// World-space tangent from the MikkTSpace basis.
    @location(2) world_t: vec3<f32>,
    /// World-space bitangent signed by the resolved tangent-frame handedness.
    @location(3) world_b: vec3<f32>,
    /// Primary UV (UV0).
    @location(4) uv_primary: vec2<f32>,
    /// Secondary UV (UV1) used by per-texture UV-set selectors.
    @location(5) uv_secondary: vec2<f32>,
    /// Vertex color; alpha is repurposed by the outline pass to flag "is-outline".
    @location(6) color: vec4<f32>,
    /// Object-space position normalised to the unit sphere -- passed through for any
    /// effects that need a stable per-vertex direction.
    @location(7) obj_pos: vec3<f32>,
    /// Stereo view layer (0 = left/mono, 1 = right). Flat-interpolated for cluster lookups.
    @location(8) @interpolate(flat) view_layer: u32,
}

/// Surface attributes resolved by `surface::sample_surface_for_layout`. Consumed by the lighting
/// and outline modules.
struct SurfaceData {
    /// Albedo with alpha; alpha is the source for blend / cutout.
    albedo: vec4<f32>,
    /// Stable per-fragment alpha to clip against (uses the base mip to avoid sparkles).
    clip_alpha: f32,
    /// Saturation-adjusted diffuse color (RGB only).
    diffuse_color: vec3<f32>,
    /// Geometry normal after dual-sided back-face correction but before normal-map perturbation.
    raw_normal: vec3<f32>,
    /// Final perturbed world-space normal (post-detail blend, post-back-face flip).
    normal: vec3<f32>,
    /// World-space tangent matching `normal` after normal-map perturbation.
    tangent: vec3<f32>,
    /// World-space bitangent matching `normal` after normal-map perturbation.
    bitangent: vec3<f32>,
    /// Final metallic factor.
    metallic: f32,
    /// Remapped roughness clamped to `[0.045, 1.0]`.
    roughness: f32,
    /// Raw smoothness (for matcap LOD selection).
    smoothness: f32,
    /// Scalar reflectivity control used for the dielectric Fresnel floor.
    reflectivity: f32,
    /// `_ReflectivityMask.r`, used when blending indirect specular into the surface.
    reflectivity_mask: f32,
    /// AO color tint, lerped from `_OcclusionColor` to white by AO sample.
    occlusion: vec3<f32>,
    /// Base emission sample.
    emission: vec3<f32>,
    /// Ramp-mask row used as the V coordinate when sampling `_Ramp`.
    ramp_mask: f32,
    /// Subsurface thickness sample.
    thickness: f32,
}

/// One resolved punctual / directional light at a surface point.
struct LightSample {
    /// Direction from the surface toward the light source (unit length).
    direction: vec3<f32>,
    /// Light color (linear, pre-attenuation).
    color: vec3<f32>,
    /// Combined intensity, distance, and spot factor (already includes intensity).
    attenuation: f32,
    /// Whether the light is directional (used to scope vertex/forward base passes).
    is_directional: bool,
}

/// Returns true when a Unity material *property* (Enum or ToggleUI, not a multi_compile
/// keyword) reads "on". Reserved for scalar property flags such as `_OutlineAlbedoTint` and
/// `_OutlineLighting`; shader keywords decode through
/// [`renderide::xiexe::toon2::variant_bits`] instead.
fn prop_flag(v: f32) -> bool {
    return v > 0.5;
}

/// `clamp(v, 0, 1)` shortcut. Hand-rolled so call sites don't drag in a `saturate` HLSL
/// intrinsic that WGSL doesn't expose.
fn saturate(v: f32) -> f32 {
    return clamp(v, 0.0, 1.0);
}

/// Delegates safe vector normalization to the shared math module for sibling Xiexe modules.
fn safe_normalize(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    return rmath::safe_normalize(v, fallback);
}

/// Rec. 709 luminance for `_Saturation` desaturation.
fn grayscale(v: vec3<f32>) -> f32 {
    return dot(v, vec3<f32>(0.2125, 0.7154, 0.0721));
}

/// Lerps `c` toward its luminance by `(1 - mat._Saturation)`. Matches the Unity behaviour
/// of `_Saturation = 0` collapsing to greyscale and `_Saturation = 1` keeping full color.
fn maybe_saturate_color(c: vec3<f32>) -> vec3<f32> {
    let g = vec3<f32>(grayscale(c));
    return mix(g, c, mat._Saturation);
}

/// Selects between the primary and secondary UV sets based on a Unity `_UVSetX` scalar
/// (`0 = UV0`, anything else = UV1).
fn uv_select(uv_primary: vec2<f32>, uv_secondary: vec2<f32>, set_id: f32) -> vec2<f32> {
    return select(uv_primary, uv_secondary, set_id > 0.5);
}

/// Looks up the 8x8 Bayer threshold for a fragment-space pixel.
fn bayer_threshold(frag_xy: vec2<f32>) -> f32 {
    let x = u32(floor(frag_xy.x)) & 7u;
    let y = u32(floor(frag_xy.y)) & 7u;
    return BAYER_GRID[y * 8u + x] / 64.0;
}

/// Picks the per-eye view-projection matrix from a per-draw record. Mono builds collapse
/// to the left view; multi-view builds branch on the eye index.
fn view_projection_for_draw(d: dt::PerDrawUniforms, view_idx: u32) -> mat4x4<f32> {
    return mv::select_view_proj(d, view_idx);
}
