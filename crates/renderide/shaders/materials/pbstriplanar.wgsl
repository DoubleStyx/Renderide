//! Unity surface shader `Shader "PBSTriplanar"`: metallic Standard lighting with triplanar
//! projection sampled from world or object space.
//!
//! Each texture (`_MainTex`, `_MetallicMap`, `_EmissionMap`, `_NormalMap`, `_OcclusionMap`) is
//! sampled three times -- once per axis-aligned plane (ZY for X, XZ for Y, XY for Z) -- and blended
//! by `pow(abs(projection_normal), _TriBlendPower)`. Normal maps use Reoriented Normal Mapping
//! (RNM) per plane, after Ben Golus's 2017 example. World-space vs object-space is selected by the
//! `_OBJECTSPACE` / `_WORLDSPACE` keyword pair.
//!
//! Back faces flip the shading normal across the geometric tangent plane so dual-sided meshes
//! shade correctly when the host disables culling.
//!
//! Froox variant bits populate `_RenderideVariantBits`; this shader decodes PBSTriplanar's
//! shader-specific keyword bits locally.


//#texture_default _MainTex white
//#texture_default _NormalMap bump
//#texture_default _MetallicMap black
//#texture_default _EmissionMap black
//#texture_default _OcclusionMap white
//#mat_default _Color vec4 1.0 1.0 1.0 1.0
//#mat_default _NormalScale float 1.0
//#mat_default _TriBlendPower float 4.0
//#mat_default _Glossiness float 0.5

#import renderide::billboard::vertex as bv
#import renderide::material::variant_bits as vb
#import renderide::pbs::families::triplanar as ptri
#import renderide::pbs::lighting as plight
#import renderide::pbs::sampling as psamp
#import renderide::pbs::surface as psurf

/// Material uniforms for `PBSTriplanar`.
struct PbsTriplanarMaterial {
    /// Tint color (`Color`).
    _Color: vec4<f32>,
    /// Emission color (`EmissionColor`).
    _EmissionColor: vec4<f32>,
    /// Albedo `_ST` applied to all three projected planes.
    _MainTex_ST: vec4<f32>,
    /// Tangent-space normal scale (`Normal Scale`).
    _NormalScale: f32,
    /// Smoothness fallback when `_METALLICMAP` is disabled.
    _Glossiness: f32,
    /// Metallic fallback when `_METALLICMAP` is disabled.
    _Metallic: f32,
    /// Triplanar blend exponent -- higher values produce sharper transitions between planes.
    _TriBlendPower: f32,
    /// Renderer-reserved Froox shader variant bitmask.
    _RenderideVariantBits: u32,
    /// Host mip bias for `_MainTex`.
    _MainTex_LodBias: f32,
    /// Host mip bias for `_NormalMap`.
    _NormalMap_LodBias: f32,
    /// Host mip bias for `_MetallicMap`.
    _MetallicMap_LodBias: f32,
    /// Host mip bias for `_EmissionMap`.
    _EmissionMap_LodBias: f32,
    /// Host mip bias for `_OcclusionMap`.
    _OcclusionMap_LodBias: f32,
}

const PBSTRIPLANAR_KW_ALBEDOTEX: u32 = 1u << 0u;
const PBSTRIPLANAR_KW_EMISSIONTEX: u32 = 1u << 1u;
const PBSTRIPLANAR_KW_METALLICMAP: u32 = 1u << 2u;
const PBSTRIPLANAR_KW_NORMALMAP: u32 = 1u << 3u;
const PBSTRIPLANAR_KW_OBJECTSPACE: u32 = 1u << 4u;
const PBSTRIPLANAR_KW_OCCLUSION: u32 = 1u << 5u;
const PBSTRIPLANAR_KW_WORLDSPACE: u32 = 1u << 6u;

@group(1) @binding(0)  var<uniform> mat: PbsTriplanarMaterial;
@group(1) @binding(1)  var _MainTex: texture_2d<f32>;
@group(1) @binding(2)  var _MainTex_sampler: sampler;
@group(1) @binding(3)  var _NormalMap: texture_2d<f32>;
@group(1) @binding(4)  var _NormalMap_sampler: sampler;
@group(1) @binding(5)  var _MetallicMap: texture_2d<f32>;
@group(1) @binding(6)  var _MetallicMap_sampler: sampler;
@group(1) @binding(7)  var _EmissionMap: texture_2d<f32>;
@group(1) @binding(8)  var _EmissionMap_sampler: sampler;
@group(1) @binding(9)  var _OcclusionMap: texture_2d<f32>;
@group(1) @binding(10) var _OcclusionMap_sampler: sampler;

fn pbstriplanar_kw(mask: u32) -> bool {
    return vb::enabled(mat._RenderideVariantBits, mask);
}

fn kw_ALBEDOTEX() -> bool {
    return pbstriplanar_kw(PBSTRIPLANAR_KW_ALBEDOTEX);
}

fn kw_EMISSIONTEX() -> bool {
    return pbstriplanar_kw(PBSTRIPLANAR_KW_EMISSIONTEX);
}

fn kw_METALLICMAP() -> bool {
    return pbstriplanar_kw(PBSTRIPLANAR_KW_METALLICMAP);
}

fn kw_NORMALMAP() -> bool {
    return pbstriplanar_kw(PBSTRIPLANAR_KW_NORMALMAP);
}

fn kw_OBJECTSPACE() -> bool {
    return pbstriplanar_kw(PBSTRIPLANAR_KW_OBJECTSPACE);
}

fn kw_OCCLUSION() -> bool {
    return pbstriplanar_kw(PBSTRIPLANAR_KW_OCCLUSION);
}

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

/// Resolve the [`SurfaceData`] for a fragment, mirroring Unity's triplanar `surf` for `PBSTriplanar`.
fn sample_surface(
    world_n: vec3<f32>,
    projection_n: vec3<f32>,
    proj_pos: vec3<f32>,
    front_facing: bool,
    view_layer: u32,
) -> SurfaceData {
    let object_space = kw_OBJECTSPACE();
    let normal_map = kw_NORMALMAP();
    let uvs = ptri::build_planar_uvs(proj_pos, projection_n, mat._MainTex_ST);
    let weights = ptri::triplanar_weights(projection_n, mat._TriBlendPower);

    var c = mat._Color;
    if (kw_ALBEDOTEX()) {
        c = c * ptri::sample_rgba_biased(_MainTex, _MainTex_sampler, uvs, weights, mat._MainTex_LodBias);
    }

    var metallic = mat._Metallic;
    var smoothness = mat._Glossiness;
    if (kw_METALLICMAP()) {
        let m = ptri::sample_rgba_biased(_MetallicMap, _MetallicMap_sampler, uvs, weights, mat._MetallicMap_LodBias);
        metallic = m.r;
        smoothness = m.a;
    }
    metallic = clamp(metallic, 0.0, 1.0);
    let roughness = psamp::roughness_from_smoothness(smoothness);

    var occlusion = 1.0;
    if (kw_OCCLUSION()) {
        // Unity's reference reads occlusion from the green channel here.
        let occ = ptri::sample_rgba_biased(_OcclusionMap, _OcclusionMap_sampler, uvs, weights, mat._OcclusionMap_LodBias);
        occlusion = occ.g;
    }

    var emission = mat._EmissionColor;
    if (kw_EMISSIONTEX()) {
        emission = emission * ptri::sample_rgba_biased(_EmissionMap, _EmissionMap_sampler, uvs, weights, mat._EmissionMap_LodBias);
    }

    let n = ptri::resolve_world_normal(
        normal_map,
        object_space,
        view_layer,
        world_n,
        projection_n,
        _NormalMap,
        _NormalMap_sampler,
        uvs,
        weights,
        mat._NormalScale,
        mat._NormalMap_LodBias,
        front_facing,
    );

    return SurfaceData(
        c.rgb,
        c.a,
        metallic,
        roughness,
        occlusion,
        n,
        emission.rgb,
    );
}

/// Vertex stage: forward world position plus the projection-space position and normal selected
/// by the `_OBJECTSPACE` / `_WORLDSPACE` keywords.
@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
    @builtin(vertex_index) vertex_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
    @location(4) t: vec4<f32>,
    @location(5) uv1: vec2<f32>,
) -> ptri::VertexOutput {
#ifdef MULTIVIEW
    let view_layer = view_idx;
#else
    let view_layer = 0u;
#endif
    if (bv::kw_RENDER_BUFFER(mat._RenderideVariantBits)) {
        return ptri::billboard_vertex_main(instance_index, view_layer, pos, n, vertex_index, uv0, t, uv1, kw_OBJECTSPACE());
    } else {
        return ptri::vertex_main(instance_index, view_layer, pos, n, kw_OBJECTSPACE());
    }
}

/// Forward-base pass: ambient + directional lighting + emission.
//#pass type=forward cull=material(off) offset=material(0,0)
@fragment
fn fs_forward_base(
    @builtin(position) frag_pos: vec4<f32>,
    @builtin(front_facing) front_facing: bool,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) projection_n: vec3<f32>,
    @location(3) proj_pos: vec3<f32>,
    @location(4) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    let s = sample_surface(world_n, projection_n, proj_pos, front_facing, view_layer);
    let surface = psurf::metallic_with_geometric_normal(
        s.base_color,
        s.alpha,
        s.metallic,
        s.roughness,
        s.occlusion,
        s.normal,
        psamp::two_sided_geometric_normal(world_n, front_facing),
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
