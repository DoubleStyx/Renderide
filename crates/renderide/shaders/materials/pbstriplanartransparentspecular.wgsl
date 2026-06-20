//! Unity surface shader `Shader "PBSTriplanarTransparentSpecular"`: transparent Standard
//! SpecularSetup lighting with triplanar projection sampled from world or object space.
//!
//! This mirrors `PBSTriplanarSpecular` surface evaluation, but declares Unity alpha-style
//! transparent render-state defaults.

//#render_queue Transparent
//#texture_default _MainTex white
//#texture_default _NormalMap bump
//#texture_default _SpecularMap white
//#texture_default _EmissionMap black
//#texture_default _OcclusionMap white
//#mat_default _Color vec4 1.0 1.0 1.0 1.0
//#mat_default _NormalScale float 1.0
//#mat_default _SpecularColor vec4 1.0 1.0 1.0 0.5
//#mat_default _TriBlendPower float 4.0

#import renderide::material::variant_bits as vb
#import renderide::pbs::families::triplanar as ptri
#import renderide::pbs::lighting as plight
#import renderide::pbs::sampling as psamp
#import renderide::pbs::surface as psurf

struct PbsTriplanarTransparentSpecularMaterial {
    _Color: vec4<f32>,
    _EmissionColor: vec4<f32>,
    _SpecularColor: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _NormalScale: f32,
    _TriBlendPower: f32,
    _RenderideVariantBits: u32,
    _MainTex_LodBias: f32,
    _NormalMap_LodBias: f32,
    _SpecularMap_LodBias: f32,
    _EmissionMap_LodBias: f32,
    _OcclusionMap_LodBias: f32,
}

const PBSTRIPLANARTSPEC_KW_ALBEDOTEX: u32 = 1u << 0u;
const PBSTRIPLANARTSPEC_KW_EMISSIONTEX: u32 = 1u << 1u;
const PBSTRIPLANARTSPEC_KW_NORMALMAP: u32 = 1u << 2u;
const PBSTRIPLANARTSPEC_KW_OBJECTSPACE: u32 = 1u << 3u;
const PBSTRIPLANARTSPEC_KW_OCCLUSION: u32 = 1u << 4u;
const PBSTRIPLANARTSPEC_KW_SPECULARMAP: u32 = 1u << 5u;
const PBSTRIPLANARTSPEC_KW_WORLDSPACE: u32 = 1u << 6u;

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

fn pbstriplanartspec_kw(mask: u32) -> bool {
    return vb::enabled(mat._RenderideVariantBits, mask);
}

fn kw_ALBEDOTEX() -> bool {
    return pbstriplanartspec_kw(PBSTRIPLANARTSPEC_KW_ALBEDOTEX);
}

fn kw_EMISSIONTEX() -> bool {
    return pbstriplanartspec_kw(PBSTRIPLANARTSPEC_KW_EMISSIONTEX);
}

fn kw_NORMALMAP() -> bool {
    return pbstriplanartspec_kw(PBSTRIPLANARTSPEC_KW_NORMALMAP);
}

fn kw_OBJECTSPACE() -> bool {
    return pbstriplanartspec_kw(PBSTRIPLANARTSPEC_KW_OBJECTSPACE);
}

fn kw_OCCLUSION() -> bool {
    return pbstriplanartspec_kw(PBSTRIPLANARTSPEC_KW_OCCLUSION);
}

fn kw_SPECULARMAP() -> bool {
    return pbstriplanartspec_kw(PBSTRIPLANARTSPEC_KW_SPECULARMAP);
}

struct SurfaceData {
    base_color: vec3<f32>,
    alpha: f32,
    f0: vec3<f32>,
    roughness: f32,
    occlusion: f32,
    normal: vec3<f32>,
    emission: vec3<f32>,
}

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

    var spec = mat._SpecularColor;
    if (kw_SPECULARMAP()) {
        spec = ptri::sample_rgba_biased(_SpecularMap, _SpecularMap_sampler, uvs, weights, mat._SpecularMap_LodBias);
    }
    let f0 = clamp(spec.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    let smoothness = clamp(spec.a, 0.0, 1.0);
    let roughness = psamp::roughness_from_smoothness(smoothness);

    var occlusion = 1.0;
    if (kw_OCCLUSION()) {
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

    return SurfaceData(c.rgb, c.a, f0, roughness, occlusion, n, emission.rgb);
}

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
) -> ptri::VertexOutput {
#ifdef MULTIVIEW
    return ptri::vertex_main(instance_index, view_idx, pos, n, kw_OBJECTSPACE());
#else
    return ptri::vertex_main(instance_index, 0u, pos, n, kw_OBJECTSPACE());
#endif
}

//#pass type=forward name=forward_transparent blend=transparent_material zwrite=material(off) cull=material(off) color_mask=material(rgba)
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
    let surface = psurf::specular_with_geometric_normal(
        s.base_color,
        s.alpha,
        s.f0,
        s.roughness,
        s.occlusion,
        s.normal,
        psamp::two_sided_geometric_normal(world_n, front_facing),
        s.emission,
    );
    return plight::shade_specular_transparent_clustered(
        frag_pos.xy,
        world_pos,
        view_layer,
        surface,
        plight::default_lighting_options(),
    );
}
