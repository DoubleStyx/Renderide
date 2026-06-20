//! Unity PBS rim transparent specular (`Shader "PBSRimTransparentSpecular"`): same surface logic
//! as [`pbsrimspecular`](super::pbsrimspecular).
//!
//! Transparent blend and depth-write state is driven by the host's material properties; the forward
//! pass is back-culled to match Unity's rim-transparent material behavior.
//!
//! Variant metadata never enables `_ALBEDOTEX`, so the albedo branch is unreachable in this
//! material. `_Color` is the only base color and `_MainTex` is not bound.
//!
//! Froox variant bits populate `_RenderideVariantBits`; this shader decodes
//! PBSRimTransparentSpecular's shader-specific keyword bits locally.


//#render_queue Transparent
//#texture_default _NormalMap bump
//#texture_default _EmissionMap black
//#texture_default _OcclusionMap white
//#texture_default _SpecularMap white
//#mat_default _Color vec4 1.0 1.0 1.0 1.0
//#mat_default _NormalScale float 1.0
//#mat_default _RimColor vec4 1.0 0.0 0.0 1.0
//#mat_default _SpecularColor vec4 1.0 1.0 1.0 0.5
//#mat_default _RimPower float 3.0

#import renderide::frame::globals as rg
#import renderide::material::fresnel as mf
#import renderide::material::variant_bits as vb
#import renderide::mesh::vertex as mv
#import renderide::pbs::lighting as plight
#import renderide::pbs::sampling as psamp
#import renderide::pbs::surface as psurf
#import renderide::core::uv as uvu
#import renderide::core::texture_sampling as ts

struct PbsRimTransparentSpecularMaterial {
    _Color: vec4<f32>,
    _SpecularColor: vec4<f32>,
    _EmissionColor: vec4<f32>,
    _RimColor: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _NormalScale: f32,
    _RimPower: f32,
    _RenderideVariantBits: u32,
    _NormalMap_LodBias: f32,
    _EmissionMap_LodBias: f32,
    _OcclusionMap_LodBias: f32,
    _SpecularMap_LodBias: f32,
}

const PBSRIMTRANSPARENTSPECULAR_KW_EMISSIONTEX: u32 = 1u << 0u;
const PBSRIMTRANSPARENTSPECULAR_KW_NORMALMAP: u32 = 1u << 1u;
const PBSRIMTRANSPARENTSPECULAR_KW_OCCLUSION: u32 = 1u << 2u;
const PBSRIMTRANSPARENTSPECULAR_KW_SPECULARMAP: u32 = 1u << 3u;

@group(1) @binding(0) var<uniform> mat: PbsRimTransparentSpecularMaterial;
@group(1) @binding(1) var _NormalMap: texture_2d<f32>;
@group(1) @binding(2) var _NormalMap_sampler: sampler;
@group(1) @binding(3) var _EmissionMap: texture_2d<f32>;
@group(1) @binding(4) var _EmissionMap_sampler: sampler;
@group(1) @binding(5) var _OcclusionMap: texture_2d<f32>;
@group(1) @binding(6) var _OcclusionMap_sampler: sampler;
@group(1) @binding(7) var _SpecularMap: texture_2d<f32>;
@group(1) @binding(8) var _SpecularMap_sampler: sampler;

fn pbs_kw(mask: u32) -> bool {
    return vb::enabled(mat._RenderideVariantBits, mask);
}

fn sample_normal_world(uv_main: vec2<f32>, world_n: vec3<f32>, world_t: vec4<f32>) -> vec3<f32> {
    return psamp::sample_optional_world_normal(
        pbs_kw(PBSRIMTRANSPARENTSPECULAR_KW_NORMALMAP),
        _NormalMap,
        _NormalMap_sampler,
        uv_main,
        mat._NormalMap_LodBias,
        mat._NormalScale,
        world_n,
        world_t,
    );
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
    @location(4) t: vec4<f32>,
) -> mv::WorldVertexOutput {
#ifdef MULTIVIEW
    return mv::world_vertex_main(instance_index, view_idx, pos, n, t, uv0);
#else
    return mv::world_vertex_main(instance_index, 0u, pos, n, t, uv0);
#endif
}

//#pass type=forward name=forward_transparent_cull_back blend=transparent_material zwrite=material(off) cull=back color_mask=material(rgba)
@fragment
fn fs_main(
    @builtin(position) frag_pos: vec4<f32>,
    @builtin(front_facing) front_facing: bool,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) world_t: vec4<f32>,
    @location(3) uv0: vec2<f32>,
    @location(4) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    let uv_main = uvu::apply_st(uv0, mat._MainTex_ST);

    let base_color = mat._Color.rgb;
    let alpha = mat._Color.a;

    var spec = mat._SpecularColor;
    if (pbs_kw(PBSRIMTRANSPARENTSPECULAR_KW_SPECULARMAP)) {
        spec = ts::sample_tex_2d(_SpecularMap, _SpecularMap_sampler, uv_main, mat._SpecularMap_LodBias);
    }
    let f0 = clamp(spec.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    let smoothness = clamp(spec.a, 0.0, 1.0);
    let roughness = psamp::roughness_from_smoothness(smoothness);

    var occlusion = 1.0;
    if (pbs_kw(PBSRIMTRANSPARENTSPECULAR_KW_OCCLUSION)) {
        occlusion = ts::sample_tex_2d(_OcclusionMap, _OcclusionMap_sampler, uv_main, mat._OcclusionMap_LodBias).r;
    }

    var n = sample_normal_world(uv_main, world_n, world_t);
    if (!front_facing) {
        n = -n;
    }

    var emission = mat._EmissionColor.rgb;
    if (pbs_kw(PBSRIMTRANSPARENTSPECULAR_KW_EMISSIONTEX)) {
        emission = emission * ts::sample_tex_2d(_EmissionMap, _EmissionMap_sampler, uv_main, mat._EmissionMap_LodBias).rgb;
    }

    let view_dir = rg::view_dir_for_world_pos(world_pos, view_layer);
    let rim = mf::rim_factor(n, view_dir, mat._RimPower);
    let rim_emission = mat._RimColor.rgb * rim;
    let surface = psurf::specular_with_geometric_normal(
        base_color,
        alpha,
        f0,
        roughness,
        occlusion,
        n,
        psamp::two_sided_geometric_normal(world_n, front_facing),
        emission + rim_emission,
    );
    return plight::shade_specular_transparent_clustered(
        frag_pos.xy,
        world_pos,
        view_layer,
        surface,
        plight::default_lighting_options(),
    );
}
