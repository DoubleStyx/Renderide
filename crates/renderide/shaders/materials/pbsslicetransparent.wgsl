//! Unity surface shader `Shader "PBSSliceTransparent"`: transparent metallic Standard lighting
//! with plane-based slicing and edge blending. Unlike `PBSSlice`, Unity's transparent variant does
//! not compile the alpha-clip keyword path; only slice-plane rejection discards fragments here.
//!
//! Froox variant bits populate `_RenderideVariantBits`; PBSSliceTransparent's nine keywords
//! (sorted alphabetically) occupy bits 0-8.


//#texture_default _MainTex white
//#texture_default _NormalMap bump
//#texture_default _EmissionMap black
//#texture_default _OcclusionMap white
//#texture_default _MetallicMap black
//#texture_default _DetailAlbedoMap gray
//#texture_default _DetailNormalMap bump

#import renderide::mesh::vertex as mv
#import renderide::material::variant_bits as vb
#import renderide::pbs::families::slice as pslice
#import renderide::pbs::lighting as plight
#import renderide::pbs::normal as pnorm
#import renderide::pbs::sampling as psamp
#import renderide::pbs::surface as psurf
#import renderide::core::uv as uvu
#import renderide::core::normal_decode as nd

struct PBSSliceTransparentMaterial {
    _Color: vec4<f32>,
    _EdgeColor: vec4<f32>,
    _EmissionColor: vec4<f32>,
    _EdgeEmissionColor: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _DetailAlbedoMap_ST: vec4<f32>,
    _DetailNormalMap_ST: vec4<f32>,
    _EdgeTransitionStart: f32,
    _EdgeTransitionEnd: f32,
    _NormalScale: f32,
    _DetailNormalMapScale: f32,
    _Glossiness: f32,
    _Metallic: f32,
    _RenderideVariantBits: u32,
    _pad0: f32,
    _Slicers: array<vec4<f32>, 8>,
}

const PBSSLICETRANSPARENT_KW_ALBEDOTEX: u32 = 1u << 0u;
const PBSSLICETRANSPARENT_KW_DETAIL_ALBEDOTEX: u32 = 1u << 1u;
const PBSSLICETRANSPARENT_KW_DETAIL_NORMALMAP: u32 = 1u << 2u;
const PBSSLICETRANSPARENT_KW_EMISSIONTEX: u32 = 1u << 3u;
const PBSSLICETRANSPARENT_KW_METALLICMAP: u32 = 1u << 4u;
const PBSSLICETRANSPARENT_KW_NORMALMAP: u32 = 1u << 5u;
const PBSSLICETRANSPARENT_KW_OCCLUSION: u32 = 1u << 6u;
const PBSSLICETRANSPARENT_KW_OBJECT_SPACE: u32 = 1u << 7u;
const PBSSLICETRANSPARENT_KW_WORLD_SPACE: u32 = 1u << 8u;

@group(1) @binding(0)  var<uniform> mat: PBSSliceTransparentMaterial;
@group(1) @binding(1)  var _MainTex: texture_2d<f32>;
@group(1) @binding(2)  var _MainTex_sampler: sampler;
@group(1) @binding(3)  var _NormalMap: texture_2d<f32>;
@group(1) @binding(4)  var _NormalMap_sampler: sampler;
@group(1) @binding(5)  var _EmissionMap: texture_2d<f32>;
@group(1) @binding(6)  var _EmissionMap_sampler: sampler;
@group(1) @binding(7)  var _OcclusionMap: texture_2d<f32>;
@group(1) @binding(8)  var _OcclusionMap_sampler: sampler;
@group(1) @binding(9)  var _MetallicMap: texture_2d<f32>;
@group(1) @binding(10) var _MetallicMap_sampler: sampler;
@group(1) @binding(11) var _DetailAlbedoMap: texture_2d<f32>;
@group(1) @binding(12) var _DetailAlbedoMap_sampler: sampler;
@group(1) @binding(13) var _DetailNormalMap: texture_2d<f32>;
@group(1) @binding(14) var _DetailNormalMap_sampler: sampler;

fn pbs_kw(mask: u32) -> bool {
    return vb::enabled(mat._RenderideVariantBits, mask);
}

fn sample_albedo_color(uv_main: vec2<f32>, edge_lerp: f32) -> vec4<f32> {
    let tint = mix(mat._Color, mat._EdgeColor, edge_lerp);
    if (pbs_kw(PBSSLICETRANSPARENT_KW_ALBEDOTEX) || pbs_kw(PBSSLICETRANSPARENT_KW_DETAIL_ALBEDOTEX)) {
        return textureSample(_MainTex, _MainTex_sampler, uv_main) * tint;
    }
    return tint;
}

fn sample_normal_world(
    uv_main: vec2<f32>,
    uv_detail: vec2<f32>,
    world_n: vec3<f32>,
    world_t: vec4<f32>,
    front_facing: bool,
) -> vec3<f32> {
    var n = world_n;
    let use_normal_map = pbs_kw(PBSSLICETRANSPARENT_KW_NORMALMAP) || pbs_kw(PBSSLICETRANSPARENT_KW_DETAIL_NORMALMAP);
    if (use_normal_map) {
        let tbn = pnorm::orthonormal_tbn(world_n, world_t);
        var ts = nd::decode_ts_normal_with_placeholder_sample(
            textureSample(_NormalMap, _NormalMap_sampler, uv_main),
            mat._NormalScale,
        );
        if (pbs_kw(PBSSLICETRANSPARENT_KW_DETAIL_NORMALMAP)) {
            let detail = nd::decode_ts_normal_with_placeholder_sample(
                textureSample(_DetailNormalMap, _DetailNormalMap_sampler, uv_detail),
                mat._DetailNormalMapScale,
            );
            ts = pslice::blend_detail_normal(ts, detail);
        }
        if (!front_facing) {
            ts = vec3<f32>(ts.x, ts.y, -ts.z);
        }
        return normalize(tbn * ts);
    }
    if (!front_facing) {
        n = -n;
    }
    return normalize(n);
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
) -> mv::WorldObjectVertexOutput {
#ifdef MULTIVIEW
    return mv::world_object_vertex_main(instance_index, view_idx, pos, n, t, uv0);
#else
    return mv::world_object_vertex_main(instance_index, 0u, pos, n, t, uv0);
#endif
}

//#pass forward_transparent
@fragment
fn fs_main(
    @builtin(position) frag_pos: vec4<f32>,
    @builtin(front_facing) front_facing: bool,
    @location(0) world_pos: vec3<f32>,
    @location(1) object_pos: vec3<f32>,
    @location(2) world_n: vec3<f32>,
    @location(3) world_t: vec4<f32>,
    @location(4) uv0: vec2<f32>,
    @location(5) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    let uv_main = uvu::apply_st(uv0, mat._MainTex_ST);
    let uv_detail_albedo = uvu::apply_st(uv0, mat._DetailAlbedoMap_ST);
    let uv_detail_normal = uvu::apply_st(uv0, mat._DetailNormalMap_ST);

    let slice_p = pslice::slice_position(
        world_pos,
        object_pos,
        pbs_kw(PBSSLICETRANSPARENT_KW_WORLD_SPACE),
        pbs_kw(PBSSLICETRANSPARENT_KW_OBJECT_SPACE),
    );
    let slice = pslice::evaluate_planes(
        mat._Slicers,
        slice_p,
        mat._EdgeTransitionStart,
        mat._EdgeTransitionEnd,
    );
    if (slice.min_distance < 0.0) {
        discard;
    }
    let edge_lerp = slice.edge_lerp;

    var c = sample_albedo_color(uv_main, edge_lerp);
    if (pbs_kw(PBSSLICETRANSPARENT_KW_DETAIL_ALBEDOTEX)) {
        let detail = textureSample(_DetailAlbedoMap, _DetailAlbedoMap_sampler, uv_detail_albedo).rgb * 2.0;
        c = vec4<f32>(c.rgb * detail, c.a);
    }

    let n = sample_normal_world(uv_main, uv_detail_normal, world_n, world_t, front_facing);

    var occlusion: f32 = 1.0;
    if (pbs_kw(PBSSLICETRANSPARENT_KW_OCCLUSION)) {
        occlusion = textureSample(_OcclusionMap, _OcclusionMap_sampler, uv_main).r;
    }

    var metallic = mat._Metallic;
    var smoothness = mat._Glossiness;
    if (pbs_kw(PBSSLICETRANSPARENT_KW_METALLICMAP)) {
        let m = textureSample(_MetallicMap, _MetallicMap_sampler, uv_main);
        metallic = m.r;
        smoothness = m.a;
    }
    metallic = clamp(metallic, 0.0, 1.0);
    let roughness = psamp::roughness_from_smoothness(clamp(smoothness, 0.0, 1.0));

    var emission = mat._EmissionColor.rgb;
    if (pbs_kw(PBSSLICETRANSPARENT_KW_EMISSIONTEX)) {
        emission = emission * textureSample(_EmissionMap, _EmissionMap_sampler, uv_main).rgb;
    }
    let edge_emission = mix(emission, mat._EdgeEmissionColor.rgb, edge_lerp);

    let surface = psurf::metallic(c.rgb, c.a, metallic, roughness, occlusion, n, edge_emission);
    return plight::shade_metallic_transparent_clustered(
        frag_pos.xy,
        world_pos,
        view_layer,
        surface,
        plight::default_lighting_options(),
    );
}
