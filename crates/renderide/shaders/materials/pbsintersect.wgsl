//! Unity shader `Shader "PBSIntersect"`: transparent metallic Standard lighting with scene-depth
//! driven intersection tint/emission (`_BeginTransition*` / `_EndTransition*` band).
//!
//! Metallic-workflow counterpart of [`pbsintersectspecular`]. Depth is sampled from the opaque
//! scene-depth snapshot bound at `@group(0)` by the intersection subpass -- see
//! [`crate::backend::frame_gpu::FrameGpuResources::copy_scene_depth_snapshot`].
//!
//! Froox variant bits populate `_RenderideVariantBits`; this shader decodes PBSIntersect's
//! shader-specific keyword bits locally.

//#texture_default _MainTex white
//#texture_default _NormalMap bump
//#texture_default _EmissionMap black
//#texture_default _OcclusionMap white
//#texture_default _MetallicMap black
//#mat_default _Color vec4 1.0 1.0 1.0 1.0
//#mat_default _EndTransitionEnd float 0.1
//#mat_default _EndTransitionStart float 0.1
//#mat_default _IntersectColor vec4 1.0 1.0 1.0 1.0
//#mat_default _IntersectEmissionColor vec4 1.0 0.0 0.0 1.0
//#mat_default _NormalScale float 1.0

#import renderide::core::math as rmath
#import renderide::material::variant_bits as vb
#import renderide::mesh::vertex as mv
#import renderide::pbs::lighting as plight
#import renderide::pbs::normal as pnorm
#import renderide::pbs::sampling as psamp
#import renderide::pbs::surface as psurf
#import renderide::frame::scene_depth_sample as sds
#import renderide::core::uv as uvu
#import renderide::core::normal_decode as nd

struct PbsIntersectMaterial {
    _Color: vec4<f32>,
    _IntersectColor: vec4<f32>,
    _IntersectEmissionColor: vec4<f32>,
    _EmissionColor: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _BeginTransitionStart: f32,
    _BeginTransitionEnd: f32,
    _EndTransitionStart: f32,
    _EndTransitionEnd: f32,
    _NormalScale: f32,
    _Glossiness: f32,
    _Metallic: f32,
    _RenderideVariantBits: u32,
}

const PBSINTERSECT_KW_ALBEDOTEX: u32 = 1u << 0u;
const PBSINTERSECT_KW_EMISSIONTEX: u32 = 1u << 1u;
const PBSINTERSECT_KW_METALLICMAP: u32 = 1u << 2u;
const PBSINTERSECT_KW_NORMALMAP: u32 = 1u << 3u;
const PBSINTERSECT_KW_OCCLUSION: u32 = 1u << 4u;

@group(1) @binding(0)  var<uniform> mat: PbsIntersectMaterial;
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

fn pbs_kw(mask: u32) -> bool {
    return vb::enabled(mat._RenderideVariantBits, mask);
}

fn sample_normal_world(uv_main: vec2<f32>, world_n: vec3<f32>, world_t: vec4<f32>, front_facing: bool) -> vec3<f32> {
    var n = world_n;
    if (pbs_kw(PBSINTERSECT_KW_NORMALMAP)) {
        let tbn = pnorm::orthonormal_tbn(n, world_t);
        var ts_n = nd::decode_ts_normal_with_placeholder_sample(
            textureSample(_NormalMap, _NormalMap_sampler, uv_main),
            mat._NormalScale,
        );
        if (!front_facing) {
            ts_n = vec3<f32>(ts_n.x, ts_n.y, -ts_n.z);
        }
        return normalize(tbn * ts_n);
    }
    if (!front_facing) {
        n = -n;
    }
    return n;
}

fn intersection_lerp(frag_pos: vec4<f32>, world_pos: vec3<f32>, view_layer: u32) -> f32 {
    let diff = sds::scene_linear_depth(frag_pos, view_layer) - sds::fragment_linear_depth(world_pos, view_layer);
    if (diff < mat._EndTransitionStart) {
        return rmath::safe_linear_factor(mat._BeginTransitionStart, mat._BeginTransitionEnd, diff);
    }
    return 1.0 - rmath::safe_linear_factor(mat._EndTransitionStart, mat._EndTransitionEnd, diff);
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

//#pass forward_transparent
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
    let intersect_lerp = intersection_lerp(frag_pos, world_pos, view_layer);

    var c0 = mix(mat._Color, mat._IntersectColor, intersect_lerp);
    if (pbs_kw(PBSINTERSECT_KW_ALBEDOTEX)) {
        c0 = c0 * textureSample(_MainTex, _MainTex_sampler, uv_main);
    }
    let base_color = c0.rgb;
    let alpha = c0.a;

    let n = sample_normal_world(uv_main, world_n, world_t, front_facing);

    var occlusion = 1.0;
    if (pbs_kw(PBSINTERSECT_KW_OCCLUSION)) {
        occlusion = textureSample(_OcclusionMap, _OcclusionMap_sampler, uv_main).r;
    }

    var metallic = mat._Metallic;
    var smoothness = mat._Glossiness;
    if (pbs_kw(PBSINTERSECT_KW_METALLICMAP)) {
        let m = textureSample(_MetallicMap, _MetallicMap_sampler, uv_main);
        metallic = m.r;
        smoothness = m.a;
    }
    metallic = clamp(metallic, 0.0, 1.0);
    smoothness = clamp(smoothness, 0.0, 1.0);
    let roughness = psamp::roughness_from_smoothness(smoothness);

    var emission = mat._EmissionColor.rgb;
    if (pbs_kw(PBSINTERSECT_KW_EMISSIONTEX)) {
        emission = emission * textureSample(_EmissionMap, _EmissionMap_sampler, uv_main).rgb;
    }
    emission = emission + mat._IntersectEmissionColor.rgb * intersect_lerp;

    let surface = psurf::metallic_with_geometric_normal(
        base_color,
        alpha,
        metallic,
        roughness,
        occlusion,
        n,
        psamp::two_sided_geometric_normal(world_n, front_facing),
        emission,
    );
    return plight::shade_metallic_transparent_clustered(
        frag_pos.xy,
        world_pos,
        view_layer,
        surface,
        plight::default_lighting_options(),
    );
}
