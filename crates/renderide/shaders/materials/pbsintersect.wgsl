//! Unity shader `Shader "PBSIntersect"`: transparent metallic Standard lighting with scene-depth
//! driven intersection tint/emission (`_BeginTransition*` / `_EndTransition*` band).
//!
//! Metallic-workflow counterpart of [`pbsintersectspecular`]. Depth is sampled from the opaque
//! scene-depth snapshot bound at `@group(0)` by the intersection subpass — see
//! [`crate::backend::frame_gpu::FrameGpuResources::copy_scene_depth_snapshot`].

#import renderide::math as rmath
#import renderide::mesh::vertex as mv
#import renderide::pbs::lighting as plight
#import renderide::pbs::normal as pnorm
#import renderide::pbs::sampling as psamp
#import renderide::pbs::surface as psurf
#import renderide::scene_depth_sample as sds
#import renderide::uv_utils as uvu
#import renderide::normal_decode as nd

struct PbsIntersectMaterial {
    _Color: vec4<f32>,
    _IntersectColor: vec4<f32>,
    _IntersectEmissionColor: vec4<f32>,
    _EmissionColor: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _MainTex_StorageVInverted: f32,
    _BeginTransitionStart: f32,
    _BeginTransitionEnd: f32,
    _EndTransitionStart: f32,
    _EndTransitionEnd: f32,
    _NormalScale: f32,
    _Glossiness: f32,
    _Metallic: f32,
    _ALBEDOTEX: f32,
    _EMISSIONTEX: f32,
    _NORMALMAP: f32,
    _METALLICMAP: f32,
    _OCCLUSION: f32,
    _pad0: f32,
    _pad1: f32,
}

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

fn sample_normal_world(uv_main: vec2<f32>, world_n: vec3<f32>, world_t: vec4<f32>, front_facing: bool) -> vec3<f32> {
    var n = world_n;
    if (uvu::kw_enabled(mat._NORMALMAP)) {
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

//#pass forward
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
    let uv_main = uvu::apply_st_for_storage(uv0, mat._MainTex_ST, mat._MainTex_StorageVInverted);
    let intersect_lerp = intersection_lerp(frag_pos, world_pos, view_layer);

    var c0 = mix(mat._Color, mat._IntersectColor, intersect_lerp);
    if (uvu::kw_enabled(mat._ALBEDOTEX)) {
        c0 = c0 * textureSample(_MainTex, _MainTex_sampler, uv_main);
    }
    let base_color = c0.rgb;
    let alpha = c0.a;

    let n = sample_normal_world(uv_main, world_n, world_t, front_facing);

    var occlusion = 1.0;
    if (uvu::kw_enabled(mat._OCCLUSION)) {
        occlusion = textureSample(_OcclusionMap, _OcclusionMap_sampler, uv_main).r;
    }

    var metallic = mat._Metallic;
    var smoothness = mat._Glossiness;
    if (uvu::kw_enabled(mat._METALLICMAP)) {
        let m = textureSample(_MetallicMap, _MetallicMap_sampler, uv_main);
        metallic = m.r;
        smoothness = m.a;
    }
    metallic = clamp(metallic, 0.0, 1.0);
    smoothness = clamp(smoothness, 0.0, 1.0);
    let roughness = psamp::roughness_from_smoothness(smoothness);

    var emission = mat._EmissionColor.rgb;
    if (uvu::kw_enabled(mat._EMISSIONTEX)) {
        emission = emission * textureSample(_EmissionMap, _EmissionMap_sampler, uv_main).rgb;
    }
    emission = emission + mat._IntersectEmissionColor.rgb * intersect_lerp;

    let surface = psurf::metallic(base_color, alpha, metallic, roughness, occlusion, n, emission);
    let color = plight::shade_metallic_clustered(
        frag_pos.xy,
        world_pos,
        view_layer,
        surface,
        plight::default_lighting_options(),
    );
    return vec4<f32>(color, alpha);
}
