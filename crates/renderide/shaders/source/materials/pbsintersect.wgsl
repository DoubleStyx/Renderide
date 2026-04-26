//! Unity shader `Shader "PBSIntersect"`: transparent metallic Standard lighting with scene-depth
//! driven intersection tint/emission (`_BeginTransition*` / `_EndTransition*` band).
//!
//! Metallic-workflow counterpart of [`pbsintersectspecular`]. Depth is sampled from the opaque
//! scene-depth snapshot bound at `@group(0)` by the intersection subpass — see
//! [`crate::backend::frame_gpu::FrameGpuResources::copy_scene_depth_snapshot`].

// unity-shader-name: PBSIntersect

#import renderide::globals as rg
#import renderide::per_draw as pd
#import renderide::pbs::brdf as brdf
#import renderide::pbs::cluster as pcls
#import renderide::uv_utils as uvu
#import renderide::normal_decode as nd

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

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) @interpolate(flat) view_layer: u32,
}

fn sample_normal_world(uv_main: vec2<f32>, world_n: vec3<f32>, front_facing: bool) -> vec3<f32> {
    var n = normalize(world_n);
    if (uvu::kw_enabled(mat._NORMALMAP)) {
        let tbn = brdf::orthonormal_tbn(n);
        var ts_n = nd::decode_ts_normal_with_placeholder(
            textureSample(_NormalMap, _NormalMap_sampler, uv_main).xyz,
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

fn safe_linear_factor(a: f32, b: f32, value: f32) -> f32 {
    let denom = b - a;
    if (abs(denom) < 1e-6) {
        return select(0.0, 1.0, value >= b);
    }
    return clamp((value - a) / denom, 0.0, 1.0);
}

fn scene_linear_depth(frag_pos: vec4<f32>, view_layer: u32) -> f32 {
    let max_xy = vec2<i32>(
        i32(rg::frame.viewport_width) - 1,
        i32(rg::frame.viewport_height) - 1,
    );
    let xy = clamp(vec2<i32>(frag_pos.xy), vec2<i32>(0, 0), max_xy);
#ifdef MULTIVIEW
    let raw_depth = textureLoad(rg::scene_depth_array, xy, i32(view_layer), 0);
#else
    let raw_depth = textureLoad(rg::scene_depth, xy, 0);
#endif
    let denom = max(
        raw_depth * (rg::frame.far_clip - rg::frame.near_clip) + rg::frame.near_clip,
        1e-6,
    );
    return (rg::frame.near_clip * rg::frame.far_clip) / denom;
}

fn fragment_linear_depth(world_pos: vec3<f32>, view_layer: u32) -> f32 {
    let z_coeffs = select(rg::frame.view_space_z_coeffs, rg::frame.view_space_z_coeffs_right, view_layer != 0u);
    let view_z = dot(z_coeffs.xyz, world_pos) + z_coeffs.w;
    return -view_z;
}

fn intersection_lerp(frag_pos: vec4<f32>, world_pos: vec3<f32>, view_layer: u32) -> f32 {
    let diff = scene_linear_depth(frag_pos, view_layer) - fragment_linear_depth(world_pos, view_layer);
    if (diff < mat._EndTransitionStart) {
        return safe_linear_factor(mat._BeginTransitionStart, mat._BeginTransitionEnd, diff);
    }
    return 1.0 - safe_linear_factor(mat._EndTransitionStart, mat._EndTransitionEnd, diff);
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
#ifdef MULTIVIEW
    out.view_layer = view_idx;
#else
    out.view_layer = 0u;
#endif
    return out;
}

//#material forward
@fragment
fn fs_main(
    @builtin(position) frag_pos: vec4<f32>,
    @builtin(front_facing) front_facing: bool,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    let uv_main = uvu::apply_st(uv0, mat._MainTex_ST);
    let intersect_lerp = intersection_lerp(frag_pos, world_pos, view_layer);

    var c0 = mix(mat._Color, mat._IntersectColor, intersect_lerp);
    if (uvu::kw_enabled(mat._ALBEDOTEX)) {
        c0 = c0 * textureSample(_MainTex, _MainTex_sampler, uv_main);
    }
    let base_color = c0.rgb;
    let alpha = c0.a;

    let n = sample_normal_world(uv_main, world_n, front_facing);

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
    let roughness = clamp(1.0 - smoothness, 0.045, 1.0);
    let f0 = mix(vec3<f32>(0.04), base_color, metallic);

    var emission = mat._EmissionColor.rgb;
    if (uvu::kw_enabled(mat._EMISSIONTEX)) {
        emission = emission * textureSample(_EmissionMap, _EmissionMap_sampler, uv_main).rgb;
    }
    emission = emission + mat._IntersectEmissionColor.rgb * intersect_lerp;

    let cam = rg::frame.camera_world_pos.xyz;
    let v = normalize(cam - world_pos);

    let cluster_id = pcls::cluster_id_from_frag(
        frag_pos.xy,
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
    var lo = vec3<f32>(0.0);
    let i_max = min(count, pcls::MAX_LIGHTS_PER_TILE);
    for (var i = 0u; i < i_max; i++) {
        let li = pcls::cluster_light_index_at(cluster_id, i);
        if (li >= rg::frame.light_count) {
            continue;
        }
        let light = rg::lights[li];
        lo = lo + brdf::direct_radiance_metallic(
            light,
            world_pos,
            n,
            v,
            roughness,
            metallic,
            base_color,
            f0,
        );
    }

    let amb = vec3<f32>(0.03);
    let color = (amb * base_color * occlusion + lo * occlusion) + emission;
    return vec4<f32>(color, alpha);
}
