//! Unity surface shader `Shader "PBSSlice"`: metallic Standard lighting with plane-based slicing.
//!
//! Up to eight per-material `_Slicers[i]` planes (xyz = world/object-space plane normal,
//! w = signed offset) are evaluated in the fragment shader; fragments whose minimum plane distance
//! is negative are discarded. Within the `[_EdgeTransitionStart, _EdgeTransitionEnd]` band we blend
//! toward `_EdgeColor` and `_EdgeEmissionColor` to highlight the cut edge. Slice coordinates can be
//! driven from world space or object space via `_WORLD_SPACE` / `_OBJECT_SPACE` keyword floats
//! (world space is the default when neither is set).


#import renderide::math as rmath
#import renderide::mesh::vertex as mv
#import renderide::pbs::lighting as plight
#import renderide::pbs::normal as pnorm
#import renderide::pbs::sampling as psamp
#import renderide::pbs::surface as psurf
#import renderide::uv_utils as uvu
#import renderide::normal_decode as nd

struct PBSSliceMaterial {
    _Color: vec4<f32>,
    _EdgeColor: vec4<f32>,
    _EmissionColor: vec4<f32>,
    _EdgeEmissionColor: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _MainTex_StorageVInverted: f32,
    _DetailAlbedoMap_ST: vec4<f32>,
    _DetailNormalMap_ST: vec4<f32>,
    _EdgeTransitionStart: f32,
    _EdgeTransitionEnd: f32,
    _NormalScale: f32,
    _DetailNormalMapScale: f32,
    _Glossiness: f32,
    _Metallic: f32,
    _AlphaClip: f32,
    _WORLD_SPACE: f32,
    _OBJECT_SPACE: f32,
    _ALPHACLIP: f32,
    _ALBEDOTEX: f32,
    _DETAIL_ALBEDOTEX: f32,
    _NORMALMAP: f32,
    _DETAIL_NORMALMAP: f32,
    _EMISSIONTEX: f32,
    _METALLICMAP: f32,
    _OCCLUSION: f32,
    _pad0: f32,
    _Slicers: array<vec4<f32>, 8>,
}

@group(1) @binding(0)  var<uniform> mat: PBSSliceMaterial;
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

fn plane_distance(p: vec3<f32>, normal: vec3<f32>, offset: f32) -> f32 {
    return dot(p, normal) + offset;
}

fn slice_position(world_pos: vec3<f32>, object_pos: vec3<f32>) -> vec3<f32> {
    // Unity `#pragma multi_compile WORLD_SPACE OBJECT_SPACE` — default to world space when neither
    // keyword is set, honor explicit OBJECT_SPACE only when WORLD_SPACE is off.
    let use_world = uvu::kw_enabled(mat._WORLD_SPACE) || (!uvu::kw_enabled(mat._OBJECT_SPACE));
    return select(object_pos, world_pos, use_world);
}

fn blend_detail_normal(base_ts: vec3<f32>, detail_ts: vec3<f32>) -> vec3<f32> {
    return normalize(vec3<f32>(base_ts.xy + detail_ts.xy, base_ts.z * detail_ts.z));
}

fn sample_albedo_color(uv_main: vec2<f32>, edge_lerp: f32) -> vec4<f32> {
    let tint = mix(mat._Color, mat._EdgeColor, edge_lerp);
    if (uvu::kw_enabled(mat._ALBEDOTEX)) {
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
    let use_normal_map = uvu::kw_enabled(mat._NORMALMAP) || uvu::kw_enabled(mat._DETAIL_NORMALMAP);
    if (use_normal_map) {
        let tbn = pnorm::orthonormal_tbn(world_n, world_t);
        var ts = nd::decode_ts_normal_with_placeholder_sample(
            textureSample(_NormalMap, _NormalMap_sampler, uv_main),
            mat._NormalScale,
        );
        if (uvu::kw_enabled(mat._DETAIL_NORMALMAP)) {
            let detail = nd::decode_ts_normal_with_placeholder_sample(
                textureSample(_DetailNormalMap, _DetailNormalMap_sampler, uv_detail),
                mat._DetailNormalMapScale,
            );
            ts = blend_detail_normal(ts, detail);
        }
        // Unity surface shader path flips tangent-space Z on back faces.
        if (!front_facing) {
            ts = vec3<f32>(ts.x, ts.y, -ts.z);
        }
        return normalize(tbn * ts);
    }
    var n = normalize(world_n);
    if (!front_facing) {
        n = -n;
    }
    return n;
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

//#pass forward
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
    let uv_main = uvu::apply_st_for_storage(uv0, mat._MainTex_ST, mat._MainTex_StorageVInverted);
    let uv_detail_albedo = uvu::apply_st(uv0, mat._DetailAlbedoMap_ST);
    let uv_detail_normal = uvu::apply_st(uv0, mat._DetailNormalMap_ST);

    let slice_p = slice_position(world_pos, object_pos);
    var min_distance: f32 = 60000.0;
    for (var si: i32 = 0; si < 8; si = si + 1) {
        let slicer = mat._Slicers[si];
        if (all(slicer.xyz == vec3<f32>(0.0))) {
            break;
        }
        min_distance = min(min_distance, plane_distance(slice_p, slicer.xyz, slicer.w));
    }
    if (min_distance < 0.0) {
        discard;
    }
    let edge_lerp = 1.0 - rmath::safe_lerp_factor(mat._EdgeTransitionStart, mat._EdgeTransitionEnd, min_distance);

    var c = sample_albedo_color(uv_main, edge_lerp);
    if (uvu::kw_enabled(mat._DETAIL_ALBEDOTEX)) {
        let detail = textureSample(_DetailAlbedoMap, _DetailAlbedoMap_sampler, uv_detail_albedo).rgb * 2.0;
        c = vec4<f32>(c.rgb * detail, c.a);
    }

    if (uvu::kw_enabled(mat._ALPHACLIP) && c.a <= mat._AlphaClip) {
        discard;
    }

    let base_color = c.rgb;
    let alpha = c.a;
    let n = sample_normal_world(uv_main, uv_detail_normal, world_n, world_t, front_facing);

    var occlusion: f32 = 1.0;
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
    let edge_emission = mix(emission, mat._EdgeEmissionColor.rgb, edge_lerp);

    let surface = psurf::metallic(base_color, alpha, metallic, roughness, occlusion, n, edge_emission);
    let color = plight::shade_metallic_clustered(
        frag_pos.xy,
        world_pos,
        view_layer,
        surface,
        plight::default_lighting_options(),
    );
    return vec4<f32>(color, alpha);
}
