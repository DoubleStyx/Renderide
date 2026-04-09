//! Unity PBS rim transparent (`Shader "PBSRimTransparent"`): same surface logic as `pbsrim`.
//!
//! The current renderer's mesh-forward pipeline is still opaque/fixed-state, so this shader only preserves
//! the material's alpha in its output; proper transparent blending depends on future pipeline-state work.

// unity-shader-name: PBSRimTransparent

#import renderide::globals as rg
#import renderide::per_draw as pd
#import renderide::pbs::brdf as brdf
#import renderide::pbs::cluster as pcls

struct PbsRimTransparentMaterial {
    _Color: vec4<f32>,
    _EmissionColor: vec4<f32>,
    _RimColor: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _Glossiness: f32,
    _Metallic: f32,
    _NormalScale: f32,
    _RimPower: f32,
    _SrcBlend: f32,
    _DstBlend: f32,
    _ZWrite: f32,
    _Cull: f32,
}

@group(1) @binding(0)  var<uniform> mat: PbsRimTransparentMaterial;
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
}

fn apply_st(uv: vec2<f32>, st: vec4<f32>) -> vec2<f32> {
    let uv_st = uv * st.xy + st.zw;
    return vec2<f32>(uv_st.x, 1.0 - uv_st.y);
}

fn decode_ts_normal(raw: vec3<f32>, scale: f32) -> vec3<f32> {
    if (all(raw > vec3<f32>(0.99, 0.99, 0.99))) {
        return vec3<f32>(0.0, 0.0, 1.0);
    }
    let nm_xy = (raw.xy * 2.0 - 1.0) * scale;
    let z = max(sqrt(max(1.0 - dot(nm_xy, nm_xy), 0.0)), 1e-6);
    return normalize(vec3<f32>(nm_xy, z));
}

fn sample_normal_world(uv_main: vec2<f32>, world_n: vec3<f32>) -> vec3<f32> {
    let tbn = brdf::orthonormal_tbn(world_n);
    let ts_n = decode_ts_normal(
        textureSample(_NormalMap, _NormalMap_sampler, uv_main).xyz,
        mat._NormalScale,
    );
    return normalize(tbn * ts_n);
}

fn metallic_roughness(uv: vec2<f32>) -> vec2<f32> {
    let mg = textureSample(_MetallicMap, _MetallicMap_sampler, uv);
    let metallic = clamp(mat._Metallic * mg.x, 0.0, 1.0);
    let smoothness = clamp(mat._Glossiness * mg.w, 0.0, 1.0);
    let roughness = clamp(1.0 - smoothness, 0.045, 1.0);
    return vec2<f32>(metallic, roughness);
}

@vertex
fn vs_main(
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
) -> VertexOutput {
    let world_p = pd::draw.model * vec4<f32>(pos.xyz, 1.0);
    let wn = normalize((pd::draw.model * vec4<f32>(n.xyz, 0.0)).xyz);
#ifdef MULTIVIEW
    var vp: mat4x4<f32>;
    if (view_idx == 0u) {
        vp = pd::draw.view_proj_left;
    } else {
        vp = pd::draw.view_proj_right;
    }
#else
    let vp = pd::draw.view_proj_left;
#endif
    var out: VertexOutput;
    out.clip_pos = vp * world_p;
    out.world_pos = world_p.xyz;
    out.world_n = wn;
    out.uv0 = uv0;
    return out;
}

@fragment
fn fs_main(
    @builtin(position) frag_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0: vec2<f32>,
) -> @location(0) vec4<f32> {
    let uv_main = apply_st(uv0, mat._MainTex_ST);

    let albedo_s = textureSample(_MainTex, _MainTex_sampler, uv_main);
    let base_color = mat._Color.xyz * albedo_s.xyz;
    let alpha = mat._Color.a * albedo_s.a;

    let mr = metallic_roughness(uv_main);
    let metallic = mr.x;
    let roughness = mr.y;

    let occ_s = textureSample(_OcclusionMap, _OcclusionMap_sampler, uv_main).x;
    let occlusion = occ_s;

    var n = normalize(world_n);
    n = sample_normal_world(uv_main, n);

    let emission = textureSample(_EmissionMap, _EmissionMap_sampler, uv_main).xyz * mat._EmissionColor.xyz;

    let cam = rg::frame.camera_world_pos.xyz;
    let v = normalize(cam - world_pos);
    let f0 = mix(vec3<f32>(0.04), base_color, metallic);

    let rim = pow(max(1.0 - clamp(dot(v, n), 0.0, 1.0), 0.0), max(mat._RimPower, 1e-4));
    let rim_emission = mat._RimColor.rgb * rim;

    let cluster_id = pcls::cluster_id_from_frag(
        frag_pos.xy,
        world_pos,
        rg::frame.view_space_z_coeffs,
        rg::frame.viewport_width,
        rg::frame.viewport_height,
        rg::frame.cluster_count_x,
        rg::frame.cluster_count_y,
        rg::frame.cluster_count_z,
        rg::frame.near_clip,
        rg::frame.far_clip,
    );

    let count = rg::cluster_light_counts[cluster_id];
    let base_idx = cluster_id * pcls::MAX_LIGHTS_PER_TILE;
    var lo = vec3<f32>(0.0);
    let i_max = min(count, pcls::MAX_LIGHTS_PER_TILE);
    for (var i = 0u; i < i_max; i++) {
        let li = rg::cluster_light_indices[base_idx + i];
        if (li >= rg::frame.light_count) {
            continue;
        }
        let light = rg::lights[li];
        lo = lo + brdf::direct_radiance_metallic(light, world_pos, n, v, roughness, metallic, base_color, f0);
    }

    let amb = vec3<f32>(0.03);
    let color = (amb * base_color * occlusion + lo * occlusion) + emission + rim_emission;
    return vec4<f32>(color, alpha);
}
