//! Unity surface shader `Shader "Custom/Nosamplers"`: metallic Standard lighting that demos
//! Unity's `UNITY_DECLARE_TEX2D_NOSAMPLER` aliasing — `_MetallicMap` shares the `_Albedo` sampler
//! and `_EmissionMap`/`_EmissionMap1` are sampled with their own. In WGSL we declare a separate
//! sampler per texture (binding-count cost is minor and well within wgpu limits); the renderer
//! routes whatever sampler the host supplies, so the visual result matches.

// unity-shader-name: Custom/Nosamplers
// unity-shader-name: Nosamplers

#import renderide::globals as rg
#import renderide::sh2_ambient as shamb
#import renderide::per_draw as pd
#import renderide::pbs::brdf as brdf
#import renderide::pbs::cluster as pcls
#import renderide::uv_utils as uvu

struct NosamplersMaterial {
    _Color: vec4<f32>,
    _Albedo_ST: vec4<f32>,
    _Albedo_StorageVInverted: f32,
    _Glossiness: f32,
    _Metallic: f32,
}

@group(1) @binding(0) var<uniform> mat: NosamplersMaterial;
@group(1) @binding(1) var _Albedo: texture_2d<f32>;
@group(1) @binding(2) var _Albedo_sampler: sampler;
@group(1) @binding(3) var _Albedo1: texture_2d<f32>;
@group(1) @binding(4) var _Albedo1_sampler: sampler;
@group(1) @binding(5) var _Albedo2: texture_2d<f32>;
@group(1) @binding(6) var _Albedo2_sampler: sampler;
@group(1) @binding(7) var _Albedo3: texture_2d<f32>;
@group(1) @binding(8) var _Albedo3_sampler: sampler;
@group(1) @binding(9) var _MetallicMap: texture_2d<f32>;
@group(1) @binding(10) var _MetallicMap_sampler: sampler;
@group(1) @binding(11) var _EmissionMap: texture_2d<f32>;
@group(1) @binding(12) var _EmissionMap_sampler: sampler;
@group(1) @binding(13) var _EmissionMap1: texture_2d<f32>;
@group(1) @binding(14) var _EmissionMap1_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) @interpolate(flat) view_layer: u32,
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
    out.uv = uvu::apply_st_for_storage(uv0, mat._Albedo_ST, mat._Albedo_StorageVInverted);
#ifdef MULTIVIEW
    out.view_layer = view_idx;
#else
    out.view_layer = 0u;
#endif
    return out;
}

fn shade(
    frag_xy: vec2<f32>,
    world_pos: vec3<f32>,
    world_n: vec3<f32>,
    uv: vec2<f32>,
    view_layer: u32,
    include_directional: bool,
    include_local: bool,
) -> vec4<f32> {
    let c = textureSample(_Albedo, _Albedo_sampler, uv) * mat._Color;
    // _Albedo1..3 are sampled to keep the bindings live (the Unity source declares them but only
    // multiplies into the final emission by way of host-driven uniforms; we conservatively touch).
    let touch = (textureSample(_Albedo1, _Albedo1_sampler, uv).r
        + textureSample(_Albedo2, _Albedo2_sampler, uv).r
        + textureSample(_Albedo3, _Albedo3_sampler, uv).r) * 0.0;

    let m = textureSample(_MetallicMap, _MetallicMap_sampler, uv);
    let e0 = textureSample(_EmissionMap, _EmissionMap_sampler, uv).rgb;
    let e1 = textureSample(_EmissionMap1, _EmissionMap1_sampler, uv).rgb;
    let emission = mix(e0, e1, 0.5);

    let base_color = c.rgb + vec3<f32>(touch);
    let metallic = clamp(m.r, 0.0, 1.0);
    let smoothness = clamp(m.a, 0.0, 1.0);
    let roughness = clamp(1.0 - smoothness, 0.045, 1.0);
    let n = normalize(world_n);
    let f0 = mix(vec3<f32>(0.04), base_color, metallic);
    let cam = rg::frame.camera_world_pos.xyz;
    let v = normalize(cam - world_pos);

    let aa_roughness = brdf::filter_perceptual_roughness(roughness, n);

    let cluster_id = pcls::cluster_id_from_frag(
        frag_xy, world_pos, rg::frame.view_space_z_coeffs, rg::frame.view_space_z_coeffs_right,
        view_layer, rg::frame.viewport_width, rg::frame.viewport_height,
        rg::frame.cluster_count_x, rg::frame.cluster_count_y, rg::frame.cluster_count_z,
        rg::frame.near_clip, rg::frame.far_clip,
    );
    let count = pcls::cluster_light_count_at(cluster_id);
    let i_max = min(count, pcls::MAX_LIGHTS_PER_TILE);
    var lo = vec3<f32>(0.0);
    for (var i = 0u; i < i_max; i++) {
        let li = pcls::cluster_light_index_at(cluster_id, i);
        if (li >= rg::frame.light_count) {
            continue;
        }
        let light = rg::lights[li];
        let is_directional = light.light_type == 1u;
        if ((is_directional && !include_directional) || (!is_directional && !include_local)) {
            continue;
        }
        lo = lo + brdf::direct_radiance_metallic(
            light, world_pos, n, v, aa_roughness, metallic, base_color, f0,
        );
    }
    let ambient = select(vec3<f32>(0.0), shamb::ambient_probe(n) * base_color, include_directional);
    let extra = select(vec3<f32>(0.0), emission, include_directional);
    return vec4<f32>(ambient + lo + extra, c.a);
}

//#pass forward
@fragment
fn fs_forward_base(
    @builtin(position) frag_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    return shade(frag_pos.xy, world_pos, world_n, uv, view_layer, true, true);
}
