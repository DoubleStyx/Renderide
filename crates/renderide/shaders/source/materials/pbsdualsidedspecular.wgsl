//! Unity surface shader `Shader "PBSDualSidedSpecular"`: specular Standard lighting with two-sided normals.
//!
//! Unity's `#pragma surface surf StandardSpecular fullforwardshadows addshadow` generates forward
//! base, forward additive, and shadow caster passes. This renderer has a forward color path here, so
//! the shader declares the forward base + forward additive passes and keeps culling disabled.

// unity-shader-name: PBSDualSidedSpecular

#import renderide::globals as rg
#import renderide::per_draw as pd
#import renderide::pbs::brdf as brdf
#import renderide::pbs::cluster as pcls
#import renderide::alpha_clip_sample as acs
#import renderide::uv_utils as uvu
#import renderide::normal_decode as nd

struct PbsDualSidedSpecularMaterial {
    _Color: vec4<f32>,
    _EmissionColor: vec4<f32>,
    _SpecularColor: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _NormalScale: f32,
    _AlphaClip: f32,
    _ALPHACLIP: f32,
    _ALBEDOTEX: f32,
    _EMISSIONTEX: f32,
    _NORMALMAP: f32,
    _SPECULARMAP: f32,
    _OCCLUSION: f32,
    VCOLOR_ALBEDO: f32,
    VCOLOR_EMIT: f32,
    VCOLOR_SPECULAR: f32,
}

@group(1) @binding(0)  var<uniform> mat: PbsDualSidedSpecularMaterial;
@group(1) @binding(1)  var _MainTex: texture_2d<f32>;
@group(1) @binding(2)  var _MainTex_sampler: sampler;
@group(1) @binding(3)  var _NormalMap: texture_2d<f32>;
@group(1) @binding(4)  var _NormalMap_sampler: sampler;
@group(1) @binding(5)  var _EmissionMap: texture_2d<f32>;
@group(1) @binding(6)  var _EmissionMap_sampler: sampler;
@group(1) @binding(7)  var _OcclusionMap: texture_2d<f32>;
@group(1) @binding(8)  var _OcclusionMap_sampler: sampler;
@group(1) @binding(9)  var _SpecularMap: texture_2d<f32>;
@group(1) @binding(10) var _SpecularMap_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) @interpolate(flat) view_layer: u32,
}

struct SurfaceData {
    base_color: vec3<f32>,
    alpha: f32,
    f0: vec3<f32>,
    roughness: f32,
    one_minus_reflectivity: f32,
    occlusion: f32,
    normal: vec3<f32>,
    emission: vec3<f32>,
}

fn sample_normal_world(uv_main: vec2<f32>, world_n: vec3<f32>, front_facing: bool) -> vec3<f32> {
    let tbn = brdf::orthonormal_tbn(normalize(world_n));
    var ts_n = vec3<f32>(0.0, 0.0, 1.0);
    if (uvu::kw_enabled(mat._NORMALMAP)) {
        ts_n = nd::decode_ts_normal_with_placeholder(
            textureSample(_NormalMap, _NormalMap_sampler, uv_main).xyz,
            mat._NormalScale,
        );
    }
    if (!front_facing) {
        ts_n.z = -ts_n.z;
    }
    return normalize(tbn * ts_n);
}

fn sample_surface(uv0: vec2<f32>, world_n: vec3<f32>, front_facing: bool) -> SurfaceData {
    let uv_main = uvu::apply_st(uv0, mat._MainTex_ST);

    var albedo = mat._Color;
    if (uvu::kw_enabled(mat._ALBEDOTEX)) {
        albedo = albedo * textureSample(_MainTex, _MainTex_sampler, uv_main);
    }
    let clip_alpha = select(
        albedo.a,
        mat._Color.a * acs::texture_alpha_base_mip(_MainTex, _MainTex_sampler, uv_main),
        uvu::kw_enabled(mat._ALBEDOTEX),
    );
    if (uvu::kw_enabled(mat._ALPHACLIP) && clip_alpha <= mat._AlphaClip) {
        discard;
    }

    var spec = mat._SpecularColor;
    if (uvu::kw_enabled(mat._SPECULARMAP)) {
        spec = textureSample(_SpecularMap, _SpecularMap_sampler, uv_main);
    }
    let f0 = clamp(spec.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    let smoothness = clamp(spec.a, 0.0, 1.0);
    let roughness = clamp(1.0 - smoothness, 0.045, 1.0);
    let one_minus_reflectivity = 1.0 - max(max(f0.r, f0.g), f0.b);

    var occlusion = 1.0;
    if (uvu::kw_enabled(mat._OCCLUSION)) {
        occlusion = textureSample(_OcclusionMap, _OcclusionMap_sampler, uv_main).r;
    }

    let emission_color = mat._EmissionColor.rgb;
    var emission = vec3<f32>(0.0);
    if (dot(emission_color, emission_color) > 1e-8) {
        emission = emission_color;
        if (uvu::kw_enabled(mat._EMISSIONTEX)) {
            emission = emission * textureSample(_EmissionMap, _EmissionMap_sampler, uv_main).rgb;
        }
    }

    return SurfaceData(
        albedo.rgb,
        albedo.a,
        f0,
        roughness,
        one_minus_reflectivity,
        occlusion,
        sample_normal_world(uv_main, world_n, front_facing),
        emission,
    );
}

fn clustered_direct_lighting(
    frag_xy: vec2<f32>,
    world_pos: vec3<f32>,
    view_layer: u32,
    s: SurfaceData,
    include_directional: bool,
    include_local: bool,
) -> vec3<f32> {
    let cam = rg::frame.camera_world_pos.xyz;
    let v = normalize(cam - world_pos);

    let cluster_id = pcls::cluster_id_from_frag(
        frag_xy,
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
        lo = lo + brdf::direct_radiance_specular(
            light,
            world_pos,
            s.normal,
            v,
            s.roughness,
            s.base_color,
            s.f0,
            s.one_minus_reflectivity,
        );
    }
    return lo * s.occlusion;
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
    let wn = normalize((d.model * vec4<f32>(n.xyz, 0.0)).xyz);
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
fn fs_forward_base(
    @builtin(position) frag_pos: vec4<f32>,
    @builtin(front_facing) front_facing: bool,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    let s = sample_surface(uv0, world_n, front_facing);
    let direct = clustered_direct_lighting(frag_pos.xy, world_pos, view_layer, s, true, true);
    let ambient = vec3<f32>(0.03) * s.base_color * s.occlusion;
    return vec4<f32>(ambient + direct + s.emission, s.alpha);
}
