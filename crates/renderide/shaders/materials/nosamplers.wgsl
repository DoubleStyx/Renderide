//! Unity surface shader `Shader "Custom/Nosamplers"`: metallic Standard lighting that demos
//! Unity's `UNITY_DECLARE_TEX2D_NOSAMPLER` aliasing -- `_MetallicMap` shares the `_Albedo` sampler
//! and `_EmissionMap`/`_EmissionMap1` are sampled with their own.


//#texture_default _Albedo white
//#texture_default _Albedo1 white
//#texture_default _Albedo2 white
//#texture_default _Albedo3 white
//#texture_default _MetallicMap white
//#texture_default _EmissionMap white
//#texture_default _EmissionMap1 white
//#mat_default _Color vec4 1.0 1.0 1.0 1.0

#import renderide::draw::per_draw as pd
#import renderide::mesh::vertex as mv
#import renderide::pbs::lighting as plight
#import renderide::pbs::sampling as psamp
#import renderide::pbs::surface as psurf
#import renderide::core::uv as uvu

struct NosamplersMaterial {
    _Color: vec4<f32>,
    _Albedo_ST: vec4<f32>,
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
@group(1) @binding(10) var _EmissionMap: texture_2d<f32>;
@group(1) @binding(11) var _EmissionMap_sampler: sampler;
@group(1) @binding(12) var _EmissionMap1: texture_2d<f32>;
@group(1) @binding(13) var _EmissionMap1_sampler: sampler;

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
    let world_p = mv::world_position(d, pos);
    let wn = mv::world_normal(d, n);
#ifdef MULTIVIEW
    let vp = mv::select_view_proj(d, view_idx);
#else
    let vp = mv::select_view_proj(d, 0u);
#endif
    var out: VertexOutput;
    out.clip_pos = vp * world_p;
    out.world_pos = world_p.xyz;
    out.world_n = wn;
    out.uv = uvu::apply_st(uv0, mat._Albedo_ST);
#ifdef MULTIVIEW
    out.view_layer = mv::packed_view_layer(instance_index, view_idx);
#else
    out.view_layer = mv::packed_view_layer(instance_index, 0u);
#endif
    return out;
}

fn shade(
    frag_xy: vec2<f32>,
    world_pos: vec3<f32>,
    world_n: vec3<f32>,
    uv: vec2<f32>,
    view_layer: u32,
) -> vec4<f32> {
    let c = textureSample(_Albedo, _Albedo_sampler, uv) * mat._Color;

    let m = textureSample(_MetallicMap, _Albedo_sampler, uv);
    let e0 = textureSample(_EmissionMap, _EmissionMap_sampler, uv).rgb;
    let e1 = textureSample(_EmissionMap1, _EmissionMap1_sampler, uv).rgb;
    let emission = mix(e0, e1, 0.5);

    let base_color = c.rgb;
    let metallic = clamp(m.r, 0.0, 1.0);
    let smoothness = clamp(m.a, 0.0, 1.0);
    let roughness = psamp::roughness_from_smoothness(smoothness);
    let n = normalize(world_n);
    let surface = psurf::metallic_with_geometric_normal(
        base_color,
        c.a,
        metallic,
        roughness,
        1.0,
        n,
        world_n,
        emission,
    );
    return vec4<f32>(
        plight::shade_metallic_clustered(
            frag_xy,
            world_pos,
            view_layer,
            surface,
            plight::default_lighting_options(),
        ),
        c.a,
    );
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
    return shade(frag_pos.xy, world_pos, world_n, uv, view_layer);
}
