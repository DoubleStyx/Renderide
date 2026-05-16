//! Unity surface shader `Shader "PBSDisplaceShadow"`: Standard metallic lighting on a
//! vertex-displaced mesh, with a matching shadow-caster proxy.
//!
//! The Unity asset declares `surface surf Standard fullforwardshadows vertex:vert addshadow`
//! with an empty `surf` body, so the fragment shading collapses to defaults: black albedo,
//! alpha 1, metallic 0, smoothness 0, no normal map, no emission. The vertex stage samples
//! `_VertexOffsetMap.r` and offsets along the mesh normal; the depth-only pass uses the same
//! displacement so shadow casts line up.

//#texture_default _MainTex white
//#texture_default _VertexOffsetMap black
//#mat_default _Color vec4 1.0 1.0 1.0 1.0
//#mat_default _VertexOffsetMagnitude float 0.1

#import renderide::draw::per_draw as pd
#import renderide::frame::globals as rg
#import renderide::mesh::vertex as mv
#import renderide::pbs::displace as pdisp
#import renderide::pbs::lighting as plight
#import renderide::pbs::surface as psurf

struct PbsDisplaceShadowMaterial {
    _Color: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _VertexOffsetMap_ST: vec4<f32>,
    _VertexOffsetMagnitude: f32,
    _VertexOffsetBias: f32,
}

@group(1) @binding(0) var<uniform> mat: PbsDisplaceShadowMaterial;
@group(1) @binding(1) var _MainTex: texture_2d<f32>;
@group(1) @binding(2) var _MainTex_sampler: sampler;
@group(1) @binding(3) var _VertexOffsetMap: texture_2d<f32>;
@group(1) @binding(4) var _VertexOffsetMap_sampler: sampler;

/// Vertex stage: displace along the mesh normal using `_VertexOffsetMap.r`, then forward the
/// usual world-space PBS payload. Tangents are passed through unchanged; the empty Unity
/// `surf` body does not sample a normal map, so the tangent space is only forwarded for the
/// shared `WorldVertexOutput` shape.
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
    let draw = pd::get_draw(instance_index);
    let displaced_uv = pdisp::apply_vertex_offsets(
        pos.xyz,
        n.xyz,
        uv0,
        draw.model,
        true,
        false,
        false,
        mat._VertexOffsetMap_ST,
        vec4<f32>(1.0, 1.0, 0.0, 0.0),
        vec2<f32>(0.0),
        mat._VertexOffsetMagnitude,
        mat._VertexOffsetBias,
        _VertexOffsetMap,
        _VertexOffsetMap_sampler,
        _VertexOffsetMap,
        _VertexOffsetMap_sampler,
    );
#ifdef MULTIVIEW
    let view_layer = view_idx;
#else
    let view_layer = 0u;
#endif
    return mv::world_vertex_main(
        instance_index,
        view_layer,
        vec4<f32>(displaced_uv.position, 1.0),
        n,
        t,
        displaced_uv.uv,
    );
}

/// Forward pass: Standard metallic shading with Unity's empty-surface defaults.
//#pass forward
@fragment
fn fs_forward_base(in: mv::WorldVertexOutput) -> @location(0) vec4<f32> {
    let surface = psurf::metallic_with_geometric_normal(
        vec3<f32>(0.0),
        1.0,
        0.0,
        1.0,
        1.0,
        normalize(in.world_n),
        in.world_n,
        vec3<f32>(0.0),
    );
    let shaded = plight::shade_metallic_clustered(
        in.clip_pos.xy,
        in.world_pos,
        in.view_layer,
        surface,
        plight::default_lighting_options(),
    );
    return vec4<f32>(shaded, 1.0);
}

/// Depth-only proxy pass for the `addshadow` shadow caster: emit a zero color that retains
/// the per-frame global bindings without writing albedo.
//#pass depth_prepass
@fragment
fn fs_depth_only(in: mv::WorldVertexOutput) -> @location(0) vec4<f32> {
    let touch = in.primary_uv.x * 0.0;
    return rg::retain_globals_additive(vec4<f32>(touch, touch, touch, 0.0));
}
