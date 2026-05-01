//! Unity surface shader `Shader "PBSDisplaceShadow"`: depth-only displaced proxy.
//!
//! The Unity asset has an empty `surf` body and only offsets vertices along their normals from
//! `_VertexOffsetMap`, so this stem writes depth without carrying the full PBS lighting path.

#import renderide::globals as rg
#import renderide::mesh::vertex as mv
#import renderide::per_draw as pd
#import renderide::uv_utils as uvu

struct PbsDisplaceShadowMaterial {
    _Color: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _VertexOffsetMap_ST: vec4<f32>,
    _MainTex_StorageVInverted: f32,
    _VertexOffsetMap_StorageVInverted: f32,
    _VertexOffsetMagnitude: f32,
    _VertexOffsetBias: f32,
}

@group(1) @binding(0) var<uniform> mat: PbsDisplaceShadowMaterial;
@group(1) @binding(1) var _MainTex: texture_2d<f32>;
@group(1) @binding(2) var _MainTex_sampler: sampler;
@group(1) @binding(3) var _VertexOffsetMap: texture_2d<f32>;
@group(1) @binding(4) var _VertexOffsetMap_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv0: vec2<f32>,
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
    let vertex_offset_uv = uvu::apply_st_for_storage(
        uv0,
        mat._VertexOffsetMap_ST,
        mat._VertexOffsetMap_StorageVInverted,
    );
    let height = textureSampleLevel(
        _VertexOffsetMap,
        _VertexOffsetMap_sampler,
        vertex_offset_uv,
        0.0,
    ).r;
    let displaced = pos.xyz + n.xyz * (height * mat._VertexOffsetMagnitude + mat._VertexOffsetBias);
    let world_p = d.model * vec4<f32>(displaced, 1.0);
#ifdef MULTIVIEW
    let vp = mv::select_view_proj(d, view_idx);
#else
    let vp = mv::select_view_proj(d, 0u);
#endif

    var out: VertexOutput;
    out.clip_pos = vp * world_p;
    out.uv0 = uv0;
    return out;
}

//#pass depth_prepass
@fragment
fn fs_depth_only(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv_main = uvu::apply_st_for_storage(in.uv0, mat._MainTex_ST, mat._MainTex_StorageVInverted);
    let albedo = textureSample(_MainTex, _MainTex_sampler, uv_main) * mat._Color;
    let touch = (albedo.x + in.uv0.x) * 0.0;
    return rg::retain_globals_additive(vec4<f32>(touch, touch, touch, 0.0));
}
