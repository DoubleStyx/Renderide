//! Plane-transition unlit wireframe material (`Shader "PlaneTransition/WireframeUnlit"`).

//#render_queue Transparent
//#wgpu_feature shader_barycentrics
//#texture_default _MainTex white
//#mat_default _LineColor vec4 0.0 1.0 1.0 1.0
//#mat_default _FillColor vec4 0.0 0.2 0.2 1.0
//#mat_default _Thickness float 2.0
//#mat_default _PlaneDir vec4 0.0 1.0 0.0 0.0
//#mat_default _FillTransitionRange float 0.1
//#mat_default _FillTransitionExp float 1.0
//#mat_default _FillPlaneOffset float 0.0
//#mat_default _WireTransitionRange float 0.1
//#mat_default _WireTransitionExp float 1.0
//#mat_default _WirePlaneOffset float 0.0

#import renderide::core::math as rmath
#import renderide::core::texture_sampling as ts
#import renderide::core::uv as uvu
#import renderide::draw::per_draw as pd
#import renderide::frame::globals as rg
#import renderide::mesh::vertex as mv
#import renderide::mesh::wireframe as wf

struct WireframeUnlitTransitionMaterial {
    _LineColor: vec4<f32>,
    _FillColor: vec4<f32>,
    _PlaneDir: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _Thickness: f32,
    _FillTransitionRange: f32,
    _FillTransitionExp: f32,
    _FillPlaneOffset: f32,
    _WireTransitionRange: f32,
    _WireTransitionExp: f32,
    _WirePlaneOffset: f32,
    _MainTex_LodBias: f32,
}

@group(1) @binding(0) var<uniform> mat: WireframeUnlitTransitionMaterial;
@group(1) @binding(1) var _MainTex: texture_2d<f32>;
@group(1) @binding(2) var _MainTex_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) object_distance: f32,
    @location(3) @interpolate(flat) view_layer: u32,
}

fn plane_axis() -> vec3<f32> {
    return rmath::safe_normalize(mat._PlaneDir.xyz, vec3<f32>(0.0, 1.0, 0.0));
}

fn transition_lerp(distance: f32, offset: f32, range: f32, exponent: f32, invert: bool) -> f32 {
    let denom = max(abs(range) * 0.5, 1e-6);
    var t = (distance + offset) / denom;
    t = (t + 1.0) * 0.5;
    if (invert) {
        t = 1.0 - t;
    }
    return pow(rmath::saturate(t), max(exponent, 1e-4));
}

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(2) uv0: vec2<f32>,
) -> VertexOutput {
    let draw = pd::get_draw(instance_index);
    let world_p = mv::world_position(draw, pos);
#ifdef MULTIVIEW
    let layer = view_idx;
#else
    let layer = 0u;
#endif
    let vp = mv::select_view_proj(draw, layer);

    var out: VertexOutput;
    out.clip_pos = vp * world_p;
    out.world_pos = world_p.xyz;
    out.uv = uvu::apply_st(uv0, mat._MainTex_ST);
    out.object_distance = dot(plane_axis(), pos.xyz);
    out.view_layer = (instance_index << 1u) | (layer & 1u);
    return out;
}

//#pass type=depth_prepass name=depth blend=off zwrite=on ztest=main cull=back color_mask=0
@fragment
fn fs_depth(in: VertexOutput) -> @location(0) vec4<f32> {
    let touch = (in.world_pos.x + in.uv.x + in.object_distance + f32(in.view_layer)) * 0.0;
    return rg::retain_globals_additive(vec4<f32>(touch, touch, touch, 0.0));
}

//#pass type=forward name=fill blend=alpha zwrite=on ztest=main cull=back color_mask=rgba
@fragment
fn fs_fill(in: VertexOutput) -> @location(0) vec4<f32> {
    var col = ts::sample_tex_2d(_MainTex, _MainTex_sampler, in.uv, mat._MainTex_LodBias);
    let fill = transition_lerp(
        in.object_distance,
        mat._FillPlaneOffset,
        mat._FillTransitionRange,
        mat._FillTransitionExp,
        false,
    );
    if (fill <= 0.0) {
        discard;
    }
    col.a = col.a * fill;
    return rg::retain_globals_additive(col);
}

//#pass type=forward name=wire blend=additive zwrite=on ztest=main cull=back color_mask=rgba
@fragment
fn fs_wire(
    in: VertexOutput,
    @builtin(barycentric) barycentric: vec3<f32>,
) -> @location(0) vec4<f32> {
    let edge = wf::unity_edge_lerp(barycentric, in.world_pos, mat._Thickness, false);
    let fill = transition_lerp(
        in.object_distance,
        mat._FillPlaneOffset,
        mat._FillTransitionRange,
        mat._FillTransitionExp,
        true,
    );
    let wire = transition_lerp(
        in.object_distance,
        mat._WirePlaneOffset,
        mat._WireTransitionRange,
        mat._WireTransitionExp,
        true,
    );
    return rg::retain_globals_additive(mat._FillColor * fill + mat._LineColor * wire * edge);
}
