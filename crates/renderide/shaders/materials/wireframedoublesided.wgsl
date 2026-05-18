//! Two-pass surface wireframe material (`Shader "WireframeDoubleSided"`).

//#wgpu_feature shader_barycentrics
//#texture_default _MainTex white
//#mat_default _LineColor vec4 1.0 1.0 1.0 1.0
//#mat_default _FillColor vec4 1.0 1.0 1.0 0.0
//#mat_default _InnerLineColor vec4 1.0 1.0 1.0 1.0
//#mat_default _InnerFillColor vec4 1.0 1.0 1.0 0.0
//#mat_default _LineFarColor vec4 1.0 1.0 1.0 0.0
//#mat_default _FillFarColor vec4 0.0 0.0 0.0 0.0
//#mat_default _InnerLineFarColor vec4 1.0 1.0 1.0 0.0
//#mat_default _InnerFillFarColor vec4 0.0 0.0 0.0 0.0
//#mat_default _Exp float 1.0
//#mat_default _Thickness float 1.0

#import renderide::core::texture_sampling as ts
#import renderide::core::uv as uvu
#import renderide::draw::per_draw as pd
#import renderide::frame::globals as rg
#import renderide::material::variant_bits as vb
#import renderide::mesh::vertex as mv
#import renderide::mesh::wireframe as wf

struct WireframeDoubleSidedMaterial {
    _LineColor: vec4<f32>,
    _FillColor: vec4<f32>,
    _InnerLineColor: vec4<f32>,
    _InnerFillColor: vec4<f32>,
    _LineFarColor: vec4<f32>,
    _FillFarColor: vec4<f32>,
    _InnerLineFarColor: vec4<f32>,
    _InnerFillFarColor: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _Exp: f32,
    _Thickness: f32,
    _RenderideVariantBits: u32,
    _MainTex_LodBias: f32,
}

const WIREFRAME_KW_FRESNEL: u32 = 1u << 0u;
const WIREFRAME_KW_SCREENSPACE: u32 = 1u << 1u;

@group(1) @binding(0) var<uniform> mat: WireframeDoubleSidedMaterial;
@group(1) @binding(1) var _MainTex: texture_2d<f32>;
@group(1) @binding(2) var _MainTex_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) @interpolate(flat) view_layer: u32,
}

fn wireframe_kw(mask: u32) -> bool {
    return vb::enabled(mat._RenderideVariantBits, mask);
}

fn kw_FRESNEL() -> bool {
    return wireframe_kw(WIREFRAME_KW_FRESNEL);
}

fn kw_SCREENSPACE() -> bool {
    return wireframe_kw(WIREFRAME_KW_SCREENSPACE);
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
    out.world_n = mv::world_normal(draw, n);
    out.uv = uvu::apply_st(uv0, mat._MainTex_ST);
    out.view_layer = (instance_index << 1u) | (layer & 1u);
    return out;
}

fn pass_color(in: VertexOutput, edge: f32, inner: bool) -> vec4<f32> {
    var fill_color = mat._FillColor;
    var line_color = mat._LineColor;
    if (inner) {
        fill_color = mat._InnerFillColor;
        line_color = mat._InnerLineColor;
    }
    if (kw_FRESNEL()) {
        let view_dir = rg::view_dir_for_world_pos(in.world_pos, in.view_layer);
        let fresnel = wf::fresnel_factor(in.world_n, view_dir, mat._Exp);
        var fill_far = mat._FillFarColor;
        var line_far = mat._LineFarColor;
        if (inner) {
            fill_far = mat._InnerFillFarColor;
            line_far = mat._InnerLineFarColor;
        }
        fill_color = mix(fill_color, fill_far, fresnel);
        line_color = mix(line_color, line_far, fresnel);
    }
    return mix(fill_color, line_color, edge);
}

fn fragment_wire(in: VertexOutput, barycentric: vec3<f32>, inner: bool) -> vec4<f32> {
    let edge = wf::edge_lerp(barycentric, in.world_pos, mat._Thickness, kw_SCREENSPACE());
    let tex = ts::sample_tex_2d(_MainTex, _MainTex_sampler, in.uv, mat._MainTex_LodBias);
    return rg::retain_globals_additive(pass_color(in, edge, inner) * tex);
}

//#pass type=forward name=inner blend=alpha zwrite=material(off) ztest=main cull=front color_mask=rgba offset=material(0,0)
@fragment
fn fs_inner(
    in: VertexOutput,
    @builtin(barycentric) barycentric: vec3<f32>,
) -> @location(0) vec4<f32> {
    return fragment_wire(in, barycentric, true);
}

//#pass type=forward name=outer blend=alpha zwrite=material(off) ztest=main cull=back color_mask=rgba offset=material(0,0)
@fragment
fn fs_outer(
    in: VertexOutput,
    @builtin(barycentric) barycentric: vec3<f32>,
) -> @location(0) vec4<f32> {
    return fragment_wire(in, barycentric, false);
}
