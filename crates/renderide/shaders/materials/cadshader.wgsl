//! Unity `Shader "CADShader"`: two-pass emissive shader with a normal-extruded outline shell.
//!
//! Pass 1 (`outline`, cull-front): expand vertex along normal by `_OutlineWidth`, output
//! `_OutlineColor`. Pass 2 (`forward_base`): standard cull-back emissive `_Color` output. Mirrors
//! the `xstoon2.0-outlined.wgsl` pass structure (`PassKind::Outline` + `PassKind::ForwardBase`).


//#mat_default _Color vec4 1.0 1.0 1.0 1.0
//#mat_default _OutlineColor vec4 0.0 0.0 0.0 1.0
//#mat_default _OutlineWidth float 0.1

#import renderide::frame::globals as rg
#import renderide::draw::per_draw as pd
#import renderide::draw::types as dt
#import renderide::mesh::vertex as mv

struct CadShaderMaterial {
    _Color: vec4<f32>,
    _OutlineColor: vec4<f32>,
    _OutlineWidth: f32,
}

@group(1) @binding(0) var<uniform> mat: CadShaderMaterial;

fn project(world_p: vec4<f32>, view_idx: u32, d: dt::PerDrawUniforms) -> vec4<f32> {
    return mv::select_view_proj(d, view_idx) * world_p;
}

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) _n: vec4<f32>,
) -> mv::ClipVertexOutput {
    let d = pd::get_draw(instance_index);
    let world_p = mv::world_position(d, pos);
#ifdef MULTIVIEW
    let clip = project(world_p, view_idx, d);
#else
    let clip = project(world_p, 0u, d);
#endif
    var out: mv::ClipVertexOutput;
    out.clip_pos = clip;
    return out;
}

@vertex
fn vs_outline(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
) -> mv::ClipVertexOutput {
    let d = pd::get_draw(instance_index);
    let extruded = pos.xyz + n.xyz * mat._OutlineWidth;
    let world_p = mv::world_position(d, vec4<f32>(extruded, 1.0));
#ifdef MULTIVIEW
    let clip = project(world_p, view_idx, d);
#else
    let clip = project(world_p, 0u, d);
#endif
    var out: mv::ClipVertexOutput;
    out.clip_pos = clip;
    return out;
}

//#pass outline vs=vs_outline
@fragment
fn fs_outline() -> @location(0) vec4<f32> {
    return rg::retain_globals_additive(vec4<f32>(mat._OutlineColor.rgb, 0.0));
}

//#pass forward
@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return rg::retain_globals_additive(vec4<f32>(mat._Color.rgb, 1.0));
}
