//! Unity `Shader "CADShader"`: two-pass emissive shader with a normal-extruded outline shell.
//!
//! Pass 1 (`outline`, cull-front): expand vertex along normal by `_OutlineWidth`, output
//! `_OutlineColor`. Pass 2 (`forward_base`): standard cull-back emissive `_Color` output. Mirrors
//! the `xstoon2.0-outlined.wgsl` pass structure (`PassKind::Outline` + `PassKind::ForwardBase`).

// unity-shader-name: CADShader

#import renderide::globals as rg
#import renderide::per_draw as pd

struct CadShaderMaterial {
    _Color: vec4<f32>,
    _OutlineColor: vec4<f32>,
    _OutlineWidth: f32,
}

@group(1) @binding(0) var<uniform> mat: CadShaderMaterial;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
}

fn project(world_p: vec4<f32>, view_idx: u32, d: pd::PerDrawUniforms) -> vec4<f32> {
    var vp: mat4x4<f32>;
    if (view_idx == 0u) {
        vp = d.view_proj_left;
    } else {
        vp = d.view_proj_right;
    }
    return vp * world_p;
}

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) _n: vec4<f32>,
) -> VertexOutput {
    let d = pd::get_draw(instance_index);
    let world_p = d.model * vec4<f32>(pos.xyz, 1.0);
#ifdef MULTIVIEW
    let clip = project(world_p, view_idx, d);
#else
    let clip = d.view_proj_left * world_p;
#endif
    var out: VertexOutput;
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
) -> VertexOutput {
    let d = pd::get_draw(instance_index);
    let extruded = pos.xyz + n.xyz * mat._OutlineWidth;
    let world_p = d.model * vec4<f32>(extruded, 1.0);
#ifdef MULTIVIEW
    let clip = project(world_p, view_idx, d);
#else
    let clip = d.view_proj_left * world_p;
#endif
    var out: VertexOutput;
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
