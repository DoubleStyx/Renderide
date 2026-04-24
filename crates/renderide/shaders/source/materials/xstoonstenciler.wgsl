//! Xiexe Toon 2.0 stenciler (`Shader "Xiexe/Toon2.0/XSToonStenciler"`).

// unity-shader-name: Xiexe/Toon2.0/XSToonStenciler

#import renderide::globals as rg
#import renderide::per_draw as pd

struct XiexeStencilerMaterial {
    _Offset: f32,
}

@group(1) @binding(0) var<uniform> mat: XiexeStencilerMaterial;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
}

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
) -> VertexOutput {
    let d = pd::get_draw(instance_index);
    let world_p = d.model * vec4<f32>(pos.xyz + n.xyz * mat._Offset, 1.0);
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
    return out;
}

//#material stencil
@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return rg::retain_globals_additive(vec4<f32>(0.0));
}
