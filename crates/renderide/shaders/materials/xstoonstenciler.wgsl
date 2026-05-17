//! Xiexe Toon 2.0 stenciler (`Shader "Xiexe/Toon2.0/XSToonStenciler"`).


#import renderide::frame::globals as rg
#import renderide::draw::per_draw as pd
#import renderide::mesh::vertex as mv

struct XiexeStencilerMaterial {
    _Offset: f32,
}

@group(1) @binding(0) var<uniform> mat: XiexeStencilerMaterial;

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
) -> mv::ClipVertexOutput {
    let d = pd::get_draw(instance_index);
    let world_p = mv::world_position(d, vec4<f32>(pos.xyz, 1.0));
#ifdef MULTIVIEW
    let vp = mv::select_view_proj(d, view_idx);
#else
    let vp = mv::select_view_proj(d, 0u);
#endif
    var out: mv::ClipVertexOutput;
    out.clip_pos = vp * world_p;
    return out;
}

//#pass type=forward name=stencil blend=off zwrite=material(off) ztest=material_unity(main) cull=material(front) color_mask=material(rgba) offset=0,0 stencil=material
@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return rg::retain_globals_additive(vec4<f32>(0.0));
}
