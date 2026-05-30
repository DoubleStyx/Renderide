//! Unity `Shader "Invisible"`: vertex collapses to origin and the fragment unconditionally
//! discards. Used as a hit-volume material that contributes nothing to color or depth.


#import renderide::frame::globals as rg
#import renderide::draw::per_draw as pd
#import renderide::mesh::vertex as mv

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) _pos: vec4<f32>,
    @location(1) _n: vec4<f32>,
) -> mv::ClipVertexOutput {
    let d = pd::get_draw(instance_index);
#ifdef MULTIVIEW
    let view_zero = f32(view_idx) * 0.0;
#else
    let view_zero = 0.0;
#endif
    var out: mv::ClipVertexOutput;
    out.clip_pos = d.model[0] * 0.0 + vec4<f32>(view_zero);
    return out;
}

//#render_queue Transparent
//#pass type=forward
@fragment
fn fs_main() -> @location(0) vec4<f32> {
    discard;
    return rg::retain_globals_additive(vec4<f32>(0.0));
}
