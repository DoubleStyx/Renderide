//! Fullscreen pass: samples HDR [`scene_color_hdr`] and writes the displayable color target.
//! `#ifdef MULTIVIEW` selects the per-eye layer; the non-multiview path samples layer 0.
//! Future exposure / tonemap / grading hook.

#import renderide::fullscreen as fs

@group(0) @binding(0) var scene_color_hdr: texture_2d_array<f32>;
@group(0) @binding(1) var scene_color_sampler: sampler;

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> fs::FullscreenVertexOutput {
    return fs::vertex_main(vid);
}

@fragment
fn fs_main(
    in: fs::FullscreenVertexOutput,
#ifdef MULTIVIEW
    @builtin(view_index) view: u32,
#endif
) -> @location(0) vec4<f32> {
#ifdef MULTIVIEW
    let layer = i32(view);
#else
    let layer = 0;
#endif
    return textureSample(scene_color_hdr, scene_color_sampler, in.uv, layer);
}
