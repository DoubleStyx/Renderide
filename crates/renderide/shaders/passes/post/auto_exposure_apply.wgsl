//! Fullscreen pass: applies the current auto-exposure EV to HDR scene color.

#import renderide::fullscreen as fs

@group(0) @binding(0) var scene_color_hdr: texture_2d_array<f32>;
@group(0) @binding(1) var scene_color_sampler: sampler;
@group(0) @binding(2) var<storage, read> exposure_ev: f32;

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> fs::FullscreenVertexOutput {
    return fs::vertex_main(vid);
}

fn apply_auto_exposure(hdr: vec4<f32>) -> vec4<f32> {
    let scale = exp2(clamp(exposure_ev, -32.0, 32.0));
    return vec4<f32>(hdr.rgb * scale, hdr.a);
}

#ifdef MULTIVIEW
@fragment
fn fs_main(in: fs::FullscreenVertexOutput, @builtin(view_index) view: u32) -> @location(0) vec4<f32> {
    return apply_auto_exposure(textureSample(scene_color_hdr, scene_color_sampler, in.uv, view));
}
#else
@fragment
fn fs_main(in: fs::FullscreenVertexOutput) -> @location(0) vec4<f32> {
    return apply_auto_exposure(textureSample(scene_color_hdr, scene_color_sampler, in.uv, 0u));
}
#endif
