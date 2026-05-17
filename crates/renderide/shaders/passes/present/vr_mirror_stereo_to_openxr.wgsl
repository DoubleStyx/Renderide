//! Fullscreen stereo copy from renderer-owned HMD color into the acquired OpenXR swapchain.

#import renderide::core::fullscreen as fs

@group(0) @binding(0) var t: texture_2d_array<f32>;
@group(0) @binding(1) var s: sampler;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> fs::FullscreenVertexOutput {
    return fs::vertex_main(vi);
}

@fragment
fn fs_main(
    in: fs::FullscreenVertexOutput,
    @builtin(view_index) view: u32,
) -> @location(0) vec4<f32> {
    return textureSample(t, s, in.uv, i32(view));
}
