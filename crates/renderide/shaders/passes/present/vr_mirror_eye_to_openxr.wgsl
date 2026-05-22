//! Fullscreen copy from one renderer-owned HMD eye into one OpenXR swapchain layer.

#import renderide::core::fullscreen as fs

@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> fs::FullscreenVertexOutput {
    return fs::vertex_main(vi);
}

@fragment
fn fs_main(in: fs::FullscreenVertexOutput) -> @location(0) vec4<f32> {
    return textureSample(t, s, in.uv);
}
