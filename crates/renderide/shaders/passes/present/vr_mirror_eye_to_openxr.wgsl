//! Fullscreen copy from renderer-owned HMD color into the acquired OpenXR swapchain.

#import renderide::core::fullscreen as fs

#ifdef MULTIVIEW
@group(0) @binding(0) var t: texture_2d_array<f32>;
#else
@group(0) @binding(0) var t: texture_2d<f32>;
#endif
@group(0) @binding(1) var s: sampler;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> fs::FullscreenVertexOutput {
    return fs::vertex_main(vi);
}

@fragment
fn fs_main(
    in: fs::FullscreenVertexOutput,
#ifdef MULTIVIEW
    @builtin(view_index) view: u32,
#endif
) -> @location(0) vec4<f32> {
#ifdef MULTIVIEW
    return textureSample(t, s, in.uv, i32(view));
#else
    return textureSample(t, s, in.uv);
#endif
}
