//! Fullscreen blit from staging to window with **cover** UV mapping (object-fit: cover):
//! uniform scale, crop texture edges so the window is filled (no letterboxing).

#import renderide::fullscreen as fs

@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> uv_params: vec4<f32>;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> fs::FullscreenVertexOutput {
    return fs::vertex_flipped_y_main(vi);
}

@fragment
fn fs_main(in: fs::FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let tuv = in.uv * uv_params.xy + uv_params.zw;
    return textureSample(t, samp, tuv);
}
