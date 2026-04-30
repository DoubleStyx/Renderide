//! Fullscreen pass: applies Stephen Hill ACES Fitted tonemap (HDR linear -> display-referred
//! linear in [0, 1]) to the HDR scene color and writes the post-processing chain output.
//!
//! Build script composes this source into `aces_tonemap_default` (mono) and
//! `aces_tonemap_multiview` (stereo, `view_index` selects array layer) targets — see the build
//! script's post-shader composition loop and [`crate::embedded_shaders`].

#import renderide::fullscreen as fs

@group(0) @binding(0) var scene_color_hdr: texture_2d_array<f32>;
@group(0) @binding(1) var scene_color_sampler: sampler;

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> fs::FullscreenVertexOutput {
    return fs::vertex_main(vid);
}

// sRGB -> AP1, RRT_SAT.
fn aces_input_matrix() -> mat3x3<f32> {
    return mat3x3<f32>(
        vec3<f32>(0.59719, 0.07600, 0.02840),
        vec3<f32>(0.35458, 0.90834, 0.13383),
        vec3<f32>(0.04823, 0.01566, 0.83777),
    );
}

// ODT_SAT -> AP1 -> sRGB.
fn aces_output_matrix() -> mat3x3<f32> {
    return mat3x3<f32>(
        vec3<f32>( 1.60475, -0.10208, -0.00327),
        vec3<f32>(-0.53108,  1.10813, -0.07276),
        vec3<f32>(-0.07367, -0.00605,  1.07602),
    );
}

// Fitted polynomial of the combined ACES Reference Rendering Transform and Output Device
// Transform. Operates per-channel on AP1 values.
fn rrt_and_odt_fit(v: vec3<f32>) -> vec3<f32> {
    let a = v * (v + vec3<f32>(0.0245786)) - vec3<f32>(0.000090537);
    let b = v * (0.983729 * v + vec3<f32>(0.4329510)) + vec3<f32>(0.238081);
    return a / b;
}

fn aces_fitted(color_linear: vec3<f32>) -> vec3<f32> {
    let c_ap1 = aces_input_matrix() * color_linear;
    let c_curve = rrt_and_odt_fit(c_ap1);
    let c_srgb = aces_output_matrix() * c_curve;
    return clamp(c_srgb, vec3<f32>(0.0), vec3<f32>(1.0));
}

#ifdef MULTIVIEW
@fragment
fn fs_main(in: fs::FullscreenVertexOutput, @builtin(view_index) view: u32) -> @location(0) vec4<f32> {
    let hdr = textureSample(scene_color_hdr, scene_color_sampler, in.uv, view);
    let ldr = aces_fitted(hdr.rgb);
    return vec4<f32>(ldr, hdr.a);
}
#else
@fragment
fn fs_main(in: fs::FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let hdr = textureSample(scene_color_hdr, scene_color_sampler, in.uv, 0u);
    let ldr = aces_fitted(hdr.rgb);
    return vec4<f32>(ldr, hdr.a);
}
#endif
