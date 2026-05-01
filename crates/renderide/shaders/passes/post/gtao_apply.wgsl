//! Fullscreen pass: GTAO depth-aware denoise (final iteration) + HDR modulation.
//!
//! Reads the post-processing chain's HDR scene-color input plus the AO term and packed edges
//! (from `gtao_main` directly when `denoise_passes ∈ {0, 1}`, or from the intermediate
//! denoise ping-pong target when `denoise_passes == 2`). Runs XeGTAO's edge-preserving 3×3
//! bilateral kernel at the full `denoise_blur_beta` (`XeGTAO_Denoise` with
//! `finalApply = true`), multiplies the resulting AO factor by `OCCLUSION_TERM_SCALE` to
//! recover the true visibility (the production pass stored `visibility / 1.5` for kernel
//! headroom), then modulates HDR scene color and writes the chain's HDR output.
//!
//! The shader short-circuits the kernel when `denoise_blur_beta <= 0` so
//! `denoise_passes == 0` collapses to a "modulate by raw production AO" path without
//! re-binding a different pipeline. The `OCCLUSION_TERM_SCALE` recovery still runs in that
//! path so the user-visible visibility is correct regardless of the denoise setting.
//!
//! Build script composes this into `gtao_apply_default` (mono) and `gtao_apply_multiview`
//! (stereo). The Rust side caches one pipeline per `(output_format, multiview_stereo)`.
//!
//! Bind group (`@group(0)`):
//! - `@binding(0)` HDR scene color (`texture_2d_array<f32>`).
//! - `@binding(1)` linear-clamp sampler.
//! - `@binding(2)` AO term (`texture_2d_array<f32>`).
//! - `@binding(3)` packed edges (`texture_2d_array<f32>`).
//! - `@binding(4)` `GtaoParams` uniform.

#import renderide::fullscreen as fs
#import renderide::gtao_filter as gf

@group(0) @binding(0) var scene_color_hdr: texture_2d_array<f32>;
@group(0) @binding(1) var linear_clamp: sampler;
@group(0) @binding(2) var ao_term: texture_2d_array<f32>;
@group(0) @binding(3) var ao_edges: texture_2d_array<f32>;

struct GtaoParams {
    radius_world: f32,
    max_pixel_radius: f32,
    intensity: f32,
    step_count: u32,
    falloff_range: f32,
    albedo_multibounce: f32,
    denoise_blur_beta: f32,
    final_apply: u32,
}

@group(0) @binding(4) var<uniform> gtao: GtaoParams;

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> fs::FullscreenVertexOutput {
    return fs::vertex_main(vid);
}

fn load_ao(pix: vec2<i32>, view_layer: u32, viewport_max: vec2<i32>) -> f32 {
    let p = clamp(pix, vec2<i32>(0), viewport_max);
    return textureLoad(ao_term, p, i32(view_layer), 0).r;
}

fn load_edges_lrtb(pix: vec2<i32>, view_layer: u32, viewport_max: vec2<i32>) -> vec4<f32> {
    let p = clamp(pix, vec2<i32>(0), viewport_max);
    let packed = textureLoad(ao_edges, p, i32(view_layer), 0).r;
    return gf::gtao_unpack_edges(packed);
}

/// Runs the XeGTAO bilateral kernel at the full `denoise_blur_beta` (XeGTAO's
/// `XeGTAO_Denoise` with `finalApply = true`). Returns the denoised AO term in the
/// production-scaled representation; the caller multiplies by `OCCLUSION_TERM_SCALE` to
/// recover the true visibility before modulating HDR.
fn final_denoise_at(pix: vec2<i32>, view_layer: u32, viewport_max: vec2<i32>) -> f32 {
    let edges_c = load_edges_lrtb(pix, view_layer, viewport_max);
    let edges_l = load_edges_lrtb(pix + vec2<i32>(-1, 0), view_layer, viewport_max);
    let edges_r = load_edges_lrtb(pix + vec2<i32>( 1, 0), view_layer, viewport_max);
    let edges_t = load_edges_lrtb(pix + vec2<i32>(0, -1), view_layer, viewport_max);
    let edges_b = load_edges_lrtb(pix + vec2<i32>(0,  1), view_layer, viewport_max);

    var edges_c_sym = gf::gtao_symmetricise_edges(edges_c, edges_l, edges_r, edges_t, edges_b);
    edges_c_sym = gf::gtao_apply_edge_leak(edges_c_sym);
    let diag = gf::gtao_diagonal_weights(edges_c_sym, edges_l, edges_r, edges_t, edges_b);

    let ao = gf::GtaoKernelAo(
        load_ao(pix, view_layer, viewport_max),
        load_ao(pix + vec2<i32>(-1, 0), view_layer, viewport_max),
        load_ao(pix + vec2<i32>( 1, 0), view_layer, viewport_max),
        load_ao(pix + vec2<i32>(0, -1), view_layer, viewport_max),
        load_ao(pix + vec2<i32>(0,  1), view_layer, viewport_max),
        load_ao(pix + vec2<i32>(-1, -1), view_layer, viewport_max),
        load_ao(pix + vec2<i32>( 1, -1), view_layer, viewport_max),
        load_ao(pix + vec2<i32>(-1,  1), view_layer, viewport_max),
        load_ao(pix + vec2<i32>( 1,  1), view_layer, viewport_max),
    );

    let blur_amount = max(gtao.denoise_blur_beta, 1e-4);
    return clamp(
        gf::gtao_denoise_kernel(edges_c_sym, diag, ao, blur_amount),
        0.0,
        1.0,
    );
}

/// Selects between "no denoise" (raw center AO) and "final-apply denoise". Either way the
/// returned value is still in the production-scaled `[0, 1] / OCCLUSION_TERM_SCALE`
/// representation; the caller scales back up.
fn ao_factor_scaled(pix: vec2<i32>, view_layer: u32, viewport_max: vec2<i32>) -> f32 {
    if (gtao.denoise_blur_beta <= 0.0) {
        return load_ao(pix, view_layer, viewport_max);
    }
    return final_denoise_at(pix, view_layer, viewport_max);
}

#ifdef MULTIVIEW
@fragment
fn fs_main(in: fs::FullscreenVertexOutput, @builtin(view_index) view: u32) -> @location(0) vec4<f32> {
    let dim = textureDimensions(ao_term);
    let viewport_max = vec2<i32>(i32(dim.x) - 1, i32(dim.y) - 1);
    let pix = vec2<i32>(in.clip_pos.xy);
    let ao_scaled = ao_factor_scaled(pix, view, viewport_max);
    let ao = clamp(ao_scaled * gf::GTAO_OCCLUSION_TERM_SCALE, 0.0, 1.0);
    let hdr = textureSample(scene_color_hdr, linear_clamp, in.uv, view);
    return vec4<f32>(hdr.rgb * ao, hdr.a);
}
#else
@fragment
fn fs_main(in: fs::FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let dim = textureDimensions(ao_term);
    let viewport_max = vec2<i32>(i32(dim.x) - 1, i32(dim.y) - 1);
    let pix = vec2<i32>(in.clip_pos.xy);
    let ao_scaled = ao_factor_scaled(pix, 0u, viewport_max);
    let ao = clamp(ao_scaled * gf::GTAO_OCCLUSION_TERM_SCALE, 0.0, 1.0);
    let hdr = textureSample(scene_color_hdr, linear_clamp, in.uv, 0u);
    return vec4<f32>(hdr.rgb * ao, hdr.a);
}
#endif
