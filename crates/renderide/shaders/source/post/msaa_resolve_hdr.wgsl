//! HDR-aware MSAA color resolve for the world-mesh forward path.
//!
//! Replaces wgpu's automatic linear `resolve_target` average. A linear average of HDR samples is
//! perceptually wrong at stark contrast edges: a pixel partially covered by a very bright sample
//! and partially by a very dark sample averages to a value that, after tonemapping, looks too
//! bright (or too dark on the inverse case), producing harshly aliased silhouettes between bright
//! emissives / specular sparks and dark surfaces even with high MSAA.
//!
//! This pass implements the Karis bracket used by Filament's `customResolveAsSubpass`: each sample
//! is compressed by `x / (1 + max3(x))` (a reversible squish into roughly `[0, 1]`), the
//! compressed values are linearly averaged, and the result is decompressed by `y / (1 - max3(y))`.
//! The compress/uncompress sandwich approximates "tonemap each sample, average, untonemap" while
//! preserving an HDR result for downstream bloom and tonemap to consume.

/// Per-pass parameters for [`fs_main`]: only the runtime sample count of `src_msaa`.
struct ResolveParams {
    /// MSAA sample count of `src_msaa` for this frame (1, 2, 4, or 8). The shader is a no-op
    /// for `sample_count == 1`; the pass should be skipped at the graph level in that case.
    sample_count: u32,
}

@group(0) @binding(0) var<uniform> params: ResolveParams;

// Multisampled arrays are not yet available in naga (`texture_multisampled_2d_array` is unbound),
// so the stereo path binds two single-layer multisampled views and picks between them with
// `@builtin(view_index)`. View index is uniform within a draw, so the branch is compile-time-
// uniform from the GPU's point of view.
#ifdef MULTIVIEW
@group(0) @binding(1) var src_msaa_left: texture_multisampled_2d<f32>;
@group(0) @binding(2) var src_msaa_right: texture_multisampled_2d<f32>;
#else
@group(0) @binding(1) var src_msaa: texture_multisampled_2d<f32>;
#endif

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    let p = vec2<f32>(f32((vid << 1u) & 2u), f32(vid & 2u));
    var out: VsOut;
    out.clip_pos = vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

//#pass forward
@fragment
fn fs_main(
    @builtin(position) frag_pos: vec4<f32>,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
) -> @location(0) vec4<f32> {
    let coord = vec2<i32>(frag_pos.xy);
    let n = i32(params.sample_count);
    var rgb_acc = vec3<f32>(0.0);
    var alpha_acc = 0.0;
    for (var i = 0; i < n; i = i + 1) {
#ifdef MULTIVIEW
        var s: vec4<f32>;
        if (view_idx == 0u) {
            s = textureLoad(src_msaa_left, coord, i);
        } else {
            s = textureLoad(src_msaa_right, coord, i);
        }
#else
        let s = textureLoad(src_msaa, coord, i);
#endif
        let m = max(max(s.r, s.g), s.b);
        let w = 1.0 / (1.0 + m);
        rgb_acc = rgb_acc + s.rgb * w;
        alpha_acc = alpha_acc + s.a;
    }
    let inv_n = 1.0 / f32(n);
    let avg_rgb = rgb_acc * inv_n;
    let m_avg = max(max(avg_rgb.r, avg_rgb.g), avg_rgb.b);
    // Clamp the inverse so a fully-saturated compressed average doesn't blow up the divide.
    let denom = max(1.0 - m_avg, 1e-4);
    let alpha = alpha_acc * inv_n;
    return vec4<f32>(avg_rgb / denom, alpha);
}
