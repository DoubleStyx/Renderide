//! Physically-based bloom (COD: Advanced Warfare dual-filter; Bevy port).
//!
//! Provides four `@fragment` entry points sharing the same vertex shader:
//!
//! * `fs_downsample_first` — first downsample of the HDR scene color with Karis firefly
//!   reduction and optional soft-knee threshold. Writes bloom mip 0.
//! * `fs_downsample` — 13-tap downsample between bloom mips N and N+1.
//! * `fs_upsample` — 3×3 tent upsample; pipelines using this entry point enable
//!   constant-factor blending so the output accumulates into the mip above.
//! * `fs_composite` — final combine pass reading the scene HDR (group 0) and bloom mip 0
//!   (group 1) and writing the chain output; performs the composite math in-shader so the
//!   pipeline can use the default `Replace` blend state.
//!
//! Kernel weights, Karis average, and soft-threshold math are direct ports of Bevy's
//! `crates/bevy_post_process/src/bloom/bloom.wgsl`. Build script emits `bloom_default` and
//! `bloom_multiview` targets — the multiview variant substitutes `@builtin(view_index)` for
//! array sampling so the left/right stereo layers are scattered independently.

#import renderide::fullscreen as fs

/// Per-frame bloom parameters shared across all four entry points.
struct BloomUniforms {
    /// `[threshold, threshold - knee, 2 * knee, 0.25 / (knee + 1e-4)]`, precomputed on CPU
    /// (Unity-style quadratic soft-knee curve). `threshold <= 0` disables the prefilter.
    threshold_precomputations: vec4<f32>,
    /// Composite intensity (linear scatter factor). `0.0` disables bloom (chain gates before
    /// the pass).
    intensity: f32,
    /// `1.0` → energy-conserving composite; `0.0` → additive composite.
    energy_conserving: f32,
    /// Padding to 16-byte alignment (std140-compatible).
    _pad: vec2<f32>,
}

@group(0) @binding(0) var src_texture: texture_2d_array<f32>;
@group(0) @binding(1) var src_sampler: sampler;
@group(0) @binding(2) var<uniform> uniforms: BloomUniforms;

/// Composite-only: bloom mip 0 sampled during the final combine.
@group(1) @binding(0) var bloom_texture: texture_2d_array<f32>;

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> fs::FullscreenVertexOutput {
    return fs::vertex_main(vid);
}

// Rec. 709 luminance (linear space).
fn rec709_luminance(v: vec3<f32>) -> f32 {
    return dot(v, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Karis firefly reduction weight (1 / (1 + luma/4)), applied per-group on the first downsample.
fn karis_average(color: vec3<f32>) -> f32 {
    let luma = rec709_luminance(color) / 4.0;
    return 1.0 / (1.0 + luma);
}

// Unity-style quadratic soft-knee threshold.
fn soft_threshold(color: vec3<f32>) -> vec3<f32> {
    let brightness = max(color.r, max(color.g, color.b));
    var softness = brightness - uniforms.threshold_precomputations.y;
    softness = clamp(softness, 0.0, uniforms.threshold_precomputations.z);
    softness = softness * softness * uniforms.threshold_precomputations.w;
    var contribution = max(brightness - uniforms.threshold_precomputations.x, softness);
    contribution /= max(brightness, 1e-5);
    return color * contribution;
}

/// Holds the 5 weighted sample groups the 13-tap kernel produces. `fs_downsample_first` applies
/// Karis per-group before summing; `fs_downsample` sums directly.
struct Tap13Groups {
    g0: vec3<f32>,
    g1: vec3<f32>,
    g2: vec3<f32>,
    g3: vec3<f32>,
    g4: vec3<f32>,
}

fn sample_13_groups(uv: vec2<f32>, view: u32) -> Tap13Groups {
    let a = textureSample(src_texture, src_sampler, uv, view, vec2<i32>(-2,  2)).rgb;
    let b = textureSample(src_texture, src_sampler, uv, view, vec2<i32>( 0,  2)).rgb;
    let c = textureSample(src_texture, src_sampler, uv, view, vec2<i32>( 2,  2)).rgb;
    let d = textureSample(src_texture, src_sampler, uv, view, vec2<i32>(-2,  0)).rgb;
    let e = textureSample(src_texture, src_sampler, uv, view).rgb;
    let f = textureSample(src_texture, src_sampler, uv, view, vec2<i32>( 2,  0)).rgb;
    let g = textureSample(src_texture, src_sampler, uv, view, vec2<i32>(-2, -2)).rgb;
    let h = textureSample(src_texture, src_sampler, uv, view, vec2<i32>( 0, -2)).rgb;
    let i = textureSample(src_texture, src_sampler, uv, view, vec2<i32>( 2, -2)).rgb;
    let j = textureSample(src_texture, src_sampler, uv, view, vec2<i32>(-1,  1)).rgb;
    let k = textureSample(src_texture, src_sampler, uv, view, vec2<i32>( 1,  1)).rgb;
    let l = textureSample(src_texture, src_sampler, uv, view, vec2<i32>(-1, -1)).rgb;
    let m = textureSample(src_texture, src_sampler, uv, view, vec2<i32>( 1, -1)).rgb;
    var out: Tap13Groups;
    out.g0 = (a + b + d + e) * (0.125 / 4.0);
    out.g1 = (b + c + e + f) * (0.125 / 4.0);
    out.g2 = (d + e + g + h) * (0.125 / 4.0);
    out.g3 = (e + f + h + i) * (0.125 / 4.0);
    out.g4 = (j + k + l + m) * (0.5   / 4.0);
    return out;
}

fn sample_plain_13_tap(uv: vec2<f32>, view: u32) -> vec3<f32> {
    let a = textureSample(src_texture, src_sampler, uv, view, vec2<i32>(-2,  2)).rgb;
    let b = textureSample(src_texture, src_sampler, uv, view, vec2<i32>( 0,  2)).rgb;
    let c = textureSample(src_texture, src_sampler, uv, view, vec2<i32>( 2,  2)).rgb;
    let d = textureSample(src_texture, src_sampler, uv, view, vec2<i32>(-2,  0)).rgb;
    let e = textureSample(src_texture, src_sampler, uv, view).rgb;
    let f = textureSample(src_texture, src_sampler, uv, view, vec2<i32>( 2,  0)).rgb;
    let g = textureSample(src_texture, src_sampler, uv, view, vec2<i32>(-2, -2)).rgb;
    let h = textureSample(src_texture, src_sampler, uv, view, vec2<i32>( 0, -2)).rgb;
    let i = textureSample(src_texture, src_sampler, uv, view, vec2<i32>( 2, -2)).rgb;
    let j = textureSample(src_texture, src_sampler, uv, view, vec2<i32>(-1,  1)).rgb;
    let k = textureSample(src_texture, src_sampler, uv, view, vec2<i32>( 1,  1)).rgb;
    let l = textureSample(src_texture, src_sampler, uv, view, vec2<i32>(-1, -1)).rgb;
    let m = textureSample(src_texture, src_sampler, uv, view, vec2<i32>( 1, -1)).rgb;
    var sample = (a + c + g + i) * 0.03125;
    sample += (b + d + f + h) * 0.0625;
    sample += (e + j + k + l + m) * 0.125;
    return sample;
}

fn sample_tent_3x3(uv: vec2<f32>, view: u32) -> vec3<f32> {
    let a = textureSample(src_texture, src_sampler, uv, view, vec2<i32>(-1,  1)).rgb;
    let b = textureSample(src_texture, src_sampler, uv, view, vec2<i32>( 0,  1)).rgb;
    let c = textureSample(src_texture, src_sampler, uv, view, vec2<i32>( 1,  1)).rgb;
    let d = textureSample(src_texture, src_sampler, uv, view, vec2<i32>(-1,  0)).rgb;
    let e = textureSample(src_texture, src_sampler, uv, view).rgb;
    let f = textureSample(src_texture, src_sampler, uv, view, vec2<i32>( 1,  0)).rgb;
    let g = textureSample(src_texture, src_sampler, uv, view, vec2<i32>(-1, -1)).rgb;
    let h = textureSample(src_texture, src_sampler, uv, view, vec2<i32>( 0, -1)).rgb;
    let i = textureSample(src_texture, src_sampler, uv, view, vec2<i32>( 1, -1)).rgb;
    var sample = e * 0.25;
    sample += (b + d + f + h) * 0.125;
    sample += (a + c + g + i) * 0.0625;
    return sample;
}

#ifdef MULTIVIEW
@fragment
fn fs_downsample_first(in: fs::FullscreenVertexOutput, @builtin(view_index) view: u32) -> @location(0) vec4<f32> {
    var groups = sample_13_groups(in.uv, view);
#else
@fragment
fn fs_downsample_first(in: fs::FullscreenVertexOutput) -> @location(0) vec4<f32> {
    var groups = sample_13_groups(in.uv, 0u);
#endif
    groups.g0 *= karis_average(groups.g0);
    groups.g1 *= karis_average(groups.g1);
    groups.g2 *= karis_average(groups.g2);
    groups.g3 *= karis_average(groups.g3);
    groups.g4 *= karis_average(groups.g4);
    var sample = groups.g0 + groups.g1 + groups.g2 + groups.g3 + groups.g4;
    // Clamp below f32::MAX to prevent NaN propagation through the downscale/upscale chain.
    sample = clamp(sample, vec3<f32>(1e-4), vec3<f32>(3.40282347e+37));
    if (uniforms.threshold_precomputations.x > 0.0) {
        sample = soft_threshold(sample);
    }
    return vec4<f32>(sample, 1.0);
}

#ifdef MULTIVIEW
@fragment
fn fs_downsample(in: fs::FullscreenVertexOutput, @builtin(view_index) view: u32) -> @location(0) vec4<f32> {
    return vec4<f32>(sample_plain_13_tap(in.uv, view), 1.0);
}
#else
@fragment
fn fs_downsample(in: fs::FullscreenVertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(sample_plain_13_tap(in.uv, 0u), 1.0);
}
#endif

#ifdef MULTIVIEW
@fragment
fn fs_upsample(in: fs::FullscreenVertexOutput, @builtin(view_index) view: u32) -> @location(0) vec4<f32> {
    return vec4<f32>(sample_tent_3x3(in.uv, view), 1.0);
}
#else
@fragment
fn fs_upsample(in: fs::FullscreenVertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(sample_tent_3x3(in.uv, 0u), 1.0);
}
#endif

#ifdef MULTIVIEW
@fragment
fn fs_composite(in: fs::FullscreenVertexOutput, @builtin(view_index) view: u32) -> @location(0) vec4<f32> {
    let scene = textureSample(src_texture, src_sampler, in.uv, view);
    let bloom = textureSample(bloom_texture, src_sampler, in.uv, view).rgb;
#else
@fragment
fn fs_composite(in: fs::FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let scene = textureSample(src_texture, src_sampler, in.uv, 0u);
    let bloom = textureSample(bloom_texture, src_sampler, in.uv, 0u).rgb;
#endif
    let t = clamp(uniforms.intensity, 0.0, 1.0);
    // Energy-conserving: out = t * bloom + (1 - t) * scene.
    let energy = t * bloom + (1.0 - t) * scene.rgb;
    // Additive: out = scene + t * bloom.
    let additive = scene.rgb + uniforms.intensity * bloom;
    let rgb = mix(additive, energy, uniforms.energy_conserving);
    return vec4<f32>(rgb, scene.a);
}
