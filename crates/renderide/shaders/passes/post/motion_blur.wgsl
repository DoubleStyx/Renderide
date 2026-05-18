//! Screen-space motion vector production and HDR motion blur resolve.
//!
//! Build script emits `motion_blur_default` and `motion_blur_multiview` targets. The velocity
//! pass writes UV-space motion into an `rg16float` render target. The blur pass samples HDR color
//! along that vector before tonemapping.

#import renderide::core::fullscreen as fs

const MAX_MOTION_BLUR_SAMPLES: u32 = 16u;

struct MotionVectorUniforms {
    current_clip_to_prev_clip_left: mat4x4<f32>,
    current_clip_to_prev_clip_right: mat4x4<f32>,
    viewport_px: vec2<f32>,
    history_valid: f32,
    _pad0: f32,
}

struct MotionBlurUniforms {
    shutter_angle: f32,
    max_velocity_pixels: f32,
    sample_count: u32,
    enabled: u32,
    viewport_px: vec2<f32>,
    _pad0: vec2<f32>,
}

#ifdef MULTIVIEW
@group(0) @binding(0) var frame_depth: texture_depth_2d_array;
#else
@group(0) @binding(0) var frame_depth: texture_depth_2d;
#endif
@group(0) @binding(1) var<uniform> motion_vectors: MotionVectorUniforms;

@group(1) @binding(0) var scene_color_hdr: texture_2d_array<f32>;
@group(1) @binding(1) var scene_color_sampler: sampler;
@group(1) @binding(2) var velocity_texture: texture_2d_array<f32>;
@group(1) @binding(3) var<uniform> blur: MotionBlurUniforms;

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> fs::FullscreenVertexOutput {
    return fs::vertex_main(vid);
}

fn uv_to_clip_xy(uv: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
}

fn clip_xy_to_uv(ndc: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
}

fn motion_matrix_for_view(view: u32) -> mat4x4<f32> {
    if (view == 1u) {
        return motion_vectors.current_clip_to_prev_clip_right;
    }
    return motion_vectors.current_clip_to_prev_clip_left;
}

fn velocity_from_depth(uv: vec2<f32>, depth: f32, view: u32) -> vec2<f32> {
    if (motion_vectors.history_valid <= 0.0) {
        return vec2<f32>(0.0);
    }

    let current_clip = vec4<f32>(uv_to_clip_xy(uv), depth, 1.0);
    let previous_clip = motion_matrix_for_view(view) * current_clip;
    if (abs(previous_clip.w) < 1e-6) {
        return vec2<f32>(0.0);
    }

    let previous_ndc = previous_clip.xy / previous_clip.w;
    let previous_uv = clip_xy_to_uv(previous_ndc);
    let velocity = previous_uv - uv;
    return velocity;
}

#ifdef MULTIVIEW
@fragment
fn fs_motion_vectors(
    in: fs::FullscreenVertexOutput,
    @builtin(view_index) view: u32,
) -> @location(0) vec2<f32> {
    let pixel = vec2<i32>(in.clip_pos.xy);
    let depth = textureLoad(frame_depth, pixel, i32(view), 0);
    return velocity_from_depth(in.uv, depth, view);
}
#else
@fragment
fn fs_motion_vectors(
    in: fs::FullscreenVertexOutput,
) -> @location(0) vec2<f32> {
    let pixel = vec2<i32>(in.clip_pos.xy);
    let depth = textureLoad(frame_depth, pixel, 0);
    return velocity_from_depth(in.uv, depth, 0u);
}
#endif

fn hash12(p: vec2<f32>) -> f32 {
    let h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453123);
}

fn clamp_velocity_to_radius(velocity_uv: vec2<f32>) -> vec2<f32> {
    let pixel_velocity = velocity_uv * blur.viewport_px;
    let pixel_len = length(pixel_velocity);
    if (pixel_len <= blur.max_velocity_pixels || pixel_len <= 1e-5) {
        return velocity_uv;
    }
    return (pixel_velocity * (blur.max_velocity_pixels / pixel_len)) / blur.viewport_px;
}

fn motion_blur_sample(uv: vec2<f32>, view: u32) -> vec4<f32> {
    let center = textureSample(scene_color_hdr, scene_color_sampler, uv, view);
    if (blur.enabled == 0u || blur.sample_count == 0u || blur.shutter_angle <= 0.0) {
        return center;
    }

    var velocity = textureSample(velocity_texture, scene_color_sampler, uv, view).xy;
    velocity = clamp_velocity_to_radius(velocity * blur.shutter_angle);
    if (length(velocity * blur.viewport_px) < 1.0) {
        return center;
    }

    let count = min(blur.sample_count, MAX_MOTION_BLUR_SAMPLES);
    let denom = f32(max(count, 1u));
    let jitter = hash12(uv * blur.viewport_px) - 0.5;

    var sum = vec4<f32>(0.0);
    var weight = 0.0;
    for (var i = 0u; i < MAX_MOTION_BLUR_SAMPLES; i = i + 1u) {
        if (i >= count) {
            continue;
        }
        let t = (f32(i) + 0.5 + jitter) / denom - 0.5;
        let sample_uv = uv + velocity * t;
        if (any(sample_uv < vec2<f32>(0.0)) || any(sample_uv > vec2<f32>(1.0))) {
            continue;
        }
        sum = sum + textureSample(scene_color_hdr, scene_color_sampler, sample_uv, view);
        weight = weight + 1.0;
    }

    if (weight <= 0.0) {
        return center;
    }
    return sum / weight;
}

#ifdef MULTIVIEW
@fragment
fn fs_motion_blur(in: fs::FullscreenVertexOutput, @builtin(view_index) view: u32) -> @location(0) vec4<f32> {
    return motion_blur_sample(in.uv, view);
}
#else
@fragment
fn fs_motion_blur(in: fs::FullscreenVertexOutput) -> @location(0) vec4<f32> {
    return motion_blur_sample(in.uv, 0u);
}
#endif
