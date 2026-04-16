//! Shared per-frame bindings (`@group(0)`) for all raster materials.
//! Import with `#import renderide::globals` from `source/materials/*.wgsl`.
//!
//! Composed materials must reference every binding below (e.g. a no-op `cluster_light_counts` /
//! `cluster_light_indices` touch in the fragment shader if unused); the composer drops unused globals,
//! which breaks the fixed [`FrameGpuResources`] bind group layout at pipeline creation.
//!
//! CPU packing must match [`crate::gpu::frame_globals::FrameGpuUniforms`],
//! [`crate::backend::light_gpu::GpuLight`], and [`crate::backend::cluster_gpu`] cluster buffers.

#define_import_path renderide::globals

struct GpuLight {
    position: vec3<f32>,
    align_pad_vec3_pos: f32,
    direction: vec3<f32>,
    align_pad_vec3_dir: f32,
    color: vec3<f32>,
    intensity: f32,
    range: f32,
    spot_cos_half_angle: f32,
    light_type: u32,
    align_pad_before_shadow: u32,
    shadow_strength: f32,
    shadow_near_plane: f32,
    shadow_bias: f32,
    shadow_normal_bias: f32,
    shadow_type: u32,
    align_pad_vec3_tail: vec3<u32>,
}

/// Per-frame scene + clustered grid (matches [`crate::gpu::frame_globals::FrameGpuUniforms`]).
struct FrameGlobals {
    camera_world_pos: vec4<f32>,
    /// Left-eye (or mono) world -> view-space X coefficients.
    view_space_x_coeffs: vec4<f32>,
    /// Left-eye (or mono) world -> view-space Y coefficients.
    view_space_y_coeffs: vec4<f32>,
    /// Left-eye (or mono) world -> view-space Z coefficients.
    view_space_z_coeffs: vec4<f32>,
    /// Right-eye world -> view-space X coefficients (equals left in mono mode).
    view_space_x_coeffs_right: vec4<f32>,
    /// Right-eye world -> view-space Y coefficients (equals left in mono mode).
    view_space_y_coeffs_right: vec4<f32>,
    /// Right-eye world -> view-space Z coefficients (equals left in mono mode).
    view_space_z_coeffs_right: vec4<f32>,
    cluster_count_x: u32,
    cluster_count_y: u32,
    cluster_count_z: u32,
    near_clip: f32,
    far_clip: f32,
    light_count: u32,
    viewport_width: u32,
    viewport_height: u32,
}

@group(0) @binding(0) var<uniform> frame: FrameGlobals;
@group(0) @binding(1) var<storage, read> lights: array<GpuLight>;
@group(0) @binding(2) var<storage, read> cluster_light_counts: array<u32>;
@group(0) @binding(3) var<storage, read> cluster_light_indices: array<u32>;
@group(0) @binding(4) var scene_depth: texture_depth_2d;
@group(0) @binding(5) var scene_depth_array: texture_depth_2d_array;
@group(0) @binding(6) var scene_color: texture_2d<f32>;
@group(0) @binding(7) var scene_color_array: texture_2d_array<f32>;
@group(0) @binding(8) var scene_color_sampler: sampler;

/// Adds infinitesimal terms tied to lights/cluster storage so every frame binding stays referenced
/// when a material would otherwise not touch storage (naga-oil drops unused globals).
fn retain_globals_additive(color: vec4<f32>) -> vec4<f32> {
    var lit: u32 = 0u;
    if (frame.light_count > 0u) {
        lit = lights[0].light_type;
    }
    let cluster_touch =
        f32(cluster_light_counts[0u] & 255u) * 1e-10 +
        f32(cluster_light_indices[0u] & 255u) * 1e-10;
    return color + vec4<f32>(vec3<f32>(f32(lit) * 1e-10 + cluster_touch), 0.0);
}
