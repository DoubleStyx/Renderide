//! Shared per-frame bindings (`@group(0)`) for all raster materials.
//! Import with `#import renderide::globals` from `shaders/materials/*.wgsl`.
//!
//! Composed materials that do not otherwise touch lighting storage must retain `lights`,
//! `cluster_light_counts`, and `cluster_light_indices` (e.g. through `retain_globals_additive`);
//! the composer drops unused globals, which breaks the fixed [`FrameGpuResources`] bind group
//! layout at pipeline creation for storage-backed frame resources.
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
    /// Right-eye world-space camera position (equals left/mono outside stereo multiview).
    camera_world_pos_right: vec4<f32>,
    /// Left-eye (or mono) world -> view-space Z coefficients.
    view_space_z_coeffs: vec4<f32>,
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
    /// Left-eye (or mono) projection coefficients `(P[0][0], P[1][1], P[0][2], P[1][2])`.
    /// Screen-to-view unprojection (view Z known):
    /// `view_x = (ndc_x - c.z) * view_z / c.x`, `view_y = (ndc_y - c.w) * view_z / c.y`.
    proj_params_left: vec4<f32>,
    /// Right-eye projection coefficients (equals left in mono mode).
    proj_params_right: vec4<f32>,
    /// Packed tail: `.x` is the monotonic frame index (for temporal / jittered screen-space
    /// effects); `.yzw` are reserved padding to keep the trailing `vec4` slot aligned to 16 bytes.
    frame_tail: vec4<u32>,
    /// Skybox indirect specular: `.x` max resident LOD, `.y` enabled flag,
    /// `.z` source kind tag, `.w` reserved padding.
    skybox_specular: vec4<f32>,
    /// Ambient SH2 coefficient 0, padded to a vec4 slot.
    ambient_sh_a: vec4<f32>,
    /// Ambient SH2 coefficient 1, padded to a vec4 slot.
    ambient_sh_b: vec4<f32>,
    /// Ambient SH2 coefficient 2, padded to a vec4 slot.
    ambient_sh_c: vec4<f32>,
    /// Ambient SH2 coefficient 3, padded to a vec4 slot.
    ambient_sh_d: vec4<f32>,
    /// Ambient SH2 coefficient 4, padded to a vec4 slot.
    ambient_sh_e: vec4<f32>,
    /// Ambient SH2 coefficient 5, padded to a vec4 slot.
    ambient_sh_f: vec4<f32>,
    /// Ambient SH2 coefficient 6, padded to a vec4 slot.
    ambient_sh_g: vec4<f32>,
    /// Ambient SH2 coefficient 7, padded to a vec4 slot.
    ambient_sh_h: vec4<f32>,
    /// Ambient SH2 coefficient 8, padded to a vec4 slot.
    ambient_sh_i: vec4<f32>,
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
@group(0) @binding(9) var skybox_specular: texture_cube<f32>;
@group(0) @binding(10) var skybox_specular_sampler: sampler;
@group(0) @binding(11) var ibl_dfg_lut: texture_2d<f32>;

/// World-space camera position for the current view layer.
fn camera_world_pos_for_view(view_layer: u32) -> vec3<f32> {
#ifdef MULTIVIEW
    if (view_layer != 0u) {
        return frame.camera_world_pos_right.xyz;
    }
#endif
    return frame.camera_world_pos.xyz;
}

/// Unit vector from `world_pos` toward the current view-layer camera.
fn view_dir_for_world_pos(world_pos: vec3<f32>, view_layer: u32) -> vec3<f32> {
    return normalize(camera_world_pos_for_view(view_layer) - world_pos);
}

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
