//! Shared per-frame bindings (`@group(0)`) for all raster materials.
//! Import with `#import renderide::frame::globals` from `shaders/materials/*.wgsl`.
//!
//! Composed materials that do not otherwise touch lighting storage must retain `lights`,
//! `cluster_light_ranges`, and `cluster_light_indices` (e.g. through `retain_globals_additive`);
//! the composer drops unused globals, which breaks the fixed [`FrameGpuResources`] bind group
//! layout at pipeline creation for storage-backed frame resources.
//!
//! CPU packing must match [`crate::gpu::frame_globals::FrameGpuUniforms`],
//! [`crate::backend::light_gpu::GpuLight`], and [`crate::backend::cluster_gpu`] cluster buffers.

#define_import_path renderide::frame::globals

#import renderide::frame::types as ft

@group(0) @binding(0) var<uniform> frame: ft::FrameGlobals;
@group(0) @binding(1) var<storage, read> lights: array<ft::GpuLight>;
@group(0) @binding(2) var<storage, read> cluster_light_ranges: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read> cluster_light_indices: array<u32>;
@group(0) @binding(4) var scene_depth: texture_depth_2d;
@group(0) @binding(5) var scene_depth_array: texture_depth_2d_array;
@group(0) @binding(6) var scene_color: texture_2d<f32>;
@group(0) @binding(7) var scene_color_array: texture_2d_array<f32>;
@group(0) @binding(8) var scene_color_sampler: sampler;
@group(0) @binding(9) var reflection_probe_specular: texture_2d_array<f32>;
@group(0) @binding(10) var reflection_probe_specular_sampler: sampler;
@group(0) @binding(11) var ibl_dfg_lut: texture_2d<f32>;
@group(0) @binding(12) var<storage, read> reflection_probes: array<ft::GpuReflectionProbe>;

/// View index encoded in a material varying.
fn view_index_from_layer(view_layer: u32) -> u32 {
    return view_layer & 1u;
}

/// Draw row encoded in a material varying.
fn draw_index_from_layer(view_layer: u32) -> u32 {
    return view_layer >> 1u;
}

/// World-space camera position for the current view layer.
fn camera_world_pos_for_view(view_layer: u32) -> vec3<f32> {
#ifdef MULTIVIEW
    if (view_index_from_layer(view_layer) != 0u) {
        return frame.camera_world_pos_right.xyz;
    }
#endif
    return frame.camera_world_pos.xyz;
}

/// World-space stereo-center camera position for effects that must stay identical between eyes.
fn stereo_center_camera_world_pos() -> vec3<f32> {
#ifdef MULTIVIEW
    return (frame.camera_world_pos.xyz + frame.camera_world_pos_right.xyz) * 0.5;
#else
    return frame.camera_world_pos.xyz;
#endif
}

/// World -> view-space Z coefficients for the current view layer.
fn view_space_z_coeffs_for_view(view_layer: u32) -> vec4<f32> {
#ifdef MULTIVIEW
    if (view_index_from_layer(view_layer) != 0u) {
        return frame.view_space_z_coeffs_right;
    }
#endif
    return frame.view_space_z_coeffs;
}

/// Projection flags for the current view layer.
fn projection_flags_for_view(view_layer: u32) -> u32 {
#ifdef MULTIVIEW
    if (view_index_from_layer(view_layer) != 0u) {
        return frame.frame_tail.z;
    }
#endif
    return frame.frame_tail.y;
}

/// Raster sample count for the current frame target.
fn frame_sample_count() -> u32 {
    let sample_count = (frame.frame_tail.w & ft::FRAME_TAIL_SAMPLE_COUNT_MASK) >> ft::FRAME_TAIL_SAMPLE_COUNT_SHIFT;
    if (sample_count == 2u || sample_count == 4u || sample_count == 8u) {
        return sample_count;
    }
    return 1u;
}

/// Returns true when the current view layer is orthographic.
fn view_is_orthographic(view_layer: u32) -> bool {
    return (projection_flags_for_view(view_layer) & ft::FRAME_PROJECTION_FLAG_ORTHOGRAPHIC) != 0u;
}

fn safe_normalize_or(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let len = length(v);
    if (len <= 0.000001) {
        return fallback;
    }
    return v / len;
}

/// Unit vector from a world position toward the orthographic camera plane.
fn orthographic_view_dir_for_view(view_layer: u32) -> vec3<f32> {
    let z_coeffs = view_space_z_coeffs_for_view(view_layer);
    return safe_normalize_or(z_coeffs.xyz, vec3<f32>(0.0, 0.0, -1.0));
}

/// Unit vector from `world_pos` toward the current view-layer camera.
fn view_dir_for_world_pos(world_pos: vec3<f32>, view_layer: u32) -> vec3<f32> {
    let fallback = orthographic_view_dir_for_view(view_layer);
    if (view_is_orthographic(view_layer)) {
        return fallback;
    }
    return safe_normalize_or(camera_world_pos_for_view(view_layer) - world_pos, fallback);
}

/// Unit vector from `world_pos` toward the stereo-center camera.
fn stereo_center_view_dir_for_world_pos(world_pos: vec3<f32>, view_layer: u32) -> vec3<f32> {
    let fallback = orthographic_view_dir_for_view(view_layer);
    if (view_is_orthographic(view_layer)) {
        return fallback;
    }
    return safe_normalize_or(stereo_center_camera_world_pos() - world_pos, fallback);
}

/// Adds infinitesimal terms tied to lights/cluster storage so every frame binding stays referenced
/// when a material would otherwise not touch storage (naga-oil drops unused globals).
fn retain_globals_additive(color: vec4<f32>) -> vec4<f32> {
    var lit: u32 = 0u;
    if (frame.light_count > 0u) {
        lit = lights[0].light_type;
    }
    let cluster_touch =
        f32(cluster_light_ranges[0u].y & 255u) * 1e-10 +
        f32(cluster_light_indices[0u] & 255u) * 1e-10;
    let probe_touch = reflection_probes[0u].params.x * 1e-10;
    return color + vec4<f32>(vec3<f32>(f32(lit) * 1e-10 + cluster_touch + probe_touch), 0.0);
}
