//! Shared per-frame bindings (`@group(0)`) for all raster materials.
//! Import with `#import renderide::globals` from `source/materials/*.wgsl`.
//!
//! CPU packing must match [`crate::gpu::frame_globals::FrameGpuUniforms`] and
//! [`crate::backend::light_gpu::GpuLight`].

#define_import_path renderide::globals

struct GpuLight {
    position: vec3<f32>,
    _pad0: f32,
    direction: vec3<f32>,
    _pad1: f32,
    color: vec3<f32>,
    intensity: f32,
    range: f32,
    spot_cos_half_angle: f32,
    light_type: u32,
    _pad_before_shadow_params: u32,
    shadow_strength: f32,
    shadow_near_plane: f32,
    shadow_bias: f32,
    shadow_normal_bias: f32,
    shadow_type: u32,
    _pad_trailing: vec3<u32>,
}

struct FrameGlobals {
    camera_world_pos: vec4<f32>,
    light_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> frame: FrameGlobals;
@group(0) @binding(1) var<storage, read> lights: array<GpuLight>;
