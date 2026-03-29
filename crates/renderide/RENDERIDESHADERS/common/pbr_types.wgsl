#define_import_path renderide_pbr_types

/// Clustered forward lighting: GPU light record (matches Rust `GpuLight` layout).
struct GpuLight {
    position: vec3f,
    align_pad_a: f32,
    direction: vec3f,
    align_pad_b: f32,
    color: vec3f,
    intensity: f32,
    range: f32,
    spot_cos_half_angle: f32,
    light_type: u32,
    align_pad_shadow: u32,
    shadow_strength: f32,
    shadow_near_plane: f32,
    shadow_bias: f32,
    shadow_normal_bias: f32,
    shadow_type: u32,
    align_pad_tail: array<u32, 3>,
}

/// Per-frame scene + cluster grid parameters for clustered lighting.
struct SceneUniforms {
    view_position: vec3f,
    align_pad_view: f32,
    view_space_z_coeffs: vec4f,
    cluster_count_x: u32,
    cluster_count_y: u32,
    cluster_count_z: u32,
    near_clip: f32,
    far_clip: f32,
    light_count: u32,
    viewport_width: u32,
    viewport_height: u32,
}

const TILE_SIZE: u32 = 16u;
const MAX_LIGHTS_PER_TILE: u32 = 32u;
