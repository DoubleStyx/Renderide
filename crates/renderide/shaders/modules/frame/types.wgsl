//! Shared per-frame GPU data layouts and constants.

#define_import_path renderide::frame::types

const FRAME_PROJECTION_FLAG_ORTHOGRAPHIC: u32 = 1u;
const FRAME_TAIL_SAMPLE_COUNT_SHIFT: u32 = 1u;
const FRAME_TAIL_SAMPLE_COUNT_MASK: u32 = 30u;
const LIGHT_COOKIE_KIND_NONE: u32 = 0u;
const LIGHT_COOKIE_KIND_SPOT_2D: u32 = 1u;
const LIGHT_COOKIE_KIND_POINT_CUBE: u32 = 2u;

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
    spot_angle_scale: f32,
    shadow_strength: f32,
    shadow_near_plane: f32,
    shadow_bias: f32,
    shadow_normal_bias: f32,
    shadow_type: u32,
    cookie_kind: u32,
    cookie_layer: u32,
    cookie_reserved: u32,
    cookie_right_tan_half_angle: vec4<f32>,
    cookie_up: vec4<f32>,
}

struct GpuReflectionProbe {
    /// World-space AABB minimum; `.w` stores the sanitized blend distance.
    box_min: vec4<f32>,
    box_max: vec4<f32>,
    position: vec4<f32>,
    params: vec4<f32>,
    sh2_a: vec4<f32>,
    sh2_b: vec4<f32>,
    sh2_c: vec4<f32>,
    sh2_d: vec4<f32>,
    sh2_e: vec4<f32>,
    sh2_f: vec4<f32>,
    sh2_g: vec4<f32>,
    sh2_h: vec4<f32>,
    sh2_i: vec4<f32>,
}

/// Per-frame scene + clustered grid.
struct FrameGlobals {
    camera_world_pos: vec4<f32>,
    /// Right-eye world-space camera position (equals left/mono outside stereo multiview).
    camera_world_pos_right: vec4<f32>,
    /// Left-eye (or mono) world -> view-space Z coefficients.
    view_space_z_coeffs: vec4<f32>,
    /// Right-eye world -> view-space Z coefficients (equals left in mono mode).
    view_space_z_coeffs_right: vec4<f32>,
    /// Left-eye (or mono) view -> world-space Y coefficients.
    view_to_world_y_coeffs: vec4<f32>,
    /// Right-eye view -> world-space Y coefficients (equals left in mono mode).
    view_to_world_y_coeffs_right: vec4<f32>,
    cluster_count_x: u32,
    cluster_count_y: u32,
    cluster_count_z: u32,
    near_clip: f32,
    far_clip: f32,
    light_count: u32,
    viewport_width: u32,
    viewport_height: u32,
    /// Left-eye (or mono) projection coefficients.
    proj_params_left: vec4<f32>,
    /// Right-eye projection coefficients (equals left in mono mode).
    proj_params_right: vec4<f32>,
    /// Packed frame metadata, projection flags, ambient-light flags, and raster sample count.
    frame_tail: vec4<u32>,
    /// Reserved. Skybox specular lighting is supplied by reflection probes.
    skybox_specular: vec4<f32>,
    /// Frame time values: `.x` is elapsed renderer seconds and `.yzw` are reserved.
    frame_time: vec4<f32>,
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
    /// Fog color in `.rgb` and fog mode in `.w`; zero mode disables fog.
    fog_color_mode: vec4<f32>,
    /// Unity-style fog parameters consumed by `renderide::frame::fog`.
    fog_params: vec4<f32>,
}
