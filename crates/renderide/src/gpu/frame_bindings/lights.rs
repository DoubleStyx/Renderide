//! GPU light row layout and the per-frame light buffer cap consumed by `@group(0)` shaders.

use bytemuck::{Pod, Zeroable};

/// Max lights copied into the frame light buffer.
pub const MAX_LIGHTS: usize = 65536;

/// No light cookie is bound.
pub const LIGHT_COOKIE_KIND_NONE: u32 = 0;
/// A spotlight cookie sampled from the 2D cookie atlas.
pub const LIGHT_COOKIE_KIND_SPOT_2D: u32 = 1;
/// A point-light cookie sampled from the cubemap-face atlas.
pub const LIGHT_COOKIE_KIND_POINT_CUBE: u32 = 2;

/// GPU-facing light record for a storage buffer upload.
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct GpuLight {
    /// Light position in world space.
    pub position: [f32; 3],
    /// Aligns `position` to 16 bytes for WGSL `vec3` storage rules.
    pub _pad0: f32,
    /// Forward axis for spot/directional lights.
    pub direction: [f32; 3],
    /// Aligns `direction` to 16 bytes.
    pub _pad1: f32,
    /// Host-authored sRGB/gamma light color.
    pub color: [f32; 3],
    /// Scalar brightness multiplier applied before shader-side light color transfer conversion.
    pub intensity: f32,
    /// Attenuation range in world units.
    pub range: f32,
    /// Cosine of the spot half-angle.
    pub spot_cos_half_angle: f32,
    /// Light type as a `u32`.
    pub light_type: u32,
    /// Multiplier for projected radial spot cone attenuation.
    pub spot_angle_scale: f32,
    /// Shadow strength / visibility factor.
    pub shadow_strength: f32,
    /// Shadow projection near plane.
    pub shadow_near_plane: f32,
    /// Depth bias for shadow sampling.
    pub shadow_bias: f32,
    /// Normal offset bias for shadowing.
    pub shadow_normal_bias: f32,
    /// Shadow type as a `u32`.
    pub shadow_type: u32,
    /// Cookie kind, matching `LIGHT_COOKIE_KIND_*`.
    pub cookie_kind: u32,
    /// Spot atlas layer or first point-cubemap face layer.
    pub cookie_layer: u32,
    /// Reserved for future cookie flags.
    pub _cookie_reserved: u32,
    /// World-space local +X axis; `.w` stores the spot half-angle tangent.
    pub cookie_right_tan_half_angle: [f32; 4],
    /// World-space local +Y axis; `.w` is reserved.
    pub cookie_up: [f32; 4],
}

impl Default for GpuLight {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            _pad0: 0.0,
            direction: [0.0, 0.0, 1.0],
            _pad1: 0.0,
            color: [1.0; 3],
            intensity: 1.0,
            range: 10.0,
            spot_cos_half_angle: 1.0,
            light_type: 0,
            spot_angle_scale: 0.0,
            shadow_strength: 0.0,
            shadow_near_plane: 0.0,
            shadow_bias: 0.0,
            shadow_normal_bias: 0.0,
            shadow_type: 0,
            cookie_kind: LIGHT_COOKIE_KIND_NONE,
            cookie_layer: 0,
            _cookie_reserved: 0,
            cookie_right_tan_half_angle: [1.0, 0.0, 0.0, 1.0],
            cookie_up: [0.0, 1.0, 0.0, 0.0],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;

    #[test]
    fn gpu_light_row_size_matches_wgsl_storage_stride() {
        assert_eq!(size_of::<GpuLight>(), 128);
    }
}
