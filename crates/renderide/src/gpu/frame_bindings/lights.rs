//! GPU light row layout and the per-frame light buffer cap consumed by `@group(0)` shaders.

use bytemuck::{Pod, Zeroable};

/// Max lights copied into the frame light buffer.
pub const MAX_LIGHTS: usize = 65536;
/// Max shadow-view records copied into the frame shadow metadata buffer.
pub const MAX_SHADOW_VIEWS: usize = 512;

/// No light cookie is bound.
pub const LIGHT_COOKIE_KIND_NONE: u32 = 0;
/// A spotlight cookie sampled from the 2D cookie atlas.
pub const LIGHT_COOKIE_KIND_SPOT_2D: u32 = 1;
/// A point-light cookie sampled from the cubemap-face atlas.
pub const LIGHT_COOKIE_KIND_POINT_CUBE: u32 = 2;
/// A directional light cookie sampled from the 2D cookie atlas.
pub const LIGHT_COOKIE_KIND_DIRECTIONAL_2D: u32 = 3;

/// No shadow-view record is bound.
pub const SHADOW_VIEW_KIND_NONE: u32 = 0;
/// Directional-light cascaded shadow-view record.
pub const SHADOW_VIEW_KIND_DIRECTIONAL: u32 = 1;
/// Spot-light projected shadow-view record.
pub const SHADOW_VIEW_KIND_SPOT: u32 = 2;
/// Point-light cubemap-face shadow-view record.
pub const SHADOW_VIEW_KIND_POINT: u32 = 3;

/// Cookie U-axis wrap mode bit shift.
pub const LIGHT_COOKIE_WRAP_U_SHIFT: u32 = 0;
/// Cookie V-axis wrap mode bit shift.
pub const LIGHT_COOKIE_WRAP_V_SHIFT: u32 = 2;
/// Cookie wrap mode bit mask for one axis.
pub const LIGHT_COOKIE_WRAP_MODE_MASK: u32 = 0b11;
/// Repeating cookie address mode.
pub const LIGHT_COOKIE_WRAP_MODE_REPEAT: u32 = 0;
/// Clamp-to-edge cookie address mode.
pub const LIGHT_COOKIE_WRAP_MODE_CLAMP: u32 = 1;
/// Mirrored repeat cookie address mode.
pub const LIGHT_COOKIE_WRAP_MODE_MIRROR: u32 = 2;
/// Mirror-once cookie address mode.
pub const LIGHT_COOKIE_WRAP_MODE_MIRROR_ONCE: u32 = 3;

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
    /// First row in the frame shadow-view storage buffer.
    pub shadow_view_start: u32,
    /// Number of shadow-view rows assigned to this light.
    pub shadow_view_count: u32,
    /// Packed shadow metadata flags.
    pub shadow_flags: u32,
    /// Aligns shadow metadata to a 16-byte lane.
    pub _shadow_reserved: u32,
    /// Cookie kind, matching `LIGHT_COOKIE_KIND_*`.
    pub cookie_kind: u32,
    /// 2D atlas layer or first point-cubemap face layer.
    pub cookie_layer: u32,
    /// Packed cookie wrap modes for 2D cookie sampling.
    pub _cookie_reserved: u32,
    /// World-space local +X axis; `.w` stores the spot half-angle tangent or directional cookie size.
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
            shadow_view_start: 0,
            shadow_view_count: 0,
            shadow_flags: 0,
            _shadow_reserved: 0,
            cookie_kind: LIGHT_COOKIE_KIND_NONE,
            cookie_layer: 0,
            _cookie_reserved: 0,
            cookie_right_tan_half_angle: [1.0, 0.0, 0.0, 1.0],
            cookie_up: [0.0, 1.0, 0.0, 0.0],
        }
    }
}

/// GPU-facing metadata for one rendered shadow-map view.
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct GpuShadowView {
    /// Matrix from world space to shadow clip space.
    pub world_to_shadow: [[f32; 4]; 4],
    /// Atlas UV rect as `origin.xy, scale.xy`.
    pub atlas_rect: [f32; 4],
    /// Atlas layer, texel size, cascade near, and cascade far.
    pub params: [f32; 4],
    /// Light type, point face, normal bias, and depth bias.
    pub light_params: [f32; 4],
}

impl Default for GpuShadowView {
    fn default() -> Self {
        Self {
            world_to_shadow: glam::Mat4::IDENTITY.to_cols_array_2d(),
            atlas_rect: [0.0, 0.0, 1.0, 1.0],
            params: [0.0, 1.0, 0.0, 1.0],
            light_params: [SHADOW_VIEW_KIND_NONE as f32, 0.0, 0.0, 0.0],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;

    #[test]
    fn gpu_light_row_size_matches_wgsl_storage_stride() {
        assert_eq!(size_of::<GpuLight>(), 144);
    }

    #[test]
    fn gpu_shadow_view_row_size_matches_wgsl_storage_stride() {
        assert_eq!(size_of::<GpuShadowView>(), 112);
    }
}
