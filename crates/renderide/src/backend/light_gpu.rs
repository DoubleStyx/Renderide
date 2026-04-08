//! GPU packing for scene lights (`storage` buffer layout / WGSL `struct` alignment).
//!
//! [`GpuLight`] uses 16-byte alignment for `vec3` slots to match typical WGSL storage rules.
//! [`LightType`](crate::shared::LightType) and [`ShadowType`](crate::shared::ShadowType) are stored as `u32`
//! with the same numeric values as `repr(u8)` on the wire.

use bytemuck::{Pod, Zeroable};

use crate::scene::ResolvedLight;
use crate::shared::{LightType, ShadowType};

/// Max lights copied into the scratch buffer (`prepare_lights_from_scene`).
pub const MAX_LIGHTS: usize = 256;

/// GPU-facing light record for a storage buffer upload.
///
/// Layout must match `GpuLight` in `shaders/source/modules/globals.wgsl`: WGSL aligns `vec3<u32>`
/// (`_pad_trailing`) to **16 bytes**, so 4 bytes of implicit padding follow `shadow_type`, and the
/// struct size rounds up to **112** bytes (see naga / WebGPU validation).
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct GpuLight {
    pub position: [f32; 3],
    pub _pad0: f32,
    pub direction: [f32; 3],
    pub _pad1: f32,
    pub color: [f32; 3],
    pub intensity: f32,
    pub range: f32,
    pub spot_cos_half_angle: f32,
    pub light_type: u32,
    pub _pad_before_shadow_params: u32,
    pub shadow_strength: f32,
    pub shadow_near_plane: f32,
    pub shadow_bias: f32,
    pub shadow_normal_bias: f32,
    pub shadow_type: u32,
    /// Padding so `_pad_trailing` starts at byte offset 88 (16-byte aligned for `vec3<u32>`).
    pub _pad_align_vec3_trailing: [u8; 4],
    pub _pad_trailing: [u32; 3],
    /// Pads struct size to 112 bytes (WGSL struct alignment).
    pub _pad_struct_end: [u8; 12],
}

unsafe impl Pod for GpuLight {}
unsafe impl Zeroable for GpuLight {}

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
            _pad_before_shadow_params: 0,
            shadow_strength: 0.0,
            shadow_near_plane: 0.0,
            shadow_bias: 0.0,
            shadow_normal_bias: 0.0,
            shadow_type: 0,
            _pad_align_vec3_trailing: [0; 4],
            _pad_trailing: [0; 3],
            _pad_struct_end: [0; 12],
        }
    }
}

impl GpuLight {
    /// Packs a [`ResolvedLight`] for GPU consumption.
    pub fn from_resolved(light: &ResolvedLight) -> Self {
        let spot_cos_half_angle = if light.spot_angle > 0.0 && light.spot_angle < 180.0 {
            (light.spot_angle.to_radians() / 2.0).cos()
        } else {
            1.0
        };
        Self {
            position: [
                light.world_position.x,
                light.world_position.y,
                light.world_position.z,
            ],
            _pad0: 0.0,
            direction: [
                light.world_direction.x,
                light.world_direction.y,
                light.world_direction.z,
            ],
            _pad1: 0.0,
            color: [light.color.x, light.color.y, light.color.z],
            intensity: light.intensity,
            range: light.range.max(0.001),
            spot_cos_half_angle,
            light_type: light_type_u32(light.light_type),
            _pad_before_shadow_params: 0,
            shadow_strength: light.shadow_strength,
            shadow_near_plane: light.shadow_near_plane,
            shadow_bias: light.shadow_bias,
            shadow_normal_bias: light.shadow_normal_bias,
            shadow_type: shadow_type_u32(light.shadow_type),
            _pad_align_vec3_trailing: [0; 4],
            _pad_trailing: [0; 3],
            _pad_struct_end: [0; 12],
        }
    }
}

fn light_type_u32(ty: LightType) -> u32 {
    match ty {
        LightType::point => 0,
        LightType::directional => 1,
        LightType::spot => 2,
    }
}

fn shadow_type_u32(ty: ShadowType) -> u32 {
    match ty {
        ShadowType::none => 0,
        ShadowType::hard => 1,
        ShadowType::soft => 2,
    }
}

/// Directional lights first (clustered forward compatibility); then point/spot; stable within bucket.
pub fn order_lights_for_clustered_shading(lights: &[ResolvedLight]) -> Vec<ResolvedLight> {
    let mut v: Vec<ResolvedLight> = lights.iter().take(MAX_LIGHTS).cloned().collect();
    v.sort_by_key(|l| match l.light_type {
        LightType::directional => 0u8,
        LightType::point | LightType::spot => 1,
    });
    v
}

#[cfg(test)]
mod layout_tests {
    use super::GpuLight;

    #[test]
    fn gpu_light_stride_matches_wgsl() {
        assert_eq!(
            std::mem::size_of::<GpuLight>(),
            112,
            "must match WGSL storage layout for `array<GpuLight>` (naga stride)"
        );
    }
}
