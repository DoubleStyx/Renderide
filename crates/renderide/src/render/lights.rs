//! GPU light buffer and clustered light infrastructure.
//!
//! Provides [`GpuLight`] for shader upload and [`LightBufferCache`] for per-frame
//! light buffer management.
//!
//! ## Clustered forward and light order
//!
//! The clustered light compute pass keeps at most 32 lights per cluster in buffer index order.
//! [`order_lights_for_clustered_shading`] places [`LightType::directional`] entries first so sun /
//! key directionals are not dropped when many local lights fill the per-cluster cap.
//!
//! ## `light_type` in WGSL
//!
//! Must match [`crate::shared::LightType`] `repr(u8)` and host wire values: `point = 0`,
//! `directional = 1`, `spot = 2`.
//!
//! ## `direction` field
//!
//! World-space **propagation** direction (local +Z / host `transform.forward` after resolve).
//! Directional shading uses `normalize(-direction)` for the **to-light** vector in WGSL.

use std::mem::size_of;

use bytemuck::{Pod, Zeroable};

use crate::scene::ResolvedLight;
use crate::shared::LightType;

/// Maximum number of lights in the GPU buffer.
pub const MAX_LIGHTS: usize = 256;

/// GPU-friendly light struct for shader upload.
/// Aligned for WGSL (vec3 = 16 bytes). Total size 80 to match WGSL storage buffer stride.
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
    /// Padding for WGSL std430: vec4u requires 16-byte alignment after light_type (offset 60).
    pub _pad_before_vec4: [u32; 1],
    pub _pad2: [u32; 4],
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
            _pad_before_vec4: [0; 1],
            _pad2: [0; 4],
        }
    }
}

impl GpuLight {
    /// Converts a [`ResolvedLight`] to [`GpuLight`] for GPU upload.
    ///
    /// `light_type` is encoded for WGSL: `0` = point, `1` = directional, `2` = spot (same as
    /// [`LightType`] on the wire).
    pub fn from_resolved(light: &ResolvedLight) -> Self {
        let spot_cos_half_angle = if light.spot_angle > 0.0 && light.spot_angle < 180.0 {
            (light.spot_angle.to_radians() / 2.0).cos()
        } else {
            1.0
        };
        let light_type = light_type_u32(light.light_type);
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
            light_type,
            _pad_before_vec4: [0; 1],
            _pad2: [0; 4],
        }
    }
}

/// Returns the `light_type` field stored in [`GpuLight`] / WGSL (0 = point, 1 = directional, 2 = spot).
pub fn light_type_u32(ty: LightType) -> u32 {
    match ty {
        LightType::point => 0,
        LightType::directional => 1,
        LightType::spot => 2,
    }
}

/// Copies up to [`MAX_LIGHTS`] lights with all [`LightType::directional`] entries first (stable).
///
/// Clustered shading assigns lights to clusters in buffer index order and caps at 32 per cluster;
/// directionals must appear early so they are not evicted by dense local lights.
pub fn order_lights_for_clustered_shading(lights: &[ResolvedLight]) -> Vec<ResolvedLight> {
    let mut v: Vec<ResolvedLight> = lights.iter().take(MAX_LIGHTS).cloned().collect();
    v.sort_by_key(|l| match l.light_type {
        LightType::directional => 0u8,
        LightType::point | LightType::spot => 1,
    });
    v
}

/// Cache for the light storage buffer. Recreates only when light count exceeds capacity.
pub struct LightBufferCache {
    buffer: Option<wgpu::Buffer>,
    cached_capacity: usize,
    /// Incremented when buffer is recreated. Used for invalidating PBR bind group cache.
    pub version: u64,
}

impl LightBufferCache {
    /// Creates a new empty cache.
    pub fn new() -> Self {
        Self {
            buffer: None,
            cached_capacity: 0,
            version: 0,
        }
    }

    /// Ensures the buffer has capacity for at least `light_count` lights.
    /// Returns a reference to the buffer, or None if light_count is 0.
    pub fn ensure_buffer(
        &mut self,
        device: &wgpu::Device,
        light_count: usize,
    ) -> Option<&wgpu::Buffer> {
        if light_count == 0 {
            return None;
        }
        let capacity = light_count.min(MAX_LIGHTS);
        if self.buffer.is_none() || capacity > self.cached_capacity {
            self.version = self.version.wrapping_add(1);
            let size = (capacity * size_of::<GpuLight>()).max(size_of::<GpuLight>()) as u64;
            self.buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("light storage buffer"),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.cached_capacity = capacity;
        }
        self.buffer.as_ref()
    }

    /// Uploads lights to the GPU buffer. Call ensure_buffer first.
    pub fn upload(&self, queue: &wgpu::Queue, lights: &[ResolvedLight]) {
        if lights.is_empty() {
            return;
        }
        let Some(ref buffer) = self.buffer else {
            return;
        };
        let gpu_lights: Vec<GpuLight> = lights
            .iter()
            .take(MAX_LIGHTS)
            .map(GpuLight::from_resolved)
            .collect();
        queue.write_buffer(buffer, 0, bytemuck::cast_slice(&gpu_lights));
    }
}

impl Default for LightBufferCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies GpuLight matches WGSL std430 layout (80 bytes).
    #[test]
    fn gpu_light_size_matches_wgsl() {
        assert_eq!(
            size_of::<GpuLight>(),
            80,
            "GpuLight must be 80 bytes to match WGSL storage buffer stride"
        );
    }

    #[test]
    fn light_type_u32_matches_wgsl_and_shared_repr() {
        use glam::Vec3;

        assert_eq!(light_type_u32(LightType::point), 0);
        assert_eq!(light_type_u32(LightType::directional), 1);
        assert_eq!(light_type_u32(LightType::spot), 2);
        let p = ResolvedLight {
            world_position: Vec3::ZERO,
            world_direction: Vec3::Z,
            color: Vec3::ONE,
            intensity: 1.0,
            range: 1.0,
            spot_angle: 0.0,
            light_type: LightType::directional,
            global_unique_id: 0,
        };
        assert_eq!(GpuLight::from_resolved(&p).light_type, 1);
    }

    #[test]
    fn order_lights_for_clustered_shading_puts_directionals_first_preserving_stable_tail() {
        use glam::Vec3;

        fn dummy(idx: u8, ty: LightType) -> ResolvedLight {
            ResolvedLight {
                world_position: Vec3::splat(idx as f32),
                world_direction: Vec3::Z,
                color: Vec3::ONE,
                intensity: 1.0,
                range: 10.0,
                spot_angle: 45.0,
                light_type: ty,
                global_unique_id: idx as i32,
            }
        }

        let mut lights: Vec<ResolvedLight> =
            (0u8..40).map(|i| dummy(i, LightType::point)).collect();
        lights.push(dummy(40, LightType::directional));

        let ordered = order_lights_for_clustered_shading(&lights);
        assert_eq!(ordered.len(), 41);
        assert_eq!(ordered[0].light_type, LightType::directional);
        assert_eq!(ordered[0].global_unique_id, 40);
        for w in ordered.iter().skip(1) {
            assert_eq!(w.light_type, LightType::point);
        }
        let tail_ids: Vec<i32> = ordered.iter().skip(1).map(|l| l.global_unique_id).collect();
        assert_eq!(tail_ids, (0..40).collect::<Vec<_>>());
    }
}
