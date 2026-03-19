//! GPU light buffer and clustered light infrastructure.
//!
//! Provides GpuLight struct for shader upload and LightBufferCache for per-frame
//! light buffer management.

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
            direction: [0.0, 0.0, -1.0],
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
    /// Converts a ResolvedLight to GpuLight for GPU upload.
    pub fn from_resolved(light: &ResolvedLight) -> Self {
        let spot_cos_half_angle = if light.spot_angle > 0.0 && light.spot_angle < 180.0 {
            (light.spot_angle.to_radians() / 2.0).cos()
        } else {
            1.0
        };
        let light_type = match light.light_type {
            LightType::point => 0u32,
            LightType::directional => 1u32,
            LightType::spot => 2u32,
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
            light_type,
            _pad_before_vec4: [0; 1],
            _pad2: [0; 4],
        }
    }
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
}
