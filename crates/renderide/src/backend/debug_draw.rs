//! Uniform slab and bind group for [`crate::pipelines::raster::DebugWorldNormalsFamily`] draws.

use std::num::NonZeroU64;
use std::sync::Arc;

use crate::backend::mesh_deform::{INITIAL_PER_DRAW_UNIFORM_SLOTS, PER_DRAW_UNIFORM_STRIDE};
use crate::pipelines::raster::DebugWorldNormalsFamily;

/// Per-frame uniform slab: one 256-byte [`crate::backend::mesh_deform::PaddedPerDrawUniforms`] slot per mesh draw.
pub struct DebugDrawResources {
    /// Packed rows (`slot_count * 256` bytes).
    pub per_draw_uniforms: wgpu::Buffer,
    /// Bind group wiring `per_draw_uniforms` for [`DebugWorldNormalsFamily`].
    pub bind_group: Arc<wgpu::BindGroup>,
    slot_count: usize,
}

impl DebugDrawResources {
    /// Allocates [`INITIAL_PER_DRAW_UNIFORM_SLOTS`] slots (256 bytes each).
    pub fn new(device: &wgpu::Device) -> Self {
        let layout = DebugWorldNormalsFamily::per_draw_bind_group_layout(device);
        let slot_count = INITIAL_PER_DRAW_UNIFORM_SLOTS;
        let size = (slot_count * PER_DRAW_UNIFORM_STRIDE) as u64;
        let per_draw_uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("debug_world_normals_per_draw_uniforms"),
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = Arc::new(Self::make_bind_group(device, &layout, &per_draw_uniforms));
        Self {
            per_draw_uniforms,
            bind_group,
            slot_count,
        }
    }

    fn make_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        slab: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("debug_world_normals_bind_group"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: slab,
                    offset: 0,
                    size: Some(
                        NonZeroU64::new(PER_DRAW_UNIFORM_STRIDE as u64).expect("stride positive"),
                    ),
                }),
            }],
        })
    }

    /// Ensures at least `need_slots` rows; grows the slab and recreates the bind group when needed.
    pub fn ensure_draw_slot_capacity(&mut self, device: &wgpu::Device, need_slots: usize) {
        if need_slots <= self.slot_count {
            return;
        }
        let next = need_slots
            .next_power_of_two()
            .max(INITIAL_PER_DRAW_UNIFORM_SLOTS);
        let size = (next * PER_DRAW_UNIFORM_STRIDE) as u64;
        let per_draw_uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("debug_world_normals_per_draw_uniforms"),
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layout = DebugWorldNormalsFamily::per_draw_bind_group_layout(device);
        let bind_group = Arc::new(Self::make_bind_group(device, &layout, &per_draw_uniforms));
        self.per_draw_uniforms = per_draw_uniforms;
        self.bind_group = bind_group;
        self.slot_count = next;
    }
}
