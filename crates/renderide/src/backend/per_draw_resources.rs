//! Per-draw instance storage slab (`@group(2)`) for mesh forward passes.

use std::sync::Arc;

use crate::backend::mesh_deform::{INITIAL_PER_DRAW_UNIFORM_SLOTS, PER_DRAW_UNIFORM_STRIDE};
use crate::pipelines::raster::DebugWorldNormalsFamily;

/// GPU storage slab: one [`crate::backend::mesh_deform::PaddedPerDrawUniforms`] slot (256 bytes) per
/// mesh draw; vertex shaders index via `@builtin(instance_index)` from `draw_indexed` instance ranges.
pub struct PerDrawResources {
    /// Packed rows (`slot_count * 256` bytes), `STORAGE | COPY_DST`.
    pub per_draw_storage: wgpu::Buffer,
    /// Bind group wiring `per_draw_storage` for raster mesh pipelines (`@group(2)`).
    pub bind_group: Arc<wgpu::BindGroup>,
    slot_count: usize,
}

impl PerDrawResources {
    /// Allocates [`INITIAL_PER_DRAW_UNIFORM_SLOTS`] slots (256 bytes each).
    pub fn new(device: &wgpu::Device) -> Self {
        let layout = DebugWorldNormalsFamily::per_draw_bind_group_layout(device);
        let slot_count = INITIAL_PER_DRAW_UNIFORM_SLOTS;
        let size = (slot_count * PER_DRAW_UNIFORM_STRIDE) as u64;
        let per_draw_storage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh_forward_per_draw_storage"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = Arc::new(Self::make_bind_group(
            device,
            &layout,
            &per_draw_storage,
            size,
        ));
        Self {
            per_draw_storage,
            bind_group,
            slot_count,
        }
    }

    fn make_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        slab: &wgpu::Buffer,
        byte_size: u64,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mesh_forward_per_draw_bind_group"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: slab,
                    offset: 0,
                    size: std::num::NonZeroU64::new(byte_size),
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
        let size_u64 = (next * PER_DRAW_UNIFORM_STRIDE) as u64;
        let per_draw_storage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh_forward_per_draw_storage"),
            size: size_u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layout = DebugWorldNormalsFamily::per_draw_bind_group_layout(device);
        let bind_group = Arc::new(Self::make_bind_group(
            device,
            &layout,
            &per_draw_storage,
            size_u64,
        ));
        self.per_draw_storage = per_draw_storage;
        self.bind_group = bind_group;
        self.slot_count = next;
    }
}
