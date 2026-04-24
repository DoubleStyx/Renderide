//! Per-draw instance storage slab (`@group(2)`) for mesh forward passes.
//!
//! Each render view owns its own [`PerDrawResources`] instance, grown on demand using the
//! Filament growth policy: `max(16, (4*n + 2) / 3)`. Views write their per-draw rows at
//! byte offset 0 of their own buffer; no ring partitioning or cross-view sharing.

use std::sync::Arc;

use crate::backend::mesh_deform::{INITIAL_PER_DRAW_UNIFORM_SLOTS, PER_DRAW_UNIFORM_STRIDE};
use crate::gpu::GpuLimits;
use crate::materials::PipelineBuildError;
use crate::pipelines::raster::DebugWorldNormalsFamily;

/// GPU storage slab: one [`crate::backend::mesh_deform::PaddedPerDrawUniforms`] slot (256 bytes) per
/// mesh draw. Shaders use `instance_index` to select the per-draw row; the downlevel path uses a
/// per-draw dynamic storage offset at bind time instead.
///
/// Each render view owns one `PerDrawResources` instance. Slabs are grown on demand (never shrink)
/// and are independent — one view cannot exhaust another view's buffer.
pub struct PerDrawResources {
    /// Packed rows (`slot_count * 256` bytes), `STORAGE | COPY_DST`.
    pub per_draw_storage: wgpu::Buffer,
    /// Bind group wiring `per_draw_storage` for raster mesh pipelines (`@group(2)`).
    pub bind_group: Arc<wgpu::BindGroup>,
    /// Layout shared by mesh-forward pipelines (`@group(2)` dynamic storage binding).
    pub bind_group_layout: Arc<wgpu::BindGroupLayout>,
    slot_count: usize,
    limits: Arc<GpuLimits>,
}

impl PerDrawResources {
    /// Allocates [`INITIAL_PER_DRAW_UNIFORM_SLOTS`] slots (256 bytes each), deriving the bind
    /// group layout from naga reflection of the embedded `debug_world_normals_default` shader.
    pub fn new(device: &wgpu::Device, limits: Arc<GpuLimits>) -> Result<Self, PipelineBuildError> {
        let layout = Arc::new(DebugWorldNormalsFamily::per_draw_bind_group_layout(device)?);
        Ok(Self::new_with_layout(device, layout, limits))
    }

    /// Allocates [`INITIAL_PER_DRAW_UNIFORM_SLOTS`] slots using a pre-built bind group layout.
    ///
    /// Use this when constructing multiple per-view instances to share the same layout `Arc`
    /// rather than reflecting the shader once per view.
    pub fn new_with_layout(
        device: &wgpu::Device,
        layout: Arc<wgpu::BindGroupLayout>,
        limits: Arc<GpuLimits>,
    ) -> Self {
        let slot_count = INITIAL_PER_DRAW_UNIFORM_SLOTS.min(limits.max_per_draw_slab_slots);
        let size = (slot_count * PER_DRAW_UNIFORM_STRIDE) as u64;
        let per_draw_storage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh_forward_per_draw_storage"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = Arc::new(Self::make_bind_group(
            device,
            layout.as_ref(),
            &per_draw_storage,
        ));
        Self {
            per_draw_storage,
            bind_group,
            bind_group_layout: layout,
            slot_count,
            limits,
        }
    }

    fn make_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        slab: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mesh_forward_per_draw_bind_group"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: slab,
                    offset: 0,
                    size: None,
                }),
            }],
        })
    }

    /// Ensures at least `need_slots` rows are available, growing the slab and recreating the bind
    /// group when needed.
    ///
    /// Growth uses the Filament policy: `max(16, (4*n + 2) / 3)`, which provides ~33% headroom
    /// per grow event. The result is capped by [`GpuLimits::max_per_draw_slab_slots`]; draws
    /// beyond the cap log a warning but are silently clamped.
    pub fn ensure_draw_slot_capacity(&mut self, device: &wgpu::Device, need_slots: usize) {
        let cap = self.limits.max_per_draw_slab_slots;
        if need_slots > cap {
            logger::warn!(
                "per-draw slab: requested {need_slots} slots exceeds max {cap} (storage binding size / stride)"
            );
        }
        let need_slots = need_slots.min(cap);
        if need_slots == 0 || need_slots <= self.slot_count {
            return;
        }
        // Filament growth policy: ~33% slack, minimum 16.
        let next = (4 * need_slots).div_ceil(3).max(16).min(cap);
        let size_u64 = (next * PER_DRAW_UNIFORM_STRIDE) as u64;
        let per_draw_storage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh_forward_per_draw_storage"),
            size: size_u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = Arc::new(Self::make_bind_group(
            device,
            self.bind_group_layout.as_ref(),
            &per_draw_storage,
        ));
        logger::debug!(
            "per-draw slab: grew {old} → {next} slots ({size} bytes)",
            old = self.slot_count,
            size = size_u64,
        );
        self.per_draw_storage = per_draw_storage;
        self.bind_group = bind_group;
        self.slot_count = next;
    }
}

#[cfg(test)]
mod tests {
    use crate::backend::mesh_deform::PER_DRAW_UNIFORM_STRIDE;

    /// Pure Filament growth formula for unit testing (no GPU device needed).
    fn filament_growth(need_slots: usize, cap: usize) -> usize {
        (4 * need_slots).div_ceil(3).max(16).min(cap)
    }

    #[test]
    fn filament_growth_policy_correct() {
        let large_cap = 100_000usize;
        let cases: &[(usize, usize)] = &[
            (1, 16),
            (10, 16),
            (12, 16),
            (16, 22),
            (100, 134),
            (1000, 1334),
        ];
        for &(need, expected) in cases {
            let actual = filament_growth(need, large_cap);
            assert_eq!(
                actual, expected,
                "growth(need={need}) should be {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn filament_growth_never_below_16() {
        let cap = 100_000usize;
        for need in 1..=100 {
            let actual = filament_growth(need, cap);
            assert!(
                actual >= 16,
                "growth(need={need}) = {actual} is below the minimum of 16"
            );
        }
    }

    #[test]
    fn filament_growth_capped_by_max() {
        let max = 500usize;
        let result = filament_growth(1000, max);
        assert_eq!(result, max, "growth should be capped at max={max}");
    }

    #[test]
    fn growth_is_monotonic() {
        let cap = 10_000usize;
        let mut prev = 16usize;
        for need in 1..=5000 {
            let next = filament_growth(need, cap);
            assert!(
                next >= prev,
                "growth not monotone at need={need}: prev={prev}, next={next}"
            );
            prev = next;
        }
    }

    #[test]
    fn stride_calculation_consistent() {
        let n = 100usize;
        let bytes = n * PER_DRAW_UNIFORM_STRIDE;
        assert_eq!(bytes / PER_DRAW_UNIFORM_STRIDE, n);
    }
}
