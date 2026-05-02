//! GPU limits validation policy.
//!
//! Pure policy: given a [`wgpu::Device`] and [`wgpu::Adapter`], either build a
//! [`GpuLimits`] snapshot or reject the device with a [`GpuLimitsError::Requirement`].
//! Callers go through [`GpuLimits::try_new`] in `super`, which forwards into [`try_new`]
//! here -- keeping the public entry point byte-identical with prior call sites in
//! [`crate::gpu::context::GpuContext`].

use std::sync::Arc;

use super::{GpuLimits, GpuLimitsError};

/// Per-draw row size in bytes; must match [`crate::mesh_deform::PER_DRAW_UNIFORM_STRIDE`].
pub(super) const PER_DRAW_UNIFORM_STRIDE: usize = 256;
/// Initial slab row count; must match [`crate::mesh_deform::INITIAL_PER_DRAW_UNIFORM_SLOTS`].
pub(super) const INITIAL_PER_DRAW_UNIFORM_SLOTS: usize = 256;

/// Builds a [`GpuLimits`] snapshot from a device and adapter (downlevel flags from `adapter`).
///
/// Fails when core WebGPU-style minimums for this codebase are not met (bind groups, storage
/// binding size for the per-draw slab, texture dimensions).
pub(super) fn try_new(
    device: &wgpu::Device,
    adapter: &wgpu::Adapter,
) -> Result<Arc<GpuLimits>, GpuLimitsError> {
    let wgpu_limits = device.limits();
    let features = device.features();
    let down = adapter.get_downlevel_capabilities();

    validate_wgpu_minimums(&wgpu_limits)?;

    // Non-WebGPU-compliant stacks (e.g. some GLES/WebGL paths) may not implement `first_instance`
    // for `draw_indexed` batching the same way; disable merged instance batches there.
    // wgpu 29 removed the dedicated BASE_INSTANCE DownlevelFlag; is_webgpu_compliant() is the
    // correct proxy.
    let supports_base_instance = down.is_webgpu_compliant();
    let supports_multiview = features.contains(wgpu::Features::MULTIVIEW);
    let supports_float32_filterable = features.contains(wgpu::Features::FLOAT32_FILTERABLE);
    let texture_compression_features = features
        & (wgpu::Features::TEXTURE_COMPRESSION_BC
            | wgpu::Features::TEXTURE_COMPRESSION_ETC2
            | wgpu::Features::TEXTURE_COMPRESSION_ASTC);

    let max_binding = wgpu_limits.max_storage_buffer_binding_size;
    let stride = PER_DRAW_UNIFORM_STRIDE as u64;
    let max_per_draw_slab_slots = (max_binding / stride) as usize;

    if max_per_draw_slab_slots < INITIAL_PER_DRAW_UNIFORM_SLOTS {
        return Err(GpuLimitsError::Requirement(
            "max_storage_buffer_binding_size too small for initial per-draw slab (256x256 B rows)",
        ));
    }

    let limits = GpuLimits {
        wgpu: wgpu_limits,
        supports_base_instance,
        supports_multiview,
        supports_float32_filterable,
        texture_compression_features,
        max_per_draw_slab_slots,
    };

    logger::info!(
        "GPU limits: max_texture_2d={} max_buffer={} max_storage_binding={} max_compute_wg_per_dim={} max_samplers_stage={} max_sampled_textures_stage={} base_instance={} multiview={}",
        limits.wgpu.max_texture_dimension_2d,
        limits.wgpu.max_buffer_size,
        limits.wgpu.max_storage_buffer_binding_size,
        limits.wgpu.max_compute_workgroups_per_dimension,
        limits.wgpu.max_samplers_per_shader_stage,
        limits.wgpu.max_sampled_textures_per_shader_stage,
        supports_base_instance,
        supports_multiview
    );

    Ok(Arc::new(limits))
}

fn validate_wgpu_minimums(l: &wgpu::Limits) -> Result<(), GpuLimitsError> {
    if l.max_bind_groups < 4 {
        return Err(GpuLimitsError::Requirement(
            "max_bind_groups must be at least 4 (frame / material / per-draw / ...)",
        ));
    }
    if l.max_texture_dimension_2d < 1024 {
        return Err(GpuLimitsError::Requirement(
            "max_texture_dimension_2d must be at least 1024",
        ));
    }
    let min_slab = (INITIAL_PER_DRAW_UNIFORM_SLOTS * PER_DRAW_UNIFORM_STRIDE) as u64;
    if l.max_storage_buffer_binding_size < min_slab {
        return Err(GpuLimitsError::Requirement(
            "max_storage_buffer_binding_size must fit initial per-draw slab (65536 bytes)",
        ));
    }
    if l.min_storage_buffer_offset_alignment > PER_DRAW_UNIFORM_STRIDE as u32 {
        return Err(GpuLimitsError::Requirement(
            "min_storage_buffer_offset_alignment must be <= 256 (per-draw slab stride)",
        ));
    }
    if l.min_uniform_buffer_offset_alignment > PER_DRAW_UNIFORM_STRIDE as u32 {
        return Err(GpuLimitsError::Requirement(
            "min_uniform_buffer_offset_alignment must be <= 256 (per-draw slab stride)",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{PER_DRAW_UNIFORM_STRIDE, validate_wgpu_minimums};

    #[test]
    fn max_per_draw_slots_formula() {
        // Synthetic limits struct for math only (not a real device).
        let l = wgpu::Limits {
            max_storage_buffer_binding_size: 256 * 1024, // 256 KiB
            ..Default::default()
        };
        let max_binding = l.max_storage_buffer_binding_size;
        let stride = PER_DRAW_UNIFORM_STRIDE as u64;
        let slots = (max_binding / stride) as usize;
        assert_eq!(slots, 1024);
    }

    #[test]
    fn validate_minimums_rejects_insufficient_core_limits() {
        let mut l = wgpu::Limits {
            max_bind_groups: 3,
            max_texture_dimension_2d: 4096,
            max_storage_buffer_binding_size: 65_536,
            ..Default::default()
        };
        assert!(validate_wgpu_minimums(&l).is_err());

        l.max_bind_groups = 4;
        l.max_texture_dimension_2d = 1023;
        assert!(validate_wgpu_minimums(&l).is_err());

        l.max_texture_dimension_2d = 4096;
        l.max_storage_buffer_binding_size = 65_535;
        assert!(validate_wgpu_minimums(&l).is_err());
    }

    #[test]
    fn validate_minimums_rejects_alignment_larger_than_per_draw_stride() {
        let mut l = wgpu::Limits {
            max_bind_groups: 4,
            max_texture_dimension_2d: 4096,
            max_storage_buffer_binding_size: 65_536,
            min_storage_buffer_offset_alignment: 512,
            ..Default::default()
        };
        assert!(validate_wgpu_minimums(&l).is_err());

        l.min_storage_buffer_offset_alignment = 256;
        l.min_uniform_buffer_offset_alignment = 512;
        assert!(validate_wgpu_minimums(&l).is_err());

        l.min_uniform_buffer_offset_alignment = 256;
        assert!(validate_wgpu_minimums(&l).is_ok());
    }
}
