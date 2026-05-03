//! [`wgpu::InstanceFlags`] and [`wgpu::Limits`] helpers used when creating the GPU instance/device.

/// Renderer policy clamp on [`wgpu::Limits::max_texture_dimension_2d`]; matches the host's
/// maximum 2D texture size and avoids encouraging textures larger than the engine ever exposes.
const RENDERER_MAX_TEXTURE_DIMENSION_2D: u32 = 16384;

/// Requests [`wgpu::Limits`] for [`wgpu::Adapter::request_device`].
///
/// Starts from the adapter's full reported [`wgpu::Limits`] so the device is granted everything
/// the GPU advertises (wgpu only hands the device the fields explicitly listed in
/// `required_limits` -- the adapter's headroom is not implicit). Then applies renderer-policy
/// clamps where we deliberately stay below the adapter's max for stability or to match the host:
/// [`wgpu::Limits::max_texture_dimension_2d`] is capped at
/// [`RENDERER_MAX_TEXTURE_DIMENSION_2D`].
///
/// Floor checks (minimum required by the renderer such as `max_bind_groups >= 4` and the
/// per-draw slab fitting `max_storage_buffer_binding_size`) are enforced later in
/// [`crate::gpu::GpuLimits::try_new`].
pub(crate) fn required_limits_for_adapter(adapter: &wgpu::Adapter) -> wgpu::Limits {
    required_limits_from_adapter_limits(adapter.limits())
}

/// Pure logic core of [`required_limits_for_adapter`]; takes adapter-reported limits and applies
/// the renderer's policy clamps.
pub(crate) fn required_limits_from_adapter_limits(adapter_limits: wgpu::Limits) -> wgpu::Limits {
    let mut limits = adapter_limits;
    limits.max_texture_dimension_2d = limits
        .max_texture_dimension_2d
        .min(RENDERER_MAX_TEXTURE_DIMENSION_2D);
    limits
}

/// Base flags from the renderer config (validation), before [`wgpu::InstanceFlags::with_env`].
pub(crate) fn instance_flags_base(gpu_validation_layers: bool) -> wgpu::InstanceFlags {
    let mut flags = wgpu::InstanceFlags::empty();
    if gpu_validation_layers {
        flags.insert(wgpu::InstanceFlags::VALIDATION);
    }
    flags
}

/// Builds [`wgpu::InstanceFlags`] for desktop GPU init: optional `VALIDATION`, then
/// [`wgpu::InstanceFlags::with_env`] so `WGPU_VALIDATION` and related variables can override at
/// process start.
pub fn instance_flags_for_gpu_init(gpu_validation_layers: bool) -> wgpu::InstanceFlags {
    instance_flags_base(gpu_validation_layers).with_env()
}

#[cfg(test)]
mod tests {
    use super::{
        RENDERER_MAX_TEXTURE_DIMENSION_2D, instance_flags_base, required_limits_from_adapter_limits,
    };
    use wgpu::InstanceFlags;

    #[test]
    fn instance_flags_base_toggles_validation() {
        assert!(!instance_flags_base(false).contains(InstanceFlags::VALIDATION));
        assert!(instance_flags_base(true).contains(InstanceFlags::VALIDATION));
    }

    #[test]
    fn required_limits_take_adapter_max_for_material_binding_caps() {
        let adapter_limits = wgpu::Limits {
            max_samplers_per_shader_stage: 64,
            max_sampled_textures_per_shader_stage: 64,
            ..wgpu::Limits::default()
        };

        let required = required_limits_from_adapter_limits(adapter_limits);

        assert_eq!(required.max_samplers_per_shader_stage, 64);
        assert_eq!(required.max_sampled_textures_per_shader_stage, 64);
    }

    #[test]
    fn required_limits_propagate_low_adapter_caps() {
        let adapter_limits = wgpu::Limits {
            max_samplers_per_shader_stage: 18,
            max_sampled_textures_per_shader_stage: 18,
            ..wgpu::Limits::default()
        };

        let required = required_limits_from_adapter_limits(adapter_limits);

        assert_eq!(required.max_samplers_per_shader_stage, 18);
        assert_eq!(required.max_sampled_textures_per_shader_stage, 18);
    }

    #[test]
    fn required_limits_propagate_non_default_compute_caps() {
        let adapter_limits = wgpu::Limits {
            max_compute_invocations_per_workgroup: 1024,
            max_compute_workgroup_size_x: 1024,
            max_compute_workgroup_size_y: 1024,
            max_compute_workgroup_size_z: 64,
            max_compute_workgroups_per_dimension: 65535,
            max_texture_dimension_3d: 16384,
            max_texture_array_layers: 2048,
            max_uniform_buffer_binding_size: 1 << 30,
            max_vertex_buffers: 16,
            max_vertex_attributes: 32,
            max_color_attachments: 8,
            max_bindings_per_bind_group: 1000,
            ..wgpu::Limits::default()
        };

        let required = required_limits_from_adapter_limits(adapter_limits);

        assert_eq!(required.max_compute_invocations_per_workgroup, 1024);
        assert_eq!(required.max_compute_workgroup_size_x, 1024);
        assert_eq!(required.max_compute_workgroup_size_y, 1024);
        assert_eq!(required.max_compute_workgroup_size_z, 64);
        assert_eq!(required.max_texture_dimension_3d, 16384);
        assert_eq!(required.max_texture_array_layers, 2048);
        assert_eq!(required.max_uniform_buffer_binding_size, 1 << 30);
        assert_eq!(required.max_vertex_buffers, 16);
        assert_eq!(required.max_vertex_attributes, 32);
        assert_eq!(required.max_color_attachments, 8);
        assert_eq!(required.max_bindings_per_bind_group, 1000);
    }

    #[test]
    fn required_limits_clamp_max_texture_dimension_2d_to_renderer_policy() {
        let adapter_limits = wgpu::Limits {
            max_texture_dimension_2d: 32768,
            ..wgpu::Limits::default()
        };

        let required = required_limits_from_adapter_limits(adapter_limits);

        assert_eq!(
            required.max_texture_dimension_2d,
            RENDERER_MAX_TEXTURE_DIMENSION_2D
        );
    }

    #[test]
    fn required_limits_keep_lower_adapter_max_texture_dimension_2d() {
        let adapter_limits = wgpu::Limits {
            max_texture_dimension_2d: 4096,
            ..wgpu::Limits::default()
        };

        let required = required_limits_from_adapter_limits(adapter_limits);

        assert_eq!(required.max_texture_dimension_2d, 4096);
    }

    #[test]
    fn required_limits_take_adapter_max_buffer_size() {
        let adapter_limits = wgpu::Limits {
            max_buffer_size: 4 * 1024 * 1024 * 1024,
            max_storage_buffer_binding_size: 2 * 1024 * 1024 * 1024,
            ..wgpu::Limits::default()
        };

        let required = required_limits_from_adapter_limits(adapter_limits);

        assert_eq!(required.max_buffer_size, 4 * 1024 * 1024 * 1024);
        assert_eq!(
            required.max_storage_buffer_binding_size,
            2 * 1024 * 1024 * 1024
        );
    }
}
