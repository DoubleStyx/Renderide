//! [`wgpu::InstanceFlags`] and [`wgpu::Limits`] helpers used when creating the GPU instance/device.

/// Requests [`wgpu::Limits`] for [`wgpu::Adapter::request_device`].
///
/// Starts from WebGPU-tier [`wgpu::Limits::default`], raises renderer-critical material binding
/// caps where native adapters expose them, then clamps every field to what the adapter supports via
/// [`wgpu::Limits::or_worse_values_from`] so GPUs with lower caps (for example
/// [`wgpu::Limits::max_color_attachments`] below 8 on some ARM/Mali stacks) do not fail device
/// creation.
///
/// After clamping, [`wgpu::Limits::max_buffer_size`] and
/// [`wgpu::Limits::max_storage_buffer_binding_size`] are set from the adapter so large mesh uploads
/// (blendshape packs, etc.) can use the full reported allowance—WebGPU defaults alone cap
/// [`wgpu::Limits::max_buffer_size`] at 256 MiB while the adapter often allows more.
/// [`wgpu::Limits::max_texture_dimension_2d`] is capped at **16384** when the adapter allows it,
/// matching the host’s maximum 2D texture size.
pub(crate) fn required_limits_for_adapter(adapter: &wgpu::Adapter) -> wgpu::Limits {
    required_limits_from_adapter_limits(adapter.limits())
}

pub(crate) fn required_limits_from_adapter_limits(adapter_limits: wgpu::Limits) -> wgpu::Limits {
    let al = adapter_limits;
    let mut desired = wgpu::Limits::default();
    desired.max_samplers_per_shader_stage = desired.max_samplers_per_shader_stage.max(32);
    desired.max_sampled_textures_per_shader_stage =
        desired.max_sampled_textures_per_shader_stage.max(32);

    let mut limits = desired.or_worse_values_from(&al);
    limits.max_buffer_size = al.max_buffer_size;
    limits.max_storage_buffer_binding_size = al.max_storage_buffer_binding_size;

    limits.max_texture_dimension_2d = std::cmp::min(al.max_texture_dimension_2d, 16384);
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
    use super::{instance_flags_base, required_limits_from_adapter_limits};
    use wgpu::InstanceFlags;

    #[test]
    fn instance_flags_base_toggles_validation() {
        assert!(!instance_flags_base(false).contains(InstanceFlags::VALIDATION));
        assert!(instance_flags_base(true).contains(InstanceFlags::VALIDATION));
    }

    #[test]
    fn required_limits_raise_material_texture_budget_when_adapter_allows() {
        let adapter_limits = wgpu::Limits {
            max_samplers_per_shader_stage: 64,
            max_sampled_textures_per_shader_stage: 64,
            ..wgpu::Limits::default()
        };

        let required = required_limits_from_adapter_limits(adapter_limits);

        assert_eq!(required.max_samplers_per_shader_stage, 32);
        assert_eq!(required.max_sampled_textures_per_shader_stage, 32);
    }

    #[test]
    fn required_limits_clamp_material_texture_budget_to_adapter() {
        let adapter_limits = wgpu::Limits {
            max_samplers_per_shader_stage: 18,
            max_sampled_textures_per_shader_stage: 18,
            ..wgpu::Limits::default()
        };

        let required = required_limits_from_adapter_limits(adapter_limits);

        assert_eq!(required.max_samplers_per_shader_stage, 18);
        assert_eq!(required.max_sampled_textures_per_shader_stage, 18);
    }
}
