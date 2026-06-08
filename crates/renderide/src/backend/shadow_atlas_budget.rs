//! Shadow-atlas allocation policy shared by planning and GPU resource synchronization.

use crate::gpu::GpuLimits;

/// Maximum resident realtime shadow atlas allocation in bytes.
pub(in crate::backend) const SHADOW_ATLAS_MAX_BYTES: u64 = 512 * 1024 * 1024;

const SHADOW_ATLAS_CONSERVATIVE_BYTES_PER_TEXEL: u64 = 4;

/// Clamps a shadow atlas edge to device dimensions and the internal atlas allocation budget.
#[must_use]
pub(in crate::backend) fn clamp_shadow_atlas_resolution(
    limits: &GpuLimits,
    requested_resolution: u32,
    layers: u32,
    format: wgpu::TextureFormat,
) -> u32 {
    clamp_shadow_atlas_resolution_for_texel_size(
        limits,
        requested_resolution,
        layers,
        shadow_atlas_format_bytes_per_texel(format),
    )
}

fn clamp_shadow_atlas_resolution_for_texel_size(
    limits: &GpuLimits,
    requested_resolution: u32,
    layers: u32,
    bytes_per_texel: u64,
) -> u32 {
    let device_cap = limits.max_texture_dimension_2d().max(1);
    let budget_cap = shadow_atlas_budget_resolution_cap(layers, bytes_per_texel);
    requested_resolution.max(1).min(device_cap).min(budget_cap)
}

const fn shadow_atlas_format_bytes_per_texel(format: wgpu::TextureFormat) -> u64 {
    match format {
        wgpu::TextureFormat::Depth16Unorm => 2,
        wgpu::TextureFormat::Depth24Plus | wgpu::TextureFormat::Depth32Float => 4,
        _ => SHADOW_ATLAS_CONSERVATIVE_BYTES_PER_TEXEL,
    }
}

fn shadow_atlas_budget_resolution_cap(layers: u32, bytes_per_texel: u64) -> u32 {
    let layer_count = u64::from(layers.max(1));
    let texel_bytes = bytes_per_texel.max(1);
    let texels_per_layer = SHADOW_ATLAS_MAX_BYTES / layer_count / texel_bytes;
    floor_sqrt_u64(texels_per_layer).max(1)
}

fn floor_sqrt_u64(value: u64) -> u32 {
    if value == 0 {
        return 0;
    }
    let mut root = (value as f64).sqrt() as u64;
    while root < u64::from(u32::MAX) {
        let next = root + 1;
        if next > value / next {
            break;
        }
        root = next;
    }
    while root > value / root {
        root -= 1;
    }
    root.min(u64::from(u32::MAX)) as u32
}

#[cfg(test)]
mod tests {
    use hashbrown::HashMap;

    use super::clamp_shadow_atlas_resolution;

    fn limits(max_texture_dimension_2d: u32) -> crate::gpu::GpuLimits {
        crate::gpu::GpuLimits::synthetic_for_tests(
            wgpu::Limits {
                max_texture_dimension_2d,
                ..Default::default()
            },
            wgpu::Features::empty(),
            HashMap::new(),
        )
    }

    #[test]
    fn shadow_atlas_budget_allows_four_ultra_depth32_layers() {
        let limits = limits(8192);

        assert_eq!(
            clamp_shadow_atlas_resolution(&limits, 4096, 4, wgpu::TextureFormat::Depth32Float),
            4096
        );
    }

    #[test]
    fn shadow_atlas_budget_clamps_many_depth32_layers() {
        let limits = limits(8192);

        assert_eq!(
            clamp_shadow_atlas_resolution(&limits, 4096, 16, wgpu::TextureFormat::Depth32Float),
            2896
        );
    }

    #[test]
    fn shadow_atlas_budget_accounts_for_depth16_texel_size() {
        let limits = limits(8192);

        assert_eq!(
            clamp_shadow_atlas_resolution(&limits, 4096, 16, wgpu::TextureFormat::Depth16Unorm),
            4096
        );
    }

    #[test]
    fn shadow_atlas_budget_respects_device_texture_limit() {
        let limits = limits(2048);

        assert_eq!(
            clamp_shadow_atlas_resolution(&limits, 4096, 1, wgpu::TextureFormat::Depth32Float),
            2048
        );
    }
}
