use crate::gpu::GpuLimits;

const SHADOW_ATLAS_CANDIDATES: [wgpu::TextureFormat; 3] = [
    wgpu::TextureFormat::Depth32Float,
    wgpu::TextureFormat::Depth24Plus,
    wgpu::TextureFormat::Depth16Unorm,
];

fn texture_binding_usage() -> wgpu::TextureUsages {
    wgpu::TextureUsages::TEXTURE_BINDING
}

fn renderable_shadow_usage() -> wgpu::TextureUsages {
    wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT
}

/// Selects the preferred shadow atlas format that can be both sampled and rendered.
pub(in crate::backend) fn select_shadow_atlas_format(
    limits: &GpuLimits,
) -> Option<wgpu::TextureFormat> {
    SHADOW_ATLAS_CANDIDATES
        .into_iter()
        .find(|&format| shadow_atlas_format_supports(limits, format, renderable_shadow_usage()))
}

/// Selects the preferred depth format that can be bound by frame globals as a fallback atlas.
pub(in crate::backend) fn select_shadow_atlas_binding_format(
    limits: &GpuLimits,
) -> Option<wgpu::TextureFormat> {
    SHADOW_ATLAS_CANDIDATES
        .into_iter()
        .find(|&format| shadow_atlas_format_supports(limits, format, texture_binding_usage()))
}

fn shadow_atlas_format_supports(
    limits: &GpuLimits,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> bool {
    limits
        .texture_format_features(format)
        .allowed_usages
        .contains(usage)
}

#[cfg(test)]
mod tests {
    use hashbrown::HashMap;

    use super::{select_shadow_atlas_binding_format, select_shadow_atlas_format};

    fn limits_with_format_features<const N: usize>(
        features: [(wgpu::TextureFormat, wgpu::TextureUsages); N],
    ) -> crate::gpu::GpuLimits {
        let mut format_features = HashMap::new();
        for (format, allowed_usages) in features {
            format_features.insert(
                format,
                wgpu::TextureFormatFeatures {
                    allowed_usages,
                    flags: wgpu::TextureFormatFeatureFlags::empty(),
                },
            );
        }
        crate::gpu::GpuLimits::synthetic_for_tests(
            wgpu::Limits {
                max_texture_dimension_2d: 4096,
                max_texture_array_layers: 64,
                max_storage_buffer_binding_size: 256 * 1024,
                max_buffer_size: 256 * 1024,
                ..Default::default()
            },
            wgpu::Features::empty(),
            format_features,
        )
    }

    #[test]
    fn shadow_atlas_format_prefers_depth32_float() {
        let limits = limits_with_format_features([(
            wgpu::TextureFormat::Depth32Float,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
        )]);

        assert_eq!(
            select_shadow_atlas_format(&limits),
            Some(wgpu::TextureFormat::Depth32Float)
        );
    }

    #[test]
    fn shadow_atlas_format_falls_back_to_depth24_plus() {
        let limits = limits_with_format_features([
            (
                wgpu::TextureFormat::Depth32Float,
                wgpu::TextureUsages::TEXTURE_BINDING,
            ),
            (
                wgpu::TextureFormat::Depth24Plus,
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            ),
        ]);

        assert_eq!(
            select_shadow_atlas_format(&limits),
            Some(wgpu::TextureFormat::Depth24Plus)
        );
    }

    #[test]
    fn shadow_atlas_format_falls_back_to_depth16_unorm() {
        let limits = limits_with_format_features([
            (
                wgpu::TextureFormat::Depth32Float,
                wgpu::TextureUsages::TEXTURE_BINDING,
            ),
            (
                wgpu::TextureFormat::Depth24Plus,
                wgpu::TextureUsages::TEXTURE_BINDING,
            ),
            (
                wgpu::TextureFormat::Depth16Unorm,
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            ),
        ]);

        assert_eq!(
            select_shadow_atlas_format(&limits),
            Some(wgpu::TextureFormat::Depth16Unorm)
        );
    }

    #[test]
    fn shadow_atlas_format_reports_unsupported_without_required_usages() {
        let limits = limits_with_format_features([
            (
                wgpu::TextureFormat::Depth32Float,
                wgpu::TextureUsages::TEXTURE_BINDING,
            ),
            (
                wgpu::TextureFormat::Depth24Plus,
                wgpu::TextureUsages::RENDER_ATTACHMENT,
            ),
            (
                wgpu::TextureFormat::Depth16Unorm,
                wgpu::TextureUsages::empty(),
            ),
        ]);

        assert_eq!(select_shadow_atlas_format(&limits), None);
        assert_eq!(
            select_shadow_atlas_binding_format(&limits),
            Some(wgpu::TextureFormat::Depth32Float)
        );
    }
}
