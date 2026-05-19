//! Persistent realtime shadow-map resources bound through frame `@group(0)`.

use std::sync::Arc;

use crate::gpu::{
    GpuLimits, GpuShadowLight, GpuShadowView, MAX_LIGHTS, MAX_SHADOW_VIEWS, SHADOW_ARRAY_LAYERS,
    SHADOW_DEPTH_FORMAT,
};
use crate::shared::ShadowResolutionMode;

/// Fallback and live shadow-map resources.
pub(super) struct ShadowMapResources {
    /// Depth texture backing all shadow-map array layers.
    _texture: Arc<wgpu::Texture>,
    /// Array view sampled by forward lighting.
    array_view: Arc<wgpu::TextureView>,
    /// Placeholder depth texture bound while writing live shadow-map layers.
    _writer_placeholder_texture: Arc<wgpu::Texture>,
    /// Placeholder array view that avoids sampling the live atlas during shadow-map writes.
    writer_placeholder_array_view: Arc<wgpu::TextureView>,
    /// One single-layer view per shadow-map layer for depth rendering.
    layer_views: Vec<Arc<wgpu::TextureView>>,
    /// Hardware depth-comparison sampler.
    sampler: Arc<wgpu::Sampler>,
    /// Current square edge in pixels.
    resolution: u32,
    /// Version incremented whenever the texture allocation changes.
    version: u64,
}

/// Borrowed shadow resources needed while building frame bind groups.
pub(super) struct ShadowBindGroupResources<'a> {
    /// Shadow map array view.
    pub(super) array_view: &'a wgpu::TextureView,
    /// Depth-comparison sampler.
    pub(super) sampler: &'a wgpu::Sampler,
}

impl ShadowMapResources {
    /// Creates a minimal valid shadow resource set.
    pub(super) fn new(device: &wgpu::Device) -> Self {
        Self::create(device, 1, 0)
    }

    /// Current resource version.
    pub(super) fn version(&self) -> u64 {
        self.version
    }

    /// Current shadow-map edge in pixels.
    pub(super) fn resolution(&self) -> u32 {
        self.resolution
    }

    /// Returns the layer view for `layer`.
    pub(super) fn layer_view(&self, layer: usize) -> Option<Arc<wgpu::TextureView>> {
        self.layer_views.get(layer).cloned()
    }

    /// Bind-group resources for shadow sampling.
    pub(super) fn bind_group_resources(&self) -> ShadowBindGroupResources<'_> {
        ShadowBindGroupResources {
            array_view: self.array_view.as_ref(),
            sampler: self.sampler.as_ref(),
        }
    }

    /// Bind-group resources for shadow-map writer passes.
    pub(super) fn writer_bind_group_resources(&self) -> ShadowBindGroupResources<'_> {
        ShadowBindGroupResources {
            array_view: self.writer_placeholder_array_view.as_ref(),
            sampler: self.sampler.as_ref(),
        }
    }

    /// Ensures the texture matches the host quality setting and device limits.
    pub(super) fn sync_resolution(
        &mut self,
        device: &wgpu::Device,
        limits: &GpuLimits,
        mode: ShadowResolutionMode,
    ) -> bool {
        let resolution = shadow_resolution_for_quality(mode, limits);
        if resolution == self.resolution {
            return false;
        }
        *self = Self::create(device, resolution, self.version.saturating_add(1));
        true
    }

    fn create(device: &wgpu::Device, resolution: u32, version: u64) -> Self {
        let texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow_map_array"),
            size: wgpu::Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: SHADOW_ARRAY_LAYERS,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SHADOW_DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }));
        crate::profiling::note_resource_churn!(Texture, "backend::shadow_map_array");

        let array_view = Arc::new(texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("shadow_map_array_view"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            base_array_layer: 0,
            array_layer_count: Some(SHADOW_ARRAY_LAYERS),
            ..Default::default()
        }));
        crate::profiling::note_resource_churn!(TextureView, "backend::shadow_map_array_view");

        let writer_placeholder_texture =
            Arc::new(device.create_texture(&wgpu::TextureDescriptor {
                label: Some("shadow_map_writer_placeholder"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: SHADOW_ARRAY_LAYERS,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: SHADOW_DEPTH_FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            }));
        crate::profiling::note_resource_churn!(Texture, "backend::shadow_map_writer_placeholder");

        let writer_placeholder_array_view = Arc::new(writer_placeholder_texture.create_view(
            &wgpu::TextureViewDescriptor {
                label: Some("shadow_map_writer_placeholder_view"),
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                base_array_layer: 0,
                array_layer_count: Some(SHADOW_ARRAY_LAYERS),
                ..Default::default()
            },
        ));
        crate::profiling::note_resource_churn!(
            TextureView,
            "backend::shadow_map_writer_placeholder_view"
        );

        let layer_views = (0..SHADOW_ARRAY_LAYERS)
            .map(|layer| {
                Arc::new(texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("shadow_map_layer_view"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    ..Default::default()
                }))
            })
            .inspect(|_| {
                crate::profiling::note_resource_churn!(
                    TextureView,
                    "backend::shadow_map_layer_view"
                );
            })
            .collect();

        let sampler = Arc::new(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow_map_compare_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            compare: Some(crate::gpu::MAIN_FORWARD_DEPTH_COMPARE),
            lod_min_clamp: 0.0,
            lod_max_clamp: 0.0,
            ..Default::default()
        }));
        crate::profiling::note_resource_churn!(Sampler, "backend::shadow_map_compare_sampler");

        Self {
            _texture: texture,
            array_view,
            _writer_placeholder_texture: writer_placeholder_texture,
            writer_placeholder_array_view,
            layer_views,
            sampler,
            resolution,
            version,
        }
    }
}

/// Creates a shadow-light metadata storage buffer.
pub(super) fn create_shadow_light_buffer(
    device: &wgpu::Device,
    label: &'static str,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (MAX_LIGHTS * size_of::<GpuShadowLight>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Creates a shadow-view metadata storage buffer.
pub(super) fn create_shadow_view_buffer(
    device: &wgpu::Device,
    label: &'static str,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (MAX_SHADOW_VIEWS * size_of::<GpuShadowView>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Maps host quality to a practical portable shadow-map edge.
pub fn shadow_resolution_for_quality(mode: ShadowResolutionMode, limits: &GpuLimits) -> u32 {
    let requested = match mode {
        ShadowResolutionMode::Low => 512,
        ShadowResolutionMode::Medium => 1024,
        ShadowResolutionMode::High | ShadowResolutionMode::Ultra => 2048,
    };
    requested.min(limits.wgpu.max_texture_dimension_2d).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hashbrown::HashMap;

    fn limits(max_texture_dimension_2d: u32) -> GpuLimits {
        GpuLimits::synthetic_for_tests(
            wgpu::Limits {
                max_texture_dimension_2d,
                ..Default::default()
            },
            wgpu::Features::empty(),
            HashMap::new(),
        )
    }

    #[test]
    fn quality_resolution_maps_to_birp_style_edges_and_device_cap() {
        let generous_limits = limits(4096);
        assert_eq!(
            shadow_resolution_for_quality(ShadowResolutionMode::Low, &generous_limits),
            512
        );
        assert_eq!(
            shadow_resolution_for_quality(ShadowResolutionMode::Medium, &generous_limits),
            1024
        );
        assert_eq!(
            shadow_resolution_for_quality(ShadowResolutionMode::High, &generous_limits),
            2048
        );
        assert_eq!(
            shadow_resolution_for_quality(ShadowResolutionMode::Ultra, &generous_limits),
            2048
        );

        assert_eq!(
            shadow_resolution_for_quality(ShadowResolutionMode::High, &limits(1024)),
            1024
        );
    }
}
