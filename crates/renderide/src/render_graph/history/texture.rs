//! Texture half-slot types and per-mip / per-layer view tables.

use std::sync::Arc;

/// Texture history slot declaration.
#[derive(Clone, Debug)]
pub struct TextureHistorySpec {
    /// Debug label used for the allocated `wgpu::Texture`.
    pub label: &'static str,
    /// Texture format.
    pub format: wgpu::TextureFormat,
    /// Texture extent.
    pub extent: wgpu::Extent3d,
    /// Texture usage flags.
    pub usage: wgpu::TextureUsages,
    /// Mip level count.
    pub mip_level_count: u32,
    /// Sample count.
    pub sample_count: u32,
    /// Texture dimension.
    pub dimension: wgpu::TextureDimension,
}

/// Per-layer/per-mip views created for one texture-history allocation.
#[derive(Clone, Debug)]
pub struct HistoryTextureMipViews {
    /// Views grouped as `layers[layer][mip]`.
    layers: Arc<[Arc<[wgpu::TextureView]>]>,
}

impl HistoryTextureMipViews {
    /// Returns all mip views for one array layer.
    pub fn layer_mip_views(&self, layer: u32) -> Option<&[wgpu::TextureView]> {
        self.layers.get(layer as usize).map(AsRef::as_ref)
    }

    /// Iterates every cached mip view.
    pub(crate) fn iter_views(&self) -> impl Iterator<Item = &wgpu::TextureView> {
        self.layers.iter().flat_map(|layer| layer.as_ref().iter())
    }
}

/// Pure shape data for texture-history view tables.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HistoryTextureViewShape {
    /// Number of array layers that should receive per-mip D2 views.
    pub layer_count: u32,
    /// Number of mip levels represented per layer.
    pub mip_level_count: u32,
}

/// One half of a ping-pong texture pair: a texture plus reusable views.
#[derive(Clone)]
pub struct HistoryTexture {
    /// Allocated texture.
    pub texture: wgpu::Texture,
    /// Default full-resource view.
    pub view: wgpu::TextureView,
    /// Per-layer/per-mip views for passes that write or sample explicit subresources.
    pub mip_views: HistoryTextureMipViews,
}

/// Ping-pong texture history slot.
pub struct TextureHistorySlot {
    pub(super) spec: TextureHistorySpec,
    pub(super) pair: [Option<HistoryTexture>; 2],
    pub(super) generation: u64,
}

impl TextureHistorySlot {
    pub(super) fn ensure(&mut self, device: &wgpu::Device) {
        for slot in &mut self.pair {
            if slot.is_none() {
                self.generation = self.generation.wrapping_add(1).max(1);
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(self.spec.label),
                    size: self.spec.extent,
                    mip_level_count: self.spec.mip_level_count.max(1),
                    sample_count: self.spec.sample_count.max(1),
                    dimension: self.spec.dimension,
                    format: self.spec.format,
                    usage: self.spec.usage,
                    view_formats: &[],
                });
                let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                crate::profiling::note_resource_churn!(
                    TextureView,
                    "render_graph::history_texture_default_view"
                );
                let mip_views = create_texture_history_mip_views(&texture, &self.spec);
                *slot = Some(HistoryTexture {
                    texture,
                    view,
                    mip_views,
                });
            }
        }
    }

    /// Current spec used for allocation; compared against reallocation requests.
    #[cfg(test)]
    pub fn spec(&self) -> &TextureHistorySpec {
        &self.spec
    }

    /// Borrows a half of the ping-pong pair; returns [`None`] until the first
    /// [`crate::render_graph::history::HistoryRegistry::ensure_resources`] call has allocated it.
    pub fn half(&self, index: usize) -> Option<&HistoryTexture> {
        self.pair.get(index)?.as_ref()
    }

    /// Current allocation generation for this history slot.
    #[cfg(test)]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub(super) fn reset_for_spec(&mut self, spec: TextureHistorySpec) {
        self.spec = spec;
        self.pair = [None, None];
        self.generation = self.generation.wrapping_add(1).max(1);
    }
}

pub(super) fn texture_specs_equivalent(a: &TextureHistorySpec, b: &TextureHistorySpec) -> bool {
    a.label == b.label
        && a.format == b.format
        && a.extent == b.extent
        && a.usage == b.usage
        && a.mip_level_count == b.mip_level_count
        && a.sample_count == b.sample_count
        && a.dimension == b.dimension
}

/// Computes the per-layer/per-mip view table shape for a texture history spec.
fn texture_history_view_shape(spec: &TextureHistorySpec) -> HistoryTextureViewShape {
    let layer_count = match spec.dimension {
        wgpu::TextureDimension::D2 => spec.extent.depth_or_array_layers.max(1),
        wgpu::TextureDimension::D1 | wgpu::TextureDimension::D3 => 0,
    };
    HistoryTextureViewShape {
        layer_count,
        mip_level_count: spec.mip_level_count.max(1),
    }
}

/// Creates D2 per-mip views for every array layer of a texture history allocation.
fn create_texture_history_mip_views(
    texture: &wgpu::Texture,
    spec: &TextureHistorySpec,
) -> HistoryTextureMipViews {
    let shape = texture_history_view_shape(spec);
    let mut layers = Vec::with_capacity(shape.layer_count as usize);
    for layer in 0..shape.layer_count {
        let mut mips = Vec::with_capacity(shape.mip_level_count as usize);
        for mip in 0..shape.mip_level_count {
            mips.push(texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("history_texture_mip_layer_view"),
                format: Some(spec.format),
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::All,
                base_mip_level: mip,
                mip_level_count: Some(1),
                base_array_layer: layer,
                array_layer_count: Some(1),
                ..Default::default()
            }));
            crate::profiling::note_resource_churn!(
                TextureView,
                "render_graph::history_texture_mip_layer_view"
            );
        }
        layers.push(Arc::<[wgpu::TextureView]>::from(mips));
    }
    HistoryTextureMipViews {
        layers: Arc::<[Arc<[wgpu::TextureView]>]>::from(layers),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tex_spec() -> TextureHistorySpec {
        TextureHistorySpec {
            label: "test_tex",
            format: wgpu::TextureFormat::Rgba16Float,
            extent: wgpu::Extent3d {
                width: 64,
                height: 64,
                depth_or_array_layers: 1,
            },
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
        }
    }

    #[test]
    fn texture_history_view_shape_tracks_d2_layers_and_mips() {
        let mut spec = tex_spec();
        spec.extent.depth_or_array_layers = 2;
        spec.mip_level_count = 6;

        assert_eq!(
            texture_history_view_shape(&spec),
            HistoryTextureViewShape {
                layer_count: 2,
                mip_level_count: 6,
            }
        );
    }

    #[test]
    fn texture_history_view_shape_skips_non_d2_subresource_tables() {
        let mut spec = tex_spec();
        spec.dimension = wgpu::TextureDimension::D3;
        spec.extent.depth_or_array_layers = 4;
        spec.mip_level_count = 3;

        assert_eq!(
            texture_history_view_shape(&spec),
            HistoryTextureViewShape {
                layer_count: 0,
                mip_level_count: 3,
            }
        );
    }
}
