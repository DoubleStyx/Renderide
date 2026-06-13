//! Shared texture-history view tables used by render graph history and graph-driven passes.

use std::sync::Arc;

/// Per-layer/per-mip views created for one texture-history allocation.
#[derive(Clone, Debug)]
pub struct HistoryTextureMipViews {
    /// Views grouped as `layers[layer][mip]`.
    layers: Arc<[Arc<[wgpu::TextureView]>]>,
}

impl HistoryTextureMipViews {
    /// Creates a history mip-view table from already grouped layer views.
    pub(crate) fn from_layers(layers: Arc<[Arc<[wgpu::TextureView]>]>) -> Self {
        Self { layers }
    }

    /// Returns all mip views for one array layer.
    pub fn layer_mip_views(&self, layer: u32) -> Option<&[wgpu::TextureView]> {
        self.layers.get(layer as usize).map(AsRef::as_ref)
    }

    /// Iterates every cached mip view.
    pub(crate) fn iter_views(&self) -> impl Iterator<Item = &wgpu::TextureView> {
        self.layers.iter().flat_map(|layer| layer.as_ref().iter())
    }
}
