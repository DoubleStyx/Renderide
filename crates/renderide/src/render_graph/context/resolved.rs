//! Resolved transient/imported resource lookup table used during graph execution.

use std::collections::HashSet;
use std::sync::Arc;

use super::super::HistoryTextureMipViews;
use super::super::pool::TransientPool;
use super::super::resources::{
    BufferHandle, ImportedBufferHandle, ImportedTextureHandle, SubresourceHandle, TextureHandle,
    TextureResourceHandle,
};
use crate::gpu::GpuRetainedResources;
use crate::gpu_resource::TextureViewCache;

/// Resolved transient texture for one graph execution scope.
#[derive(Clone, Debug)]
pub struct ResolvedGraphTexture {
    /// Transient pool entry id.
    pub pool_id: usize,
    /// Texture handle.
    pub texture: wgpu::Texture,
    /// Default texture view.
    pub view: wgpu::TextureView,
    /// Compatible derived-view cache owned by the transient pool slot.
    pub view_cache: Arc<TextureViewCache>,
    /// Resource generation for the backing texture.
    pub resource_generation: u64,
    /// Resolved width in pixels.
    pub width: u32,
    /// Resolved height in pixels.
    pub height: u32,
    /// Per-layer D2 views for array textures.
    pub layer_views: Vec<wgpu::TextureView>,
    /// Resolved mip count for this execution scope.
    pub mip_levels: u32,
    /// Resolved array-layer/depth count for this execution scope.
    pub array_layers: u32,
    /// Resolved texture dimension.
    pub dimension: wgpu::TextureDimension,
}

/// Resolved transient buffer for one graph execution scope.
#[derive(Clone, Debug)]
pub struct ResolvedGraphBuffer {
    /// Transient pool entry id.
    pub pool_id: usize,
    /// Buffer handle.
    pub buffer: wgpu::Buffer,
}

/// Imported texture resolved from the current frame target or backend history.
#[derive(Clone, Debug)]
pub struct ResolvedImportedTexture {
    /// Texture view available to graph-owned pass descriptors.
    pub view: wgpu::TextureView,
    /// Backing history texture and subresource views when this import resolves a ping-pong slot.
    pub history: Option<ResolvedImportedHistoryTexture>,
}

/// Resolved ping-pong texture import with backing texture access for explicit subresource writes.
#[derive(Clone, Debug)]
pub struct ResolvedImportedHistoryTexture {
    /// Backing ping-pong texture for the selected current or previous half.
    pub texture: wgpu::Texture,
    /// Per-layer/per-mip views created by the history registry for this texture half.
    pub mip_views: HistoryTextureMipViews,
}

/// Imported buffer resolved from backend frame resources or external state.
#[derive(Clone, Debug)]
pub struct ResolvedImportedBuffer {
    /// Buffer handle.
    pub buffer: wgpu::Buffer,
}

/// Execute-time resource lookup table built by [`crate::render_graph::compiled::CompiledRenderGraph`].
#[derive(Clone, Debug, Default)]
pub struct GraphResolvedResources {
    transient_textures: Vec<Option<ResolvedGraphTexture>>,
    transient_buffers: Vec<Option<ResolvedGraphBuffer>>,
    imported_textures: Vec<Option<ResolvedImportedTexture>>,
    imported_buffers: Vec<Option<ResolvedImportedBuffer>>,
    /// Resolved subresource views, populated eagerly per frame from the parent transient texture.
    /// Index parallels [`crate::render_graph::compiled::CompiledRenderGraph::subresources`].
    subresource_views: Vec<Option<wgpu::TextureView>>,
}

impl GraphResolvedResources {
    /// Creates a lookup table with fixed handle capacities.
    pub fn with_capacity(
        transient_texture_count: usize,
        transient_buffer_count: usize,
        imported_texture_count: usize,
        imported_buffer_count: usize,
        subresource_count: usize,
    ) -> Self {
        Self {
            transient_textures: std::iter::repeat_with(|| None)
                .take(transient_texture_count)
                .collect(),
            transient_buffers: std::iter::repeat_with(|| None)
                .take(transient_buffer_count)
                .collect(),
            imported_textures: std::iter::repeat_with(|| None)
                .take(imported_texture_count)
                .collect(),
            imported_buffers: std::iter::repeat_with(|| None)
                .take(imported_buffer_count)
                .collect(),
            subresource_views: std::iter::repeat_with(|| None)
                .take(subresource_count)
                .collect(),
        }
    }

    /// Inserts a transient texture.
    pub fn set_transient_texture(&mut self, handle: TextureHandle, texture: ResolvedGraphTexture) {
        if let Some(slot) = self.transient_textures.get_mut(handle.index()) {
            *slot = Some(texture);
        }
    }

    /// Inserts a transient buffer.
    pub fn set_transient_buffer(&mut self, handle: BufferHandle, buffer: ResolvedGraphBuffer) {
        if let Some(slot) = self.transient_buffers.get_mut(handle.index()) {
            *slot = Some(buffer);
        }
    }

    /// Inserts an imported texture.
    pub fn set_imported_texture(
        &mut self,
        handle: ImportedTextureHandle,
        texture: ResolvedImportedTexture,
    ) {
        if let Some(slot) = self.imported_textures.get_mut(handle.index()) {
            *slot = Some(texture);
        }
    }

    /// Inserts an imported buffer.
    pub fn set_imported_buffer(
        &mut self,
        handle: ImportedBufferHandle,
        buffer: ResolvedImportedBuffer,
    ) {
        if let Some(slot) = self.imported_buffers.get_mut(handle.index()) {
            *slot = Some(buffer);
        }
    }

    /// Looks up a transient texture.
    pub fn transient_texture(&self, handle: TextureHandle) -> Option<&ResolvedGraphTexture> {
        self.transient_textures.get(handle.index())?.as_ref()
    }

    /// Looks up a transient buffer.
    #[cfg(test)]
    pub fn transient_buffer(&self, handle: BufferHandle) -> Option<&ResolvedGraphBuffer> {
        self.transient_buffers.get(handle.index())?.as_ref()
    }

    /// Looks up an imported texture.
    pub fn imported_texture(
        &self,
        handle: ImportedTextureHandle,
    ) -> Option<&ResolvedImportedTexture> {
        self.imported_textures.get(handle.index())?.as_ref()
    }

    /// Looks up an imported buffer.
    pub fn imported_buffer(&self, handle: ImportedBufferHandle) -> Option<&ResolvedImportedBuffer> {
        self.imported_buffers.get(handle.index())?.as_ref()
    }

    /// Inserts a resolved subresource view. Called by the executor at resolve time.
    pub fn set_subresource_view(&mut self, handle: SubresourceHandle, view: wgpu::TextureView) {
        if let Some(slot) = self.subresource_views.get_mut(handle.index()) {
            *slot = Some(view);
        }
    }

    /// Looks up a resolved subresource view.
    ///
    /// Returns [`None`] when the subresource index is out of range or the view has not been
    /// resolved for this frame yet. Pass this directly into bind groups or attachment
    /// descriptors the way you would any other `wgpu::TextureView`.
    pub fn subresource_view(&self, handle: SubresourceHandle) -> Option<&wgpu::TextureView> {
        self.subresource_views.get(handle.index())?.as_ref()
    }

    pub(crate) fn texture_view(&self, handle: TextureResourceHandle) -> Option<&wgpu::TextureView> {
        match handle {
            TextureResourceHandle::Transient(handle) => Some(&self.transient_texture(handle)?.view),
            TextureResourceHandle::Imported(handle) => Some(&self.imported_texture(handle)?.view),
        }
    }

    pub(crate) fn release_to_pool(&self, pool: &mut TransientPool) {
        let mut texture_ids = HashSet::new();
        for texture in self.transient_textures.iter().flatten() {
            if texture_ids.insert(texture.pool_id) {
                pool.release_texture(texture.pool_id);
            }
        }
        let mut buffer_ids = HashSet::new();
        for buffer in self.transient_buffers.iter().flatten() {
            if buffer_ids.insert(buffer.pool_id) {
                pool.release_buffer(buffer.pool_id);
            }
        }
    }

    /// Appends all GPU handles resolved for this graph scope to `out`.
    pub(crate) fn retain_submit_resources(&self, out: &mut GpuRetainedResources) {
        for texture in self.transient_textures.iter().flatten() {
            out.retain_texture(texture.texture.clone());
            out.retain_texture_view(texture.view.clone());
            out.retain_texture_views(texture.layer_views.iter().cloned());
        }
        for buffer in self.transient_buffers.iter().flatten() {
            out.retain_buffer(buffer.buffer.clone());
        }
        for texture in self.imported_textures.iter().flatten() {
            out.retain_texture_view(texture.view.clone());
            if let Some(history) = &texture.history {
                out.retain_texture(history.texture.clone());
                out.retain_texture_views(history.mip_views.iter_views().cloned());
            }
        }
        for buffer in self.imported_buffers.iter().flatten() {
            out.retain_buffer(buffer.buffer.clone());
        }
        out.retain_texture_views(self.subresource_views.iter().flatten().cloned());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_resources_missing_and_out_of_range_handles_return_none() {
        let resources = GraphResolvedResources::with_capacity(1, 1, 1, 1, 1);

        assert!(resources.transient_texture(TextureHandle(0)).is_none());
        assert!(resources.transient_texture(TextureHandle(1)).is_none());
        assert!(resources.transient_buffer(BufferHandle(0)).is_none());
        assert!(resources.transient_buffer(BufferHandle(1)).is_none());
        assert!(
            resources
                .imported_texture(ImportedTextureHandle(0))
                .is_none()
        );
        assert!(
            resources
                .imported_texture(ImportedTextureHandle(1))
                .is_none()
        );
        assert!(resources.imported_buffer(ImportedBufferHandle(0)).is_none());
        assert!(resources.imported_buffer(ImportedBufferHandle(1)).is_none());
        assert!(resources.subresource_view(SubresourceHandle(0)).is_none());
        assert!(resources.subresource_view(SubresourceHandle(1)).is_none());
    }

    #[test]
    fn graph_texture_view_lookup_returns_none_for_unresolved_resources() {
        let resources = GraphResolvedResources::with_capacity(1, 0, 1, 0, 0);

        assert!(
            resources
                .texture_view(TextureResourceHandle::Transient(TextureHandle(0)))
                .is_none()
        );
        assert!(
            resources
                .texture_view(TextureResourceHandle::Imported(ImportedTextureHandle(0)))
                .is_none()
        );
        assert!(
            resources
                .texture_view(TextureResourceHandle::Transient(TextureHandle(2)))
                .is_none()
        );
    }
}
