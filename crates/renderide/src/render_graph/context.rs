//! Per-frame context passed to each [`super::RenderPass`].

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use crate::gpu::GpuLimits;

use super::frame_params::FrameRenderParams;
use super::resources::{
    BufferHandle, ImportedBufferHandle, ImportedTextureHandle, TextureHandle, TextureResourceHandle,
};
use super::transient_pool::TransientPool;

/// Resolved transient texture for one graph execution scope.
#[derive(Clone, Debug)]
pub struct ResolvedGraphTexture {
    /// Transient pool entry id.
    pub pool_id: usize,
    /// Compiler-assigned alias slot.
    pub physical_slot: usize,
    /// Texture handle.
    pub texture: wgpu::Texture,
    /// Default texture view.
    pub view: wgpu::TextureView,
    /// Per-layer D2 views for array textures.
    pub layer_views: Vec<wgpu::TextureView>,
}

/// Resolved transient buffer for one graph execution scope.
#[derive(Clone, Debug)]
pub struct ResolvedGraphBuffer {
    /// Transient pool entry id.
    pub pool_id: usize,
    /// Compiler-assigned alias slot.
    pub physical_slot: usize,
    /// Buffer handle.
    pub buffer: wgpu::Buffer,
    /// Buffer size in bytes.
    pub size: u64,
}

/// Imported texture resolved from the current frame target or backend history.
#[derive(Clone, Debug)]
pub struct ResolvedImportedTexture {
    /// Texture view available to graph-owned pass descriptors.
    pub view: wgpu::TextureView,
}

/// Imported buffer resolved from backend frame resources or external state.
#[derive(Clone, Debug)]
pub struct ResolvedImportedBuffer {
    /// Buffer handle.
    pub buffer: wgpu::Buffer,
}

/// Execute-time resource lookup table built by [`super::CompiledRenderGraph`].
#[derive(Debug, Default)]
pub struct GraphResolvedResources {
    transient_textures: Vec<Option<ResolvedGraphTexture>>,
    transient_buffers: Vec<Option<ResolvedGraphBuffer>>,
    imported_textures: Vec<Option<ResolvedImportedTexture>>,
    imported_buffers: Vec<Option<ResolvedImportedBuffer>>,
}

impl GraphResolvedResources {
    /// Creates a lookup table with fixed handle capacities.
    pub fn with_capacity(
        transient_texture_count: usize,
        transient_buffer_count: usize,
        imported_texture_count: usize,
        imported_buffer_count: usize,
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
}

/// Immutable GPU handles and mutable encoder for one frame’s recording.
pub struct RenderPassContext<'a, 'encoder, 'frame> {
    /// WGPU device.
    pub device: &'a wgpu::Device,
    /// Effective limits for this frame (from [`crate::gpu::GpuContext::limits`]).
    pub gpu_limits: &'a GpuLimits,
    /// Submission queue (same mutex as [`crate::gpu::GpuContext::queue`]).
    pub queue: &'a Arc<Mutex<wgpu::Queue>>,
    /// Command encoder for this frame (all passes share one encoder in v1).
    pub encoder: &'encoder mut wgpu::CommandEncoder,
    /// Swapchain view when this frame acquired the surface; [`None`] for offscreen-only graphs.
    pub backbuffer: Option<&'a wgpu::TextureView>,
    /// Depth attachment for the main forward pass when configured.
    pub depth_view: Option<&'a wgpu::TextureView>,
    /// Scene + backend when the graph participates in mesh drawing.
    pub frame: Option<&'frame mut FrameRenderParams<'a>>,
    /// Typed graph resources resolved for this execution scope.
    pub graph_resources: Option<&'a GraphResolvedResources>,
}

/// Context for raster passes whose `wgpu::RenderPass` is opened by the graph.
pub struct GraphRasterPassContext<'a, 'frame> {
    /// WGPU device.
    pub device: &'a wgpu::Device,
    /// Effective limits for this frame (from [`crate::gpu::GpuContext::limits`]).
    pub gpu_limits: &'a GpuLimits,
    /// Submission queue (same mutex as [`crate::gpu::GpuContext::queue`]).
    pub queue: &'a Arc<Mutex<wgpu::Queue>>,
    /// Swapchain view when this frame acquired the surface; [`None`] for offscreen-only graphs.
    pub backbuffer: Option<&'a wgpu::TextureView>,
    /// Depth attachment for the main forward pass when configured.
    pub depth_view: Option<&'a wgpu::TextureView>,
    /// Scene + backend when the graph participates in mesh drawing.
    pub frame: Option<&'frame mut FrameRenderParams<'a>>,
    /// Typed graph resources resolved for this execution scope.
    pub graph_resources: Option<&'a GraphResolvedResources>,
}
