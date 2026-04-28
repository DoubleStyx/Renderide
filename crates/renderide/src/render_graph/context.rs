//! Context types passed to each pass's recording method for one encoder slice.
//!
//! Four context types correspond to the four pass kinds in [`super::pass::PassNode`]:
//! - [`RasterPassCtx`] — graph has already opened `wgpu::RenderPass`; pass records draws.
//! - [`ComputePassCtx`] — pass receives the raw `wgpu::CommandEncoder` for compute work.
//! - [`CopyPassCtx`] — same as compute, semantically restricted to copy operations.
//! - [`CallbackCtx`] — no encoder; pass runs CPU prep, Queue writes, and blackboard mutations.
//!
//! [`PostSubmitContext`] is shared across all pass kinds for post-submit hooks.
//!
//! ## Lifetime parameters
//!
//! Contexts use up to three lifetime parameters:
//! - `'a` — immutable GPU handles (device, limits, queue, graph resources, views).
//! - `'encoder` — mutable encoder borrow (only compute/copy contexts).
//! - `'frame` — mutable scene/backend frame params borrow.
//!
//! Multi-view graph execution creates a **separate** encoder (and thus a separate context) for
//! frame-global work vs each view; passes never share one encoder across those slices.

use std::collections::HashSet;
use std::sync::Arc;

use crate::backend::HistoryTextureMipViews;
use crate::gpu::GpuLimits;

use super::blackboard::Blackboard;
use super::frame_params::{FrameRenderParams, FrameRenderParamsView, FrameSystemsShared};
use super::frame_upload_batch::FrameUploadBatch;
use super::resources::{
    BufferHandle, ImportedBufferHandle, ImportedTextureHandle, SubresourceHandle, TextureHandle,
    TextureResourceHandle,
};
use super::transient_pool::TransientPool;

// ─────────────────────────────────────────────────────────────────────────────
// Resolved resource types
// ─────────────────────────────────────────────────────────────────────────────

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

// ─────────────────────────────────────────────────────────────────────────────
// Resolved resource lookup table
// ─────────────────────────────────────────────────────────────────────────────

/// Execute-time resource lookup table built by [`super::compiled::CompiledRenderGraph`].
#[derive(Clone, Debug, Default)]
pub struct GraphResolvedResources {
    transient_textures: Vec<Option<ResolvedGraphTexture>>,
    transient_buffers: Vec<Option<ResolvedGraphBuffer>>,
    imported_textures: Vec<Option<ResolvedImportedTexture>>,
    imported_buffers: Vec<Option<ResolvedImportedBuffer>>,
    /// Resolved subresource views, populated eagerly per frame from the parent transient texture.
    /// Index parallels [`super::compiled::CompiledRenderGraph::subresources`].
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
}

// ─────────────────────────────────────────────────────────────────────────────
// New typed context structs (Phases 1–2)
// ─────────────────────────────────────────────────────────────────────────────

/// Context for [`super::pass::RasterPass::record`].
///
/// The graph has already opened a [`wgpu::RenderPass`] from the compiled attachment template;
/// the pass records draw commands into it. No encoder is exposed since the encoder is borrowed
/// by the open render pass.
pub struct RasterPassCtx<'a, 'frame> {
    /// WGPU device.
    pub device: &'a wgpu::Device,
    /// Effective limits for this frame.
    pub gpu_limits: &'a GpuLimits,
    /// Submission queue for resource creation paths that still require wgpu queue access.
    pub queue: &'a Arc<wgpu::Queue>,
    /// Swapchain view when the frame acquired the surface; [`None`] for offscreen-only graphs.
    pub backbuffer: Option<&'a wgpu::TextureView>,
    /// Depth attachment for the main forward pass.
    pub depth_view: Option<&'a wgpu::TextureView>,
    /// Scene + backend frame params for this view (serial path; `None` in parallel path).
    pub frame: Option<&'frame mut FrameRenderParams<'a>>,
    /// Shared system handles (parallel path; `None` in serial path — use `frame.shared`).
    pub frame_shared: Option<&'frame FrameSystemsShared<'a>>,
    /// Per-view surface state (parallel path; `None` in serial path — use `frame.view`).
    pub frame_view: Option<&'frame FrameRenderParamsView<'a>>,
    /// Deferred [`wgpu::Queue::write_buffer`] sink; drained on the main thread after all per-view
    /// encoding completes and before submit.
    pub upload_batch: &'frame FrameUploadBatch,
    /// Typed graph resources resolved for this execution scope.
    pub graph_resources: Option<&'a GraphResolvedResources>,
    /// Per-scope typed blackboard (read/write; populated by prior callback passes this scope).
    pub blackboard: &'frame mut Blackboard,
    /// GPU profiler handle for pass-level timestamp queries.
    ///
    /// [`None`] when the `tracy` feature is off or when the adapter lacks
    /// [`wgpu::Features::TIMESTAMP_QUERY`]. Pass bodies that open a render pass should call
    /// [`crate::profiling::GpuProfilerHandle::begin_pass_query`] and feed
    /// [`crate::profiling::render_pass_timestamp_writes`] into their descriptor when this is
    /// [`Some`], then close the query with
    /// [`crate::profiling::GpuProfilerHandle::end_query`] after the pass drops.
    pub profiler: Option<&'a crate::profiling::GpuProfilerHandle>,
}

impl RasterPassCtx<'_, '_> {
    /// Records a deferred buffer upload through the graph-owned upload recorder.
    pub fn write_buffer(&self, buffer: &wgpu::Buffer, offset: u64, data: &[u8]) {
        self.upload_batch.write_buffer(buffer, offset, data);
    }
}

/// Context for [`super::pass::ComputePass::record`].
///
/// The pass receives the raw [`wgpu::CommandEncoder`] and dispatches compute workgroups or
/// issues other encoder-level commands.
pub struct ComputePassCtx<'a, 'encoder, 'frame> {
    /// WGPU device.
    pub device: &'a wgpu::Device,
    /// Effective limits for this frame.
    pub gpu_limits: &'a GpuLimits,
    /// Submission queue for resource creation paths that still require wgpu queue access.
    pub queue: &'a Arc<wgpu::Queue>,
    /// Active command encoder for this recording slice.
    pub encoder: &'encoder mut wgpu::CommandEncoder,
    /// Depth attachment for the main forward pass (often needed by compute passes that
    /// read or copy the depth buffer).
    pub depth_view: Option<&'a wgpu::TextureView>,
    /// Scene + backend frame params for this view (serial path; `None` in parallel path).
    pub frame: Option<&'frame mut FrameRenderParams<'a>>,
    /// Shared system handles (parallel path; `None` in serial path — use `frame.shared`).
    pub frame_shared: Option<&'frame FrameSystemsShared<'a>>,
    /// Per-view surface state (parallel path; `None` in serial path — use `frame.view`).
    pub frame_view: Option<&'frame FrameRenderParamsView<'a>>,
    /// Deferred [`wgpu::Queue::write_buffer`] sink; drained on the main thread after all per-view
    /// encoding completes and before submit.
    pub upload_batch: &'frame FrameUploadBatch,
    /// Typed graph resources resolved for this execution scope.
    pub graph_resources: Option<&'a GraphResolvedResources>,
    /// Per-scope typed blackboard (read/write; populated by prior callback passes this scope).
    pub blackboard: &'frame mut Blackboard,
    /// GPU profiler handle for pass-level timestamp queries.
    ///
    /// [`None`] when the `tracy` feature is off or when the adapter lacks
    /// [`wgpu::Features::TIMESTAMP_QUERY`]. Pass bodies that open a compute pass should call
    /// [`crate::profiling::GpuProfilerHandle::begin_pass_query`] and feed
    /// [`crate::profiling::compute_pass_timestamp_writes`] into their descriptor when this is
    /// [`Some`], then close the query with
    /// [`crate::profiling::GpuProfilerHandle::end_query`] after the pass drops.
    pub profiler: Option<&'a crate::profiling::GpuProfilerHandle>,
}

impl ComputePassCtx<'_, '_, '_> {
    /// Records a deferred buffer upload through the graph-owned upload recorder.
    pub fn write_buffer(&self, buffer: &wgpu::Buffer, offset: u64, data: &[u8]) {
        self.upload_batch.write_buffer(buffer, offset, data);
    }
}

/// Context for [`super::pass::CopyPass::record`].
///
/// Structurally identical to [`ComputePassCtx`]; separated by type to distinguish copy-only
/// intent from arbitrary compute dispatch.
pub type CopyPassCtx<'a, 'encoder, 'frame> = ComputePassCtx<'a, 'encoder, 'frame>;

/// Context for [`super::pass::CallbackPass::run`].
///
/// No encoder is provided. The pass runs as a CPU callback, records uploads through
/// [`Self::write_buffer`], and mutates `blackboard`.
pub struct CallbackCtx<'a, 'frame> {
    /// WGPU device.
    pub device: &'a wgpu::Device,
    /// Effective limits for this frame.
    pub gpu_limits: &'a GpuLimits,
    /// Submission queue for resource creation paths that still require wgpu queue access.
    pub queue: &'a Arc<wgpu::Queue>,
    /// Scene + backend frame params for this view (serial path; `None` in parallel path).
    pub frame: Option<&'frame mut FrameRenderParams<'a>>,
    /// Shared system handles (parallel path; `None` in serial path — use `frame.shared`).
    pub frame_shared: Option<&'frame FrameSystemsShared<'a>>,
    /// Per-view surface state (parallel path; `None` in serial path — use `frame.view`).
    pub frame_view: Option<&'frame FrameRenderParamsView<'a>>,
    /// Deferred [`wgpu::Queue::write_buffer`] sink; drained on the main thread after all per-view
    /// encoding completes and before submit.
    pub upload_batch: &'frame FrameUploadBatch,
    /// Typed graph resources resolved for this execution scope.
    pub graph_resources: Option<&'a GraphResolvedResources>,
    /// Per-scope typed blackboard (read/write; the primary output of callback passes).
    pub blackboard: &'frame mut Blackboard,
}

impl CallbackCtx<'_, '_> {
    /// Records a deferred buffer upload through the graph-owned upload recorder.
    pub fn write_buffer(&self, buffer: &wgpu::Buffer, offset: u64, data: &[u8]) {
        self.upload_batch.write_buffer(buffer, offset, data);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Post-submit context (unchanged)
// ─────────────────────────────────────────────────────────────────────────────

/// Context passed to `post_submit` after a per-view or frame-global submit.
///
/// Runs on the CPU **after** [`wgpu::Queue::submit`] so passes can start `map_async` work on
/// buffers they wrote this frame (e.g. Hi-Z readback staging rotation).
pub struct PostSubmitContext<'a> {
    /// WGPU device for `map_async` and device polling.
    pub device: &'a wgpu::Device,
    /// Hi-Z readback and temporal bookkeeping for this view after submit.
    pub occlusion: &'a mut crate::backend::OcclusionSystem,
    /// Which occlusion view this submit covered.
    pub occlusion_view: super::OcclusionViewId,
    /// Host camera snapshot for the view.
    pub host_camera: super::HostCameraFrame,
}

// ─────────────────────────────────────────────────────────────────────────────
// Compatibility context types (kept for test compatibility; callers should migrate)
// ─────────────────────────────────────────────────────────────────────────────

/// Compatibility encoder-driven pass context. Prefer [`ComputePassCtx`] for new code.
///
/// Kept as an alias for incremental migration of tests and helper functions that reference the
/// compatibility type. Will be removed when all callers are updated.
pub struct RenderPassContext<'a, 'encoder, 'frame> {
    /// WGPU device.
    pub device: &'a wgpu::Device,
    /// Effective limits for this frame.
    pub gpu_limits: &'a GpuLimits,
    /// Submission queue.
    pub queue: &'a Arc<wgpu::Queue>,
    /// Active command encoder.
    pub encoder: &'encoder mut wgpu::CommandEncoder,
    /// Swapchain view when this frame acquired the surface.
    pub backbuffer: Option<&'a wgpu::TextureView>,
    /// Depth attachment for the main forward pass.
    pub depth_view: Option<&'a wgpu::TextureView>,
    /// Scene + backend frame params.
    pub frame: Option<&'frame mut FrameRenderParams<'a>>,
    /// Typed graph resources resolved for this execution scope.
    pub graph_resources: Option<&'a GraphResolvedResources>,
}

/// Compatibility graph-raster pass context. Prefer [`RasterPassCtx`] for new code.
///
/// Kept for incremental migration of tests and setup/compose pass helpers.
pub struct GraphRasterPassContext<'a, 'frame> {
    /// WGPU device.
    pub device: &'a wgpu::Device,
    /// Effective limits for this frame.
    pub gpu_limits: &'a GpuLimits,
    /// Submission queue.
    pub queue: &'a Arc<wgpu::Queue>,
    /// Swapchain view when this frame acquired the surface.
    pub backbuffer: Option<&'a wgpu::TextureView>,
    /// Depth attachment for the main forward pass.
    pub depth_view: Option<&'a wgpu::TextureView>,
    /// Scene + backend frame params.
    pub frame: Option<&'frame mut FrameRenderParams<'a>>,
    /// Typed graph resources resolved for this execution scope.
    pub graph_resources: Option<&'a GraphResolvedResources>,
}
