//! Per-pass record-time contexts (`RasterPassCtx`, `ComputePassCtx`, `PostSubmitContext`).

use super::super::blackboard::Blackboard;
use super::super::frame_params::GraphPassFrame;
use super::super::frame_upload_batch::GraphUploadSink;
use super::resolved::GraphResolvedResources;
use crate::gpu::GpuLimits;

/// Context for [`crate::render_graph::pass::RasterPass::record`].
///
/// The graph has already opened a [`wgpu::RenderPass`] from the compiled attachment template;
/// the pass records draw commands into it. No encoder is exposed since the encoder is borrowed
/// by the open render pass.
pub struct RasterPassCtx<'a, 'frame> {
    /// WGPU device.
    pub device: &'a wgpu::Device,
    /// Scene, backend system handles, and per-view frame state for this pass.
    pub pass_frame: &'frame mut GraphPassFrame<'a>,
    /// Deferred graph upload sink drained before submit.
    pub uploads: GraphUploadSink<'frame>,
    /// Typed graph resources resolved for this execution scope.
    pub graph_resources: &'a GraphResolvedResources,
    /// Per-scope typed blackboard (read/write; populated before or during this scope).
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
        self.uploads.write_buffer(buffer, offset, data);
    }
}

/// Context for [`crate::render_graph::pass::ComputePass::record`].
///
/// The pass receives the raw [`wgpu::CommandEncoder`] and dispatches compute workgroups or
/// issues other encoder-level commands.
pub struct ComputePassCtx<'a, 'encoder, 'frame> {
    /// WGPU device.
    pub device: &'a wgpu::Device,
    /// Effective limits for this frame.
    pub gpu_limits: &'a GpuLimits,
    /// Active command encoder for this recording slice.
    pub encoder: &'encoder mut wgpu::CommandEncoder,
    /// Depth attachment for the main forward pass (often needed by compute passes that
    /// read or copy the depth buffer).
    pub depth_view: Option<&'a wgpu::TextureView>,
    /// Scene, backend system handles, and per-view frame state for this pass.
    pub pass_frame: &'frame mut GraphPassFrame<'a>,
    /// Deferred graph upload sink drained before submit.
    pub uploads: GraphUploadSink<'frame>,
    /// Typed graph resources resolved for this execution scope.
    pub graph_resources: &'a GraphResolvedResources,
    /// Per-scope typed blackboard (read/write; populated before or during this scope).
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
        self.uploads.write_buffer(buffer, offset, data);
    }
}

/// Context for [`crate::render_graph::pass::EncoderPass::record`].
///
/// The pass receives the raw [`wgpu::CommandEncoder`] and may issue copies, resolves, or manually
/// opened render/compute passes. Resource dependencies still come from the pass setup declarations.
pub struct EncoderPassCtx<'a, 'encoder, 'frame> {
    /// WGPU device.
    pub device: &'a wgpu::Device,
    /// Active command encoder for this recording slice.
    pub encoder: &'encoder mut wgpu::CommandEncoder,
    /// Scene, backend system handles, and per-view frame state for this pass.
    pub pass_frame: &'frame mut GraphPassFrame<'a>,
    /// Deferred graph upload sink drained before submit.
    pub uploads: GraphUploadSink<'frame>,
    /// Typed graph resources resolved for this execution scope.
    pub graph_resources: &'a GraphResolvedResources,
    /// Per-scope typed blackboard (read/write; populated before or during this scope).
    pub blackboard: &'frame mut Blackboard,
    /// GPU profiler handle for encoder-level timestamp queries.
    pub profiler: Option<&'a crate::profiling::GpuProfilerHandle>,
}

/// Context passed to `post_submit` after a per-view or frame-global submit.
///
/// Runs on the CPU **after** [`wgpu::Queue::submit`] so passes can start `map_async` work on
/// buffers they wrote this frame (e.g. Hi-Z readback staging rotation).
pub struct PostSubmitContext<'a> {
    /// WGPU device for `map_async` and device polling.
    pub _device: &'a wgpu::Device,
    /// Hi-Z readback and temporal bookkeeping for this view after submit.
    pub _occlusion: &'a mut dyn crate::occlusion::OcclusionGraphHook,
    /// Which occlusion view this submit covered.
    pub _view_id: crate::camera::ViewId,
    /// Host camera snapshot for the view.
    pub _host_camera: crate::camera::HostCameraFrame,
}
