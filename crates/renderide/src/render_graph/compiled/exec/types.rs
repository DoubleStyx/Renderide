//! Shared execution data structures for compiled render graph execution.

use hashbrown::HashMap;

use crate::camera::{HostCameraFrame, ViewId};
use crate::diagnostics::PerViewHudOutputs;
use crate::scene::SceneCoordinator;

use super::super::super::context::GraphResolvedResources;
use super::super::super::frame_params::FrameViewClear;
use super::super::super::frame_upload_batch::{FrameUploadBatch, FrameUploadBatchStats};
use super::super::{FrameView, ResolvedView, WorldMeshDrawPlan};

/// Key for reusing transient pool allocations across [`FrameView`]s with identical surface layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct GraphResolveKey {
    pub(super) viewport_px: (u32, u32),
    pub(super) surface_format: wgpu::TextureFormat,
    pub(super) depth_stencil_format: wgpu::TextureFormat,
    pub(super) sample_count: u32,
    pub(super) multiview_stereo: bool,
}

/// CPU-side outputs collected while recording one per-view command buffer.
pub(super) struct PerViewEncodeOutput {
    /// Encoded GPU work for the view.
    pub(super) command_buffer: wgpu::CommandBuffer,
    /// Deferred HUD payload merged on the main thread after recording.
    pub(super) hud_outputs: Option<PerViewHudOutputs>,
    /// CPU time spent before this view's encoder finish.
    pub(super) encode_ms: f64,
    /// CPU time spent inside this view's encoder finish.
    pub(super) finish_ms: f64,
    /// World-mesh command counts captured from the final per-view blackboard.
    pub(super) world_mesh: WorldMeshCommandStats,
}

/// Completed per-view recording result, including ordering metadata for single-submit assembly.
pub(super) struct PerViewRecordOutput {
    /// Stable occlusion slot used by post-submit hooks.
    pub(super) view_id: ViewId,
    /// Host camera snapshot paired with the view.
    pub(super) host_camera: HostCameraFrame,
    /// Encoded GPU work for the view.
    pub(super) command_buffer: wgpu::CommandBuffer,
    /// Deferred HUD payload merged on the main thread after recording.
    pub(super) hud_outputs: Option<PerViewHudOutputs>,
    /// CPU time spent before this view's encoder finish.
    pub(super) encode_ms: f64,
    /// CPU time spent inside this view's encoder finish.
    pub(super) finish_ms: f64,
    /// World-mesh command counts captured from this view.
    pub(super) world_mesh: WorldMeshCommandStats,
}

/// Command buffer plus CPU timings for one encoder.
pub(super) struct TimedCommandBuffer {
    /// Encoded GPU work.
    pub(super) command_buffer: wgpu::CommandBuffer,
    /// CPU time spent before `finish`.
    pub(super) encode_ms: f64,
    /// CPU time spent inside `finish`.
    pub(super) finish_ms: f64,
}

/// Lightweight world-mesh counts attached to command-encoding diagnostics.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct WorldMeshCommandStats {
    /// Sorted draw items recorded for the view.
    pub(super) draws: usize,
    /// Indexed draw groups emitted across the forward subpasses.
    pub(super) instance_batches: usize,
    /// Pipeline-pass draw submissions after material multi-pass expansion.
    pub(super) pipeline_pass_submits: usize,
}

impl WorldMeshCommandStats {
    pub(super) fn add(&mut self, other: Self) {
        self.draws = self.draws.saturating_add(other.draws);
        self.instance_batches = self.instance_batches.saturating_add(other.instance_batches);
        self.pipeline_pass_submits = self
            .pipeline_pass_submits
            .saturating_add(other.pipeline_pass_submits);
    }
}

/// Owned clone of a resolved view so per-view workers can borrow it without touching [`GpuContext`].
#[derive(Clone)]
pub(super) struct OwnedResolvedView {
    /// Depth texture backing the view.
    pub(super) depth_texture: wgpu::Texture,
    /// Depth view used by raster and compute passes.
    pub(super) depth_view: wgpu::TextureView,
    /// Optional color attachment view.
    pub(super) backbuffer: Option<wgpu::TextureView>,
    /// Surface format for pipeline resolution.
    pub(super) surface_format: wgpu::TextureFormat,
    /// Pixel viewport for the view.
    pub(super) viewport_px: (u32, u32),
    /// Whether the view targets multiview stereo attachments.
    pub(super) multiview_stereo: bool,
    /// Optional offscreen render-texture asset id being written this pass.
    pub(super) offscreen_write_render_texture_asset_id: Option<i32>,
    /// Stable occlusion slot for the view.
    pub(super) view_id: ViewId,
    /// Effective sample count for the view.
    pub(super) sample_count: u32,
}

impl OwnedResolvedView {
    /// Borrows this owned snapshot as the executor's standard [`ResolvedView`] shape.
    pub(super) fn as_resolved(&self) -> ResolvedView<'_> {
        ResolvedView {
            depth_texture: &self.depth_texture,
            depth_view: &self.depth_view,
            backbuffer: self.backbuffer.as_ref(),
            surface_format: self.surface_format,
            viewport_px: self.viewport_px,
            multiview_stereo: self.multiview_stereo,
            offscreen_write_render_texture_asset_id: self.offscreen_write_render_texture_asset_id,
            view_id: self.view_id,
            sample_count: self.sample_count,
        }
    }
}

/// Serially prepared per-view input that can later be recorded on any rayon worker.
pub(super) struct PerViewWorkItem {
    /// Original input order for submit stability.
    pub(super) view_idx: usize,
    /// Host camera snapshot for the view.
    pub(super) host_camera: HostCameraFrame,
    /// Stable occlusion slot used by post-submit hooks.
    pub(super) view_id: ViewId,
    /// Optional secondary-camera transform filter.
    pub(super) draw_filter: Option<crate::world_mesh::draw_prep::CameraTransformDrawFilter>,
    /// Background clear/skybox behavior for this view.
    pub(super) clear: FrameViewClear,
    /// Explicit draw plan moved out of [`FrameView`] before fan-out.
    pub(super) world_mesh_draw_plan: WorldMeshDrawPlan,
    /// Owned resolved view snapshot safe to move to a worker thread.
    pub(super) resolved: OwnedResolvedView,
    /// Optional per-view `@group(0)` bind group and uniform buffer.
    pub(super) per_view_frame_bg_and_buf: Option<(std::sync::Arc<wgpu::BindGroup>, wgpu::Buffer)>,
}

/// Immutable shared inputs required to record one per-view command buffer.
pub(super) struct PerViewRecordShared<'a> {
    /// Scene after cache flush for the frame.
    pub(super) scene: &'a SceneCoordinator,
    /// Device used to build encoders and any lazily created views.
    pub(super) device: &'a wgpu::Device,
    /// Effective device limits for this frame.
    pub(super) gpu_limits: &'a crate::gpu::GpuLimits,
    /// Submission queue used by deferred uploads and pass callbacks.
    pub(super) queue_arc: &'a std::sync::Arc<wgpu::Queue>,
    /// Shared occlusion system for Hi-Z snapshots and temporal state.
    pub(super) occlusion: &'a crate::occlusion::OcclusionSystem,
    /// Shared frame resources for bind groups, lights, and per-view slabs.
    pub(super) frame_resources: &'a crate::backend::FrameResourceManager,
    /// Persistent history resources resolved for ping-pong graph imports.
    pub(super) history: &'a crate::backend::HistoryRegistry,
    /// Shared material system for pipeline and bind lookups.
    pub(super) materials: &'a crate::materials::MaterialSystem,
    /// Shared asset pools for meshes and textures.
    pub(super) asset_transfers: &'a crate::assets::asset_transfer_queue::AssetTransferQueue,
    /// Optional mesh preprocess pipelines (unused in per-view recording, kept for completeness).
    pub(super) mesh_preprocess: Option<&'a crate::mesh_deform::MeshPreprocessPipelines>,
    /// Optional read-only skin cache for deformed forward draws.
    pub(super) skin_cache: Option<&'a crate::mesh_deform::GpuSkinCache>,
    /// Read-only HUD capture switches for deferred per-view diagnostics.
    pub(super) debug_hud: crate::diagnostics::PerViewHudConfig,
    /// Scene-color format selected for the frame.
    pub(super) scene_color_format: wgpu::TextureFormat,
    /// GPU limits snapshot cloned into per-view frame params.
    pub(super) gpu_limits_arc: Option<std::sync::Arc<crate::gpu::GpuLimits>>,
    /// Optional MSAA depth-resolve resources for the frame.
    pub(super) msaa_depth_resolve: Option<std::sync::Arc<crate::gpu::MsaaDepthResolveResources>>,
    /// Live GTAO settings snapshot for the frame, seeded into each view's blackboard so the
    /// GTAO chain ([`crate::passes::post_processing::GtaoEffect`]) can read the current slider
    /// values without rebuilding the compiled render graph.
    pub(super) live_gtao_settings: crate::config::GtaoSettings,
    /// Live bloom settings snapshot for the frame, seeded into each view's blackboard so the
    /// bloom passes' UBO / blend constants / pipeline variants reflect current slider values.
    pub(super) live_bloom_settings: crate::config::BloomSettings,
    /// Live auto-exposure settings snapshot for the frame, seeded into each view's blackboard.
    pub(super) live_auto_exposure_settings: crate::config::AutoExposureSettings,
    /// Wall-frame delta used by temporal post-processing effects.
    pub(super) wall_frame_delta_seconds: f32,
}

impl GraphResolveKey {
    pub(super) fn from_resolved(resolved: &ResolvedView<'_>) -> Self {
        Self {
            viewport_px: resolved.viewport_px,
            surface_format: resolved.surface_format,
            depth_stencil_format: resolved.depth_texture.format(),
            sample_count: resolved.sample_count,
            multiview_stereo: resolved.multiview_stereo,
        }
    }
}

/// Immutable shared inputs threaded into [`CompiledRenderGraph::record_per_view_outputs`] for
/// both the serial and rayon-fan-out recording paths.
pub(super) struct PerViewRecordInputs<'a> {
    /// Pre-resolved transient pool leases keyed by view layout.
    pub(super) transient_by_key: &'a HashMap<GraphResolveKey, GraphResolvedResources>,
    /// Deferred upload sink drained on the main thread after recording.
    pub(super) upload_batch: &'a FrameUploadBatch,
    /// Shared frame systems and view-independent GPU state.
    pub(super) per_view_shared: &'a PerViewRecordShared<'a>,
    /// Optional GPU profiler handle that must be shared across workers by reference.
    pub(super) profiler: Option<&'a crate::profiling::GpuProfilerHandle>,
}

/// Per-view recording outputs, split into submission-parallel vectors consumed by [`SubmitFrameInputs`].
pub(super) struct RecordedPerViewBatch {
    /// One command buffer per view in input order.
    pub(super) per_view_cmds: Vec<wgpu::CommandBuffer>,
    /// Per-view occlusion slot + host camera pairs used for Hi-Z callbacks and post-submit hooks.
    pub(super) per_view_occlusion_info: Vec<(ViewId, HostCameraFrame)>,
    /// HUD payloads to apply after submit, parallel to `per_view_cmds`.
    pub(super) per_view_hud_outputs: Vec<Option<PerViewHudOutputs>>,
    /// Optional command buffer that resolves per-view GPU profiler queries.
    pub(super) per_view_profiler_cmd: Option<wgpu::CommandBuffer>,
    /// Aggregate CPU time spent before per-view encoder finishes.
    pub(super) encode_ms: f64,
    /// Aggregate CPU time spent inside per-view encoder finishes.
    pub(super) finish_ms: f64,
    /// Largest single per-view encoder finish.
    pub(super) max_finish_ms: f64,
    /// Aggregate world-mesh command counts across views.
    pub(super) world_mesh: WorldMeshCommandStats,
}

/// Submit-batch timings and upload counters captured after recording.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct SubmitFrameBatchStats {
    /// CPU time spent draining deferred uploads.
    pub(super) upload_drain_ms: f64,
    /// CPU time spent inside the upload encoder finish.
    pub(super) upload_finish_ms: f64,
    /// CPU time spent allocating and assembling command buffers.
    pub(super) command_batch_assembly_ms: f64,
    /// CPU time spent enqueueing the submit batch to the driver thread.
    pub(super) submit_enqueue_ms: f64,
    /// Number of command buffers submitted.
    pub(super) command_buffer_count: usize,
    /// Deferred upload traffic.
    pub(super) upload_stats: FrameUploadBatchStats,
}

#[derive(Debug)]
pub(super) struct DrainedUploadCommand {
    pub(super) command_buffer: Option<wgpu::CommandBuffer>,
    pub(super) stats: FrameUploadBatchStats,
    pub(super) drain_ms: f64,
}

/// Inputs threaded from [`CompiledRenderGraph::execute_multi_view`] into
/// [`CompiledRenderGraph::submit_frame_batch`].
///
/// Bundles the command buffers produced by each phase, the per-view metadata needed for Hi-Z
/// callbacks and HUD output application, and the swapchain/queue handles consumed by the single
/// submit.
pub(super) struct SubmitFrameInputs<'a> {
    /// Per-view targets in the input order (used for swapchain detection).
    pub(super) views: &'a [FrameView<'a>],
    /// Optional command buffer produced by frame-global passes.
    pub(super) frame_global_cmd: Option<wgpu::CommandBuffer>,
    /// One command buffer per view in input order.
    pub(super) per_view_cmds: Vec<wgpu::CommandBuffer>,
    /// Optional command buffer that resolves per-view GPU profiler queries.
    pub(super) per_view_profiler_cmd: Option<wgpu::CommandBuffer>,
    /// HUD payloads to apply after submit, parallel to `per_view_cmds`.
    pub(super) per_view_hud_outputs: Vec<Option<PerViewHudOutputs>>,
    /// Per-view occlusion slot + host camera pairs used for Hi-Z callbacks.
    pub(super) per_view_occlusion_info: &'a [(ViewId, HostCameraFrame)],
    /// Swapchain scope whose acquired texture (if any) is taken on submit.
    pub(super) swapchain_scope: &'a mut super::super::super::swapchain_scope::SwapchainScope,
    /// Optional swapchain backbuffer view for the HUD encoder.
    pub(super) backbuffer_view_holder: &'a Option<wgpu::TextureView>,
    /// Deferred upload batch drained before submit.
    pub(super) upload_batch: &'a FrameUploadBatch,
    /// Shared queue handle used for the HUD encoder.
    pub(super) queue_arc: &'a std::sync::Arc<wgpu::Queue>,
}

/// View surface properties used when resolving transient [`TextureKey`] values for a graph view.
pub(crate) struct TransientTextureResolveSurfaceParams {
    /// Viewport extent in pixels.
    pub viewport_px: (u32, u32),
    /// Swapchain or offscreen color format for format resolution.
    pub surface_format: wgpu::TextureFormat,
    /// Depth attachment format for format resolution.
    pub depth_stencil_format: wgpu::TextureFormat,
    /// HDR scene-color format ([`crate::config::RenderingSettings::scene_color_format`]).
    pub scene_color_format: wgpu::TextureFormat,
    /// MSAA sample count for the view.
    pub sample_count: u32,
    /// Stereo multiview (two layers) vs single-view.
    pub multiview_stereo: bool,
}
