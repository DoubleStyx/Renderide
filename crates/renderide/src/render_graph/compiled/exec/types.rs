//! Shared execution data structures for compiled render graph execution.

use hashbrown::HashMap;
use std::sync::Arc;

use crate::camera::{HostCameraFrame, ViewId};
use crate::diagnostics::PerViewHudOutputs;
use crate::gpu::{GpuLimits, MsaaDepthResolveResources};
use crate::graph_inputs::{
    FrameSystemsShared, FrameViewClear, GraphPassFrame, GraphPassFrameView, GraphSceneView,
    OffscreenWriteTarget, PerViewFramePlan, ViewWinding,
};
use crate::occlusion::gpu::HiZGpuState;
use crate::shared::RenderingContext;

use super::super::super::blackboard::{Blackboard, GraphCommandStats};
use super::super::super::context::GraphResolvedResources;
use super::super::super::frame_upload_batch::{FrameUploadBatch, FrameUploadBatchStats};
use super::super::super::history::HistoryRegistry;
use super::super::super::{GraphAssetResources, GraphFrameResources};
use super::super::{FrameGlobalView, FrameView, ResolvedView, ViewPostProcessing};
use super::recording_path::GraphCommandRecordingStrategy;

/// Key for reusing transient pool allocations across [`FrameView`]s with identical surface layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct GraphResolveKey {
    pub(super) viewport_px: (u32, u32),
    pub(super) surface_format: wgpu::TextureFormat,
    pub(super) depth_stencil_format: wgpu::TextureFormat,
    pub(super) sample_count: u32,
    pub(super) multiview_stereo: bool,
}

/// CPU-side outputs collected while recording one view's graph work.
pub(super) struct PerViewEncodeOutput {
    /// Encoded GPU work for the view, in deterministic submit order.
    pub(super) command_buffers: Vec<wgpu::CommandBuffer>,
    /// Deferred HUD payload merged on the main thread after recording.
    pub(super) hud_outputs: Option<PerViewHudOutputs>,
    /// CPU time spent before this view's encoder finish.
    pub(super) encode_ms: f64,
    /// CPU time spent inside this view's encoder finish.
    pub(super) finish_ms: f64,
    /// Command counts captured from the final per-view blackboard.
    pub(super) command_stats: GraphCommandStats,
}

/// Completed per-view recording result, including ordering metadata for single-submit assembly.
pub(super) struct PerViewRecordOutput {
    /// Stable occlusion slot used by post-submit hooks.
    pub(super) view_id: ViewId,
    /// Host camera snapshot paired with the view.
    pub(super) host_camera: HostCameraFrame,
    /// Encoded GPU work for the view, in deterministic submit order.
    pub(super) command_buffers: Vec<wgpu::CommandBuffer>,
    /// Deferred HUD payload merged on the main thread after recording.
    pub(super) hud_outputs: Option<PerViewHudOutputs>,
    /// CPU time spent before this view's encoder finish.
    pub(super) encode_ms: f64,
    /// CPU time spent inside this view's encoder finish.
    pub(super) finish_ms: f64,
    /// Command counts captured from this view.
    pub(super) command_stats: GraphCommandStats,
}

/// Copy metadata for a partial offscreen camera viewport.
#[derive(Clone)]
pub(super) struct ResolvedOffscreenColorCopy {
    /// Source texture rendered by this view.
    pub(super) source_texture: wgpu::Texture,
    /// Destination host render texture receiving the partial viewport.
    pub(super) destination_texture: wgpu::Texture,
    /// Destination origin in render-texture storage coordinates.
    pub(super) destination_origin_px: (u32, u32),
    /// Copy extent in pixels.
    pub(super) extent_px: (u32, u32),
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

/// Command recording strategy selected for the graph work in this frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum GraphCommandRecordingPath {
    /// Record frame-global and per-view graph work using the existing phase-specific command buffers.
    StandardCommandBuffers,
    /// Record frame-global work plus one serial swapchain view into one command encoder.
    SingleSwapchainEncoder,
}

impl GraphCommandRecordingPath {
    /// Numeric value used by Tracy plots and compact diagnostics.
    pub(super) const fn as_plot_value(self) -> u64 {
        match self {
            Self::StandardCommandBuffers => 0,
            Self::SingleSwapchainEncoder => 1,
        }
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
    /// Offscreen target currently being written by this view.
    pub(super) offscreen_write_target: OffscreenWriteTarget,
    /// Per-view winding policy before draw-local transform parity is applied.
    pub(super) view_winding: ViewWinding,
    /// Stable occlusion slot for the view.
    pub(super) view_id: ViewId,
    /// Effective sample count for the view.
    pub(super) sample_count: u32,
    /// Post-processing permissions requested by this view.
    pub(super) post_processing: ViewPostProcessing,
    /// Optional color copy into a host render texture after this view's graph passes.
    pub(super) offscreen_color_copy: Option<ResolvedOffscreenColorCopy>,
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
            offscreen_write_target: self.offscreen_write_target,
            view_winding: self.view_winding,
            view_id: self.view_id,
            sample_count: self.sample_count,
            post_processing: self.post_processing,
        }
    }

    /// Installs the late-acquired swapchain backbuffer on a previously metadata-only view.
    pub(super) fn attach_backbuffer(&mut self, backbuffer: &wgpu::TextureView) {
        self.backbuffer = Some(backbuffer.clone());
    }
}

/// Prepared view-local frame inputs reused by blackboard preparation and command recording.
pub(super) struct PreparedPerViewFrameInput {
    /// Depth-only view used by compute passes that sample the main depth texture.
    pub(super) depth_sample_view: wgpu::TextureView,
    /// Per-view frame bind group and backing uniform buffer seeded into the graph blackboard.
    pub(super) frame_plan: PerViewFramePlan,
    /// GPU capability limits snapshot exposed through [`GraphPassFrameView`].
    pub(super) gpu_limits: Option<Arc<GpuLimits>>,
    /// Optional MSAA depth-resolve helpers exposed through [`GraphPassFrameView`].
    pub(super) msaa_depth_resolve: Option<Arc<MsaaDepthResolveResources>>,
    /// Per-camera Hi-Z state slot exposed through [`GraphPassFrameView`].
    pub(super) hi_z_slot: Arc<parking_lot::Mutex<HiZGpuState>>,
}

/// Per-view fields used to build a pass-facing [`GraphPassFrame`].
pub(super) struct PreparedPerViewFrameParams<'a, 'view> {
    /// Resolved surface targets, viewport, and view flags for this view.
    pub(super) resolved: &'view ResolvedView<'a>,
    /// Scene color format selected for the frame.
    pub(super) scene_color_format: wgpu::TextureFormat,
    /// Host camera snapshot for the view.
    pub(super) host_camera: &'view HostCameraFrame,
    /// Render-context override scope used by this view.
    pub(super) render_context: RenderingContext,
    /// Elapsed renderer runtime in seconds for Unity-style shader time inputs.
    pub(super) frame_time_seconds: f32,
    /// Background clear/skybox behavior for this view.
    pub(super) clear: FrameViewClear,
    /// Post-processing permissions requested by this view.
    pub(super) post_processing: ViewPostProcessing,
}

impl PreparedPerViewFrameInput {
    /// Builds cached view-local inputs from a resolved view and backend-owned frame resources.
    pub(super) fn from_resolved(
        resolved: &ResolvedView<'_>,
        frame_plan: PerViewFramePlan,
        gpu_limits: Option<Arc<GpuLimits>>,
        msaa_depth_resolve: Option<Arc<MsaaDepthResolveResources>>,
        hi_z_slot: Arc<parking_lot::Mutex<HiZGpuState>>,
    ) -> Self {
        let depth_sample_view = resolved
            .depth_texture
            .create_view(&wgpu::TextureViewDescriptor {
                label: Some("depth_sample"),
                aspect: wgpu::TextureAspect::DepthOnly,
                ..Default::default()
            });
        crate::profiling::note_resource_churn!(
            TextureView,
            "render_graph::frame_depth_sample_view"
        );
        Self {
            depth_sample_view,
            frame_plan,
            gpu_limits,
            msaa_depth_resolve,
            hi_z_slot,
        }
    }

    /// Builds pass-facing frame parameters around the cached view-local resources.
    pub(super) fn frame_params<'a>(
        &self,
        shared: FrameSystemsShared<'a>,
        inputs: PreparedPerViewFrameParams<'a, '_>,
    ) -> GraphPassFrame<'a> {
        GraphPassFrame {
            shared,
            view: GraphPassFrameView {
                depth_texture: inputs.resolved.depth_texture,
                depth_view: inputs.resolved.depth_view,
                depth_sample_view: Some(self.depth_sample_view.clone()),
                surface_format: inputs.resolved.surface_format,
                scene_color_format: inputs.scene_color_format,
                viewport_px: inputs.resolved.viewport_px,
                host_camera: *inputs.host_camera,
                render_context: inputs.render_context,
                frame_time_seconds: inputs.frame_time_seconds,
                multiview_stereo: inputs.resolved.multiview_stereo,
                offscreen_write_target: inputs.resolved.offscreen_write_target,
                view_winding: inputs.resolved.view_winding,
                view_id: inputs.resolved.view_id,
                hi_z_slot: Arc::clone(&self.hi_z_slot),
                sample_count: inputs.resolved.sample_count,
                gpu_limits: self.gpu_limits.clone(),
                msaa_depth_resolve: self.msaa_depth_resolve.clone(),
                clear: inputs.clear,
                post_processing: inputs.post_processing,
            },
        }
    }
}

/// Serially prepared per-view input that can later be recorded on any rayon worker.
pub(super) struct PerViewWorkItem {
    /// Original input order for submit stability.
    pub(super) view_idx: usize,
    /// Host camera snapshot for the view.
    pub(super) host_camera: HostCameraFrame,
    /// Render-context override scope used by this view.
    pub(super) render_context: RenderingContext,
    /// Elapsed renderer runtime in seconds for Unity-style shader time inputs.
    pub(super) frame_time_seconds: f32,
    /// Stable occlusion slot used by post-submit hooks.
    pub(super) view_id: ViewId,
    /// Background clear/skybox behavior for this view.
    pub(super) clear: FrameViewClear,
    /// Post-processing permissions requested by this view.
    pub(super) post_processing: ViewPostProcessing,
    /// Whether this work item targets the desktop swapchain.
    pub(super) target_is_swapchain: bool,
    /// Caller-seeded blackboard moved out of the frame view before pass recording.
    pub(super) initial_blackboard: Blackboard,
    /// Owned resolved view snapshot safe to move to a worker thread.
    pub(super) resolved: OwnedResolvedView,
    /// Prepared view-local frame input reused by pre-record and command recording.
    pub(super) frame_input: PreparedPerViewFrameInput,
    /// Estimated world-mesh draw work captured before blackboard preparation consumes draw slots.
    pub(super) estimated_draw_count: usize,
}

/// Immutable shared inputs required to record one view's graph work.
pub(super) struct PerViewRecordShared<'a> {
    /// Scene after cache flush for the frame.
    pub(super) scene: GraphSceneView<'a>,
    /// Device used to build encoders and any lazily created views.
    pub(super) device: &'a wgpu::Device,
    /// Effective device limits for this frame.
    pub(super) gpu_limits: &'a GpuLimits,
    /// Shared occlusion system for Hi-Z snapshots and temporal state.
    pub(super) occlusion: &'a dyn crate::occlusion::OcclusionGraphHook,
    /// Shared frame resources for bind groups, lights, and per-view slabs.
    pub(super) frame_resources: &'a dyn GraphFrameResources,
    /// Persistent history resources resolved for ping-pong graph imports.
    pub(super) history: &'a HistoryRegistry,
    /// Shared material system for pipeline and bind lookups.
    pub(super) materials: &'a crate::materials::MaterialSystem,
    /// Shared asset pools for meshes and textures.
    pub(super) asset_resources: &'a dyn GraphAssetResources,
    /// Optional mesh preprocess pipelines (unused in per-view recording, kept for completeness).
    pub(super) mesh_preprocess: Option<&'a crate::mesh_deform::MeshPreprocessPipelines>,
    /// Optional read-only skin cache for deformed forward draws.
    pub(super) skin_cache: Option<&'a crate::mesh_deform::GpuSkinCache>,
    /// Host-owned skin influence mode for mesh deform compute.
    pub(super) skin_weight_mode: crate::shared::SkinWeightMode,
    /// Read-only HUD capture switches for deferred per-view diagnostics.
    pub(super) debug_hud: crate::diagnostics::PerViewHudConfig,
    /// Scene-color format selected for the frame.
    pub(super) scene_color_format: wgpu::TextureFormat,
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
    /// Frame-level strategy that decides which recording parallelism layer may use Rayon.
    pub(super) strategy: GraphCommandRecordingStrategy,
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
    /// Aggregate command counts across views.
    pub(super) command_stats: GraphCommandStats,
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
    /// Whether the submit included the window swapchain target.
    pub(super) target_is_swapchain: bool,
    /// Deferred upload traffic.
    pub(super) upload_stats: FrameUploadBatchStats,
}

pub(super) struct DrainedUploadCommand {
    pub(super) command_buffer: Option<wgpu::CommandBuffer>,
    pub(super) on_submitted_work_done: Option<Box<dyn FnOnce() + Send + 'static>>,
    pub(super) stats: FrameUploadBatchStats,
    pub(super) drain_ms: f64,
}

/// Inputs for recording frame-global passes into an existing encoder.
pub(super) struct FrameGlobalPassRecordInputs<'a, 'view> {
    /// Per-view targets used to locate the frame-global anchor view and secondary fallback layout.
    pub(super) views: &'a [FrameView<'view>],
    /// Primary-view metadata used for frame-global pass parameters.
    pub(super) frame_global: &'a FrameGlobalView,
    /// Shared transient resources keyed by the resolved view layout.
    pub(super) transient_by_key: &'a mut HashMap<GraphResolveKey, GraphResolvedResources>,
    /// Command encoder receiving frame-global work.
    pub(super) encoder: &'a mut wgpu::CommandEncoder,
    /// Deferred upload batch for scoped graph uploads.
    pub(super) upload_batch: &'a FrameUploadBatch,
    /// Optional profiler handle for pass GPU scopes.
    pub(super) pass_profiler: Option<&'a crate::profiling::GpuProfilerHandle>,
}

/// Inputs threaded from [`CompiledRenderGraph::execute_multi_view`] into
/// [`CompiledRenderGraph::submit_frame_batch`].
///
/// Bundles the command buffers produced by each phase, the per-view metadata needed for Hi-Z
/// callbacks and HUD output application, and the swapchain/queue handles consumed by the single
/// submit.
pub(super) struct SubmitFrameInputs<'a, 'view> {
    /// Per-view targets in the input order (used for swapchain detection).
    pub(super) views: &'a [FrameView<'view>],
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
    pub(super) queue_arc: &'a Arc<wgpu::Queue>,
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
