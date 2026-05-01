//! [`CompiledRenderGraph`] execution: multi-view scheduling, resource resolution, and submits.
//!
//! ## Submit model
//!
//! Multi-view execution records optional frame-global work plus one command buffer per view, then
//! submits the whole batch through a single [`wgpu::Queue::submit`] call. Per-view
//! `Queue::write_buffer` uploads (per-draw slab, frame uniforms, cluster params) are drained
//! before submit, so each view's GPU commands see coherent buffer contents. Each view owns its
//! own per-draw slab buffer, so views never compete for per-draw storage capacity.
//!
//! ## Pass dispatch
//!
//! Each retained pass is a [`super::super::pass::PassNode`] enum. The executor matches on the
//! variant to call the correct record method:
//! - `Raster` → graph opens `wgpu::RenderPass` from template; calls `record_raster`.
//! - `Compute` → passes receive raw encoder; calls `record_compute`.
//! - `Copy` → same as compute; calls `record_copy`.
//! - `Callback` → no encoder; calls `run_callback`.

use hashbrown::HashMap;

use crate::backend::BackendGraphAccess;
use crate::gpu::GpuContext;
use crate::scene::SceneCoordinator;

use super::super::context::{GraphResolvedResources, PostSubmitContext};
use super::super::error::GraphExecuteError;
use super::super::frame_params::FrameViewClear;
use super::{
    CompiledRenderGraph, FrameView, FrameViewTarget, MultiViewExecutionContext, ResolvedView,
    WorldMeshDrawPlan,
};
use crate::camera::{HostCameraFrame, ViewId};
use crate::diagnostics::PerViewHudOutputs;

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
struct PerViewRecordInputs<'a> {
    /// Pre-resolved transient pool leases keyed by view layout.
    transient_by_key: &'a HashMap<GraphResolveKey, GraphResolvedResources>,
    /// Deferred upload sink drained on the main thread after recording.
    upload_batch: &'a super::super::frame_upload_batch::FrameUploadBatch,
    /// Shared frame systems and view-independent GPU state.
    per_view_shared: &'a PerViewRecordShared<'a>,
    /// Optional GPU profiler handle that must be shared across workers by reference.
    profiler: Option<&'a crate::profiling::GpuProfilerHandle>,
}

/// Per-view recording outputs, split into submission-parallel vectors consumed by [`SubmitFrameInputs`].
struct RecordedPerViewBatch {
    /// One command buffer per view in input order.
    per_view_cmds: Vec<wgpu::CommandBuffer>,
    /// Per-view occlusion slot + host camera pairs used for Hi-Z callbacks and post-submit hooks.
    per_view_occlusion_info: Vec<(ViewId, HostCameraFrame)>,
    /// HUD payloads to apply after submit, parallel to `per_view_cmds`.
    per_view_hud_outputs: Vec<Option<PerViewHudOutputs>>,
    /// Optional command buffer that resolves per-view GPU profiler queries.
    per_view_profiler_cmd: Option<wgpu::CommandBuffer>,
}

/// Releases all transient resource leases back to the pool and ticks the global GC counter.
fn release_transients_and_gc(
    mv_ctx: &mut MultiViewExecutionContext<'_, '_>,
    transient_by_key: HashMap<GraphResolveKey, GraphResolvedResources>,
) {
    let pool = mv_ctx.backend.transient_pool_mut();
    for (_, resources) in transient_by_key {
        resources.release_to_pool(pool);
    }
    profiling::scope!("render::transient_gc");
    pool.gc_tick(120);
}

/// Inputs threaded from [`CompiledRenderGraph::execute_multi_view`] into
/// [`CompiledRenderGraph::submit_frame_batch`].
///
/// Bundles the command buffers produced by each phase, the per-view metadata needed for Hi-Z
/// callbacks and HUD output application, and the swapchain/queue handles consumed by the single
/// submit.
struct SubmitFrameInputs<'a> {
    /// Per-view targets in the input order (used for swapchain detection).
    views: &'a [FrameView<'a>],
    /// Optional command buffer produced by frame-global passes.
    frame_global_cmd: Option<wgpu::CommandBuffer>,
    /// One command buffer per view in input order.
    per_view_cmds: Vec<wgpu::CommandBuffer>,
    /// Optional command buffer that resolves per-view GPU profiler queries.
    per_view_profiler_cmd: Option<wgpu::CommandBuffer>,
    /// HUD payloads to apply after submit, parallel to `per_view_cmds`.
    per_view_hud_outputs: Vec<Option<PerViewHudOutputs>>,
    /// Per-view occlusion slot + host camera pairs used for Hi-Z callbacks.
    per_view_occlusion_info: &'a [(ViewId, HostCameraFrame)],
    /// Swapchain scope whose acquired texture (if any) is taken on submit.
    swapchain_scope: &'a mut super::super::swapchain_scope::SwapchainScope,
    /// Optional swapchain backbuffer view for the HUD encoder.
    backbuffer_view_holder: &'a Option<wgpu::TextureView>,
    /// Deferred upload batch drained before submit.
    upload_batch: &'a super::super::frame_upload_batch::FrameUploadBatch,
    /// Shared queue handle used for the HUD encoder.
    queue_arc: &'a std::sync::Arc<wgpu::Queue>,
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

impl CompiledRenderGraph {
    /// Ordered pass count.
    pub fn pass_count(&self) -> usize {
        self.passes.len()
    }

    /// Whether this graph targets the swapchain this frame.
    pub fn needs_surface_acquire(&self) -> bool {
        self.needs_surface_acquire
    }

    /// Returns a CPU-side snapshot of the compiled schedule for the debug HUD.
    ///
    /// Captures pass count, wave count, phase distribution, and per-wave pass counts.
    pub fn schedule_hud_snapshot(&self) -> super::super::schedule::ScheduleHudSnapshot {
        super::super::schedule::ScheduleHudSnapshot::from_schedule(&self.schedule)
    }

    /// Validates the compiled schedule for structural invariants
    /// (frame-global before per-view, monotonic waves, wave ranges cover steps).
    ///
    /// Called by tests; production code can use this to surface graph build failures early.
    pub fn validate_schedule(&self) -> Result<(), super::super::schedule::ScheduleValidationError> {
        self.schedule.validate()
    }

    /// Records all views into separate command encoders and submits them in a single
    /// [`wgpu::Queue::submit`] call alongside the frame-global encoder.
    ///
    /// ## Per-view write ordering
    ///
    /// Per-view `Queue::write_buffer` calls (per-draw slab, frame uniforms, cluster params) happen
    /// during per-view callback passes. Since all writes are issued BEFORE the single submit, wgpu
    /// guarantees they are visible to every GPU command in that submit. Each view owns its own
    /// per-draw slab buffer (keyed by [`ViewId`]), so views never compete for buffer
    /// space.
    ///
    /// ## Per-view frame plan
    ///
    /// A [`super::super::frame_params::PerViewFramePlanSlot`] is inserted into each view's
    /// per-view blackboard carrying the per-view `@group(0)` frame bind group and uniform buffer.
    pub(crate) fn execute_multi_view(
        &mut self,
        gpu: &mut GpuContext,
        scene: &SceneCoordinator,
        backend: &mut BackendGraphAccess<'_>,
        views: &mut [FrameView<'_>],
    ) -> Result<(), GraphExecuteError> {
        profiling::scope!("graph::execute_multi_view");
        if views.is_empty() {
            return Ok(());
        }

        let Some((mut swapchain_scope, backbuffer_view_holder)) =
            self.enter_swapchain_scope_for_views(gpu, views)?
        else {
            return Ok(());
        };

        let device_arc = gpu.device().clone();
        let queue_arc = gpu.queue().clone();
        let limits_arc = gpu.limits().clone();
        let device = device_arc.as_ref();
        let gpu_limits = limits_arc.as_ref();

        backend.transient_pool_mut().begin_generation();

        let mut mv_ctx = MultiViewExecutionContext {
            gpu,
            scene,
            backend,
            device,
            gpu_limits,
            queue_arc: &queue_arc,
            backbuffer_view_holder: &backbuffer_view_holder,
        };

        let mut transient_by_key: HashMap<GraphResolveKey, GraphResolvedResources> = HashMap::new();

        // Pre-resolve transient textures and buffers for every unique view key before any
        // per-view recording begins. The record loop then reads `transient_by_key` without
        // touching the shared transient pool.
        self.pre_resolve_transients_for_views(&mut mv_ctx, views, &mut transient_by_key)?;

        // Deferred `queue.write_buffer` sink shared by frame-global and per-view record paths.
        // Drained onto the main thread after all recording completes and before submit.
        let upload_batch = super::super::frame_upload_batch::FrameUploadBatch::new();

        // Shared frame resources, per-view slots, mesh extended streams, and material pipelines
        // are warmed up front so later per-view recording can run with read-only shared state
        // plus per-view interior mutability.
        Self::prepare_view_resources_for_views(&mut mv_ctx, views)?;

        // ── Frame-global pass (optional) ─────────────────────────────────────────────────────
        let frame_global_cmd = self.encode_frame_global_passes(
            &mut mv_ctx,
            views,
            &mut transient_by_key,
            &upload_batch,
        )?;
        let per_view_work_items = self.prepare_per_view_work_items(&mut mv_ctx, views)?;

        // ── Per-view recording (no submit per view) ──────────────────────────────────────────
        let RecordedPerViewBatch {
            per_view_cmds,
            per_view_occlusion_info,
            per_view_hud_outputs,
            per_view_profiler_cmd,
        } = self.record_per_view_batch(
            &mut mv_ctx,
            per_view_work_items,
            &transient_by_key,
            &upload_batch,
        )?;

        self.submit_frame_batch(
            &mut mv_ctx,
            SubmitFrameInputs {
                views,
                frame_global_cmd,
                per_view_cmds,
                per_view_profiler_cmd,
                per_view_hud_outputs,
                per_view_occlusion_info: &per_view_occlusion_info,
                swapchain_scope: &mut swapchain_scope,
                backbuffer_view_holder: &backbuffer_view_holder,
                upload_batch: &upload_batch,
                queue_arc: &queue_arc,
            },
        )?;

        self.run_post_submit_passes(&mut mv_ctx, views, device, &per_view_occlusion_info)?;

        release_transients_and_gc(&mut mv_ctx, transient_by_key);

        Ok(())
    }

    /// Enters [`SwapchainScope`] for `views` if any target the swapchain; `Ok(None)` signals a frame skip.
    ///
    /// The scope holds the [`wgpu::SurfaceTexture`] for the entire frame. After all encoders
    /// are finished, the texture is taken out of the scope via
    /// [`super::super::swapchain_scope::SwapchainScope::take_surface_texture`] and handed to
    /// the driver thread for `Queue::submit` + `SurfaceTexture::present`. On any early return
    /// before the handoff, the scope still presents on drop so the wgpu Vulkan acquire
    /// semaphore is returned to the pool.
    fn enter_swapchain_scope_for_views(
        &self,
        gpu: &mut GpuContext,
        views: &[FrameView<'_>],
    ) -> Result<
        Option<(
            super::super::swapchain_scope::SwapchainScope,
            Option<wgpu::TextureView>,
        )>,
        GraphExecuteError,
    > {
        let needs_swapchain = views
            .iter()
            .any(|v| matches!(v.target, FrameViewTarget::Swapchain));
        match super::super::swapchain_scope::SwapchainScope::enter(
            needs_swapchain,
            self.needs_surface_acquire,
            gpu,
        )? {
            super::super::swapchain_scope::SwapchainEnterOutcome::NotNeeded => Ok(Some((
                super::super::swapchain_scope::SwapchainScope::none(),
                None,
            ))),
            super::super::swapchain_scope::SwapchainEnterOutcome::SkipFrame => Ok(None),
            super::super::swapchain_scope::SwapchainEnterOutcome::Acquired(scope) => {
                let bb = scope.backbuffer_view().cloned();
                Ok(Some((scope, bb)))
            }
        }
    }

    /// Records per-view command buffers and resolves per-view profiler queries.
    ///
    /// Builds [`PerViewRecordShared`] from `mv_ctx`, drives [`Self::record_per_view_outputs`], and
    /// splits the owned outputs into the parallel vectors consumed by [`SubmitFrameInputs`].
    fn record_per_view_batch(
        &self,
        mv_ctx: &mut MultiViewExecutionContext<'_, '_>,
        per_view_work_items: Vec<PerViewWorkItem>,
        transient_by_key: &HashMap<GraphResolveKey, GraphResolvedResources>,
        upload_batch: &super::super::frame_upload_batch::FrameUploadBatch,
    ) -> Result<RecordedPerViewBatch, GraphExecuteError> {
        let n_views = per_view_work_items.len();
        let device = mv_ctx.device;
        let record_parallelism = mv_ctx.backend.record_parallelism;
        let per_view_shared = PerViewRecordShared {
            scene: mv_ctx.scene,
            device,
            gpu_limits: mv_ctx.gpu_limits,
            queue_arc: mv_ctx.queue_arc,
            occlusion: mv_ctx.backend.occlusion(),
            frame_resources: mv_ctx.backend.frame_resources(),
            history: mv_ctx.backend.history_registry(),
            materials: mv_ctx.backend.materials(),
            asset_transfers: mv_ctx.backend.asset_transfers(),
            mesh_preprocess: mv_ctx.backend.mesh_preprocess(),
            skin_cache: mv_ctx.backend.skin_cache(),
            debug_hud: mv_ctx.backend.per_view_hud_config(),
            scene_color_format: mv_ctx.backend.scene_color_format_wgpu(),
            gpu_limits_arc: mv_ctx.backend.gpu_limits().cloned(),
            msaa_depth_resolve: mv_ctx.backend.msaa_depth_resolve(),
            live_gtao_settings: mv_ctx.backend.live_gtao_settings(),
            live_bloom_settings: mv_ctx.backend.live_bloom_settings(),
        };
        let mut per_view_profiler = mv_ctx.gpu.take_gpu_profiler();
        let record_result = (|| -> Result<RecordedPerViewBatch, GraphExecuteError> {
            let per_view_outputs = self.record_per_view_outputs(
                per_view_work_items,
                PerViewRecordInputs {
                    transient_by_key,
                    upload_batch,
                    per_view_shared: &per_view_shared,
                    profiler: per_view_profiler.as_ref(),
                },
                record_parallelism,
                n_views,
            )?;
            let mut per_view_cmds: Vec<wgpu::CommandBuffer> = Vec::with_capacity(n_views);
            let mut per_view_occlusion_info: Vec<(ViewId, HostCameraFrame)> =
                Vec::with_capacity(n_views);
            let mut per_view_hud_outputs: Vec<Option<PerViewHudOutputs>> =
                Vec::with_capacity(n_views);
            for output in per_view_outputs {
                per_view_cmds.push(output.command_buffer);
                per_view_occlusion_info.push((output.view_id, output.host_camera));
                per_view_hud_outputs.push(output.hud_outputs);
            }
            let per_view_profiler_cmd = per_view_profiler.as_mut().map(|profiler| {
                let mut profiler_encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("render-graph-per-view-profiler-resolve"),
                    });
                profiler.resolve_queries(&mut profiler_encoder);
                profiler_encoder.finish()
            });
            Ok(RecordedPerViewBatch {
                per_view_cmds,
                per_view_occlusion_info,
                per_view_hud_outputs,
                per_view_profiler_cmd,
            })
        })();
        mv_ctx.gpu.restore_gpu_profiler(per_view_profiler);

        record_result
    }

    /// Encodes the debug HUD overlay (swapchain path only), drains the deferred upload batch, and
    /// submits the assembled command buffers as a single batch through the GPU driver thread.
    fn submit_frame_batch(
        &self,
        mv_ctx: &mut MultiViewExecutionContext<'_, '_>,
        inputs: SubmitFrameInputs<'_>,
    ) -> Result<(), GraphExecuteError> {
        profiling::scope!("graph::single_submit");
        let SubmitFrameInputs {
            views,
            frame_global_cmd,
            per_view_cmds,
            per_view_profiler_cmd,
            per_view_hud_outputs,
            per_view_occlusion_info,
            swapchain_scope,
            backbuffer_view_holder,
            upload_batch,
            queue_arc,
        } = inputs;
        let target_is_swapchain = views
            .iter()
            .any(|v| matches!(v.target, FrameViewTarget::Swapchain));
        let queue_ref: &wgpu::Queue = queue_arc.as_ref();

        let hud_cmd = encode_swapchain_hud_overlay(
            mv_ctx,
            queue_ref,
            target_is_swapchain,
            backbuffer_view_holder.as_ref(),
        )?;

        // Drain the deferred upload arena into one staging buffer + N copy_buffer_to_buffer ops on
        // a dedicated upload encoder. Submitted at the head of the batch so subsequent encoders'
        // reads see the freshly copied uniforms / storage data.
        let upload_cmd = {
            profiling::scope!("gpu::drain_upload_batch");
            upload_batch.drain_and_flush(mv_ctx.device, queue_ref)
        };

        let all_cmds: Vec<wgpu::CommandBuffer> = upload_cmd
            .into_iter()
            .chain(frame_global_cmd)
            .chain(per_view_cmds)
            .chain(per_view_profiler_cmd)
            .chain(hud_cmd)
            .collect();

        // Hand the swapchain texture (if any) to the driver thread so `queue.submit` and
        // `SurfaceTexture::present` run off the main thread. The scope still drops cleanly — with
        // the texture taken, its `Drop` is a no-op.
        let surface_tex = if target_is_swapchain {
            swapchain_scope.take_surface_texture()
        } else {
            None
        };

        let hi_z_callbacks = collect_hi_z_submit_callbacks(mv_ctx, per_view_occlusion_info);

        {
            profiling::scope!("gpu::queue_submit");
            mv_ctx.gpu.submit_frame_batch_with_callbacks(
                all_cmds,
                surface_tex,
                None,
                hi_z_callbacks,
            );
        };

        for outputs in per_view_hud_outputs.iter().flatten() {
            mv_ctx.backend.apply_per_view_hud_outputs(outputs);
        }
        Ok(())
    }

    /// Runs frame-global and per-view `post_submit` hooks on every pass in schedule order.
    fn run_post_submit_passes(
        &mut self,
        mv_ctx: &mut MultiViewExecutionContext<'_, '_>,
        views: &[FrameView<'_>],
        device: &wgpu::Device,
        per_view_occlusion_info: &[(ViewId, HostCameraFrame)],
    ) -> Result<(), GraphExecuteError> {
        // Frame-global post-submit (uses first view's occlusion slot).
        if let Some((first_occlusion, first_hc)) = per_view_occlusion_info.first().copied() {
            profiling::scope!("graph::post_submit_frame_global");
            let mut post_ctx = PostSubmitContext {
                device,
                occlusion: mv_ctx.backend.occlusion_mut(),
                view_id: first_occlusion,
                host_camera: first_hc,
            };
            for &pass_idx in self.schedule.frame_global_pass_indices() {
                self.passes[pass_idx]
                    .post_submit(&mut post_ctx)
                    .map_err(GraphExecuteError::Pass)?;
            }
        }

        // Per-view post-submit.
        {
            profiling::scope!("graph::post_submit_per_view");
            for (view, (view_id, host_camera)) in views.iter().zip(per_view_occlusion_info.iter()) {
                let _ = view;
                let mut post_ctx = PostSubmitContext {
                    device,
                    occlusion: mv_ctx.backend.occlusion_mut(),
                    view_id: *view_id,
                    host_camera: *host_camera,
                };
                for &pass_idx in self.schedule.per_view_pass_indices() {
                    self.passes[pass_idx]
                        .post_submit(&mut post_ctx)
                        .map_err(GraphExecuteError::Pass)?;
                }
            }
        }
        Ok(())
    }

    /// Prepares owned per-view work items on the main thread before serial or parallel recording.
    fn prepare_per_view_work_items(
        &self,
        mv_ctx: &mut MultiViewExecutionContext<'_, '_>,
        views: &mut [FrameView<'_>],
    ) -> Result<Vec<PerViewWorkItem>, GraphExecuteError> {
        profiling::scope!("graph::prepare_per_view_work_items");
        let mut work_items = Vec::with_capacity(views.len());
        for (view_idx, view) in views.iter_mut().enumerate() {
            let view_id = view.view_id();
            let host_camera = view.host_camera;
            let per_view_frame_bg_and_buf = mv_ctx
                .backend
                .frame_resources
                .per_view_frame(view_id)
                .map(|state| {
                    (
                        state.frame_bind_group.clone(),
                        state.frame_uniform_buffer.clone(),
                    )
                });
            work_items.push(PerViewWorkItem {
                view_idx,
                host_camera,
                view_id,
                draw_filter: view.draw_filter.clone(),
                clear: view.clear,
                world_mesh_draw_plan: std::mem::replace(
                    &mut view.world_mesh_draw_plan,
                    WorldMeshDrawPlan::Empty,
                ),
                resolved: Self::resolve_owned_view_from_target(
                    view_id,
                    &view.target,
                    mv_ctx.gpu,
                    mv_ctx.backbuffer_view_holder.as_ref(),
                )?,
                per_view_frame_bg_and_buf,
            });
        }
        Ok(work_items)
    }

    /// Drives the per-view recording phase either serially or across a `rayon::scope` fan-out,
    /// returning one [`PerViewRecordOutput`] per input work item in submission order.
    ///
    /// `PerViewParallel` scaffolding is in place across the pass traits, frame-params split,
    /// upload batch, and transient resolve; remaining serialization points around shared backend
    /// systems are gated in [`super::super::record_parallel`].
    fn record_per_view_outputs(
        &self,
        per_view_work_items: Vec<PerViewWorkItem>,
        inputs: PerViewRecordInputs<'_>,
        record_parallelism: crate::config::RecordParallelism,
        n_views: usize,
    ) -> Result<Vec<PerViewRecordOutput>, GraphExecuteError> {
        profiling::scope!("graph::record_per_view_outputs");
        let graph: &CompiledRenderGraph = self;
        let PerViewRecordInputs {
            transient_by_key,
            upload_batch,
            per_view_shared,
            profiler,
        } = inputs;
        if record_parallelism == crate::config::RecordParallelism::PerViewParallel && n_views > 1 {
            profiling::scope!("graph::per_view_fan_out");
            let results = parking_lot::Mutex::new(
                std::iter::repeat_with(|| None)
                    .take(n_views)
                    .collect::<Vec<Option<PerViewRecordOutput>>>(),
            );
            let first_error = parking_lot::Mutex::new(None::<GraphExecuteError>);
            rayon::scope(|scope| {
                for work_item in per_view_work_items {
                    let results = &results;
                    let first_error = &first_error;
                    let shared = per_view_shared;
                    scope.spawn(move |_| {
                        if first_error.lock().is_some() {
                            return;
                        }
                        let view_idx = work_item.view_idx;
                        let view_id = work_item.view_id;
                        let host_camera = work_item.host_camera;
                        match graph.record_one_view(
                            shared,
                            work_item,
                            transient_by_key,
                            upload_batch,
                            profiler,
                        ) {
                            Ok(encoded) => {
                                results.lock()[view_idx] = Some(PerViewRecordOutput {
                                    view_id,
                                    host_camera,
                                    command_buffer: encoded.command_buffer,
                                    hud_outputs: encoded.hud_outputs,
                                });
                            }
                            Err(err) => {
                                let mut first_error = first_error.lock();
                                if first_error.is_none() {
                                    *first_error = Some(err);
                                }
                            }
                        }
                    });
                }
            });
            if let Some(err) = first_error.into_inner() {
                return Err(err);
            }
            results
                .into_inner()
                .into_iter()
                .map(|item| item.ok_or(GraphExecuteError::NoViewsInBatch))
                .collect::<Result<Vec<_>, _>>()
        } else {
            profiling::scope!("graph::per_view_serial");
            let mut outputs = Vec::with_capacity(n_views);
            for work_item in per_view_work_items {
                let view_id = work_item.view_id;
                let host_camera = work_item.host_camera;
                let encoded = graph.record_one_view(
                    per_view_shared,
                    work_item,
                    transient_by_key,
                    upload_batch,
                    profiler,
                )?;
                outputs.push(PerViewRecordOutput {
                    view_id,
                    host_camera,
                    command_buffer: encoded.command_buffer,
                    hud_outputs: encoded.hud_outputs,
                });
            }
            Ok(outputs)
        }
    }
}

mod pre_warm;
mod recording;
mod resolve;

/// Encodes the debug HUD overlay into its own command buffer for the swapchain path.
///
/// Returns `None` when the target isn't the swapchain or the HUD has no visible window for this
/// frame. In the no-content case, cached input-capture flags are cleared so a hidden HUD does not
/// block input dispatch to the world.
fn encode_swapchain_hud_overlay(
    mv_ctx: &mut MultiViewExecutionContext<'_, '_>,
    queue_ref: &wgpu::Queue,
    target_is_swapchain: bool,
    backbuffer_view: Option<&wgpu::TextureView>,
) -> Result<Option<wgpu::CommandBuffer>, GraphExecuteError> {
    if !target_is_swapchain {
        return Ok(None);
    }
    if !mv_ctx.backend.debug_hud_has_visible_content() {
        // No visible HUD content — drop cached input-capture flags so stale "want capture
        // keyboard/mouse" state from a previously visible HUD does not block input dispatch
        // to the world while the HUD is hidden.
        mv_ctx.backend.clear_debug_hud_input_capture();
        return Ok(None);
    }
    let Some(bb) = backbuffer_view else {
        return Err(GraphExecuteError::MissingSwapchainView);
    };
    let device: &wgpu::Device = mv_ctx.device;
    let viewport_px = mv_ctx.gpu.surface_extent_px();
    let mut hud_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("render-graph-hud"),
    });
    // Wrap the HUD encoder in a GPU profiler scope so Dear ImGui's per-frame draw work
    // appears as `graph::hud_imgui` on the Tracy GPU timeline. The encoder is its own
    // command buffer in the submit batch, so the resolve must happen inside the encoder
    // itself — `prof.resolve_queries` after `end_query` records the resolve copy in the
    // same buffer that wrote the timestamps, ensuring GPU ordering across the submit.
    let hud_query = mv_ctx
        .gpu
        .gpu_profiler_mut()
        .map(|p| p.begin_query("graph::hud_imgui", &mut hud_encoder));
    if let Err(e) = mv_ctx.backend.encode_debug_hud_overlay(
        device,
        queue_ref,
        &mut hud_encoder,
        bb,
        viewport_px,
    ) {
        logger::warn!("debug HUD overlay: {e}");
    }
    if let Some(query) = hud_query
        && let Some(prof) = mv_ctx.gpu.gpu_profiler_mut()
    {
        prof.end_query(&mut hud_encoder, query);
        prof.resolve_queries(&mut hud_encoder);
    }
    Ok(Some(hud_encoder.finish()))
}

/// Collects per-view Hi-Z submit-done notifications as `on_submitted_work_done` callbacks. Each
/// callback only marks the readback-ring slot as submit-done; the real `map_async` runs on the
/// main thread from the next frame's
/// [`crate::occlusion::OcclusionSystem::hi_z_begin_frame_readback`]. Doing wgpu work inside a
/// device-poll callback can deadlock against wgpu-internal locks that also serialize
/// `queue.write_texture` on the main thread (observed as a futex-wait hang inside
/// `write_one_mip`).
///
/// The encoded slot is captured out of the per-view state here (main thread, under the Hi-Z state
/// lock) and baked into the closure by value — a late-firing callback cannot consume a newer
/// frame's slot and alias two submits to the same staging buffer.
fn collect_hi_z_submit_callbacks(
    mv_ctx: &MultiViewExecutionContext<'_, '_>,
    per_view_occlusion_info: &[(ViewId, HostCameraFrame)],
) -> Vec<Box<dyn FnOnce() + Send + 'static>> {
    per_view_occlusion_info
        .iter()
        .filter_map(|(view_id, _hc)| {
            let state = mv_ctx.backend.occlusion.ensure_hi_z_state(*view_id);
            let ws = state.lock().take_encoded_slot()?;
            let cb: Box<dyn FnOnce() + Send + 'static> = Box::new(move || {
                profiling::scope!("hi_z::on_submitted_callback");
                state.lock().mark_submit_done(ws);
            });
            Some(cb)
        })
        .collect()
}
