//! [`CompiledRenderGraph`] execution: multi-view scheduling, resource resolution, and submits.
//!
//! ## Submit model
//!
//! Multi-view execution issues **one submit for frame-global work** (optional) plus
//! **one submit per view** for per-view passes. This ordering guarantees that
//! per-view `Queue::write_buffer` uploads (per-draw slab, frame uniforms, cluster params) are
//! visible to that view's GPU commands. Each view owns its own per-draw slab buffer, so views
//! never compete for per-draw storage capacity.
//!
//! ## Pass dispatch
//!
//! Each retained pass is a [`super::super::pass::PassNode`] enum. The executor matches on the
//! variant to call the correct record method:
//! - `Raster` → graph opens `wgpu::RenderPass` from template; calls `record_raster`.
//! - `Compute` → passes receive raw encoder; calls `record_compute`.
//! - `Copy` → same as compute; calls `record_copy`.
//! - `Callback` → no encoder; calls `run_callback`.

use hashbrown::hash_map::Entry;
use hashbrown::HashMap;

use crate::backend::RenderBackend;
use crate::gpu::GpuContext;
use crate::scene::SceneCoordinator;

use super::super::blackboard::Blackboard;
use super::super::context::{
    CallbackCtx, ComputePassCtx, GraphResolvedResources, PostSubmitContext, RasterPassCtx,
    ResolvedGraphBuffer, ResolvedGraphTexture, ResolvedImportedBuffer, ResolvedImportedTexture,
};
use super::super::error::GraphExecuteError;
use super::super::frame_params::{
    HostCameraFrame, MsaaViewsSlot, OcclusionViewId, PerViewFramePlan, PerViewFramePlanSlot,
    PrefetchedWorldMeshDrawsSlot,
};
use super::super::pass::PassKind;
use super::super::resources::{
    BackendFrameBufferKind, BufferImportSource, FrameTargetRole, ImportSource,
    ImportedBufferHandle, ImportedTextureHandle, TextureHandle,
};
use super::super::transient_pool::{BufferKey, TextureKey, TransientPool};
use super::helpers;
use super::{
    CompiledRenderGraph, ExternalFrameTargets, FrameView, FrameViewTarget,
    MultiViewExecutionContext, OffscreenSingleViewExecuteSpec, ResolvedView,
};

/// Key for reusing transient pool allocations across [`FrameView`]s with identical surface layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct GraphResolveKey {
    viewport_px: (u32, u32),
    surface_format: wgpu::TextureFormat,
    depth_stencil_format: wgpu::TextureFormat,
    sample_count: u32,
    multiview_stereo: bool,
}

impl GraphResolveKey {
    fn from_resolved(resolved: &ResolvedView<'_>) -> Self {
        Self {
            viewport_px: resolved.viewport_px,
            surface_format: resolved.surface_format,
            depth_stencil_format: resolved.depth_texture.format(),
            sample_count: resolved.sample_count,
            multiview_stereo: resolved.multiview_stereo,
        }
    }
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

    /// Desktop single-view entry: delegates to [`Self::execute_multi_view`] (one swapchain view).
    pub fn execute(
        &mut self,
        gpu: &mut GpuContext,
        scene: &SceneCoordinator,
        backend: &mut RenderBackend,
        host_camera: HostCameraFrame,
    ) -> Result<(), GraphExecuteError> {
        let mut single = [FrameView {
            host_camera,
            target: FrameViewTarget::Swapchain,
            draw_filter: None,
            prefetched_world_mesh_draws: None,
        }];
        self.execute_multi_view(gpu, scene, backend, &mut single)
    }

    /// Records passes against pre-built multiview array targets (OpenXR swapchain path).
    pub fn execute_external_multiview(
        &mut self,
        gpu: &mut GpuContext,
        scene: &SceneCoordinator,
        backend: &mut RenderBackend,
        host_camera: HostCameraFrame,
        external: ExternalFrameTargets<'_>,
    ) -> Result<(), GraphExecuteError> {
        let mut single = [FrameView {
            host_camera,
            target: FrameViewTarget::ExternalMultiview(external),
            draw_filter: None,
            prefetched_world_mesh_draws: None,
        }];
        self.execute_multi_view(gpu, scene, backend, &mut single)
    }

    /// Renders the graph to a single-view offscreen color/depth target (secondary camera → render texture).
    pub fn execute_offscreen_single_view(
        &mut self,
        gpu: &mut GpuContext,
        backend: &mut RenderBackend,
        spec: OffscreenSingleViewExecuteSpec<'_>,
    ) -> Result<(), GraphExecuteError> {
        let scene = spec.scene;
        let host_camera = spec.host_camera;
        let external = spec.external;
        let transform_filter = spec.transform_filter;
        let prefetched_world_mesh_draws = spec.prefetched_world_mesh_draws;
        let mut single = [FrameView {
            host_camera,
            target: FrameViewTarget::OffscreenRt(external),
            draw_filter: transform_filter,
            prefetched_world_mesh_draws,
        }];
        self.execute_multi_view(gpu, scene, backend, &mut single)
    }

    /// Records all views into separate command encoders and submits them in a single
    /// [`wgpu::Queue::submit`] call alongside the frame-global encoder.
    ///
    /// ## Per-view write ordering
    ///
    /// Per-view `Queue::write_buffer` calls (per-draw slab, frame uniforms, cluster params) happen
    /// during per-view callback passes. Since all writes are issued BEFORE the single submit, wgpu
    /// guarantees they are visible to every GPU command in that submit. Each view owns its own
    /// per-draw slab buffer (keyed by [`OcclusionViewId`]), so views never compete for buffer
    /// space.
    ///
    /// ## Per-view frame plan
    ///
    /// A [`super::super::frame_params::PerViewFramePlanSlot`] is inserted into each view's
    /// per-view blackboard carrying the per-view `@group(0)` frame bind group and uniform buffer.
    pub fn execute_multi_view<'a>(
        &mut self,
        gpu: &mut GpuContext,
        scene: &SceneCoordinator,
        backend: &mut RenderBackend,
        views: &mut [FrameView<'a>],
    ) -> Result<(), GraphExecuteError> {
        profiling::scope!("graph::execute_multi_view");
        if views.is_empty() {
            return Ok(());
        }

        let needs_swapchain = views
            .iter()
            .any(|v| matches!(v.target, FrameViewTarget::Swapchain));

        // Surface acquire + fallback present-on-drop via SwapchainScope.
        //
        // The scope holds the [`wgpu::SurfaceTexture`] for the entire frame. After all encoders
        // are finished below, the texture is taken out of the scope via
        // [`SwapchainScope::take_surface_texture`] and handed to the driver thread for
        // `Queue::submit` + `SurfaceTexture::present`. The scope's `Drop` impl tolerates the
        // texture being gone — it becomes a no-op for this frame. On any early return (error
        // or skip) before the handoff, the scope still presents on drop so the wgpu Vulkan
        // acquire semaphore is returned to the pool.
        let (mut swapchain_scope, backbuffer_view_holder): (
            super::super::swapchain_scope::SwapchainScope,
            Option<wgpu::TextureView>,
        ) = match super::super::swapchain_scope::SwapchainScope::enter(
            needs_swapchain,
            self.needs_surface_acquire,
            gpu,
        )? {
            super::super::swapchain_scope::SwapchainEnterOutcome::NotNeeded => {
                (super::super::swapchain_scope::SwapchainScope::none(), None)
            }
            super::super::swapchain_scope::SwapchainEnterOutcome::SkipFrame => return Ok(()),
            super::super::swapchain_scope::SwapchainEnterOutcome::Acquired(scope) => {
                let bb = scope.backbuffer_view().cloned();
                (scope, bb)
            }
        };

        let device_arc = gpu.device().clone();
        let queue_arc = gpu.queue().clone();
        let limits_arc = gpu.limits().clone();
        let device = device_arc.as_ref();
        let gpu_limits = limits_arc.as_ref();

        backend.transient_pool_mut().begin_generation();

        let n_views = views.len();

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

        // Pre-resolve transient textures and buffers for every unique view key before any per-view
        // recording begins. Milestone D hoists `backend.transient_pool_mut()` access out of the
        // per-view loop so that the loop becomes read-only against `transient_by_key` (except for
        // per-view imported overlays, which mutate disjoint entries today and will be split per-view
        // in Milestone E).
        self.pre_resolve_transients_for_views(&mut mv_ctx, views, &mut transient_by_key)?;

        // Deferred `queue.write_buffer` sink shared by frame-global and per-view record paths.
        // Drained onto the main thread after all recording completes and before submit.
        let upload_batch = super::super::frame_upload_batch::FrameUploadBatch::new();

        // ── Pre-warm per-view resources and material pipeline cache ─────────────────────────
        //
        // Calls `per_view_frame_or_create` and `per_view_per_draw_or_create` for every view up
        // front so subsequent per-view recording no longer mutates the per-view resource maps.
        // This also primes secondary RT cameras on first appearance (no first-frame stutter
        // when a new view id shows up mid-session) and is a structural prerequisite for moving
        // the per-view recording loop onto rayon workers in a future iteration. Pipeline cache
        // pre-warming walks every view's prefetched draw list and materializes pipelines for
        // unique batch keys so the per-view record loop sees only cache hits.
        Self::pre_warm_per_view_resources_for_views(&mut mv_ctx, views)?;
        Self::pre_warm_pipeline_cache_for_views(&mut mv_ctx, views);

        // ── Frame-global pass (optional) ─────────────────────────────────────────────────────
        let frame_global_cmd = self.encode_frame_global_passes(
            &mut mv_ctx,
            views,
            &mut transient_by_key,
            &upload_batch,
        )?;

        // ── Per-view recording (no submit per view) ──────────────────────────────────────────
        // Serial vs parallel recording is controlled by `backend.record_parallelism`.
        //
        // `PerViewParallel` scaffolding is in place: `record(&self, …)` on every pass trait,
        // `FrameSystemsShared` / `FrameRenderParamsView` split, `FrameUploadBatch` drains on the
        // main thread post-submit, transient textures/buffers pre-resolved once before the loop,
        // per-view scratch slabs (uniforms + byte slab) keyed by `OcclusionViewId`, per-view
        // resources pre-warmed above, and [`encode_per_view_to_cmd`] / [`execute_pass_node`]
        // both take `&self`. The remaining blockers for full `rayon::scope` fan-out live in
        // [`super::super::record_parallel`]: they require interior mutability on the
        // [`crate::backend::OcclusionSystem`] / [`crate::backend::FrameResourceManager`] /
        // [`crate::backend::MaterialSystem`] handles passed via [`FrameSystemsShared`], plus
        // gating around the singleton `GpuProfiler` take/restore pattern.
        let record_parallelism = mv_ctx.backend.record_parallelism;
        if record_parallelism == crate::config::RecordParallelism::PerViewParallel && n_views > 1 {
            super::super::record_parallel::warn_parallel_falls_back_once(n_views);
        }
        let mut per_view_cmds: Vec<wgpu::CommandBuffer> = Vec::with_capacity(n_views);
        let mut per_view_occlusion_info: Vec<(
            OcclusionViewId,
            super::super::frame_params::HostCameraFrame,
        )> = Vec::with_capacity(n_views);

        for (view_idx, view) in views.iter_mut().enumerate() {
            let occlusion_view = view.occlusion_view_id();
            let host_camera = view.host_camera;

            // Determine per-view viewport and stereo for per-view frame/cluster state.
            let (view_viewport, view_stereo) = match &view.target {
                FrameViewTarget::ExternalMultiview(ext) => {
                    let stereo = host_camera.vr_active && host_camera.stereo_views.is_some();
                    (ext.extent_px, stereo)
                }
                FrameViewTarget::OffscreenRt(ext) => (ext.extent_px, false),
                FrameViewTarget::Swapchain => (mv_ctx.gpu.surface_extent_px(), false),
            };

            // Ensure per-view cluster buffers and @group(0) bind group exist for this view.
            let per_view_frame_bg_and_buf = mv_ctx
                .backend
                .frame_resources
                .per_view_frame_or_create(occlusion_view, device, view_viewport, view_stereo)
                .map(|state| {
                    (
                        state.frame_bind_group.clone(),
                        state.frame_uniform_buffer.clone(),
                    )
                });

            let cmd = self.encode_per_view_to_cmd(
                &mut mv_ctx,
                view,
                view_idx,
                &mut transient_by_key,
                per_view_frame_bg_and_buf,
                &upload_batch,
            )?;
            per_view_cmds.push(cmd);
            per_view_occlusion_info.push((occlusion_view, host_camera));
        }

        // ── Single submit ────────────────────────────────────────────────────────────────────
        {
            let target_is_swapchain = views
                .iter()
                .any(|v| matches!(v.target, FrameViewTarget::Swapchain));
            let queue_ref: &wgpu::Queue = queue_arc.as_ref();

            // Debug HUD overlay encodes into the last view's encoder (swapchain path).
            // For simplicity with single-submit, we add a fresh encoder for the HUD.
            let hud_cmd = if target_is_swapchain {
                let Some(bb) = backbuffer_view_holder.as_ref() else {
                    return Err(GraphExecuteError::MissingSwapchainView);
                };
                let viewport_px = mv_ctx.gpu.surface_extent_px();
                let mut hud_encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("render-graph-hud"),
                    });
                if let Err(e) = mv_ctx.backend.encode_debug_hud_overlay(
                    device,
                    queue_ref,
                    &mut hud_encoder,
                    bb,
                    viewport_px,
                ) {
                    logger::warn!("debug HUD overlay: {e}");
                }
                Some(hud_encoder.finish())
            } else {
                None
            };

            // Drain all per-view and frame-global deferred writes onto the main thread before
            // submit so every command buffer sees a coherent queue state.
            upload_batch.drain_and_flush(queue_ref);

            let all_cmds: Vec<wgpu::CommandBuffer> = frame_global_cmd
                .into_iter()
                .chain(per_view_cmds)
                .chain(hud_cmd)
                .collect();

            // Hand the swapchain texture (if any) to the driver thread so `queue.submit` and
            // `SurfaceTexture::present` run off the main thread. The scope still drops cleanly
            // below — with the texture taken, its `Drop` is a no-op.
            let surface_tex = if target_is_swapchain {
                swapchain_scope.take_surface_texture()
            } else {
                None
            };
            let _ = queue_ref; // retained above for the HUD encoder; submit path now uses the driver
            mv_ctx.gpu.submit_frame_batch(all_cmds, surface_tex, None);
            // `submit_frame_batch` only enqueues on the driver thread. Pass `post_submit` hooks
            // (notably Hi-Z build → [`crate::backend::OcclusionSystem::hi_z_on_frame_submitted_for_view`])
            // call [`crate::render_graph::occlusion::HiZGpuState::on_frame_submitted`], which
            // `map_async`s readback staging. wgpu forbids submitting copy commands that target a
            // buffer while it is mapped, so the real [`wgpu::Queue::submit`] for this frame must
            // complete before those hooks run. [`crate::gpu::GpuContext::flush_driver`] drains the driver queue
            // through this batch (and swapchain present when applicable), which also prevents the
            // next frame's `get_current_texture` from racing a not-yet-presented surface image.
            mv_ctx.gpu.flush_driver();
        }

        // ── Post-submit hooks ────────────────────────────────────────────────────────────────
        let pv_post: Vec<usize> = self.schedule.per_view_steps().map(|s| s.pass_idx).collect();
        let fg_post: Vec<usize> = self
            .schedule
            .frame_global_steps()
            .map(|s| s.pass_idx)
            .collect();

        // Frame-global post-submit (uses first view's occlusion slot).
        if let Some((first_occlusion, first_hc)) = per_view_occlusion_info.first().copied() {
            let mut post_ctx = PostSubmitContext {
                device,
                occlusion: &mut mv_ctx.backend.occlusion,
                occlusion_view: first_occlusion,
                host_camera: first_hc,
            };
            for &pass_idx in &fg_post {
                self.passes[pass_idx]
                    .post_submit(&mut post_ctx)
                    .map_err(GraphExecuteError::Pass)?;
            }
        }

        // Per-view post-submit.
        for (view, (occlusion_view, host_camera)) in
            views.iter().zip(per_view_occlusion_info.iter())
        {
            let _ = view;
            let mut post_ctx = PostSubmitContext {
                device,
                occlusion: &mut mv_ctx.backend.occlusion,
                occlusion_view: *occlusion_view,
                host_camera: *host_camera,
            };
            for &pass_idx in &pv_post {
                self.passes[pass_idx]
                    .post_submit(&mut post_ctx)
                    .map_err(GraphExecuteError::Pass)?;
            }
        }

        // ── Transient cleanup ────────────────────────────────────────────────────────────────
        {
            let pool = mv_ctx.backend.transient_pool_mut();
            for (_, resources) in transient_by_key {
                resources.release_to_pool(pool);
            }
            {
                profiling::scope!("render::transient_gc");
                pool.gc_tick(120);
            }
        }

        Ok(())
    }

    /// Walks every view's prefetched draw list and pre-warms the material pipeline cache for
    /// each unique batch key. After this call the per-view recording loop can find every needed
    /// pipeline via cached lookup, removing the lazy `&mut self` cache-miss build path from the
    /// critical record path.
    ///
    /// Pre-warming uses [`crate::materials::MaterialPipelineDesc`] derived from each view's
    /// surface format, depth format, sample count, and multiview mask so cache keys match the
    /// keys the record path will request. Views without prefetched draws (lazy-collect path)
    /// are skipped — they will fall back to lazy cache build during recording as before.
    fn pre_warm_pipeline_cache_for_views(
        mv_ctx: &mut MultiViewExecutionContext<'_>,
        views: &[FrameView<'_>],
    ) {
        use crate::materials::MaterialPipelineDesc;
        use std::num::NonZeroU32;
        profiling::scope!("graph::pre_warm_pipelines");
        let Some(reg) = mv_ctx.backend.materials.material_registry_mut() else {
            return;
        };
        for view in views.iter() {
            let Some(collection) = view.prefetched_world_mesh_draws.as_ref() else {
                continue;
            };
            if collection.items.is_empty() {
                continue;
            }
            let host_camera = view.host_camera;
            let (viewport, multiview_stereo) = match &view.target {
                FrameViewTarget::ExternalMultiview(ext) => {
                    let stereo = host_camera.vr_active && host_camera.stereo_views.is_some();
                    (ext.extent_px, stereo)
                }
                FrameViewTarget::OffscreenRt(ext) => (ext.extent_px, false),
                FrameViewTarget::Swapchain => (mv_ctx.gpu.surface_extent_px(), false),
            };
            let _ = viewport;
            let surface_format = match &view.target {
                FrameViewTarget::ExternalMultiview(ext) => ext.surface_format,
                FrameViewTarget::OffscreenRt(ext) => ext.color_format,
                FrameViewTarget::Swapchain => mv_ctx.gpu.config_format(),
            };
            let depth_stencil_format = match &view.target {
                FrameViewTarget::ExternalMultiview(ext) => ext.depth_texture.format(),
                FrameViewTarget::OffscreenRt(ext) => ext.depth_texture.format(),
                FrameViewTarget::Swapchain => {
                    let Ok((depth_tex, _)) = mv_ctx.gpu.ensure_depth_target() else {
                        continue;
                    };
                    depth_tex.format()
                }
            };
            let sample_count = match &view.target {
                FrameViewTarget::ExternalMultiview(_) => {
                    mv_ctx.gpu.swapchain_msaa_effective_stereo().max(1)
                }
                FrameViewTarget::OffscreenRt(_) => 1,
                FrameViewTarget::Swapchain => mv_ctx.gpu.swapchain_msaa_effective().max(1),
            };
            let use_multiview = multiview_stereo
                && host_camera.vr_active
                && host_camera.stereo_view_proj.is_some()
                && mv_ctx.gpu_limits.supports_multiview;
            let pass_desc = MaterialPipelineDesc {
                surface_format,
                depth_stencil_format: Some(depth_stencil_format),
                sample_count,
                multiview_mask: if use_multiview {
                    NonZeroU32::new(3)
                } else {
                    None
                },
            };
            let shader_perm = if use_multiview {
                crate::pipelines::SHADER_PERM_MULTIVIEW_STEREO
            } else {
                crate::pipelines::ShaderPermutation(0)
            };

            // Walk unique (shader_asset_id, blend_mode, render_state) tuples to avoid duplicate
            // cache calls for draws that share the same batch key.
            let mut seen: std::collections::HashSet<(
                i32,
                crate::materials::MaterialBlendMode,
                crate::materials::MaterialRenderState,
            )> = std::collections::HashSet::new();
            for item in &collection.items {
                let key = (
                    item.batch_key.shader_asset_id,
                    item.batch_key.blend_mode,
                    item.batch_key.render_state,
                );
                if !seen.insert(key) {
                    continue;
                }
                let _ = reg.pipeline_for_shader_asset(
                    item.batch_key.shader_asset_id,
                    &pass_desc,
                    shader_perm,
                    item.batch_key.blend_mode,
                    item.batch_key.render_state,
                );
            }
        }
    }

    /// Eagerly allocates per-view frame state ([`crate::backend::FrameResourceManager::per_view_frame_or_create`])
    /// and per-view per-draw resources ([`crate::backend::FrameResourceManager::per_view_per_draw_or_create`])
    /// for every view in `views` before per-view recording begins.
    ///
    /// Hoists the lazy `&mut backend.frame_resources.*_or_create` calls out of the per-view
    /// recording loop so that loop can later borrow `backend` shared across rayon workers
    /// without colliding on the per-view resource maps (`per_view_frame`, `per_view_draw`).
    /// Also primes a freshly added secondary RT camera so its first frame does not pay the
    /// cluster-buffer / frame-uniform-buffer allocation cost mid-recording.
    fn pre_warm_per_view_resources_for_views(
        mv_ctx: &mut MultiViewExecutionContext<'_>,
        views: &[FrameView<'_>],
    ) -> Result<(), GraphExecuteError> {
        profiling::scope!("graph::pre_warm_per_view");
        for view in views.iter() {
            let occlusion_view = view.occlusion_view_id();
            let host_camera = view.host_camera;
            let (viewport, stereo) = match &view.target {
                FrameViewTarget::ExternalMultiview(ext) => {
                    let stereo = host_camera.vr_active && host_camera.stereo_views.is_some();
                    (ext.extent_px, stereo)
                }
                FrameViewTarget::OffscreenRt(ext) => (ext.extent_px, false),
                FrameViewTarget::Swapchain => (mv_ctx.gpu.surface_extent_px(), false),
            };
            let _ = mv_ctx.backend.frame_resources.per_view_frame_or_create(
                occlusion_view,
                mv_ctx.device,
                viewport,
                stereo,
            );
            let _ = mv_ctx
                .backend
                .frame_resources
                .per_view_per_draw_or_create(occlusion_view, mv_ctx.device);
        }
        Ok(())
    }

    /// Pre-resolves transient textures and buffers for every view's [`GraphResolveKey`].
    ///
    /// Hoists the transient-pool allocation out of the per-view record loop so that the loop
    /// itself no longer calls `backend.transient_pool_mut()`. This is a prerequisite for parallel
    /// per-view recording (Milestone E): concurrent workers cannot share `&mut` access to the
    /// pool, but they can share `&` access to the resulting `transient_by_key` map.
    ///
    /// Imported textures/buffers still resolve per-view inside the record loop because their
    /// bindings (backbuffer, per-view cluster refs) differ across views that share a key.
    fn pre_resolve_transients_for_views(
        &self,
        mv_ctx: &mut MultiViewExecutionContext<'_>,
        views: &mut [FrameView<'_>],
        transient_by_key: &mut HashMap<GraphResolveKey, GraphResolvedResources>,
    ) -> Result<(), GraphExecuteError> {
        profiling::scope!("render::pre_resolve_transients");
        for view in views.iter() {
            let resolved = Self::resolve_view_from_target(
                &view.target,
                mv_ctx.gpu,
                mv_ctx.backbuffer_view_holder,
            )?;
            let key = GraphResolveKey::from_resolved(&resolved);
            if let Entry::Vacant(v) = transient_by_key.entry(key) {
                let mut resources = GraphResolvedResources::with_capacity(
                    self.transient_textures.len(),
                    self.transient_buffers.len(),
                    self.imported_textures.len(),
                    self.imported_buffers.len(),
                );
                let alloc_viewport = helpers::clamp_viewport_for_transient_alloc(
                    resolved.viewport_px,
                    mv_ctx.gpu_limits.max_texture_dimension_2d(),
                );
                let scene_color_format = mv_ctx.backend.scene_color_format_wgpu();
                self.resolve_transient_textures(
                    mv_ctx.device,
                    mv_ctx.backend.transient_pool_mut(),
                    TransientTextureResolveSurfaceParams {
                        viewport_px: alloc_viewport,
                        surface_format: resolved.surface_format,
                        depth_stencil_format: resolved.depth_texture.format(),
                        scene_color_format,
                        sample_count: resolved.sample_count,
                        multiview_stereo: resolved.multiview_stereo,
                    },
                    &mut resources,
                )?;
                self.resolve_transient_buffers(
                    mv_ctx.device,
                    mv_ctx.backend.transient_pool_mut(),
                    alloc_viewport,
                    &mut resources,
                )?;
                v.insert(resources);
            }
        }
        Ok(())
    }

    /// Encodes one per-view pass into a command buffer and returns it without submitting.
    ///
    /// The caller is responsible for submitting the returned buffer (with all other per-view
    /// buffers) in a single [`wgpu::Queue::submit`] call after all per-view encoding is done.
    ///
    /// `per_view_frame_bg_and_buf` is the per-view `@group(0)` bind group + uniform buffer.
    ///
    /// Takes `&self` so per-view recording is structurally compatible with rayon fan-out at the
    /// [`CompiledRenderGraph`] layer; the remaining serialization point is the `&mut backend`
    /// borrow on [`MultiViewExecutionContext`] (see [`super::super::record_parallel`] for the
    /// remaining gating around shared mutable system handles).
    fn encode_per_view_to_cmd(
        &self,
        mv_ctx: &mut MultiViewExecutionContext<'_>,
        view: &mut FrameView<'_>,
        view_idx: usize,
        transient_by_key: &mut HashMap<GraphResolveKey, GraphResolvedResources>,
        per_view_frame_bg_and_buf: Option<(std::sync::Arc<wgpu::BindGroup>, wgpu::Buffer)>,
        upload_batch: &super::super::frame_upload_batch::FrameUploadBatch,
    ) -> Result<wgpu::CommandBuffer, GraphExecuteError> {
        profiling::scope!("graph::per_view");
        let MultiViewExecutionContext {
            gpu,
            scene,
            backend,
            device,
            gpu_limits,
            queue_arc,
            backbuffer_view_holder,
        } = mv_ctx;

        let prefetched = view.prefetched_world_mesh_draws.take();
        let draw_filter = view.draw_filter.clone();
        let host_camera = view.host_camera;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render-graph-per-view"),
        });
        let gpu_query = gpu
            .gpu_profiler_mut()
            .map(|p| p.begin_query("graph::per_view", &mut encoder));
        let mut pass_profiler = gpu.take_gpu_profiler();

        let resolved = Self::resolve_view_from_target(&view.target, gpu, backbuffer_view_holder)?;
        let key = GraphResolveKey::from_resolved(&resolved);
        // Transients were pre-resolved in `pre_resolve_transients_for_views` before the per-view
        // loop began, so a missing entry here is a bug.
        let resolved_resources = transient_by_key.get_mut(&key).ok_or_else(|| {
            logger::warn!("pre-resolve: missing transient resources for view key {key:?}");
            GraphExecuteError::MissingTransientResources
        })?;
        self.resolve_imported_textures(&resolved, resolved_resources);
        self.resolve_imported_buffers(&backend.frame_resources, &resolved, resolved_resources);
        let graph_resources: &GraphResolvedResources = &*resolved_resources;

        {
            let mut frame_params = helpers::frame_render_params_from_resolved(
                scene,
                backend,
                &resolved,
                host_camera,
                draw_filter,
            );
            // Per-view blackboard: seed with prefetched draws, ring plan, and MSAA views.
            let mut view_blackboard = Blackboard::new();

            // Resolve and insert MSAA views (replaces the removed FrameRenderParams MSAA fields).
            if let Some(msaa_views) = helpers::resolve_forward_msaa_views_from_graph_resources(
                &frame_params,
                Some(graph_resources),
                self.main_graph_msaa_transient_handles,
            ) {
                view_blackboard.insert::<MsaaViewsSlot>(msaa_views);
            }

            if let Some(draws) = prefetched {
                view_blackboard.insert::<PrefetchedWorldMeshDrawsSlot>(draws);
            }
            // Seed per-view frame plan so the prepare pass can write frame uniforms to the
            // correct per-view buffer and bind the right @group(0) bind group.
            if let Some((frame_bg, frame_buf)) = per_view_frame_bg_and_buf.clone() {
                view_blackboard.insert::<PerViewFramePlanSlot>(PerViewFramePlan {
                    frame_bind_group: frame_bg,
                    frame_uniform_buffer: frame_buf,
                    view_idx,
                });
            }

            // Collect indices from the single FrameSchedule source of truth.
            let per_view_indices: Vec<usize> =
                self.schedule.per_view_steps().map(|s| s.pass_idx).collect();

            for &pass_idx in &per_view_indices {
                let pass_name = self.passes[pass_idx].name().to_string();
                profiling::scope!("graph::pass", pass_name.as_str());

                // Open the GPU profiler query before calling execute_pass_node so we can
                // avoid capturing `encoder` in a closure while also passing it mutably.
                let pass_query = pass_profiler
                    .as_mut()
                    .map(|p| p.begin_query(pass_name.as_str(), &mut encoder));

                self.execute_pass_node(
                    pass_idx,
                    &resolved,
                    graph_resources,
                    &mut frame_params,
                    &mut view_blackboard,
                    &mut encoder,
                    device,
                    gpu_limits,
                    queue_arc,
                    upload_batch,
                )?;

                if let Some(q) = pass_query {
                    if let Some(p) = pass_profiler.as_mut() {
                        p.end_query(&mut encoder, q);
                    }
                }
            }
        }

        gpu.restore_gpu_profiler(pass_profiler);
        if let Some(query) = gpu_query {
            if let Some(prof) = gpu.gpu_profiler_mut() {
                prof.end_query(&mut encoder, query);
                prof.resolve_queries(&mut encoder);
            }
        }
        // Return the encoded command buffer WITHOUT submitting; the caller handles single submit.
        Ok(encoder.finish())
    }

    /// Encodes [`super::super::pass::PassPhase::FrameGlobal`] passes into a command buffer.
    ///
    /// Returns `None` when there are no frame-global passes (nothing to submit for this phase).
    /// The caller is responsible for including the returned buffer in the single-submit batch.
    fn encode_frame_global_passes(
        &self,
        mv_ctx: &mut MultiViewExecutionContext<'_>,
        views: &[FrameView<'_>],
        transient_by_key: &mut HashMap<GraphResolveKey, GraphResolvedResources>,
        upload_batch: &super::super::frame_upload_batch::FrameUploadBatch,
    ) -> Result<Option<wgpu::CommandBuffer>, GraphExecuteError> {
        profiling::scope!("graph::frame_global");
        let MultiViewExecutionContext {
            gpu,
            scene,
            backend,
            device,
            gpu_limits,
            queue_arc,
            backbuffer_view_holder,
        } = mv_ctx;

        if self.schedule.frame_global_steps().next().is_none() {
            return Ok(None);
        }
        let first = views.first().ok_or(GraphExecuteError::NoViewsInBatch)?;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render-graph-frame-global"),
        });
        let gpu_query = gpu
            .gpu_profiler_mut()
            .map(|p| p.begin_query("graph::frame_global", &mut encoder));
        let mut pass_profiler = gpu.take_gpu_profiler();

        {
            let resolved =
                Self::resolve_view_from_target(&first.target, gpu, backbuffer_view_holder)?;
            let key = GraphResolveKey::from_resolved(&resolved);
            let resolved_resources = match transient_by_key.entry(key) {
                Entry::Vacant(v) => {
                    profiling::scope!("render::transient_resolve");
                    let mut resources = GraphResolvedResources::with_capacity(
                        self.transient_textures.len(),
                        self.transient_buffers.len(),
                        self.imported_textures.len(),
                        self.imported_buffers.len(),
                    );
                    let alloc_viewport = helpers::clamp_viewport_for_transient_alloc(
                        resolved.viewport_px,
                        gpu_limits.max_texture_dimension_2d(),
                    );
                    let scene_color_format = backend.scene_color_format_wgpu();
                    self.resolve_transient_textures(
                        device,
                        backend.transient_pool_mut(),
                        TransientTextureResolveSurfaceParams {
                            viewport_px: alloc_viewport,
                            surface_format: resolved.surface_format,
                            depth_stencil_format: resolved.depth_texture.format(),
                            scene_color_format,
                            sample_count: resolved.sample_count,
                            multiview_stereo: resolved.multiview_stereo,
                        },
                        &mut resources,
                    )?;
                    self.resolve_transient_buffers(
                        device,
                        backend.transient_pool_mut(),
                        alloc_viewport,
                        &mut resources,
                    )?;
                    v.insert(resources)
                }
                Entry::Occupied(o) => o.into_mut(),
            };
            self.resolve_imported_textures(&resolved, resolved_resources);
            self.resolve_imported_buffers(&backend.frame_resources, &resolved, resolved_resources);
            let graph_resources: &GraphResolvedResources = &*resolved_resources;

            {
                let mut frame_params = helpers::frame_render_params_from_resolved(
                    scene,
                    backend,
                    &resolved,
                    first.host_camera,
                    first.draw_filter.clone(),
                );
                // Frame-global blackboard (one per tick).
                let mut frame_blackboard = Blackboard::new();
                // MSAA views are per-view, not frame-global; seed in per-view blackboard only.
                // Frame-global passes (e.g. mesh deform) don't need MSAA views.

                // Collect from FrameSchedule (single source of truth).
                let fg_indices: Vec<usize> = self
                    .schedule
                    .frame_global_steps()
                    .map(|s| s.pass_idx)
                    .collect();

                for &pass_idx in &fg_indices {
                    let pass_name = self.passes[pass_idx].name().to_string();
                    profiling::scope!("graph::pass", pass_name.as_str());

                    let pass_query = pass_profiler
                        .as_mut()
                        .map(|p| p.begin_query(pass_name.as_str(), &mut encoder));

                    self.execute_pass_node(
                        pass_idx,
                        &resolved,
                        graph_resources,
                        &mut frame_params,
                        &mut frame_blackboard,
                        &mut encoder,
                        device,
                        gpu_limits,
                        queue_arc,
                        upload_batch,
                    )?;

                    if let Some(q) = pass_query {
                        if let Some(p) = pass_profiler.as_mut() {
                            p.end_query(&mut encoder, q);
                        }
                    }
                }
            }
        }

        gpu.restore_gpu_profiler(pass_profiler);
        if let Some(query) = gpu_query {
            if let Some(prof) = gpu.gpu_profiler_mut() {
                prof.end_query(&mut encoder, query);
                prof.resolve_queries(&mut encoder);
            }
        }
        // Return the encoded command buffer WITHOUT submitting; the caller handles single submit.
        Ok(Some(encoder.finish()))
    }

    /// Dispatches one pass node to its correct execution path.
    ///
    /// - `Raster` → opens `wgpu::RenderPass` from template, calls `record_raster`.
    /// - `Compute` → calls `record_compute` with raw encoder.
    /// - `Copy` → calls `record_copy` with raw encoder.
    /// - `Callback` → calls `run_callback` (no encoder).
    ///
    /// Takes `&self` so per-view recording can be hoisted onto rayon workers without serialising
    /// on the [`CompiledRenderGraph`] handle. All pass `record_*` methods already require only
    /// `&self`, so the dispatch loop is structurally Send/Sync-safe at this layer.
    #[allow(clippy::too_many_arguments)]
    fn execute_pass_node<'a>(
        &self,
        pass_idx: usize,
        resolved: &'a ResolvedView<'a>,
        graph_resources: &'a GraphResolvedResources,
        frame_params: &mut crate::render_graph::frame_params::FrameRenderParams<'a>,
        blackboard: &mut Blackboard,
        // `encoder` intentionally uses no named lifetime so each call's borrow
        // ends at the call boundary, avoiding cross-iteration borrow conflicts.
        encoder: &mut wgpu::CommandEncoder,
        device: &'a wgpu::Device,
        gpu_limits: &'a crate::gpu::GpuLimits,
        queue_arc: &'a std::sync::Arc<wgpu::Queue>,
        upload_batch: &super::super::frame_upload_batch::FrameUploadBatch,
    ) -> Result<(), GraphExecuteError> {
        let kind = self.passes[pass_idx].kind();
        match kind {
            PassKind::Raster => {
                let template = helpers::pass_info_raster_template(&self.pass_info, pass_idx)?;
                let mut ctx = RasterPassCtx {
                    device,
                    gpu_limits,
                    queue: queue_arc,
                    backbuffer: resolved.backbuffer,
                    depth_view: Some(resolved.depth_view),
                    frame: Some(frame_params),
                    frame_shared: None,
                    frame_view: None,
                    upload_batch,
                    graph_resources: Some(graph_resources),
                    blackboard,
                };
                helpers::execute_graph_raster_pass_node(
                    &self.passes[pass_idx],
                    &template,
                    graph_resources,
                    encoder,
                    &mut ctx,
                )?;
            }
            PassKind::Compute => {
                // encoder is moved into ComputePassCtx; pass uses ctx.encoder.
                let mut ctx = ComputePassCtx {
                    device,
                    gpu_limits,
                    queue: queue_arc,
                    encoder,
                    depth_view: Some(resolved.depth_view),
                    frame: Some(frame_params),
                    frame_shared: None,
                    frame_view: None,
                    upload_batch,
                    graph_resources: Some(graph_resources),
                    blackboard,
                };
                self.passes[pass_idx]
                    .record_compute(&mut ctx)
                    .map_err(GraphExecuteError::Pass)?;
            }
            PassKind::Copy => {
                let mut ctx = ComputePassCtx {
                    device,
                    gpu_limits,
                    queue: queue_arc,
                    encoder,
                    depth_view: Some(resolved.depth_view),
                    frame: Some(frame_params),
                    frame_shared: None,
                    frame_view: None,
                    upload_batch,
                    graph_resources: Some(graph_resources),
                    blackboard,
                };
                self.passes[pass_idx]
                    .record_copy(&mut ctx)
                    .map_err(GraphExecuteError::Pass)?;
            }
            PassKind::Callback => {
                let mut ctx = CallbackCtx {
                    device,
                    gpu_limits,
                    queue: queue_arc,
                    frame: Some(frame_params),
                    frame_shared: None,
                    frame_view: None,
                    upload_batch,
                    graph_resources: Some(graph_resources),
                    blackboard,
                };
                self.passes[pass_idx]
                    .run_callback(&mut ctx)
                    .map_err(GraphExecuteError::Pass)?;
            }
        }
        Ok(())
    }

    #[allow(clippy::map_entry)]
    fn resolve_transient_textures(
        &self,
        device: &wgpu::Device,
        pool: &mut TransientPool,
        surface: TransientTextureResolveSurfaceParams,
        resources: &mut GraphResolvedResources,
    ) -> Result<(), GraphExecuteError> {
        let mut physical_slots: HashMap<usize, ResolvedGraphTexture> = HashMap::new();
        for (idx, compiled) in self.transient_textures.iter().enumerate() {
            if compiled.lifetime.is_none() || compiled.physical_slot == usize::MAX {
                continue;
            }
            if !physical_slots.contains_key(&compiled.physical_slot) {
                let array_layers = compiled.desc.array_layers.resolve(surface.multiview_stereo);
                let key = TextureKey {
                    format: compiled.desc.format.resolve(
                        surface.surface_format,
                        surface.depth_stencil_format,
                        surface.scene_color_format,
                    ),
                    extent: helpers::resolve_transient_extent(
                        compiled.desc.extent,
                        surface.viewport_px,
                        array_layers,
                    ),
                    mip_levels: compiled.desc.mip_levels,
                    sample_count: compiled.desc.sample_count.resolve(surface.sample_count),
                    dimension: compiled.desc.dimension,
                    array_layers,
                    usage_bits: compiled.usage.bits() as u64,
                };
                let lease = pool.acquire_texture_resource(
                    device,
                    key,
                    compiled.desc.label,
                    compiled.usage,
                )?;
                let layer_views = helpers::create_transient_layer_views(&lease.texture, key);
                physical_slots.insert(
                    compiled.physical_slot,
                    ResolvedGraphTexture {
                        pool_id: lease.pool_id,
                        physical_slot: compiled.physical_slot,
                        texture: lease.texture,
                        view: lease.view,
                        layer_views,
                    },
                );
            }
            let resolved = physical_slots[&compiled.physical_slot].clone();
            resources.set_transient_texture(TextureHandle(idx as u32), resolved);
        }
        Ok(())
    }

    #[allow(clippy::map_entry)]
    fn resolve_transient_buffers(
        &self,
        device: &wgpu::Device,
        pool: &mut TransientPool,
        viewport_px: (u32, u32),
        resources: &mut GraphResolvedResources,
    ) -> Result<(), GraphExecuteError> {
        let mut physical_slots: HashMap<usize, ResolvedGraphBuffer> = HashMap::new();
        for (idx, compiled) in self.transient_buffers.iter().enumerate() {
            if compiled.lifetime.is_none() || compiled.physical_slot == usize::MAX {
                continue;
            }
            if !physical_slots.contains_key(&compiled.physical_slot) {
                let key = BufferKey {
                    size_policy: compiled.desc.size_policy,
                    usage_bits: compiled.usage.bits() as u64,
                };
                let size = helpers::resolve_buffer_size(compiled.desc.size_policy, viewport_px);
                let lease = pool.acquire_buffer_resource(
                    device,
                    key,
                    compiled.desc.label,
                    compiled.usage,
                    size,
                )?;
                physical_slots.insert(
                    compiled.physical_slot,
                    ResolvedGraphBuffer {
                        pool_id: lease.pool_id,
                        physical_slot: compiled.physical_slot,
                        buffer: lease.buffer,
                        size: lease.size,
                    },
                );
            }
            let resolved = physical_slots[&compiled.physical_slot].clone();
            resources
                .set_transient_buffer(super::super::resources::BufferHandle(idx as u32), resolved);
        }
        Ok(())
    }

    fn resolve_imported_textures(
        &self,
        resolved: &ResolvedView<'_>,
        resources: &mut GraphResolvedResources,
    ) {
        for (idx, import) in self.imported_textures.iter().enumerate() {
            let view = match &import.source {
                ImportSource::FrameTarget(FrameTargetRole::ColorAttachment) => {
                    resolved.backbuffer.cloned()
                }
                ImportSource::FrameTarget(FrameTargetRole::DepthAttachment) => {
                    Some(resolved.depth_view.clone())
                }
                ImportSource::External | ImportSource::PingPong(_) => None,
            };
            if let Some(view) = view {
                resources.set_imported_texture(
                    ImportedTextureHandle(idx as u32),
                    ResolvedImportedTexture { view },
                );
            }
        }
    }

    fn resolve_imported_buffers(
        &self,
        frame_resources: &crate::backend::FrameResourceManager,
        resolved: &ResolvedView<'_>,
        resources: &mut GraphResolvedResources,
    ) {
        let frame_gpu = frame_resources.frame_gpu();
        // Use per-view cluster refs so each view resolves its own independent cluster buffers.
        let cluster_refs = frame_resources
            .per_view_frame(resolved.occlusion_view)
            .and_then(|state| state.cluster_buffer_refs());
        for (idx, import) in self.imported_buffers.iter().enumerate() {
            let buffer = match &import.source {
                BufferImportSource::BackendFrameResource(BackendFrameBufferKind::Lights) => {
                    frame_gpu.map(|fgpu| fgpu.lights_buffer.clone())
                }
                BufferImportSource::BackendFrameResource(BackendFrameBufferKind::FrameUniforms) => {
                    frame_gpu.map(|fgpu| fgpu.frame_uniform.clone())
                }
                BufferImportSource::BackendFrameResource(
                    BackendFrameBufferKind::ClusterLightCounts,
                ) => cluster_refs
                    .as_ref()
                    .map(|refs| refs.cluster_light_counts.clone()),
                BufferImportSource::BackendFrameResource(
                    BackendFrameBufferKind::ClusterLightIndices,
                ) => cluster_refs
                    .as_ref()
                    .map(|refs| refs.cluster_light_indices.clone()),
                BufferImportSource::BackendFrameResource(BackendFrameBufferKind::PerDrawSlab) => {
                    frame_resources
                        .per_view_per_draw(resolved.occlusion_view)
                        .map(|per_draw| per_draw.per_draw_storage.clone())
                }
                BufferImportSource::External | BufferImportSource::PingPong(_) => None,
            };
            if let Some(buffer) = buffer {
                resources.set_imported_buffer(
                    ImportedBufferHandle(idx as u32),
                    ResolvedImportedBuffer { buffer },
                );
            }
        }
    }

    fn resolve_view_from_target<'a>(
        target: &'a FrameViewTarget<'a>,
        gpu: &'a mut GpuContext,
        backbuffer_view_holder: &'a Option<wgpu::TextureView>,
    ) -> Result<ResolvedView<'a>, GraphExecuteError> {
        match target {
            FrameViewTarget::Swapchain => {
                let surface_format = gpu.config_format();
                let viewport_px = gpu.surface_extent_px();
                let bb = backbuffer_view_holder
                    .as_ref()
                    .map(|v| v as &wgpu::TextureView);
                let Some(bb_ref) = bb else {
                    return Err(GraphExecuteError::MissingSwapchainView);
                };
                let sample_count = gpu.swapchain_msaa_effective().max(1);
                let (depth_tex, depth_view) = gpu
                    .ensure_depth_target()
                    .map_err(|_| GraphExecuteError::DepthTarget)?;

                Ok(ResolvedView {
                    depth_texture: depth_tex,
                    depth_view,
                    backbuffer: Some(bb_ref),
                    surface_format,
                    viewport_px,
                    multiview_stereo: false,
                    offscreen_write_render_texture_asset_id: None,
                    occlusion_view: OcclusionViewId::Main,
                    sample_count,
                })
            }
            FrameViewTarget::ExternalMultiview(ext) => {
                let sample_count = gpu.swapchain_msaa_effective_stereo().max(1);
                Ok(ResolvedView {
                    depth_texture: ext.depth_texture,
                    depth_view: ext.depth_view,
                    backbuffer: Some(ext.color_view),
                    surface_format: ext.surface_format,
                    viewport_px: ext.extent_px,
                    multiview_stereo: true,
                    offscreen_write_render_texture_asset_id: None,
                    occlusion_view: OcclusionViewId::Main,
                    sample_count,
                })
            }
            FrameViewTarget::OffscreenRt(ext) => Ok(ResolvedView {
                depth_texture: ext.depth_texture,
                depth_view: ext.depth_view,
                backbuffer: Some(ext.color_view),
                surface_format: ext.color_format,
                viewport_px: ext.extent_px,
                multiview_stereo: false,
                offscreen_write_render_texture_asset_id: Some(ext.render_texture_asset_id),
                occlusion_view: OcclusionViewId::OffscreenRenderTexture(
                    ext.render_texture_asset_id,
                ),
                sample_count: 1,
            }),
        }
    }
}
