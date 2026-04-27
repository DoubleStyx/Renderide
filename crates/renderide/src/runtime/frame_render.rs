//! Unified per-tick render entry point: builds one `FrameView` list covering the HMD, secondary
//! render-texture cameras, and the main desktop view, then dispatches the compiled render graph
//! in a single submit.

use std::sync::Arc;

use rayon::prelude::*;

use crate::assets::material::MaterialDictionary;
use crate::backend::OcclusionSystem;
use crate::gpu::GpuContext;
use crate::materials::{MaterialPipelinePropertyIds, MaterialRouter, RasterPipelineKind};
use crate::pipelines::{ShaderPermutation, SHADER_PERM_MULTIVIEW_STEREO};
use crate::render_graph::{
    build_world_mesh_cull_proj_params, camera_state_enabled,
    collect_and_sort_world_mesh_draws_with_parallelism, draw_filter_from_camera_entry,
    host_camera_frame_for_render_texture, CameraTransformDrawFilter, DrawCollectionContext,
    ExternalFrameTargets, ExternalOffscreenTargets, FramePreparedRenderables, FrameView,
    FrameViewClear, FrameViewTarget, GraphExecuteError, HiZCullData, HiZTemporalState,
    HostCameraFrame, OcclusionViewId, OutputDepthMode, WorldMeshCullInput, WorldMeshCullProjParams,
    WorldMeshDrawCollectParallelism, WorldMeshDrawPlan,
};
use crate::scene::SceneCoordinator;

use super::RendererRuntime;

/// Which combination of views the compiled render graph records for one tick.
///
/// Encodes the three legal render-mode permutations as an enum so the illegal "desktop swapchain
/// plus OpenXR HMD" state cannot be represented.
pub(crate) enum FrameRenderMode<'a> {
    /// Non-VR path: main swapchain view plus any active secondary render-texture cameras.
    DesktopPlusSecondaries,
    /// VR path with a successfully acquired HMD swapchain; stereo multiview view plus secondaries.
    VrWithHmd(ExternalFrameTargets<'a>),
    /// VR path when the HMD swapchain acquire failed this tick; secondaries still render, the
    /// desktop mirror stays on its last frame.
    VrSecondariesOnly,
}

impl FrameRenderMode<'_> {
    /// `true` when this mode appends the main desktop swapchain view.
    fn includes_main_swapchain(&self) -> bool {
        matches!(self, FrameRenderMode::DesktopPlusSecondaries)
    }

    /// `true` when this mode prepends an HMD stereo multiview view.
    fn has_hmd(&self) -> bool {
        matches!(self, FrameRenderMode::VrWithHmd(_))
    }
}

/// Cheap-clone snapshot of [`crate::gpu::PrimaryOffscreenTargets`] used by the headless render path
/// to satisfy the borrow checker: clones are cheap (`wgpu::Texture` and `wgpu::TextureView` are
/// internally `Arc`-backed) and let the substitution borrow from a stack-local instead of a
/// long-lived `&mut gpu`. Without this split, holding `&gpu.primary_offscreen` while passing
/// `&mut gpu` to the next call is a borrow error.
struct HeadlessOffscreenSnapshot {
    /// Color attachment view for the substituted offscreen target.
    color_view: wgpu::TextureView,
    /// Backing depth texture for the substituted offscreen target.
    depth_texture: wgpu::Texture,
    /// Depth view over the substituted depth texture.
    depth_view: wgpu::TextureView,
    /// Pixel extent of the primary offscreen attachments.
    extent_px: (u32, u32),
    /// Color attachment format matching the primary offscreen target.
    color_format: wgpu::TextureFormat,
}

impl HeadlessOffscreenSnapshot {
    /// Lazily allocates the headless primary targets if needed and snapshots cheap clones of
    /// their handles. Returns [`None`] when `gpu` is windowed.
    fn from_gpu(gpu: &mut GpuContext) -> Option<Self> {
        let targets = gpu.primary_offscreen_targets()?;
        Some(Self {
            color_view: targets.color_view.clone(),
            depth_texture: targets.depth_texture.clone(),
            depth_view: targets.depth_view.clone(),
            extent_px: targets.extent_px,
            color_format: targets.color_format,
        })
    }

    /// Replaces every [`FrameViewTarget::Swapchain`] in `views` with an
    /// [`FrameViewTarget::OffscreenRt`] backed by this snapshot's owned handles. The borrowed
    /// references are valid for as long as `&self` outlives `views`, which is enforced by the
    /// `'a` lifetime.
    fn substitute_swapchain_views<'a>(&'a self, views: &mut [FrameView<'a>]) {
        for view in views.iter_mut() {
            if matches!(view.target, FrameViewTarget::Swapchain) {
                view.target = FrameViewTarget::OffscreenRt(ExternalOffscreenTargets {
                    render_texture_asset_id: -1,
                    color_view: &self.color_view,
                    depth_texture: &self.depth_texture,
                    depth_view: &self.depth_view,
                    extent_px: self.extent_px,
                    color_format: self.color_format,
                });
            }
        }
    }
}

/// Render-texture attachment handles owned by one prepared secondary view so the underlying
/// `Arc<TextureView>` / `Arc<Texture>` stay alive for the duration of the tick.
struct OffscreenRtHandles {
    /// Host render texture asset id writing this pass; used to suppress self-sampling.
    rt_id: i32,
    /// Color attachment view for this render texture.
    color_view: Arc<wgpu::TextureView>,
    /// Depth attachment backing texture.
    depth_texture: Arc<wgpu::Texture>,
    /// Depth attachment view.
    depth_view: Arc<wgpu::TextureView>,
    /// Color attachment format (must match pipeline targets).
    color_format: wgpu::TextureFormat,
}

/// Target-specific payload for a [`PreparedView`].
enum PreparedViewKind<'a> {
    /// HMD stereo multiview view; targets are external (pre-acquired by the XR driver).
    Hmd(ExternalFrameTargets<'a>),
    /// Secondary render-texture camera; owns the RT color/depth handles for the tick.
    SecondaryRt(OffscreenRtHandles),
    /// Main desktop swapchain view.
    MainSwapchain,
}

/// One CPU-prepared view ready to become a [`FrameView`] for the compiled render graph.
///
/// Built by [`RendererRuntime::collect_prepared_views`] for every active view in the tick —
/// HMD stereo multiview, secondary render-texture cameras, and the main desktop swapchain —
/// so the per-view draw and cull pipelines downstream do not need to branch on view kind.
struct PreparedView<'a> {
    /// Per-view camera parameters (clip planes, matrices, stereo, overrides).
    host_camera: HostCameraFrame,
    /// Optional selective/exclude filter; present for secondary cameras only.
    draw_filter: Option<CameraTransformDrawFilter>,
    /// Hi-Z / occlusion slot identity for this view.
    occlusion_view_id: OcclusionViewId,
    /// Attachment extent in pixels for this view.
    viewport_px: (u32, u32),
    /// Background clear/skybox behavior for this view.
    clear: FrameViewClear,
    /// Target-specific payload (HMD, secondary RT, main swapchain).
    kind: PreparedViewKind<'a>,
}

impl PreparedView<'_> {
    /// Builds the [`FrameViewTarget`] for this view, borrowing from `self` when target handles
    /// live on the prepared view (secondary RT path).
    fn target(&self) -> FrameViewTarget<'_> {
        match &self.kind {
            PreparedViewKind::Hmd(ext) => {
                FrameViewTarget::ExternalMultiview(ExternalFrameTargets {
                    color_view: ext.color_view,
                    depth_texture: ext.depth_texture,
                    depth_view: ext.depth_view,
                    extent_px: ext.extent_px,
                    surface_format: ext.surface_format,
                })
            }
            PreparedViewKind::SecondaryRt(handles) => {
                FrameViewTarget::OffscreenRt(ExternalOffscreenTargets {
                    render_texture_asset_id: handles.rt_id,
                    color_view: handles.color_view.as_ref(),
                    depth_texture: handles.depth_texture.as_ref(),
                    depth_view: handles.depth_view.as_ref(),
                    extent_px: self.viewport_px,
                    color_format: handles.color_format,
                })
            }
            PreparedViewKind::MainSwapchain => FrameViewTarget::Swapchain,
        }
    }

    /// Back-to-front sort origin for transparent draws.
    ///
    /// Preference order matches the world-mesh forward path: explicit camera world position
    /// (secondary RT cameras) → main-space eye position → head-output translation as a last-ditch
    /// fallback. Sorting from the render-space *root* instead of the eye produced visually wrong
    /// transparency ordering whenever the host enabled `override_view_position`.
    fn view_origin_world(&self) -> glam::Vec3 {
        self.host_camera
            .explicit_camera_world_position
            .or(self.host_camera.eye_world_position)
            .unwrap_or_else(|| self.host_camera.head_output_transform.col(3).truncate())
    }

    /// `true` when this prepared view records the OpenXR stereo multiview draw path.
    fn is_multiview_stereo_active(&self) -> bool {
        matches!(self.kind, PreparedViewKind::Hmd(_))
            && self.host_camera.vr_active
            && self.host_camera.stereo.is_some()
    }

    /// Shader permutation used by CPU draw collection and material metadata for this view.
    fn shader_permutation(&self) -> ShaderPermutation {
        if self.is_multiview_stereo_active() {
            SHADER_PERM_MULTIVIEW_STEREO
        } else {
            ShaderPermutation(0)
        }
    }

    /// Depth output layout used for Hi-Z and occlusion data sampled during CPU culling.
    fn output_depth_mode(&self) -> OutputDepthMode {
        OutputDepthMode::from_multiview_stereo(self.is_multiview_stereo_active())
    }
}

/// Frustum + Hi-Z cull inputs for one prepared view.
struct ViewCullSnapshot {
    /// Projection parameters matching the view's camera/viewport.
    proj: WorldMeshCullProjParams,
    /// CPU-side Hi-Z snapshot for this view's occlusion slot.
    hi_z: Option<HiZCullData>,
    /// Temporal Hi-Z state captured after the prior frame's depth pyramid author pass.
    hi_z_temporal: Option<HiZTemporalState>,
}

/// Immutable per-frame references shared by every per-view draw-collection call.
///
/// Grouped into one struct so the `par_iter().map(...)` closure in [`RendererRuntime::render_frame`]
/// can close over a single reference rather than shuttling seven individual bindings through the
/// rayon worker boundary.
struct FrameCollectionContext<'a> {
    /// Scene after cache flush — used for world-matrix lookups and cull evaluation.
    scene: &'a SceneCoordinator,
    /// Mesh GPU asset pool, queried for bounds and skinning metadata during draw collection.
    mesh_pool: &'a crate::resources::MeshPool,
    /// Property store backing `MaterialDictionary::new` plus pipeline lookup.
    property_store: &'a crate::assets::material::MaterialPropertyStore,
    /// Resolved raster pipeline selection for embedded materials.
    router: &'a MaterialRouter,
    /// Registry of renderer-side property ids used by the pipeline selector.
    pipeline_property_ids: MaterialPipelinePropertyIds,
    /// Mono/stereo/overlay render context applied this tick.
    render_context: crate::shared::RenderingContext,
    /// Persistent mono material batch cache, refreshed once at the start of [`RendererRuntime::render_frame`].
    material_cache: &'a crate::render_graph::FrameMaterialBatchCache,
    /// Dense per-frame walk of renderables pre-expanded once before per-view collection.
    prepared: &'a FramePreparedRenderables,
    /// Rayon parallelism tier for each view's inner walk.
    inner_parallelism: WorldMeshDrawCollectParallelism,
}

/// Collects and sorts world-mesh draws for every prepared view in parallel.
///
/// Returns one explicit [`WorldMeshDrawPlan`] per prepared view, preserving input order so the
/// compiled graph never has to infer whether draws were intentionally omitted or merely missing.
fn collect_view_draws(
    ctx: &FrameCollectionContext<'_>,
    prepared: &[PreparedView<'_>],
    cull_snapshots: &[Option<ViewCullSnapshot>],
) -> Vec<WorldMeshDrawPlan> {
    profiling::scope!("render::collect_view_draws");
    prepared
        .par_iter()
        .zip(cull_snapshots.par_iter())
        .map(|(prep, snap)| {
            let shader_perm = prep.shader_permutation();
            let material_cache =
                (shader_perm == ShaderPermutation(0)).then_some(ctx.material_cache);
            let dict = MaterialDictionary::new(ctx.property_store);
            let culling = snap.as_ref().map(|s| WorldMeshCullInput {
                proj: s.proj,
                host_camera: &prep.host_camera,
                hi_z: s.hi_z.clone(),
                hi_z_temporal: s.hi_z_temporal.clone(),
            });
            WorldMeshDrawPlan::Prefetched(collect_and_sort_world_mesh_draws_with_parallelism(
                &DrawCollectionContext {
                    scene: ctx.scene,
                    mesh_pool: ctx.mesh_pool,
                    material_dict: &dict,
                    material_router: ctx.router,
                    pipeline_property_ids: &ctx.pipeline_property_ids,
                    shader_perm,
                    render_context: ctx.render_context,
                    head_output_transform: prep.host_camera.head_output_transform,
                    view_origin_world: prep.view_origin_world(),
                    culling: culling.as_ref(),
                    transform_filter: prep.draw_filter.as_ref(),
                    material_cache,
                    prepared: Some(ctx.prepared),
                },
                ctx.inner_parallelism,
            ))
        })
        .collect()
}

/// Selects the per-view inner-walk parallelism tier for a tick based on how many views will
/// collect draws. Keeps rayon from oversubscribing when several views each spawn worker-level
/// parallelism.
fn select_inner_parallelism(prepared: &[PreparedView<'_>]) -> WorldMeshDrawCollectParallelism {
    if prepared.len() > 1 {
        WorldMeshDrawCollectParallelism::SerialInnerForNestedBatch
    } else {
        WorldMeshDrawCollectParallelism::Full
    }
}

/// Builds frustum + Hi-Z cull inputs for one prepared view.
///
/// Returns [`None`] when the view has explicitly suppressed temporal occlusion (selective
/// secondary cameras). Safe to call in parallel across views:
/// [`OcclusionSystem`] is `Sync` because its internal readback channel uses `crossbeam_channel`.
fn cull_snapshot_for_view(
    scene: &SceneCoordinator,
    occlusion: &OcclusionSystem,
    prep: &PreparedView<'_>,
) -> Option<ViewCullSnapshot> {
    if prep.host_camera.suppress_occlusion_temporal {
        return None;
    }
    let proj = build_world_mesh_cull_proj_params(scene, prep.viewport_px, &prep.host_camera);
    let depth_mode = prep.output_depth_mode();
    Some(ViewCullSnapshot {
        proj,
        hi_z: occlusion.hi_z_cull_data(depth_mode, prep.occlusion_view_id),
        hi_z_temporal: occlusion.hi_z_temporal_snapshot(prep.occlusion_view_id),
    })
}

impl RendererRuntime {
    /// Desktop entry point: renders the main swapchain view plus any active secondary render-texture
    /// cameras in a single submit. Used when OpenXR is not active.
    ///
    /// See [`Self::render_frame`] for the shared implementation that also powers the VR entry
    /// points on [`crate::xr::XrFrameRenderer`].
    pub fn render_desktop_frame(&mut self, gpu: &mut GpuContext) -> Result<(), GraphExecuteError> {
        self.render_frame(gpu, FrameRenderMode::DesktopPlusSecondaries)
    }

    /// Unified per-tick world render entry point.
    ///
    /// Builds a single prepared-view list (HMD first when present, secondary RTs in depth order,
    /// main swapchain last when requested) and dispatches the compiled render graph in one
    /// [`RenderBackend::execute_multi_view_frame`](crate::backend::RenderBackend::execute_multi_view_frame)
    /// call. Hi-Z readback has already been drained once at the top of the tick (see
    /// [`Self::drain_hi_z_readback`]), so the caller always skips the readback pass here.
    ///
    /// Callers should not invoke this directly; use [`Self::render_desktop_frame`] for desktop or
    /// the [`crate::xr::XrFrameRenderer`] trait methods for VR paths.
    ///
    /// In headless mode (`gpu.is_headless()`) the main `Swapchain` view is transparently
    /// substituted for an `OffscreenRt` view backed by [`GpuContext::primary_offscreen_targets`]
    /// so the render graph stack stays oblivious to output mode.
    pub(crate) fn render_frame(
        &mut self,
        gpu: &mut GpuContext,
        mode: FrameRenderMode<'_>,
    ) -> Result<(), GraphExecuteError> {
        profiling::scope!("render::render_frame");
        {
            profiling::scope!("render::prepare_lights_from_scene");
            self.backend
                .frame_resources
                .prepare_lights_from_scene(&self.scene);
        }
        self.sync_debug_hud_diagnostics_from_settings();
        self.setup_msaa_for_mode(gpu, &mode);

        let includes_main = mode.includes_main_swapchain();
        // Capture the swapchain extent before the per-view collection. The main desktop view's
        // CPU cull projection (`build_world_mesh_cull_proj_params`) runs against this extent
        // before the render graph dispatches, so passing a stale/zero value produces a degenerate
        // frustum and randomly culls scene objects.
        let swapchain_extent_px = gpu.surface_extent_px();
        let prepared = {
            profiling::scope!("render::collect_prepared_views");
            self.collect_prepared_views(mode, swapchain_extent_px)
        };
        if prepared.is_empty() {
            return Ok(());
        }

        let scene_ref: &SceneCoordinator = &self.scene;
        let fallback_router = MaterialRouter::new(RasterPipelineKind::Null);
        let render_context = scene_ref.active_main_render_context();
        // Direct field access enables the split-borrow against `material_batch_cache` below —
        // routing through `self.backend.material_property_store()` would borrow the whole
        // `RenderBackend` and block the subsequent `&mut material_batch_cache`.
        let property_store = self.backend.materials.material_property_store();
        let router_ref = self
            .backend
            .materials
            .material_registry()
            .map(|r| &r.router)
            .unwrap_or(&fallback_router);
        let pipeline_property_ids =
            MaterialPipelinePropertyIds::new(self.backend.materials.property_id_registry());
        let mesh_pool = &self.backend.asset_transfers.mesh_pool;
        let occlusion_ref: &OcclusionSystem = &self.backend.occlusion;
        let inner_parallelism = select_inner_parallelism(&prepared);

        // Refresh the persistent material batch cache **once** before any per-view collection.
        // Every per-view call reuses the same immutable cache via `material_cache`, removing the
        // N+1 dictionary/router resolution walks that used to dominate multi-view frame cost.
        {
            profiling::scope!("render::build_frame_material_cache");
            let dict = MaterialDictionary::new(property_store);
            self.backend.material_batch_cache.refresh_for_frame(
                scene_ref,
                &dict,
                router_ref,
                &pipeline_property_ids,
                ShaderPermutation(0),
            );
        }

        // Pre-expand the scene walk into a dense draw list once per frame. Every per-view
        // collection iterates this shared list instead of walking each active render space
        // independently. `render_context` is the same for every view in this tick.
        let frame_prepared = {
            profiling::scope!("render::build_frame_prepared_renderables");
            FramePreparedRenderables::build_for_frame(scene_ref, mesh_pool, render_context)
        };

        let collect_ctx = FrameCollectionContext {
            scene: scene_ref,
            mesh_pool,
            property_store,
            router: router_ref,
            pipeline_property_ids,
            render_context,
            material_cache: &self.backend.material_batch_cache,
            prepared: &frame_prepared,
            inner_parallelism,
        };

        let cull_snapshots: Vec<Option<ViewCullSnapshot>> = {
            profiling::scope!("render::gather_view_cull_snapshots");
            prepared
                .par_iter()
                .map(|prep| cull_snapshot_for_view(scene_ref, occlusion_ref, prep))
                .collect()
        };
        let view_draws = collect_view_draws(&collect_ctx, &prepared, &cull_snapshots);

        // Headless substitution: snapshot persistent offscreen handles BEFORE building views so
        // we can borrow from a local instead of a long-lived `&mut gpu` (which would conflict
        // with the `&mut gpu` we hand to `execute_multi_view_frame`).
        let headless_snapshot = {
            profiling::scope!("render::headless_snapshot");
            if includes_main && gpu.is_headless() {
                HeadlessOffscreenSnapshot::from_gpu(gpu)
            } else {
                None
            }
        };

        let mut views: Vec<FrameView<'_>> = prepared
            .iter()
            .zip(view_draws)
            .map(|(prep, draws)| FrameView {
                host_camera: prep.host_camera,
                target: prep.target(),
                draw_filter: prep.draw_filter.clone(),
                clear: prep.clear,
                world_mesh_draw_plan: draws,
            })
            .collect();

        if let Some(snapshot) = headless_snapshot.as_ref() {
            snapshot.substitute_swapchain_views(&mut views);
        }

        self.backend
            .execute_multi_view_frame(gpu, scene_ref, &mut views, true)
    }

    /// Applies the MSAA tier for the active mode and evicts transient textures keyed by stale
    /// sample counts on a tier change.
    fn setup_msaa_for_mode(&mut self, gpu: &mut GpuContext, mode: &FrameRenderMode<'_>) {
        profiling::scope!("render::setup_msaa");
        let requested_msaa = self
            .settings
            .read()
            .map(|s| s.rendering.msaa.as_count())
            .unwrap_or(1);
        let prev_msaa = gpu.swapchain_msaa_effective();
        gpu.set_swapchain_msaa_requested(requested_msaa);
        self.transient_evict_stale_msaa_tiers_if_changed(prev_msaa, gpu.swapchain_msaa_effective());
        // Stereo MSAA tier applies to `ExternalMultiview` HMD targets; keep both tiers in sync
        // so transient textures keyed by sample count invalidate on a mode change.
        if mode.has_hmd() {
            let prev_stereo = gpu.swapchain_msaa_effective_stereo();
            gpu.set_swapchain_msaa_requested_stereo(requested_msaa);
            self.transient_evict_stale_msaa_tiers_if_changed(
                prev_stereo,
                gpu.swapchain_msaa_effective_stereo(),
            );
        }
    }

    /// Collects every active view for this tick into a single ordered list.
    ///
    /// Ordering — preserved from the pre-unification code so the mesh-deform skip flag on
    /// [`crate::backend::FrameResourceManager`] still runs deform exactly once per tick:
    /// 1. HMD stereo multiview (when `mode = VrWithHmd`).
    /// 2. Secondary render-texture cameras, sorted by camera depth.
    /// 3. Main desktop swapchain (when `mode = DesktopPlusSecondaries`).
    fn collect_prepared_views<'a>(
        &mut self,
        mode: FrameRenderMode<'a>,
        swapchain_extent_px: (u32, u32),
    ) -> Vec<PreparedView<'a>> {
        let (includes_main, hmd_target) = match mode {
            FrameRenderMode::DesktopPlusSecondaries => (true, None),
            FrameRenderMode::VrWithHmd(ext) => (false, Some(ext)),
            FrameRenderMode::VrSecondariesOnly => (false, None),
        };

        let mut secondary_views = self.collect_secondary_rt_views();
        let est_capacity =
            usize::from(hmd_target.is_some()) + secondary_views.len() + usize::from(includes_main);
        let mut views: Vec<PreparedView<'a>> = Vec::with_capacity(est_capacity);

        if let Some(ext) = hmd_target {
            let extent_px = ext.extent_px;
            views.push(PreparedView {
                host_camera: self.host_camera,
                draw_filter: None,
                occlusion_view_id: OcclusionViewId::Main,
                viewport_px: extent_px,
                clear: FrameViewClear::skybox(),
                kind: PreparedViewKind::Hmd(ext),
            });
        }

        views.append(&mut secondary_views);

        if includes_main {
            views.push(self.build_main_swapchain_view(swapchain_extent_px));
        }

        views
    }

    /// Builds prepared views for every enabled secondary render-texture camera in the scene,
    /// skipping cameras whose host render texture is not yet resident on the GPU.
    ///
    /// Reuses [`RendererRuntime::secondary_view_tasks_scratch`] for the depth-sort scratch buffer
    /// so a frame with secondary cameras does not allocate a fresh `Vec` for the sort each tick.
    fn collect_secondary_rt_views<'a>(&mut self) -> Vec<PreparedView<'a>> {
        let mut tasks = std::mem::take(&mut self.secondary_view_tasks_scratch);
        tasks.clear();
        let result = self.collect_secondary_rt_views_using(&mut tasks);
        self.secondary_view_tasks_scratch = tasks;
        result
    }

    /// Inner helper that consumes the supplied scratch `tasks` buffer; split out so the outer
    /// caller can keep the scratch field reachable across the immutable borrow taken here.
    fn collect_secondary_rt_views_using<'a>(
        &self,
        tasks: &mut Vec<(crate::scene::RenderSpaceId, f32, usize)>,
    ) -> Vec<PreparedView<'a>> {
        for sid in self.scene.render_space_ids() {
            let Some(space) = self.scene.space(sid) else {
                continue;
            };
            if !space.is_active {
                continue;
            }
            for (idx, cam) in space.cameras.iter().enumerate() {
                if !camera_state_enabled(cam.state.flags) {
                    continue;
                }
                if cam.state.render_texture_asset_id < 0 {
                    continue;
                }
                tasks.push((sid, cam.state.depth, idx));
            }
        }
        tasks.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut views: Vec<PreparedView<'a>> = Vec::with_capacity(tasks.len());
        for (sid, _, cam_idx) in tasks.drain(..) {
            let Some(space) = self.scene.space(sid) else {
                continue;
            };
            let Some(entry) = space.cameras.get(cam_idx) else {
                continue;
            };
            if !camera_state_enabled(entry.state.flags) {
                continue;
            }
            let rt_id = entry.state.render_texture_asset_id;
            let (color_view, depth_texture, depth_view, viewport, color_format) = {
                let Some(rt) = self.backend.render_texture_pool().get(rt_id) else {
                    logger::trace!(
                        "secondary camera: render texture asset {rt_id} not resident; skipping"
                    );
                    continue;
                };
                let Some(dt) = rt.depth_texture.clone() else {
                    logger::warn!("secondary camera: render texture {rt_id} missing depth");
                    continue;
                };
                let Some(dv) = rt.depth_view.clone() else {
                    logger::warn!("secondary camera: render texture {rt_id} missing depth view");
                    continue;
                };
                (
                    rt.color_view.clone(),
                    dt,
                    dv,
                    (rt.width, rt.height),
                    rt.wgpu_color_format,
                )
            };
            let Some(world_m) = self.scene.world_matrix(sid, entry.transform_id as usize) else {
                continue;
            };
            let mut hc = host_camera_frame_for_render_texture(
                &self.host_camera,
                &entry.state,
                viewport,
                world_m,
                &self.scene,
            );
            let filter = draw_filter_from_camera_entry(entry);
            // Selective secondary cameras (dashboards, in-world UI panels, mirrors on specific
            // subtrees) render tens of draws, not thousands. Hi-Z snapshots + occlusion temporal
            // cost a per-camera readback path with negligible payoff at that scale — skip them.
            if !entry.selective_transform_ids.is_empty() {
                hc.suppress_occlusion_temporal = true;
            }
            views.push(PreparedView {
                host_camera: hc,
                draw_filter: Some(filter),
                occlusion_view_id: OcclusionViewId::OffscreenRenderTexture(rt_id),
                viewport_px: viewport,
                clear: FrameViewClear::from_camera_state(&entry.state),
                kind: PreparedViewKind::SecondaryRt(OffscreenRtHandles {
                    rt_id,
                    color_view,
                    depth_texture,
                    depth_view,
                    color_format,
                }),
            });
        }
        views
    }

    /// Builds the main desktop swapchain [`PreparedView`] from the cached [`Self::host_camera`].
    ///
    /// `swapchain_extent_px` must be the current GPU surface extent: it feeds
    /// [`build_world_mesh_cull_proj_params`] on the pre-dispatch CPU cull path. A stale or zero
    /// extent produces a degenerate frustum and random scene-object culling. The render graph
    /// resolves its own rendering extent from [`FrameViewTarget::Swapchain::extent_px`] at record
    /// time — that is a separate concern from cull math, which has already run by then.
    fn build_main_swapchain_view<'a>(&self, swapchain_extent_px: (u32, u32)) -> PreparedView<'a> {
        PreparedView {
            host_camera: self.host_camera,
            draw_filter: None,
            occlusion_view_id: OcclusionViewId::Main,
            viewport_px: swapchain_extent_px,
            clear: FrameViewClear::skybox(),
            kind: PreparedViewKind::MainSwapchain,
        }
    }
}

#[cfg(test)]
mod tests {
    //! Data-only tests for [`RendererRuntime::collect_prepared_views`]. No GPU is created.

    use std::path::PathBuf;
    use std::sync::Arc;

    use super::*;
    use crate::config::{RendererSettings, RendererSettingsHandle};
    use crate::connection::ConnectionParams;

    fn build_runtime() -> RendererRuntime {
        let settings: RendererSettingsHandle =
            Arc::new(std::sync::RwLock::new(RendererSettings::default()));
        RendererRuntime::new(
            Option::<ConnectionParams>::None,
            settings,
            PathBuf::from("test_config.toml"),
        )
    }

    const TEST_EXTENT: (u32, u32) = (1920, 1080);

    #[test]
    fn empty_scene_desktop_mode_yields_only_main_view() {
        let mut runtime = build_runtime();
        let views =
            runtime.collect_prepared_views(FrameRenderMode::DesktopPlusSecondaries, TEST_EXTENT);
        assert_eq!(views.len(), 1);
        assert!(matches!(views[0].kind, PreparedViewKind::MainSwapchain));
        assert_eq!(views[0].occlusion_view_id, OcclusionViewId::Main);
        assert!(views[0].draw_filter.is_none());
    }

    #[test]
    fn empty_scene_vr_secondaries_only_yields_empty_vec() {
        let mut runtime = build_runtime();
        let views = runtime.collect_prepared_views(FrameRenderMode::VrSecondariesOnly, TEST_EXTENT);
        assert!(
            views.is_empty(),
            "no HMD, no secondaries, and main swapchain excluded — nothing to render"
        );
    }

    #[test]
    fn main_view_carries_runtime_host_camera() {
        let mut runtime = build_runtime();
        runtime.host_camera.frame_index = 42;
        runtime.host_camera.desktop_fov_degrees = 75.0;
        let views =
            runtime.collect_prepared_views(FrameRenderMode::DesktopPlusSecondaries, TEST_EXTENT);
        let main = &views[0];
        assert_eq!(main.host_camera.frame_index, 42);
        assert_eq!(main.host_camera.desktop_fov_degrees, 75.0);
    }

    /// Pins the contract from the April 2026 cull regression: the main desktop `PreparedView`
    /// must carry the swapchain extent supplied to `collect_prepared_views`. A zero or stale
    /// extent produces a degenerate `build_world_mesh_cull_proj_params` frustum and flickering
    /// scene-object culling.
    #[test]
    fn main_view_viewport_matches_supplied_swapchain_extent() {
        let mut runtime = build_runtime();
        let views =
            runtime.collect_prepared_views(FrameRenderMode::DesktopPlusSecondaries, (1280, 720));
        let main = views
            .iter()
            .find(|v| matches!(v.kind, PreparedViewKind::MainSwapchain))
            .expect("DesktopPlusSecondaries yields a MainSwapchain view");
        assert_eq!(main.viewport_px, (1280, 720));
    }

    #[test]
    fn main_view_uses_default_shader_permutation_and_depth_mode() {
        let runtime = build_runtime();
        let view = runtime.build_main_swapchain_view(TEST_EXTENT);
        assert_eq!(view.shader_permutation(), ShaderPermutation(0));
        assert_eq!(view.output_depth_mode(), OutputDepthMode::DesktopSingle);
        assert_eq!(view.clear.mode, crate::shared::CameraClearMode::Skybox);
    }

    #[test]
    fn prepared_view_helpers_honor_explicit_camera_world_position() {
        let runtime = build_runtime();
        let mut view = runtime.build_main_swapchain_view(TEST_EXTENT);
        view.host_camera.head_output_transform =
            glam::Mat4::from_translation(glam::Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(view.view_origin_world(), glam::Vec3::new(1.0, 2.0, 3.0));
        view.host_camera.explicit_camera_world_position = Some(glam::Vec3::new(7.0, 8.0, 9.0));
        assert_eq!(view.view_origin_world(), glam::Vec3::new(7.0, 8.0, 9.0));
    }

    #[test]
    fn prepared_view_helpers_prefer_eye_world_position_over_head_output() {
        let runtime = build_runtime();
        let mut view = runtime.build_main_swapchain_view(TEST_EXTENT);
        view.host_camera.head_output_transform =
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, 0.0));
        view.host_camera.eye_world_position = Some(glam::Vec3::new(4.0, 5.0, 6.0));
        assert_eq!(
            view.view_origin_world(),
            glam::Vec3::new(4.0, 5.0, 6.0),
            "eye_world_position must override the head-output (render-space root) translation \
             so PBS view-direction math sees the eye, not the floor anchor"
        );
        view.host_camera.explicit_camera_world_position = Some(glam::Vec3::new(7.0, 8.0, 9.0));
        assert_eq!(
            view.view_origin_world(),
            glam::Vec3::new(7.0, 8.0, 9.0),
            "explicit_camera_world_position still wins over eye_world_position"
        );
    }
}
