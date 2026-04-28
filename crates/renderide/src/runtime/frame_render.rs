//! Unified per-tick render entry point: builds one `FrameView` list covering the HMD, secondary
//! render-texture cameras, and the main desktop view, then dispatches the compiled render graph
//! in a single submit.

use rayon::prelude::*;

use crate::backend::FrameDrawSetup;
use crate::gpu::GpuContext;
use crate::materials::{MaterialRouter, RasterPipelineKind};
use crate::render_graph::{
    build_world_mesh_cull_proj_params, camera_state_enabled,
    collect_and_sort_world_mesh_draws_with_parallelism, draw_filter_from_camera_entry,
    host_camera_frame_for_render_texture, DrawCollectionContext, ExternalFrameTargets, FrameView,
    FrameViewClear, GraphExecuteError, HiZCullData, HiZTemporalState, OcclusionViewId,
    PrefetchedWorldMeshViewDraws, WorldMeshCullInput, WorldMeshCullProjParams,
    WorldMeshDrawCollectParallelism, WorldMeshDrawPlan,
};

use super::frame_view_plan::{
    FrameViewPlan, FrameViewPlanTarget, HeadlessOffscreenSnapshot, OffscreenRtHandles,
};
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

/// Prepared per-frame view list plus any headless swapchain substitution resources needed to
/// turn it into executable graph views.
struct PreparedFrameViews<'a> {
    /// Ordered list of planned views for this tick.
    prepared: Vec<FrameViewPlan<'a>>,
    /// Headless main-target replacement captured before backend execution borrows the GPU.
    headless_snapshot: Option<HeadlessOffscreenSnapshot>,
}

impl<'a> PreparedFrameViews<'a> {
    /// Returns `true` when no view should be rendered this tick.
    fn is_empty(&self) -> bool {
        self.prepared.is_empty()
    }

    /// Shared slice of the ordered planned views.
    fn plans(&self) -> &[FrameViewPlan<'a>] {
        &self.prepared
    }

    /// Builds executable graph views from the prepared plans and collected draw plans.
    fn build_execution_views<'b>(&'b self, draw_plans: Vec<WorldMeshDrawPlan>) -> Vec<FrameView<'b>>
    where
        'a: 'b,
    {
        let mut views: Vec<FrameView<'b>> = self
            .prepared
            .iter()
            .zip(draw_plans)
            .map(|(prep, draws)| prep.to_frame_view(draws))
            .collect();
        if let Some(snapshot) = self.headless_snapshot.as_ref() {
            snapshot.substitute_swapchain_views(&mut views);
        }
        views
    }
}

/// One collected world-mesh draw plan per prepared view, in the same order as
/// [`PreparedFrameViews::prepared`].
struct CollectedFrameDraws {
    /// Explicit draw plan for every prepared view.
    view_draws: Vec<WorldMeshDrawPlan>,
}

/// Frustum + Hi-Z cull inputs for one planned view.
struct ViewCullSnapshot {
    /// Projection parameters matching the view's camera/viewport.
    proj: WorldMeshCullProjParams,
    /// CPU-side Hi-Z snapshot for this view's occlusion slot.
    hi_z: Option<HiZCullData>,
    /// Temporal Hi-Z state captured after the prior frame's depth pyramid author pass.
    hi_z_temporal: Option<HiZTemporalState>,
}

/// Collects and sorts world-mesh draws for every prepared view in parallel.
///
/// Returns one explicit [`WorldMeshDrawPlan`] per prepared view, preserving input order so the
/// compiled graph never has to infer whether draws were intentionally omitted or merely missing.
fn collect_view_draws(
    setup: &FrameDrawSetup<'_>,
    prepared: &[FrameViewPlan<'_>],
    cull_snapshots: &[Option<ViewCullSnapshot>],
) -> Vec<WorldMeshDrawPlan> {
    profiling::scope!("render::collect_view_draws");
    prepared
        .par_iter()
        .zip(cull_snapshots.par_iter())
        .map(|(prep, snap)| {
            let shader_perm = prep.shader_permutation();
            let material_cache = (shader_perm == crate::pipelines::ShaderPermutation(0))
                .then_some(setup.material_cache);
            let dict = crate::assets::material::MaterialDictionary::new(setup.property_store);
            let cull_proj = snap.as_ref().map(|s| s.proj);
            let culling = snap.as_ref().map(|s| WorldMeshCullInput {
                proj: s.proj,
                host_camera: &prep.host_camera,
                hi_z: s.hi_z.clone(),
                hi_z_temporal: s.hi_z_temporal.clone(),
            });
            let collection = collect_and_sort_world_mesh_draws_with_parallelism(
                &DrawCollectionContext {
                    scene: setup.scene,
                    mesh_pool: setup.mesh_pool,
                    material_dict: &dict,
                    material_router: setup.router,
                    pipeline_property_ids: &setup.pipeline_property_ids,
                    shader_perm,
                    render_context: setup.render_context,
                    head_output_transform: prep.host_camera.head_output_transform,
                    view_origin_world: prep.view_origin_world(),
                    culling: culling.as_ref(),
                    transform_filter: prep.draw_filter.as_ref(),
                    material_cache,
                    prepared: Some(&setup.prepared_renderables),
                },
                setup.inner_parallelism,
            );
            WorldMeshDrawPlan::Prefetched(Box::new(PrefetchedWorldMeshViewDraws::new(
                collection, cull_proj,
            )))
        })
        .collect()
}

/// Selects the per-view inner-walk parallelism tier for a tick based on how many views will
/// collect draws. Keeps rayon from oversubscribing when several views each spawn worker-level
/// parallelism.
fn select_inner_parallelism(prepared: &[FrameViewPlan<'_>]) -> WorldMeshDrawCollectParallelism {
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
    setup: &FrameDrawSetup<'_>,
    prep: &FrameViewPlan<'_>,
) -> Option<ViewCullSnapshot> {
    if prep.host_camera.suppress_occlusion_temporal {
        return None;
    }
    let proj = build_world_mesh_cull_proj_params(setup.scene, prep.viewport_px, &prep.host_camera);
    let depth_mode = prep.output_depth_mode();
    Some(ViewCullSnapshot {
        proj,
        hi_z: setup
            .occlusion
            .hi_z_cull_data(depth_mode, prep.occlusion_view_id),
        hi_z_temporal: setup
            .occlusion
            .hi_z_temporal_snapshot(prep.occlusion_view_id),
    })
}

/// Collects world-mesh draw plans for every prepared view using the backend-owned frame draw
/// setup assembled for the current tick.
fn collect_frame_draws(
    prepared_views: &PreparedFrameViews<'_>,
    draw_setup: &FrameDrawSetup<'_>,
) -> CollectedFrameDraws {
    let cull_snapshots: Vec<Option<ViewCullSnapshot>> = {
        profiling::scope!("render::gather_view_cull_snapshots");
        prepared_views
            .plans()
            .par_iter()
            .map(|prep| cull_snapshot_for_view(draw_setup, prep))
            .collect()
    };
    CollectedFrameDraws {
        view_draws: collect_view_draws(draw_setup, prepared_views.plans(), &cull_snapshots),
    }
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
            self.backend.prepare_lights_from_scene(&self.scene);
        }
        self.sync_debug_hud_diagnostics_from_settings();
        self.setup_msaa_for_mode(gpu, &mode);

        let prepared_views = {
            profiling::scope!("render::collect_prepared_views");
            self.prepare_frame_views(gpu, mode)
        };
        if prepared_views.is_empty() {
            return Ok(());
        }

        let collected_draws = {
            let fallback_router = MaterialRouter::new(RasterPipelineKind::Null);
            let draw_setup = self.backend.prepare_frame_draw_setup(
                &self.scene,
                self.scene.active_main_render_context(),
                select_inner_parallelism(prepared_views.plans()),
                &fallback_router,
            );
            collect_frame_draws(&prepared_views, &draw_setup)
        };

        let mut views = prepared_views.build_execution_views(collected_draws.view_draws);
        self.backend
            .execute_multi_view_frame(gpu, &self.scene, &mut views, true)
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

    /// Builds the explicit prepared-view stage for this tick, including any headless main-target
    /// substitution resources that must outlive graph-view creation.
    fn prepare_frame_views<'a>(
        &mut self,
        gpu: &mut GpuContext,
        mode: FrameRenderMode<'a>,
    ) -> PreparedFrameViews<'a> {
        let includes_main = mode.includes_main_swapchain();
        // Capture the swapchain extent before the per-view collection. The main desktop view's
        // CPU cull projection (`build_world_mesh_cull_proj_params`) runs against this extent
        // before the render graph dispatches, so passing a stale/zero value produces a degenerate
        // frustum and randomly culls scene objects.
        let swapchain_extent_px = gpu.surface_extent_px();
        let prepared = self.collect_prepared_views(mode, swapchain_extent_px);
        let headless_snapshot = {
            profiling::scope!("render::headless_snapshot");
            if includes_main && gpu.is_headless() {
                HeadlessOffscreenSnapshot::from_gpu(gpu)
            } else {
                None
            }
        };
        PreparedFrameViews {
            prepared,
            headless_snapshot,
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
    ) -> Vec<FrameViewPlan<'a>> {
        let (includes_main, hmd_target) = match mode {
            FrameRenderMode::DesktopPlusSecondaries => (true, None),
            FrameRenderMode::VrWithHmd(ext) => (false, Some(ext)),
            FrameRenderMode::VrSecondariesOnly => (false, None),
        };

        let mut secondary_views = self.collect_secondary_rt_views();
        let est_capacity =
            usize::from(hmd_target.is_some()) + secondary_views.len() + usize::from(includes_main);
        let mut views: Vec<FrameViewPlan<'a>> = Vec::with_capacity(est_capacity);

        if let Some(ext) = hmd_target {
            let extent_px = ext.extent_px;
            views.push(FrameViewPlan {
                host_camera: self.host_camera,
                draw_filter: None,
                occlusion_view_id: OcclusionViewId::Main,
                viewport_px: extent_px,
                clear: FrameViewClear::skybox(),
                target: FrameViewPlanTarget::Hmd(ext),
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
    fn collect_secondary_rt_views<'a>(&mut self) -> Vec<FrameViewPlan<'a>> {
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
    ) -> Vec<FrameViewPlan<'a>> {
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

        let mut views: Vec<FrameViewPlan<'a>> = Vec::with_capacity(tasks.len());
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
            views.push(FrameViewPlan {
                host_camera: hc,
                draw_filter: Some(filter),
                occlusion_view_id: OcclusionViewId::OffscreenRenderTexture(rt_id),
                viewport_px: viewport,
                clear: FrameViewClear::from_camera_state(&entry.state),
                target: FrameViewPlanTarget::SecondaryRt(OffscreenRtHandles {
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

    /// Builds the main desktop swapchain [`FrameViewPlan`] from the cached [`Self::host_camera`].
    ///
    /// `swapchain_extent_px` must be the current GPU surface extent: it feeds
    /// [`build_world_mesh_cull_proj_params`] on the pre-dispatch CPU cull path. A stale or zero
    /// extent produces a degenerate frustum and random scene-object culling. The render graph
    /// resolves its own rendering extent from [`FrameViewTarget::Swapchain::extent_px`] at record
    /// time — that is a separate concern from cull math, which has already run by then.
    fn build_main_swapchain_view<'a>(&self, swapchain_extent_px: (u32, u32)) -> FrameViewPlan<'a> {
        FrameViewPlan {
            host_camera: self.host_camera,
            draw_filter: None,
            occlusion_view_id: OcclusionViewId::Main,
            viewport_px: swapchain_extent_px,
            clear: FrameViewClear::skybox(),
            target: FrameViewPlanTarget::MainSwapchain,
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
    use crate::pipelines::ShaderPermutation;
    use crate::render_graph::OutputDepthMode;

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
        assert!(matches!(
            views[0].target,
            FrameViewPlanTarget::MainSwapchain
        ));
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

    /// Pins the contract from the April 2026 cull regression: the main desktop `FrameViewPlan`
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
            .find(|v| matches!(v.target, FrameViewPlanTarget::MainSwapchain))
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
