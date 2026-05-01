//! Unified per-tick render entry point: builds one `FrameView` list covering the HMD, secondary
//! render-texture cameras, and the main desktop view, then dispatches the compiled render graph
//! in a single submit.

use std::fmt::Write as _;

use crate::gpu::GpuContext;
use crate::render_graph::{ExternalFrameTargets, GraphExecuteError};

use super::RendererRuntime;
use super::frame_extract::{ExtractedFrame, PreparedViews, select_inner_parallelism};
use super::frame_view_plan::{FrameViewPlan, FrameViewPlanTarget, HeadlessOffscreenSnapshot};

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
    pub(super) fn includes_main_swapchain(&self) -> bool {
        matches!(self, FrameRenderMode::DesktopPlusSecondaries)
    }

    /// `true` when this mode prepends an HMD stereo multiview view.
    fn has_hmd(&self) -> bool {
        matches!(self, FrameRenderMode::VrWithHmd(_))
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
        };
        self.sync_debug_hud_diagnostics_from_settings();
        self.setup_msaa_for_mode(gpu, &mode);

        let frame_extract = {
            profiling::scope!("render::extract_frame");
            self.extract_frame(gpu, mode)
        };
        if frame_extract.is_empty() {
            logger::trace!("render frame skipped: no prepared views");
            return Ok(());
        }

        let prepared_draws = {
            profiling::scope!("render::prepare_draws");
            frame_extract.prepare_draws()
        };
        let submit_frame = prepared_draws.into_submit_frame();
        let scene = &self.scene;
        let backend = &mut self.backend;
        submit_frame.execute(gpu, scene, backend)
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

    /// Builds the explicit frame extraction packet for this tick, including prepared views,
    /// backend draw-prep state, and any headless main-target substitution resources that must
    /// outlive graph-view creation.
    fn extract_frame<'a>(
        &mut self,
        gpu: &mut GpuContext,
        mode: FrameRenderMode<'a>,
    ) -> ExtractedFrame<'a, '_> {
        let prepared_views = {
            profiling::scope!("render::prepare_views");
            self.prepare_frame_views(gpu, mode)
        };
        self.backend
            .sync_active_views(prepared_views.plans().iter().map(|view| view.view_id));
        let shared = {
            profiling::scope!("render::extract_frame_shared");
            // Hand the per-view shader permutations through so the backend refreshes one material
            // batch cache per distinct permutation; previously every non-mono view rebuilt its own
            // cache locally inside `collect_view_draws`.
            let view_perms: Vec<crate::materials::ShaderPermutation> = prepared_views
                .plans()
                .iter()
                .map(|plan| plan.shader_permutation())
                .collect();
            self.backend.extract_frame_shared(
                &self.scene,
                self.scene.active_main_render_context(),
                select_inner_parallelism(prepared_views.plans()),
                view_perms,
            )
        };
        ExtractedFrame::new(prepared_views, shared)
    }

    /// Builds the explicit prepared-view stage for this tick, including any headless main-target
    /// substitution resources that must outlive graph-view creation.
    fn prepare_frame_views<'a>(
        &mut self,
        gpu: &mut GpuContext,
        mode: FrameRenderMode<'a>,
    ) -> PreparedViews<'a> {
        let includes_main = mode.includes_main_swapchain();
        // Capture the swapchain extent before the per-view collection. The main desktop view's
        // CPU cull projection (`build_world_mesh_cull_proj_params`) runs against this extent
        // before the render graph dispatches, so passing a stale/zero value produces a degenerate
        // frustum and randomly culls scene objects.
        let swapchain_extent_px = gpu.surface_extent_px();
        let prepared: Vec<FrameViewPlan<'a>> =
            self.collect_prepared_views(mode, swapchain_extent_px);
        trace_prepared_views(&prepared);
        let headless_snapshot = {
            profiling::scope!("render::headless_snapshot");
            if includes_main && gpu.is_headless() {
                HeadlessOffscreenSnapshot::from_gpu(gpu)
            } else {
                None
            }
        };
        PreparedViews::new(prepared, headless_snapshot)
    }
}

fn trace_prepared_views(prepared: &[FrameViewPlan<'_>]) {
    if !logger::enabled(logger::LogLevel::Trace) {
        return;
    }
    let mut hmd = 0usize;
    let mut secondary = 0usize;
    let mut main = 0usize;
    let mut details = String::new();
    for (idx, view) in prepared.iter().enumerate() {
        let label = match &view.target {
            FrameViewPlanTarget::Hmd(_) => {
                hmd += 1;
                "hmd"
            }
            FrameViewPlanTarget::SecondaryRt(_) => {
                secondary += 1;
                "secondary_rt"
            }
            FrameViewPlanTarget::MainSwapchain => {
                main += 1;
                "main_swapchain"
            }
        };
        if idx > 0 {
            details.push_str(", ");
        }
        let _ = write!(
            details,
            "#{idx}:{label} view_id={:?} extent={}x{} stereo={} filter={}",
            view.view_id,
            view.viewport_px.0,
            view.viewport_px.1,
            view.is_multiview_stereo_active(),
            view.draw_filter.is_some(),
        );
    }
    logger::trace!(
        "render prepared views: count={} hmd={} secondary_rt={} main_swapchain={} [{}]",
        prepared.len(),
        hmd,
        secondary,
        main,
        details,
    );
}
