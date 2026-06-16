//! Unified per-tick render entry point: builds one `FrameView` list covering the HMD, secondary
//! render-texture cameras, and the main desktop view, then dispatches the compiled render graph
//! in a single submit.

use std::fmt::Write as _;

use crate::gpu::GpuContext;
use crate::render_graph::{ExternalFrameTargets, GraphExecuteError};

use super::super::RendererRuntime;
use super::extract::{PreparedViews, select_inner_parallelism};
use super::schedule::{
    CpuRenderPhase, CpuRenderSchedule, RenderScheduleKind, execute_prepared_views,
    prepare_assets_for_schedule,
};
use super::view_plan::{FrameViewPlan, FrameViewPlanTarget, OffscreenTargetHandles};

/// Primary render target requested for a view-family submission.
pub(in crate::runtime) enum PrimaryViewRequest<'a> {
    /// Main desktop view.
    DesktopMain,
    /// OpenXR HMD stereo multiview view.
    HmdExternalMultiview(ExternalFrameTargets<'a>),
    /// No primary view; render secondary render-texture cameras only.
    None,
}

impl PrimaryViewRequest<'_> {
    /// `true` when this mode appends the main desktop view.
    pub(in crate::runtime) fn includes_main_view(&self) -> bool {
        matches!(self, PrimaryViewRequest::DesktopMain)
    }

    /// `true` when this mode prepends an HMD stereo multiview view.
    fn has_hmd(&self) -> bool {
        matches!(self, PrimaryViewRequest::HmdExternalMultiview(_))
    }
}

/// Complete render request for one view-family submission.
pub(crate) struct FrameViewFamilyRequest<'a> {
    primary: PrimaryViewRequest<'a>,
    schedule_kind: RenderScheduleKind,
}

impl<'a> FrameViewFamilyRequest<'a> {
    /// Desktop world render: secondary render textures plus the main desktop view.
    pub(crate) fn desktop() -> Self {
        Self {
            primary: PrimaryViewRequest::DesktopMain,
            schedule_kind: RenderScheduleKind::Desktop,
        }
    }

    /// Desktop render when another presentation path owns the swapchain.
    pub(crate) fn desktop_secondaries_only() -> Self {
        Self {
            primary: PrimaryViewRequest::None,
            schedule_kind: RenderScheduleKind::Desktop,
        }
    }

    /// OpenXR HMD render: HMD stereo view plus secondary render textures.
    pub(crate) fn hmd(hmd: ExternalFrameTargets<'a>) -> Self {
        Self {
            primary: PrimaryViewRequest::HmdExternalMultiview(hmd),
            schedule_kind: RenderScheduleKind::Hmd,
        }
    }

    /// VR tick where HMD rendering did not start and only secondary RTs should render.
    pub(crate) fn vr_secondaries_only() -> Self {
        Self {
            primary: PrimaryViewRequest::None,
            schedule_kind: RenderScheduleKind::VrSecondariesOnly,
        }
    }

    /// Primary view request for this family.
    pub(in crate::runtime) fn primary(self) -> PrimaryViewRequest<'a> {
        self.primary
    }

    /// CPU render schedule kind for this family.
    pub(in crate::runtime) const fn schedule_kind(&self) -> RenderScheduleKind {
        self.schedule_kind
    }

    /// `true` when this family appends the main desktop view.
    pub(in crate::runtime) fn includes_main_view(&self) -> bool {
        self.primary.includes_main_view()
    }

    /// Frame-global fallback profile to use when no HMD or main desktop view is submitted.
    pub(in crate::runtime) fn fallback_frame_global_profile(
        &self,
        desktop_profile: crate::render_graph::RenderPathProfile,
    ) -> crate::render_graph::RenderPathProfile {
        match self.schedule_kind {
            RenderScheduleKind::Hmd | RenderScheduleKind::VrSecondariesOnly => {
                crate::render_graph::RenderPathProfile::xr_hmd()
            }
            RenderScheduleKind::Desktop
            | RenderScheduleKind::CameraTask
            | RenderScheduleKind::Camera360Capture
            | RenderScheduleKind::ReflectionProbeCapture => desktop_profile,
        }
    }

    /// `true` when this family records an HMD stereo primary view.
    fn has_hmd(&self) -> bool {
        self.primary.has_hmd()
    }
}

impl RendererRuntime {
    /// Desktop entry point: renders the main desktop view plus any active secondary render-texture
    /// cameras in a single submit. Used when OpenXR is not active.
    ///
    /// See [`Self::render_frame`] for the shared implementation that also powers the VR entry
    /// points on [`crate::xr::XrFrameRenderer`].
    pub fn render_desktop_frame(&mut self, gpu: &mut GpuContext) -> Result<(), GraphExecuteError> {
        self.render_frame(gpu, FrameViewFamilyRequest::desktop())
    }

    /// Desktop entry point for ticks where presentation is supplied by an explicit host
    /// `BlitToDisplay`.
    ///
    /// Secondary render-texture cameras still update through the normal desktop schedule, but
    /// the main desktop world view is omitted because the display blit pass fills it later.
    pub(crate) fn render_desktop_secondaries_frame(
        &mut self,
        gpu: &mut GpuContext,
    ) -> Result<(), GraphExecuteError> {
        self.render_frame(gpu, FrameViewFamilyRequest::desktop_secondaries_only())
    }

    /// Unified per-tick world render entry point.
    ///
    /// Builds a single prepared-view list (HMD first when present, secondary RTs in depth order,
    /// main desktop view last when requested) and dispatches the compiled render graph in one
    /// [`RenderBackend::execute_multi_view_frame`](crate::backend::RenderBackend::execute_multi_view_frame)
    /// call. Hi-Z readback has already been drained once at the top of the tick (see
    /// [`Self::drain_hi_z_readback`]), so the caller always skips the readback pass here.
    ///
    /// Callers should not invoke this directly; use [`Self::render_desktop_frame`] for desktop or
    /// the [`crate::xr::XrFrameRenderer`] trait methods for VR paths.
    ///
    /// The main desktop/headless view is planned directly as an offscreen target backed by
    /// [`GpuContext::primary_offscreen_targets`].
    pub(crate) fn render_frame(
        &mut self,
        gpu: &mut GpuContext,
        request: FrameViewFamilyRequest<'_>,
    ) -> Result<(), GraphExecuteError> {
        profiling::scope!("render::render_frame");
        let schedule = CpuRenderSchedule::new(request.schedule_kind());
        schedule.run_phase(CpuRenderPhase::Extract, || {
            self.sync_debug_hud_diagnostics_from_settings();
        });
        schedule.run_phase(CpuRenderPhase::AssetPrepare, || {
            self.setup_msaa_for_request(gpu, &request);
            prepare_assets_for_schedule(&mut self.backend);
        });
        let prepared_views = schedule.run_phase(CpuRenderPhase::ViewPlanning, || {
            self.prepare_frame_views(gpu, request)
        });
        let inner_parallelism = select_inner_parallelism(prepared_views.plans());
        let scene = &self.scene;
        let backend = &mut self.backend;
        execute_prepared_views(
            schedule,
            gpu,
            backend,
            scene,
            prepared_views,
            inner_parallelism,
        )
    }

    /// Applies the MSAA tier for the active mode and evicts transient textures keyed by stale
    /// sample counts on a tier change.
    fn setup_msaa_for_request(
        &mut self,
        gpu: &mut GpuContext,
        request: &FrameViewFamilyRequest<'_>,
    ) {
        profiling::scope!("render::setup_msaa");
        self.sync_master_msaa(gpu);
        // Stereo MSAA tier applies to `ExternalMultiview` HMD targets; keep both tiers in sync
        // so transient textures keyed by sample count invalidate on a mode change.
        if request.has_hmd() {
            self.sync_stereo_msaa_from_master(gpu);
        }
    }

    /// Builds the explicit prepared-view stage for this tick, including any main-target
    /// offscreen resources that must outlive graph-view creation.
    fn prepare_frame_views<'a>(
        &mut self,
        gpu: &mut GpuContext,
        request: FrameViewFamilyRequest<'a>,
    ) -> PreparedViews<'a> {
        let includes_main = request.includes_main_view();
        // Capture the configured surface extent before the per-view collection. When no primary
        // offscreen target is needed, this still supplies the main-view CPU cull projection
        // extent before render-graph dispatch.
        let configured_extent_px = gpu.surface_extent_px();
        let main_offscreen_target =
            includes_main.then(|| OffscreenTargetHandles::from_primary_offscreen(gpu));
        let main_extent_px = main_offscreen_target
            .as_ref()
            .map_or(configured_extent_px, |target| target.extent_px);
        let main_profile = if includes_main && gpu.is_headless() {
            crate::render_graph::RenderPathProfile::headless_main()
        } else {
            crate::render_graph::RenderPathProfile::desktop_main()
        };
        let fallback_frame_global_profile = request.fallback_frame_global_profile(main_profile);
        let prepared = self.collect_prepared_views(
            gpu,
            request.primary(),
            main_extent_px,
            main_profile,
            fallback_frame_global_profile,
            main_offscreen_target,
        );
        trace_prepared_views(prepared.plans());
        self.backend
            .sync_active_views(prepared.plans().iter().flat_map(|view| {
                std::iter::once(view.view_id).chain(view.desktop_overlay_resource_view_id())
            }));
        PreparedViews::new(prepared)
    }
}

fn trace_prepared_views(prepared: &[FrameViewPlan<'_>]) {
    crate::crash_context::set_prepared_view_count(prepared.len());
    if !logger::enabled(logger::LogLevel::Trace) {
        return;
    }
    let mut hmd = 0usize;
    let mut offscreen = 0usize;
    let mut main = 0usize;
    let mut details = String::new();
    for (idx, view) in prepared.iter().enumerate() {
        let label = match &view.target {
            FrameViewPlanTarget::ExternalMultiview(_) => {
                hmd += 1;
                "hmd"
            }
            FrameViewPlanTarget::Offscreen(_) => {
                offscreen += 1;
                "offscreen"
            }
            FrameViewPlanTarget::Swapchain => {
                main += 1;
                "swapchain"
            }
        };
        if idx > 0 {
            details.push_str(", ");
        }
        let _ = write!(
            details,
            "#{idx}:{label} profile={:?} view_id={:?} extent={}x{} stereo={} post={} filter={}",
            view.profile.id(),
            view.view_id,
            view.viewport_px.0,
            view.viewport_px.1,
            view.is_multiview_stereo_active(),
            view.post_processing().is_enabled(),
            view.draw_filter.is_some(),
        );
    }
    logger::trace!(
        "render prepared views: count={} hmd={} offscreen={} swapchain={} [{}]",
        prepared.len(),
        hmd,
        offscreen,
        main,
        details,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render_graph::RenderPathProfile;
    use crate::render_graph::compiled::RenderPathProfileId;

    #[test]
    fn desktop_request_includes_main_view_on_desktop_schedule() {
        let request = FrameViewFamilyRequest::desktop();

        assert_eq!(request.schedule_kind(), RenderScheduleKind::Desktop);
        assert!(request.includes_main_view());
        assert_eq!(
            request
                .fallback_frame_global_profile(RenderPathProfile::desktop_main())
                .id(),
            RenderPathProfileId::DesktopMain
        );
    }

    #[test]
    fn desktop_secondaries_only_keeps_desktop_schedule() {
        let request = FrameViewFamilyRequest::desktop_secondaries_only();

        assert_eq!(request.schedule_kind(), RenderScheduleKind::Desktop);
        assert_eq!(
            CpuRenderSchedule::new(request.schedule_kind()).mesh_lod_bias(),
            2.0
        );
        assert!(!request.includes_main_view());
        assert_eq!(
            request
                .fallback_frame_global_profile(RenderPathProfile::headless_main())
                .id(),
            RenderPathProfileId::HeadlessMain
        );
    }

    #[test]
    fn vr_secondaries_only_keeps_vr_schedule_and_global_profile() {
        let request = FrameViewFamilyRequest::vr_secondaries_only();

        assert_eq!(
            request.schedule_kind(),
            RenderScheduleKind::VrSecondariesOnly
        );
        assert!(!request.includes_main_view());
        assert_eq!(
            request
                .fallback_frame_global_profile(RenderPathProfile::desktop_main())
                .id(),
            RenderPathProfileId::XrHmd
        );
    }
}
