//! Render result presentation and display-blit dispatch for the app driver.

use crate::backend::RenderBackend;
use crate::diagnostics::DebugHudEncodeError;
use crate::gpu::GpuContext;
use crate::present::{
    SurfaceAcquireTrace, SurfaceSubmitTrace, present_clear_frame_overlay_traced,
    present_clear_frame_overlay_traced_with_color,
};
use crate::runtime::RendererRuntime;
use crate::runtime::display::DisplayBlitSource;
use crate::scene::DesktopDashboardOverlaySource;
use crate::shared::BlitToDisplayState;
use crate::xr::OpenxrFrameTick;
use glam::Vec4;

use super::AppDriver;

/// Presentation action implied by the frame render outcome.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PresentationAction {
    /// Present the normal desktop offscreen final target to the window surface.
    DesktopFinalBlit,
    /// Present the latest HMD eye staging texture to the desktop mirror surface.
    VrMirrorBlit,
    /// Clear the VR mirror surface because no HMD frame was submitted.
    VrClear,
    /// Run the desktop display blit pass for an explicit host `BlitToDisplay`.
    ///
    /// Implies that the world-camera path skipped the main desktop view this tick (`render_views`
    /// routed only secondary RTs to the GPU); this stage acquires the swapchain, clears it to
    /// `state.background_color`, and blits the `state.texture_id` source into the centered
    /// fitted rect.
    DesktopBlitToDisplay,
}

/// Presentation action and the optional source payloads needed to execute it.
///
/// `BlitToDisplayState` does not implement `Eq`/`PartialEq` because it carries an `f32` color, so
/// tests inspect the action and selected payloads explicitly.
#[derive(Clone, Copy, Debug)]
pub(super) struct PresentationPlan {
    action: PresentationAction,
    explicit_desktop_blit: Option<BlitToDisplayState>,
    dashboard_overlay: Option<DesktopDashboardOverlaySource>,
}

impl PresentationPlan {
    /// Builds a presentation plan from current VR state and HMD submission result.
    pub(super) const fn from_frame(vr_active: bool, hmd_projection_ended: bool) -> Self {
        if !vr_active {
            Self::desktop_final_blit(None)
        } else if hmd_projection_ended {
            Self::vr_mirror_blit()
        } else {
            Self::vr_clear()
        }
    }

    const fn desktop_final_blit(dashboard_overlay: Option<DesktopDashboardOverlaySource>) -> Self {
        Self {
            action: PresentationAction::DesktopFinalBlit,
            explicit_desktop_blit: None,
            dashboard_overlay,
        }
    }

    const fn desktop_blit_to_display(state: BlitToDisplayState) -> Self {
        Self {
            action: PresentationAction::DesktopBlitToDisplay,
            explicit_desktop_blit: Some(state),
            dashboard_overlay: None,
        }
    }

    const fn vr_mirror_blit() -> Self {
        Self {
            action: PresentationAction::VrMirrorBlit,
            explicit_desktop_blit: None,
            dashboard_overlay: None,
        }
    }

    const fn vr_clear() -> Self {
        Self {
            action: PresentationAction::VrClear,
            explicit_desktop_blit: None,
            dashboard_overlay: None,
        }
    }

    fn action(self) -> PresentationAction {
        self.action
    }

    fn explicit_desktop_blit(self) -> Option<BlitToDisplayState> {
        self.explicit_desktop_blit
    }

    fn dashboard_overlay(self) -> Option<DesktopDashboardOverlaySource> {
        self.dashboard_overlay
    }
}

fn presentation_plan_from_frame_and_desktop_blit(
    vr_active: bool,
    hmd_projection_ended: bool,
    explicit_desktop_blit: Option<BlitToDisplayState>,
    dashboard_overlay: Option<DesktopDashboardOverlaySource>,
) -> PresentationPlan {
    if !vr_active && let Some(state) = explicit_desktop_blit {
        return PresentationPlan::desktop_blit_to_display(state);
    }
    if !vr_active {
        return PresentationPlan::desktop_final_blit(dashboard_overlay);
    }
    PresentationPlan::from_frame(vr_active, hmd_projection_ended)
}

impl AppDriver {
    pub(super) fn present_and_diagnostics(
        &mut self,
        xr_tick: Option<OpenxrFrameTick>,
        hmd_projection_ended: bool,
    ) {
        profiling::scope!("tick::present_and_diagnostics");
        super::tick_phase_trace("present_and_diagnostics");
        let plan = self.compute_presentation_plan(hmd_projection_ended);
        self.present_plan(plan);
        if !hmd_projection_ended {
            self.queue_empty_openxr_frame_if_needed(xr_tick);
        }
    }

    /// Builds this tick's [`PresentationPlan`] from VR state, HMD submission, and any explicit
    /// desktop display blit source.
    fn compute_presentation_plan(&self, hmd_projection_ended: bool) -> PresentationPlan {
        let vr_active = self.runtime.vr_active();
        let scene = self.runtime.scene();
        presentation_plan_from_frame_and_desktop_blit(
            vr_active,
            hmd_projection_ended,
            scene.active_blit_for_display(super::DESKTOP_DISPLAY_INDEX),
            scene.active_desktop_dashboard_overlay_source(),
        )
    }

    fn present_plan(&mut self, plan: PresentationPlan) {
        match plan.action() {
            PresentationAction::DesktopFinalBlit => {
                self.present_desktop_final_blit(plan.dashboard_overlay());
            }
            PresentationAction::VrMirrorBlit => self.present_vr_mirror_blit(),
            PresentationAction::VrClear => self.present_vr_clear(),
            PresentationAction::DesktopBlitToDisplay => {
                if let Some(state) = plan.explicit_desktop_blit() {
                    self.present_desktop_blit_to_display(state);
                }
            }
        }
    }

    /// Acquires the desktop swapchain and runs the [`crate::gpu::DisplayBlitResources`] pass for
    /// the currently active local-user `BlitToDisplay`. If the source texture is not yet GPU
    /// resident the function falls back to a `present_clear_frame` with the same background color
    /// so the swapchain still receives a presentable frame this tick.
    fn present_desktop_blit_to_display(&mut self, state: BlitToDisplayState) {
        let Some(target) = self.target.as_mut() else {
            return;
        };
        let gpu = target.gpu_mut();
        let resolved = self
            .runtime
            .resolve_blit_to_display_texture(state.texture_id);
        let Some((view_arc, tex_w, tex_h)) = resolved else {
            // Texture not yet resident: still drive a present so the swapchain does not stall on
            // the previously-presented frame.
            let bg = state.background_color;
            let runtime = &mut self.runtime;
            let clear = wgpu::Color {
                r: bg.x as f64,
                g: bg.y as f64,
                b: bg.z as f64,
                a: bg.w as f64,
            };
            if let Err(error) = present_clear_frame_overlay_traced_with_color(
                gpu,
                SurfaceAcquireTrace::DesktopBlitToDisplay,
                SurfaceSubmitTrace::DesktopBlitToDisplay,
                clear,
                |encoder, view, gpu| encode_debug_hud_overlay(runtime, gpu, encoder, view),
            ) {
                logger::debug!("display blit fallback clear failed: {error:?}");
            }
            return;
        };
        let (blit, backend) = self.runtime.display_blit_and_backend_mut();
        let source = DisplayBlitSource {
            view: view_arc.as_ref(),
            width: tex_w,
            height: tex_h,
            flip_horizontally: blit_flag_set(state.flags, 0),
            flip_vertically: blit_flag_set(state.flags, 1),
            background_color: state.background_color,
        };
        if let Err(error) = blit.present_blit_to_surface(gpu, source, |encoder, view, gpu| {
            encode_debug_hud_overlay_via_backend(backend, gpu, encoder, view)
        }) {
            logger::debug!("display blit failed: {error:?}");
        }
    }

    /// Acquires the desktop swapchain and presents the offscreen final target rendered by the
    /// normal desktop world path.
    fn present_desktop_final_blit(
        &mut self,
        dashboard_overlay: Option<DesktopDashboardOverlaySource>,
    ) {
        let Some(target) = self.target.as_mut() else {
            return;
        };
        let gpu = target.gpu_mut();
        let Some((final_view, (width, height))) = gpu.primary_offscreen_color_source() else {
            let runtime = &mut self.runtime;
            if let Err(error) = present_clear_frame_overlay_traced(
                gpu,
                SurfaceAcquireTrace::DesktopFinalBlit,
                SurfaceSubmitTrace::DesktopFinalBlit,
                |encoder, view, gpu| encode_debug_hud_overlay(runtime, gpu, encoder, view),
            ) {
                logger::debug!("desktop final fallback clear failed: {error:?}");
            }
            return;
        };
        let resolved_dashboard_overlay = dashboard_overlay.and_then(|source| {
            self.runtime
                .resolve_blit_to_display_texture(source.texture_id)
        });
        if dashboard_overlay.is_some() && resolved_dashboard_overlay.is_none() {
            logger::trace!(
                "desktop dashboard overlay source texture not resident; skipping overlay composite"
            );
        }
        let (blit, backend) = self.runtime.display_blit_and_backend_mut();
        let source = DisplayBlitSource {
            view: &final_view,
            width,
            height,
            flip_horizontally: false,
            flip_vertically: false,
            background_color: Vec4::new(0.0, 0.0, 0.0, 1.0),
        };
        let overlay_source =
            resolved_dashboard_overlay
                .as_ref()
                .map(|(view_arc, width, height)| DisplayBlitSource {
                    view: view_arc.as_ref(),
                    width: *width,
                    height: *height,
                    flip_horizontally: false,
                    flip_vertically: false,
                    background_color: Vec4::ZERO,
                });
        if let Err(error) = blit.present_blit_to_surface_traced_with_overlay(
            gpu,
            source,
            overlay_source,
            SurfaceAcquireTrace::DesktopFinalBlit,
            SurfaceSubmitTrace::DesktopFinalBlit,
            |encoder, view, gpu| encode_debug_hud_overlay_via_backend(backend, gpu, encoder, view),
        ) {
            logger::debug!("desktop final blit failed: {error:?}");
        }
    }

    fn present_vr_mirror_blit(&mut self) {
        let Some(target) = self.target.as_mut() else {
            return;
        };
        let Some((gpu, session)) = target.openxr_parts_mut() else {
            return;
        };

        let runtime = &mut self.runtime;
        if let Err(error) = session
            .mirror_blit
            .present_staging_to_surface_overlay(gpu, |encoder, view, gpu| {
                encode_debug_hud_overlay(runtime, gpu, encoder, view)
            })
        {
            logger::debug!("VR mirror blit failed: {error:?}");
            let runtime = &mut self.runtime;
            if let Err(present_error) = present_clear_frame_overlay_traced(
                gpu,
                SurfaceAcquireTrace::VrClear,
                SurfaceSubmitTrace::VrClear,
                |encoder, view, gpu| encode_debug_hud_overlay(runtime, gpu, encoder, view),
            ) {
                logger::warn!("present_clear_frame after mirror blit: {present_error:?}");
            }
        }
    }

    fn present_vr_clear(&mut self) {
        let Some(target) = self.target.as_mut() else {
            return;
        };
        let gpu = target.gpu_mut();
        let runtime = &mut self.runtime;
        if let Err(error) = present_clear_frame_overlay_traced(
            gpu,
            SurfaceAcquireTrace::VrClear,
            SurfaceSubmitTrace::VrClear,
            |encoder, view, gpu| encode_debug_hud_overlay(runtime, gpu, encoder, view),
        ) {
            logger::debug!("VR mirror clear (no HMD frame): {error:?}");
        }
    }
}

fn encode_debug_hud_overlay(
    runtime: &mut RendererRuntime,
    gpu: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
) -> Result<(), DebugHudEncodeError> {
    runtime.encode_debug_hud_overlay_on_surface(gpu, encoder, view)
}

fn encode_debug_hud_overlay_via_backend(
    backend: &mut RenderBackend,
    gpu: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
) -> Result<(), DebugHudEncodeError> {
    if !backend.debug_hud_has_visible_content() {
        backend.clear_debug_hud_input_capture();
        return Ok(());
    }
    let device = gpu.device().as_ref();
    let extent = gpu.surface_extent_px();
    let queue = gpu.queue().as_ref();
    backend.encode_debug_hud_overlay(device, queue, encoder, view, extent, gpu.gpu_profiler())
}

/// Tests bit `bit_index` (zero-based) on a [`BlitToDisplayState::flags`] byte.
///
/// `bit 0` is horizontal flip and `bit 1` is vertical flip.
fn blit_flag_set(flags: u8, bit_index: u8) -> bool {
    (flags >> bit_index) & 1 != 0
}

#[cfg(test)]
mod tests {
    use super::{
        PresentationAction, PresentationPlan, presentation_plan_from_frame_and_desktop_blit,
    };
    use crate::scene::DesktopDashboardOverlaySource;
    use crate::shared::BlitToDisplayState;
    use glam::Vec4;

    fn test_blit_state(texture_id: i32) -> BlitToDisplayState {
        BlitToDisplayState {
            renderable_index: 0,
            texture_id,
            background_color: Vec4::new(0.0, 0.0, 0.0, 1.0),
            display_index: 0,
            flags: 0,
            _padding: [0; 1],
        }
    }

    fn dashboard_source(texture_id: i32) -> DesktopDashboardOverlaySource {
        DesktopDashboardOverlaySource {
            texture_id,
            render_texture_asset_id: texture_id,
        }
    }

    #[test]
    fn desktop_uses_final_blit_presentation() {
        let normal = PresentationPlan::from_frame(false, false);
        assert_eq!(normal.action(), PresentationAction::DesktopFinalBlit);
        assert!(normal.dashboard_overlay().is_none());

        let ended = PresentationPlan::from_frame(false, true);
        assert_eq!(ended.action(), PresentationAction::DesktopFinalBlit);
        assert!(ended.dashboard_overlay().is_none());
    }

    #[test]
    fn vr_hmd_submission_uses_mirror_blit() {
        assert_eq!(
            PresentationPlan::from_frame(true, true).action(),
            PresentationAction::VrMirrorBlit
        );
    }

    #[test]
    fn vr_without_hmd_submission_clears_mirror() {
        assert_eq!(
            PresentationPlan::from_frame(true, false).action(),
            PresentationAction::VrClear
        );
    }

    #[test]
    fn desktop_explicit_blit_owns_presentation() {
        let plan = presentation_plan_from_frame_and_desktop_blit(
            false,
            false,
            Some(test_blit_state(42)),
            Some(dashboard_source(90)),
        );

        assert_eq!(plan.action(), PresentationAction::DesktopBlitToDisplay);
        assert_eq!(
            plan.explicit_desktop_blit()
                .expect("explicit blit")
                .texture_id,
            42
        );
        assert!(plan.dashboard_overlay().is_none());
    }

    #[test]
    fn desktop_without_explicit_blit_uses_final_target() {
        let plan = presentation_plan_from_frame_and_desktop_blit(false, false, None, None);

        assert_eq!(plan.action(), PresentationAction::DesktopFinalBlit);
        assert!(plan.dashboard_overlay().is_none());
    }

    #[test]
    fn desktop_dashboard_overlay_composes_over_final_target() {
        let plan = presentation_plan_from_frame_and_desktop_blit(
            false,
            false,
            None,
            Some(dashboard_source(90)),
        );

        assert_eq!(plan.action(), PresentationAction::DesktopFinalBlit);
        assert_eq!(
            plan.dashboard_overlay()
                .expect("dashboard overlay")
                .texture_id,
            90
        );
    }

    #[test]
    fn vr_ignores_desktop_blit_for_presentation() {
        let plan = presentation_plan_from_frame_and_desktop_blit(
            true,
            false,
            Some(test_blit_state(42)),
            Some(dashboard_source(90)),
        );

        assert_eq!(plan.action(), PresentationAction::VrClear);
    }
}
