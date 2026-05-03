//! Render result presentation and swapchain recovery for the app driver.

use crate::gpu::GpuContext;
use crate::present::{
    SurfaceAcquireTrace, SurfaceSubmitTrace, present_clear_frame,
    present_clear_frame_overlay_traced,
};
use crate::render_graph::GraphExecuteError;
use crate::runtime::RendererRuntime;
use crate::xr::OpenxrFrameTick;

use super::AppDriver;

/// Presentation action implied by the frame render outcome.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PresentationPlan {
    /// No explicit app-side present step is needed.
    None,
    /// Present the latest HMD eye staging texture to the desktop mirror surface.
    VrMirrorBlit,
    /// Clear the VR mirror surface because no HMD frame was submitted.
    VrClear,
}

impl PresentationPlan {
    /// Builds a presentation plan from current VR state and HMD submission result.
    pub(super) const fn from_frame(vr_active: bool, hmd_projection_ended: bool) -> Self {
        if !vr_active {
            Self::None
        } else if hmd_projection_ended {
            Self::VrMirrorBlit
        } else {
            Self::VrClear
        }
    }
}

impl AppDriver {
    pub(super) fn present_and_diagnostics(
        &mut self,
        xr_tick: Option<OpenxrFrameTick>,
        hmd_projection_ended: bool,
    ) {
        profiling::scope!("tick::present_and_diagnostics");
        super::frame::tick_phase_trace("present_and_diagnostics");
        let plan = PresentationPlan::from_frame(self.runtime.vr_active(), hmd_projection_ended);
        self.present_vr_plan(plan);
        self.end_openxr_frame_if_needed(xr_tick, hmd_projection_ended);
    }

    fn present_vr_plan(&mut self, plan: PresentationPlan) {
        match plan {
            PresentationPlan::None => {}
            PresentationPlan::VrMirrorBlit => self.present_vr_mirror_blit(),
            PresentationPlan::VrClear => self.present_vr_clear(),
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

    fn end_openxr_frame_if_needed(
        &mut self,
        xr_tick: Option<OpenxrFrameTick>,
        hmd_projection_ended: bool,
    ) {
        let Some(tick) = xr_tick else {
            return;
        };
        if hmd_projection_ended {
            return;
        }
        let Some(target) = self.target.as_mut() else {
            return;
        };
        let Some((gpu, session)) = target.openxr_parts_mut() else {
            return;
        };
        // Atomic check is intentional: the driver thread clears `frame_open` from a deferred
        // finalize, so this read is safe without holding any session mutex.
        if !session.handles.xr_session.frame_open() {
            return;
        }
        profiling::scope!("xr::end_frame_if_open");
        let (finalize, rx) = session
            .handles
            .xr_session
            .build_empty_finalize(tick.predicted_display_time);
        gpu.submit_finalize_only(finalize);
        session.handles.xr_session.set_pending_finalize(rx);
    }

    pub(super) fn handle_frame_graph_error(&mut self, error: GraphExecuteError) {
        let Some(target) = self.target.as_mut() else {
            return;
        };
        if let GraphExecuteError::NoFrameGraph = error {
            if let Err(present_error) = present_clear_frame(target.gpu_mut()) {
                logger::warn!("present fallback failed: {present_error:?}");
                target.reconfigure_for_window();
            }
        } else {
            logger::warn!("frame graph failed: {error:?}");
            target.reconfigure_for_window();
        }
    }
}

fn encode_debug_hud_overlay(
    runtime: &mut RendererRuntime,
    gpu: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
) -> Result<(), crate::diagnostics::DebugHudEncodeError> {
    runtime.encode_debug_hud_overlay_on_surface(gpu, encoder, view)
}

#[cfg(test)]
mod tests {
    use super::PresentationPlan;

    #[test]
    fn desktop_needs_no_app_presentation() {
        assert_eq!(
            PresentationPlan::from_frame(false, false),
            PresentationPlan::None
        );
        assert_eq!(
            PresentationPlan::from_frame(false, true),
            PresentationPlan::None
        );
    }

    #[test]
    fn vr_hmd_submission_uses_mirror_blit() {
        assert_eq!(
            PresentationPlan::from_frame(true, true),
            PresentationPlan::VrMirrorBlit
        );
    }

    #[test]
    fn vr_without_hmd_submission_clears_mirror() {
        assert_eq!(
            PresentationPlan::from_frame(true, false),
            PresentationPlan::VrClear
        );
    }
}
