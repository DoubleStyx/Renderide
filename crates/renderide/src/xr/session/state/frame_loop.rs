//! OpenXR frame wait, view location, and pre-begin synchronisation with deferred finalize.
//!
//! `xrEndFrame` for the previous tick runs on the renderer's driver thread (see
//! [`crate::gpu::driver_thread::run_xr_finalize`]). [`XrSessionState::wait_frame`] consumes
//! the matching finalize signal before issuing `xrBeginFrame` so the OpenXR begin/end
//! ordering invariant is preserved across the deferred handoff.

use std::sync::atomic::Ordering;
use std::time::Duration;

use openxr as xr;

use super::XrSessionState;
use crate::gpu::driver_thread::wait_for_finalize;

impl XrSessionState {
    /// Blocks until the next frame, begins the frame stream. Returns `None` if not ready or idle.
    ///
    /// Steps in order:
    /// 1. Drain any pending finalize signal from the previous tick. This is the one place
    ///    the main thread synchronises with the driver thread for VR finalize. In the
    ///    steady state the receiver is already signaled (an entire main-thread tick has
    ///    elapsed since the finalize was queued), so the wait costs nothing.
    /// 2. If the driver recorded a finalize error, surface it instead of beginning a new
    ///    frame. The existing recovery paths handle the failure one tick later.
    /// 3. Run the regular `xrWaitFrame` + `xrBeginFrame` sequence under the queue access
    ///    gate.
    ///
    /// On a successful `frame_stream.begin()` sets [`Self::frame_open`] (atomic, mirrored
    /// to the driver thread for the deferred end-frame to clear) so the outer loop knows
    /// a matching `end_frame_*` must be queued.
    pub fn wait_frame(
        &mut self,
        gpu_queue_access_gate: &crate::gpu::GpuQueueAccessGate,
    ) -> Result<Option<xr::FrameState>, xr::sys::Result> {
        if let Some(rx) = self.pending_finalize.take() {
            profiling::scope!("xr::wait_previous_finalize");
            // Timeout means the driver thread is unresponsive; existing
            // `take_pending_error` plumbing surfaces driver crashes separately so we
            // log here and fall through to the error-slot drain below.
            if wait_for_finalize(rx).is_err() {
                logger::warn!("xr: timed out waiting for previous-frame finalize");
            }
        }
        if let Some(err) = self.take_finalize_error() {
            return Err(err);
        }
        if !self.session_running {
            std::thread::sleep(Duration::from_millis(10));
            return Ok(None);
        }
        let state = self.frame_wait.wait()?;
        {
            profiling::scope!("xr::frame_stream_begin");
            let _gate = gpu_queue_access_gate.lock();
            self.frame_stream.lock().begin()?;
        };
        self.frame_open.store(true, Ordering::Release);
        Ok(Some(state))
    }

    /// Locates stereo views for the predicted display time.
    pub fn locate_views(
        &self,
        predicted_display_time: xr::Time,
    ) -> Result<Vec<xr::View>, xr::sys::Result> {
        let (_, views) = self.session.locate_views(
            xr::ViewConfigurationType::PRIMARY_STEREO,
            predicted_display_time,
            self.stage.as_ref(),
        )?;
        Ok(views)
    }

    /// Drains a pending finalize signal without beginning a new frame. Called from the
    /// shutdown path so we do not destroy the session while the driver thread is still
    /// holding `xr::FrameStream` / `xr::Swapchain` references.
    pub(in crate::xr) fn await_finalize_pending(&mut self) {
        if let Some(rx) = self.pending_finalize.take() {
            let _ = wait_for_finalize(rx);
        }
    }
}
