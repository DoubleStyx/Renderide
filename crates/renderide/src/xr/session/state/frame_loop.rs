//! OpenXR frame wait, view location, and pre-begin synchronisation with deferred finalize.
//!
//! `xrEndFrame` for the previous tick runs on the renderer's driver thread (see
//! [`crate::gpu::driver_thread::run_xr_finalize`]). [`XrSessionState::wait_frame`] consumes
//! the matching finalize signal before issuing `xrBeginFrame` so the OpenXR begin/end
//! ordering invariant is preserved across the deferred handoff.

use std::sync::atomic::Ordering;
use std::sync::mpsc::RecvTimeoutError;
use std::time::Duration;

use openxr as xr;

use super::XrSessionState;
use crate::diagnostics::gpu_flight_recorder::{
    GpuFlightCallResult, GpuFlightOpenXrCall, GpuFlightRecorder,
};
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
        flight_recorder: &GpuFlightRecorder,
    ) -> Result<Option<xr::FrameState>, xr::sys::Result> {
        self.wait_previous_finalize_if_pending(flight_recorder);
        if let Some(err) = self.take_finalize_error() {
            flight_recorder.record_openxr_call_result(
                GpuFlightOpenXrCall::WaitFrame,
                GpuFlightCallResult::failed_debug(err),
                None,
                None,
            );
            return Err(err);
        }
        if !self.session_running {
            std::thread::sleep(Duration::from_millis(10));
            return Ok(None);
        }
        let state = self.wait_openxr_frame(flight_recorder)?;
        self.begin_openxr_frame(gpu_queue_access_gate, flight_recorder, state)?;
        self.frame_open.store(true, Ordering::Release);
        Ok(Some(state))
    }

    /// Waits for the previous driver-thread finalize signal when one is pending.
    fn wait_previous_finalize_if_pending(&mut self, flight_recorder: &GpuFlightRecorder) {
        if let Some(rx) = self.pending_finalize.take() {
            profiling::scope!("xr::wait_previous_finalize");
            flight_recorder.record_openxr_call_started(
                GpuFlightOpenXrCall::WaitPreviousFinalize,
                None,
                None,
            );
            // Timeout means the driver thread is unresponsive; existing
            // `take_pending_error` plumbing surfaces driver crashes separately so we
            // log here and fall through to the error-slot drain below.
            match wait_for_finalize(rx) {
                Ok(()) => {
                    flight_recorder.record_openxr_call_result(
                        GpuFlightOpenXrCall::WaitPreviousFinalize,
                        GpuFlightCallResult::Ok,
                        None,
                        None,
                    );
                }
                Err(RecvTimeoutError::Timeout) => {
                    flight_recorder.record_openxr_call_result(
                        GpuFlightOpenXrCall::WaitPreviousFinalize,
                        GpuFlightCallResult::TimedOut("previous_finalize_timeout".to_owned()),
                        None,
                        None,
                    );
                    logger::warn!(
                        "xr: timed out waiting for previous-frame finalize (session_running={} frame_open={})",
                        self.session_running,
                        self.frame_open.load(Ordering::Acquire)
                    );
                }
                Err(RecvTimeoutError::Disconnected) => {
                    flight_recorder.record_openxr_call_result(
                        GpuFlightOpenXrCall::WaitPreviousFinalize,
                        GpuFlightCallResult::failed_static("previous_finalize_disconnected"),
                        None,
                        None,
                    );
                    logger::warn!(
                        "xr: previous-frame finalize disconnected (session_running={} frame_open={})",
                        self.session_running,
                        self.frame_open.load(Ordering::Acquire)
                    );
                }
            }
        }
    }

    /// Runs `xrWaitFrame` and records the active call in diagnostics.
    fn wait_openxr_frame(
        &mut self,
        flight_recorder: &GpuFlightRecorder,
    ) -> Result<xr::FrameState, xr::sys::Result> {
        flight_recorder.record_openxr_call_started(GpuFlightOpenXrCall::WaitFrame, None, None);
        match self.frame_wait.wait() {
            Ok(state) => {
                flight_recorder.record_openxr_call_result(
                    GpuFlightOpenXrCall::WaitFrame,
                    GpuFlightCallResult::Ok,
                    None,
                    Some(state.predicted_display_time.as_nanos()),
                );
                Ok(state)
            }
            Err(error) => {
                flight_recorder.record_openxr_call_result(
                    GpuFlightOpenXrCall::WaitFrame,
                    GpuFlightCallResult::failed_debug(error),
                    None,
                    None,
                );
                Err(error)
            }
        }
    }

    /// Runs `xrBeginFrame` under the GPU queue gate and records diagnostics.
    fn begin_openxr_frame(
        &self,
        gpu_queue_access_gate: &crate::gpu::GpuQueueAccessGate,
        flight_recorder: &GpuFlightRecorder,
        state: xr::FrameState,
    ) -> Result<(), xr::sys::Result> {
        profiling::scope!("xr::frame_stream_begin");
        flight_recorder.record_openxr_call_started(
            GpuFlightOpenXrCall::BeginFrame,
            None,
            Some(state.predicted_display_time.as_nanos()),
        );
        let _gate = gpu_queue_access_gate.lock();
        let begin_result = self.frame_stream.lock().begin();
        let begin_flight_result = begin_result.as_ref().map_or_else(
            |error| GpuFlightCallResult::failed_debug(*error),
            |()| GpuFlightCallResult::Ok,
        );
        flight_recorder.record_openxr_call_result(
            GpuFlightOpenXrCall::BeginFrame,
            begin_flight_result,
            None,
            Some(state.predicted_display_time.as_nanos()),
        );
        begin_result
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
    /// holding `xr::FrameStream` / `xr::Swapchain` references. Bounded by
    /// [`AWAIT_FINALIZE_SHUTDOWN_TIMEOUT`] so a hung compositor cannot stall the Drop
    /// chain past the main-thread watchdog threshold.
    pub(in crate::xr) fn await_finalize_pending(&mut self) {
        if let Some(rx) = self.pending_finalize.take()
            && rx.recv_timeout(AWAIT_FINALIZE_SHUTDOWN_TIMEOUT).is_err()
        {
            logger::warn!(
                "xr: shutdown finalize wait timed out after {} ms; proceeding without driver-thread ack (session_running={} frame_open={})",
                AWAIT_FINALIZE_SHUTDOWN_TIMEOUT.as_millis(),
                self.session_running,
                self.frame_open.load(Ordering::Acquire)
            );
        }
    }
}

/// Upper bound on how long [`XrSessionState::await_finalize_pending`] will block during
/// shutdown. The cooperative graceful-shutdown drain already bounds the polling loop at
/// `GRACEFUL_SHUTDOWN_TIMEOUT`; this guards the unconditional wait inside Drop so the
/// main thread cannot park here past the watchdog's hang threshold.
const AWAIT_FINALIZE_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(1);
