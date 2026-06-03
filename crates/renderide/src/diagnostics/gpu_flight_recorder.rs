//! Bounded in-memory GPU and OpenXR event recorder for crash diagnostics.

use std::collections::VecDeque;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use crate::diagnostics::crash_context;

/// Number of recent GPU/XR events retained in memory.
pub(crate) const GPU_FLIGHT_RECORDER_CAPACITY: usize = 512;
/// Maximum number of events emitted when a crash path dumps the recorder.
pub(crate) const GPU_FLIGHT_RECORDER_DUMP_LIMIT: usize = 128;

/// In-memory ring buffer for recent GPU and OpenXR lifecycle events.
pub(crate) struct GpuFlightRecorder {
    /// Instant used as the zero point for event offsets.
    started_at: Instant,
    /// Monotonic event ID assigned before each event enters the ring.
    next_id: AtomicU64,
    /// One-shot guard for crash dumps.
    dumped: AtomicBool,
    /// Maximum event count kept in the ring.
    capacity: usize,
    /// Maximum event count printed in a dump.
    dump_limit: usize,
    /// Fixed-capacity event ring protected by a small mutex.
    events: Mutex<VecDeque<GpuFlightEvent>>,
}

impl GpuFlightRecorder {
    /// Creates a recorder with the production retention and dump limits.
    pub(crate) fn new() -> Self {
        Self::with_limits(GPU_FLIGHT_RECORDER_CAPACITY, GPU_FLIGHT_RECORDER_DUMP_LIMIT)
    }

    /// Creates a recorder with explicit limits for tests.
    fn with_limits(capacity: usize, dump_limit: usize) -> Self {
        Self {
            started_at: Instant::now(),
            next_id: AtomicU64::new(0),
            dumped: AtomicBool::new(false),
            capacity,
            dump_limit,
            events: Mutex::new(VecDeque::with_capacity(capacity)),
        }
    }

    /// Adds one event to the bounded ring.
    pub(crate) fn record(&self, kind: GpuFlightEventKind) {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let event = GpuFlightEvent {
            id,
            elapsed: self.started_at.elapsed(),
            kind,
        };
        let mut events = self.events.lock();
        if events.len() == self.capacity {
            events.pop_front();
        }
        events.push_back(event);
    }

    /// Records that an OpenXR call has started and marks it active in crash context.
    pub(crate) fn record_openxr_call_started(
        &self,
        call: GpuFlightOpenXrCall,
        image_index: Option<u32>,
        predicted_display_time_nanos: Option<i64>,
    ) {
        crash_context::set_openxr_call(call.crash_context_call());
        self.record(GpuFlightEventKind::OpenXrCall {
            call,
            result: GpuFlightCallResult::Started,
            image_index,
            predicted_display_time_nanos,
        });
    }

    /// Records that an OpenXR call completed and clears the active crash-context call.
    pub(crate) fn record_openxr_call_result(
        &self,
        call: GpuFlightOpenXrCall,
        result: GpuFlightCallResult,
        image_index: Option<u32>,
        predicted_display_time_nanos: Option<i64>,
    ) {
        self.record(GpuFlightEventKind::OpenXrCall {
            call,
            result,
            image_index,
            predicted_display_time_nanos,
        });
        crash_context::clear_openxr_call_if(call.crash_context_call());
    }

    /// Records that an OpenXR call exceeded its watchdog timeout while still active.
    pub(crate) fn record_openxr_call_timeout(
        &self,
        call: GpuFlightOpenXrCall,
        reason: impl Into<String>,
        image_index: Option<u32>,
        predicted_display_time_nanos: Option<i64>,
    ) {
        self.record(GpuFlightEventKind::OpenXrCall {
            call,
            result: GpuFlightCallResult::TimedOut(reason.into()),
            image_index,
            predicted_display_time_nanos,
        });
    }

    /// Emits the recent event timeline once, returning whether a dump was written.
    pub(crate) fn dump_once(&self, reason: &'static str) -> bool {
        if self.dumped.swap(true, Ordering::AcqRel) {
            return false;
        }
        let snapshot = self.snapshot_tail(self.dump_limit);
        logger::error!(
            "GPU flight recorder dump: reason={} retained={} showing_last={}",
            reason,
            self.retained_len(),
            snapshot.len(),
        );
        for event in snapshot {
            logger::error!(
                "GPU flight event id={} +{:.3}ms {}",
                event.id,
                event.elapsed.as_secs_f64() * 1000.0,
                event.kind,
            );
        }
        true
    }

    /// Returns the current number of retained events.
    fn retained_len(&self) -> usize {
        self.events.lock().len()
    }

    /// Clones the newest `limit` events in chronological order.
    fn snapshot_tail(&self, limit: usize) -> Vec<GpuFlightEvent> {
        let events = self.events.lock();
        let start = events.len().saturating_sub(limit);
        events.iter().skip(start).cloned().collect()
    }

    /// Returns formatted dump lines once for tests.
    #[cfg(test)]
    fn dump_lines_once_for_test(&self, reason: &'static str) -> Option<Vec<String>> {
        if self.dumped.swap(true, Ordering::AcqRel) {
            return None;
        }
        let snapshot = self.snapshot_tail(self.dump_limit);
        let mut lines = Vec::with_capacity(snapshot.len() + 1);
        lines.push(format!(
            "GPU flight recorder dump: reason={} retained={} showing_last={}",
            reason,
            self.retained_len(),
            snapshot.len()
        ));
        lines.extend(snapshot.into_iter().map(|event| {
            format!(
                "GPU flight event id={} +{:.3}ms {}",
                event.id,
                event.elapsed.as_secs_f64() * 1000.0,
                event.kind
            )
        }));
        Some(lines)
    }
}

impl Default for GpuFlightRecorder {
    fn default() -> Self {
        Self::new()
    }
}

/// One retained GPU/XR event.
#[derive(Clone, Debug)]
struct GpuFlightEvent {
    /// Monotonic event identifier.
    id: u64,
    /// Elapsed time since recorder creation.
    elapsed: Duration,
    /// Event payload.
    kind: GpuFlightEventKind,
}

/// Compact GPU/XR event payload.
#[derive(Clone, Debug)]
pub(crate) enum GpuFlightEventKind {
    /// wgpu reported device loss from the device callback.
    DeviceLost {
        /// Device-loss generation assigned by `GpuDeviceHealth`.
        generation: u64,
        /// Adapter name captured at device creation.
        adapter_name: String,
        /// Adapter backend captured at device creation.
        backend: wgpu::Backend,
        /// wgpu loss reason.
        reason: String,
        /// wgpu loss message.
        message: String,
    },
    /// The app observed a new device-loss generation and is starting shutdown.
    DeviceLossObserved {
        /// Device-loss generation observed by the app loop.
        generation: u64,
        /// Active adapter name.
        adapter_name: String,
        /// Active backend.
        backend: wgpu::Backend,
    },
    /// Surface acquisition completed or failed.
    SurfaceAcquire {
        /// Surface acquire site.
        site: GpuFlightSurfaceSite,
        /// Acquire result.
        outcome: GpuFlightSurfaceAcquireOutcome,
        /// Current surface extent in pixels.
        extent: (u32, u32),
        /// Active present mode.
        present_mode: wgpu::PresentMode,
    },
    /// Surface reconfiguration completed, failed, or was skipped.
    SurfaceReconfigure {
        /// Surface reconfigure site.
        site: GpuFlightSurfaceReconfigureSite,
        /// Reconfigure result.
        outcome: GpuFlightSurfaceReconfigureOutcome,
        /// Previous extent in pixels.
        old_extent: (u32, u32),
        /// New extent in pixels.
        new_extent: (u32, u32),
        /// Active surface format.
        format: wgpu::TextureFormat,
        /// Active present mode.
        present_mode: wgpu::PresentMode,
    },
    /// Surface-carrying submit was enqueued.
    SurfaceSubmit {
        /// Surface submit site.
        site: GpuFlightSurfaceSubmitSite,
        /// Command buffers in the batch.
        command_buffers: usize,
        /// Frame sequence assigned by frame timing, or zero when untracked.
        frame_seq: u64,
    },
    /// Driver-thread batch event.
    Driver {
        /// Driver stage.
        stage: GpuFlightDriverStage,
        /// Frame sequence assigned by frame timing, or zero when untracked.
        frame_seq: u64,
        /// Command buffers in the batch.
        command_buffers: usize,
        /// Whether this batch carries a surface texture.
        has_surface: bool,
        /// Whether this batch carries OpenXR finalize work.
        has_xr_finalize: bool,
        /// Driver ring depth after the relevant action.
        ring_depth: usize,
        /// Driver backlog after the relevant action.
        backlog: u64,
        /// Optional result for stages that can fail.
        result: GpuFlightCallResult,
    },
    /// OpenXR call event.
    OpenXrCall {
        /// OpenXR call name.
        call: GpuFlightOpenXrCall,
        /// Call result.
        result: GpuFlightCallResult,
        /// Swapchain image index when the call returned one.
        image_index: Option<u32>,
        /// Predicted display time in OpenXR nanoseconds.
        predicted_display_time_nanos: Option<i64>,
    },
    /// Render-graph submit summary.
    RenderGraphSubmit {
        /// Whether the submit targets the window swapchain.
        swapchain: bool,
        /// Number of views in the submit.
        views: usize,
        /// Command buffers in the submit.
        command_buffers: usize,
        /// World draw item count.
        draw_items: usize,
        /// Pipeline pass submits recorded by world draw code.
        pipeline_pass_submits: usize,
        /// Upload bytes drained before submit.
        upload_bytes: usize,
        /// Transient texture misses this frame.
        transient_texture_misses: usize,
        /// Transient buffer misses this frame.
        transient_buffer_misses: usize,
    },
}

impl fmt::Display for GpuFlightEventKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeviceLost {
                generation,
                adapter_name,
                backend,
                reason,
                message,
            } => write!(
                f,
                "device_lost generation={generation} adapter=\"{adapter_name}\" backend={backend:?} reason={reason} message=\"{message}\""
            ),
            Self::DeviceLossObserved {
                generation,
                adapter_name,
                backend,
            } => write!(
                f,
                "device_loss_observed generation={generation} adapter=\"{adapter_name}\" backend={backend:?}"
            ),
            Self::SurfaceAcquire {
                site,
                outcome,
                extent,
                present_mode,
            } => write!(
                f,
                "surface_acquire site={site} outcome={outcome} extent={}x{} present_mode={present_mode:?}",
                extent.0, extent.1
            ),
            Self::SurfaceReconfigure {
                site,
                outcome,
                old_extent,
                new_extent,
                format,
                present_mode,
            } => write!(
                f,
                "surface_reconfigure site={site} outcome={outcome} old_extent={}x{} new_extent={}x{} format={format:?} present_mode={present_mode:?}",
                old_extent.0, old_extent.1, new_extent.0, new_extent.1
            ),
            Self::SurfaceSubmit {
                site,
                command_buffers,
                frame_seq,
            } => write!(
                f,
                "surface_submit site={site} command_buffers={command_buffers} frame_seq={frame_seq}"
            ),
            Self::Driver {
                stage,
                frame_seq,
                command_buffers,
                has_surface,
                has_xr_finalize,
                ring_depth,
                backlog,
                result,
            } => write!(
                f,
                "driver stage={stage} frame_seq={frame_seq} command_buffers={command_buffers} has_surface={has_surface} has_xr_finalize={has_xr_finalize} ring_depth={ring_depth} backlog={backlog} result={result}"
            ),
            Self::OpenXrCall {
                call,
                result,
                image_index,
                predicted_display_time_nanos,
            } => write!(
                f,
                "openxr call={call} result={result} image_index={} predicted_time_ns={}",
                optional_u32(*image_index),
                optional_i64(*predicted_display_time_nanos)
            ),
            Self::RenderGraphSubmit {
                swapchain,
                views,
                command_buffers,
                draw_items,
                pipeline_pass_submits,
                upload_bytes,
                transient_texture_misses,
                transient_buffer_misses,
            } => write!(
                f,
                "render_graph_submit swapchain={swapchain} views={views} command_buffers={command_buffers} draw_items={draw_items} pipeline_pass_submits={pipeline_pass_submits} upload_bytes={upload_bytes} transient_misses(tex/buf)={transient_texture_misses}/{transient_buffer_misses}"
            ),
        }
    }
}

/// Surface acquisition site.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GpuFlightSurfaceSite {
    /// Desktop render-graph swapchain acquire.
    DesktopGraph,
    /// Terminal desktop acquire for presenting the offscreen final target.
    DesktopFinalBlit,
    /// Desktop acquire for the host `BlitToDisplay` pass.
    DesktopBlitToDisplay,
    /// VR mirror blit acquire.
    VrMirror,
    /// VR clear fallback acquire.
    VrClear,
    /// Generic clear fallback acquire.
    ClearFallback,
}

impl fmt::Display for GpuFlightSurfaceSite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::DesktopGraph => "desktop_graph",
            Self::DesktopFinalBlit => "desktop_final_blit",
            Self::DesktopBlitToDisplay => "desktop_blit_to_display",
            Self::VrMirror => "vr_mirror",
            Self::VrClear => "vr_clear",
            Self::ClearFallback => "clear_fallback",
        })
    }
}

/// Surface acquire outcome.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GpuFlightSurfaceAcquireOutcome {
    /// A surface texture was acquired.
    Acquired,
    /// The acquire was skipped because the surface timed out or was occluded.
    Skipped,
    /// The surface was reconfigured and the frame was skipped.
    Reconfigured,
    /// The acquire failed with a current-surface status.
    Failed(GpuFlightSurfaceStatus),
}

impl fmt::Display for GpuFlightSurfaceAcquireOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Acquired => f.write_str("acquired"),
            Self::Skipped => f.write_str("skipped"),
            Self::Reconfigured => f.write_str("reconfigured"),
            Self::Failed(status) => write!(f, "failed_{status:?}"),
        }
    }
}

/// Current-surface texture status without owning a surface texture.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GpuFlightSurfaceStatus {
    /// Surface acquire timed out.
    Timeout,
    /// Window is occluded.
    Occluded,
    /// Surface configuration is outdated.
    Outdated,
    /// Surface is lost.
    Lost,
    /// Surface validation failed.
    Validation,
}

/// Surface reconfigure site.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GpuFlightSurfaceReconfigureSite {
    /// Resize or acquire-recovery reconfigure.
    Resize,
    /// Present-mode change reconfigure.
    PresentMode,
}

impl fmt::Display for GpuFlightSurfaceReconfigureSite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Resize => "resize",
            Self::PresentMode => "present_mode",
        })
    }
}

/// Surface reconfigure outcome.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum GpuFlightSurfaceReconfigureOutcome {
    /// Reconfigure succeeded.
    Succeeded,
    /// Reconfigure failed.
    Failed(String),
    /// Reconfigure was skipped because the device is already lost.
    SkippedDeviceLost,
}

impl fmt::Display for GpuFlightSurfaceReconfigureOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Succeeded => f.write_str("succeeded"),
            Self::Failed(error) => write!(f, "failed \"{error}\""),
            Self::SkippedDeviceLost => f.write_str("skipped_device_lost"),
        }
    }
}

/// Surface submit site.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GpuFlightSurfaceSubmitSite {
    /// Terminal desktop submit for presenting the offscreen final target.
    DesktopFinalBlit,
    /// Desktop submit for the host `BlitToDisplay` pass.
    DesktopBlitToDisplay,
    /// VR mirror blit submit.
    VrMirror,
    /// VR clear fallback submit.
    VrClear,
    /// Generic clear fallback submit.
    ClearFallback,
}

impl fmt::Display for GpuFlightSurfaceSubmitSite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::DesktopFinalBlit => "desktop_final_blit",
            Self::DesktopBlitToDisplay => "desktop_blit_to_display",
            Self::VrMirror => "vr_mirror",
            Self::VrClear => "vr_clear",
            Self::ClearFallback => "clear_fallback",
        })
    }
}

/// Driver-thread stage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GpuFlightDriverStage {
    /// Batch was enqueued by the producer.
    Enqueued,
    /// Batch was dropped because the driver thread exited.
    DroppedAfterExit,
    /// Queue submit is about to run.
    SubmitStart,
    /// Queue submit returned.
    SubmitDone,
    /// Surface present is about to run.
    PresentStart,
    /// Surface present returned.
    PresentDone,
    /// OpenXR finalize is about to run.
    XrFinalizeStart,
    /// OpenXR finalize returned.
    XrFinalizeDone,
}

impl fmt::Display for GpuFlightDriverStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Enqueued => "enqueued",
            Self::DroppedAfterExit => "dropped_after_exit",
            Self::SubmitStart => "submit_start",
            Self::SubmitDone => "submit_done",
            Self::PresentStart => "present_start",
            Self::PresentDone => "present_done",
            Self::XrFinalizeStart => "xr_finalize_start",
            Self::XrFinalizeDone => "xr_finalize_done",
        })
    }
}

impl GpuFlightDriverStage {
    /// Returns the crash-context equivalent of this driver stage.
    pub(crate) const fn crash_context_stage(self) -> crash_context::DriverStage {
        match self {
            Self::Enqueued => crash_context::DriverStage::Enqueued,
            Self::DroppedAfterExit => crash_context::DriverStage::DroppedAfterExit,
            Self::SubmitStart => crash_context::DriverStage::SubmitStart,
            Self::SubmitDone => crash_context::DriverStage::SubmitDone,
            Self::PresentStart => crash_context::DriverStage::PresentStart,
            Self::PresentDone => crash_context::DriverStage::PresentDone,
            Self::XrFinalizeStart => crash_context::DriverStage::XrFinalizeStart,
            Self::XrFinalizeDone => crash_context::DriverStage::XrFinalizeDone,
        }
    }
}

/// OpenXR call recorded by the flight recorder.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GpuFlightOpenXrCall {
    /// `xrPollEvent` loop.
    PollEvents,
    /// Wait for previous deferred finalize signal.
    WaitPreviousFinalize,
    /// `xrWaitFrame`.
    WaitFrame,
    /// `xrBeginFrame`.
    BeginFrame,
    /// `xrLocateViews`.
    LocateViews,
    /// `xrAcquireSwapchainImage`.
    AcquireImage,
    /// `xrWaitSwapchainImage`.
    WaitImage,
    /// `xrReleaseSwapchainImage`.
    ReleaseImage,
    /// Projection `xrEndFrame`.
    EndFrameProjection,
    /// Empty `xrEndFrame`.
    EndFrameEmpty,
}

impl fmt::Display for GpuFlightOpenXrCall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::PollEvents => "poll_events",
            Self::WaitPreviousFinalize => "wait_previous_finalize",
            Self::WaitFrame => "wait_frame",
            Self::BeginFrame => "begin_frame",
            Self::LocateViews => "locate_views",
            Self::AcquireImage => "acquire_image",
            Self::WaitImage => "wait_image",
            Self::ReleaseImage => "release_image",
            Self::EndFrameProjection => "end_frame_projection",
            Self::EndFrameEmpty => "end_frame_empty",
        })
    }
}

impl GpuFlightOpenXrCall {
    /// Returns the crash-context equivalent of this OpenXR call.
    pub(crate) const fn crash_context_call(self) -> crash_context::OpenXrCall {
        match self {
            Self::PollEvents => crash_context::OpenXrCall::PollEvents,
            Self::WaitPreviousFinalize => crash_context::OpenXrCall::WaitPreviousFinalize,
            Self::WaitFrame => crash_context::OpenXrCall::WaitFrame,
            Self::BeginFrame => crash_context::OpenXrCall::BeginFrame,
            Self::LocateViews => crash_context::OpenXrCall::LocateViews,
            Self::AcquireImage => crash_context::OpenXrCall::AcquireImage,
            Self::WaitImage => crash_context::OpenXrCall::WaitImage,
            Self::ReleaseImage => crash_context::OpenXrCall::ReleaseImage,
            Self::EndFrameProjection => crash_context::OpenXrCall::EndFrameProjection,
            Self::EndFrameEmpty => crash_context::OpenXrCall::EndFrameEmpty,
        }
    }
}

/// Compact call result stored in the flight recorder.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum GpuFlightCallResult {
    /// Call has started but has not returned yet.
    Started,
    /// Call succeeded.
    Ok,
    /// Call intentionally skipped.
    Skipped,
    /// Call failed with a compact debug string.
    Failed(String),
    /// Call exceeded a watchdog timeout while still running.
    TimedOut(String),
}

impl GpuFlightCallResult {
    /// Creates a failed result from a debug-printable error.
    pub(crate) fn failed_debug(error: impl fmt::Debug) -> Self {
        Self::Failed(format!("{error:?}"))
    }

    /// Creates a failed result from a static reason.
    pub(crate) fn failed_static(reason: &'static str) -> Self {
        Self::Failed(reason.to_owned())
    }
}

impl fmt::Display for GpuFlightCallResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Started => f.write_str("started"),
            Self::Ok => f.write_str("ok"),
            Self::Skipped => f.write_str("skipped"),
            Self::Failed(error) => write!(f, "failed \"{error}\""),
            Self::TimedOut(reason) => write!(f, "timed_out \"{reason}\""),
        }
    }
}

/// Formats an optional unsigned integer for compact event output.
fn optional_u32(value: Option<u32>) -> String {
    value.map_or_else(|| "none".to_owned(), |v| v.to_string())
}

/// Formats an optional signed integer for compact event output.
fn optional_i64(value: Option<i64>) -> String {
    value.map_or_else(|| "none".to_owned(), |v| v.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_overwrites_oldest_events_and_keeps_monotonic_ids() {
        let recorder = GpuFlightRecorder::with_limits(3, 3);
        for i in 0..5 {
            recorder.record(GpuFlightEventKind::RenderGraphSubmit {
                swapchain: false,
                views: 1,
                command_buffers: i,
                draw_items: 0,
                pipeline_pass_submits: 0,
                upload_bytes: 0,
                transient_texture_misses: 0,
                transient_buffer_misses: 0,
            });
        }

        let events = recorder.snapshot_tail(10);
        let ids: Vec<u64> = events.iter().map(|event| event.id).collect();
        assert_eq!(ids, vec![2, 3, 4]);
        assert_eq!(events.len(), 3);
    }

    #[test]
    fn dump_lines_are_bounded_and_one_shot() {
        let recorder = GpuFlightRecorder::with_limits(8, 2);
        for _ in 0..4 {
            recorder.record(GpuFlightEventKind::OpenXrCall {
                call: GpuFlightOpenXrCall::WaitFrame,
                result: GpuFlightCallResult::Ok,
                image_index: None,
                predicted_display_time_nanos: None,
            });
        }

        let lines = recorder
            .dump_lines_once_for_test("test")
            .expect("first dump succeeds");
        assert_eq!(lines.len(), 3);
        assert!(lines[0].contains("showing_last=2"));
        assert!(recorder.dump_lines_once_for_test("test").is_none());
    }

    #[test]
    fn representative_events_format_compactly() {
        let surface = GpuFlightEventKind::SurfaceAcquire {
            site: GpuFlightSurfaceSite::DesktopGraph,
            outcome: GpuFlightSurfaceAcquireOutcome::Acquired,
            extent: (1920, 1080),
            present_mode: wgpu::PresentMode::Immediate,
        }
        .to_string();
        assert!(surface.contains("surface_acquire site=desktop_graph"));
        assert!(surface.contains("extent=1920x1080"));

        let driver = GpuFlightEventKind::Driver {
            stage: GpuFlightDriverStage::XrFinalizeDone,
            frame_seq: 7,
            command_buffers: 2,
            has_surface: false,
            has_xr_finalize: true,
            ring_depth: 1,
            backlog: 1,
            result: GpuFlightCallResult::failed_static("ERROR_RUNTIME_FAILURE"),
        }
        .to_string();
        assert!(driver.contains("driver stage=xr_finalize_done"));
        assert!(driver.contains("failed \"ERROR_RUNTIME_FAILURE\""));

        let openxr = GpuFlightEventKind::OpenXrCall {
            call: GpuFlightOpenXrCall::AcquireImage,
            result: GpuFlightCallResult::Ok,
            image_index: Some(2),
            predicted_display_time_nanos: Some(123),
        }
        .to_string();
        assert!(openxr.contains("openxr call=acquire_image"));
        assert!(openxr.contains("image_index=2"));

        let openxr_started = GpuFlightEventKind::OpenXrCall {
            call: GpuFlightOpenXrCall::EndFrameProjection,
            result: GpuFlightCallResult::Started,
            image_index: Some(1),
            predicted_display_time_nanos: Some(456),
        }
        .to_string();
        assert!(openxr_started.contains("result=started"));

        let openxr_timeout = GpuFlightEventKind::OpenXrCall {
            call: GpuFlightOpenXrCall::EndFrameProjection,
            result: GpuFlightCallResult::TimedOut("runtime_stall".to_owned()),
            image_index: Some(1),
            predicted_display_time_nanos: Some(456),
        }
        .to_string();
        assert!(openxr_timeout.contains("timed_out \"runtime_stall\""));

        let device = GpuFlightEventKind::DeviceLost {
            generation: 1,
            adapter_name: "adapter".to_owned(),
            backend: wgpu::Backend::Vulkan,
            reason: "Unknown".to_owned(),
            message: "lost".to_owned(),
        }
        .to_string();
        assert!(device.contains("device_lost generation=1"));
        assert!(device.contains("backend=Vulkan"));
    }
}
