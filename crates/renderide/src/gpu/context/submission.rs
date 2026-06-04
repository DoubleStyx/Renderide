//! Submission facade methods on [`GpuContext`].
//!
//! All `Queue::submit` and `SurfaceTexture::present` calls flow through the dedicated
//! [`crate::gpu::driver_thread::DriverThread`]; these methods build
//! [`crate::gpu::driver_thread::SubmitBatch`] instances and hand them off. Primary render submits
//! attach frame-timing state here so the driver thread can update CPU/GPU intervals asynchronously.

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;

use super::GpuContext;

/// Purpose of a driver-thread submit for frame timing diagnostics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrameSubmitKind {
    /// Main render-graph work for the user-visible frame.
    PrimaryRender,
    /// Primary clear-only fallback when no normal render graph work is submitted.
    PrimaryClear,
    /// Offscreen or cache work that should be visible in detailed GPU profiler attribution but not
    /// counted as the primary frame's GPU busy time.
    BackgroundGpuWork,
    /// Mirror, compositor handoff, or other presentation work after the primary render.
    Presentation,
    /// OpenXR finalize-only work associated with compositor frame handoff.
    XrFinalize,
}

impl FrameSubmitKind {
    /// Returns `true` when the submit should contribute to compact CPU/GPU frame timing.
    const fn tracks_primary_frame_timing(self) -> bool {
        matches!(self, Self::PrimaryRender | Self::PrimaryClear)
    }

    /// Returns `true` when this submit should advance Tracy's logical render-submit frame track.
    const fn emits_render_submit_frame_mark(self) -> bool {
        self.tracks_primary_frame_timing()
    }
}

impl From<FrameSubmitKind> for crate::gpu::driver_thread::DriverSubmitKind {
    fn from(value: FrameSubmitKind) -> Self {
        match value {
            FrameSubmitKind::PrimaryRender => Self::PrimaryRender,
            FrameSubmitKind::PrimaryClear => Self::PrimaryClear,
            FrameSubmitKind::BackgroundGpuWork => Self::BackgroundGpuWork,
            FrameSubmitKind::Presentation => Self::Presentation,
            FrameSubmitKind::XrFinalize => Self::XrFinalize,
        }
    }
}

impl GpuContext {
    /// Returns a cloneable producer handle for background driver-thread submits.
    pub(crate) fn driver_submitter(&self) -> crate::gpu::driver_thread::DriverSubmitter {
        self.submission.driver_thread.submitter()
    }

    /// Hands a command-buffer batch off to the driver thread for submit + optional present.
    ///
    /// The surface texture is optional: pass `Some` for the main swapchain frame (the
    /// driver calls [`wgpu::SurfaceTexture::present`] after submit), `None` for batches
    /// that render to an offscreen target only. `kind` decides whether this batch contributes to
    /// compact primary frame timing. `wait` is an opaque oneshot used by
    /// synchronous callers (headless tests) that need to block until the driver has
    /// finished with this batch.
    pub fn submit_frame_batch(
        &self,
        kind: FrameSubmitKind,
        cmds: Vec<wgpu::CommandBuffer>,
        surface_texture: Option<wgpu::SurfaceTexture>,
        wait: Option<crate::gpu::driver_thread::SubmitWait>,
    ) {
        self.submit_frame_batch_inner(kind, cmds, surface_texture, wait, Vec::new());
    }

    /// Hands presentation-only work to the driver thread without updating compact frame timing.
    ///
    /// Use this for swapchain or compositor blits that are not the primary render workload for the
    /// tick. The GPU pass profiler still records any query scopes in the command buffers.
    pub(crate) fn submit_frame_batch_untracked(
        &self,
        kind: FrameSubmitKind,
        cmds: Vec<wgpu::CommandBuffer>,
        surface_texture: Option<wgpu::SurfaceTexture>,
        wait: Option<crate::gpu::driver_thread::SubmitWait>,
    ) {
        debug_assert!(
            !kind.tracks_primary_frame_timing(),
            "primary submits must go through submit_frame_batch"
        );
        self.submit_frame_batch_inner(kind, cmds, surface_texture, wait, Vec::new());
    }

    /// Same as [`Self::submit_frame_batch`] but attaches extra `on_submitted_work_done`
    /// callbacks that fire after the driver has submitted this batch to the queue.
    ///
    /// Use this to schedule main-thread work (e.g. `map_async` for Hi-Z readback) that
    /// depends on the submit having completed without paying a driver-ring flush.
    pub fn submit_frame_batch_with_callbacks(
        &self,
        kind: FrameSubmitKind,
        cmds: Vec<wgpu::CommandBuffer>,
        surface_texture: Option<wgpu::SurfaceTexture>,
        wait: Option<crate::gpu::driver_thread::SubmitWait>,
        extra_on_submitted_work_done: Vec<Box<dyn FnOnce() + Send + 'static>>,
    ) {
        self.submit_frame_batch_inner(
            kind,
            cmds,
            surface_texture,
            wait,
            extra_on_submitted_work_done,
        );
    }

    /// Submits final OpenXR copy work plus an OpenXR finalize payload.
    ///
    /// The command buffers still go through the normal driver thread and GPU profiler path, but
    /// they do not update the compact frame timing HUD. In VR, the HUD's GPU row should reflect the
    /// HMD multiview render graph submitted before this compositor handoff copy.
    pub fn submit_frame_batch_with_xr_finalize(
        &self,
        cmds: Vec<wgpu::CommandBuffer>,
        xr_finalize: crate::gpu::driver_thread::XrFinalizeWork,
    ) {
        self.submit_frame_batch_inner_full(
            FrameSubmitKind::XrFinalize,
            cmds,
            None,
            None,
            Vec::new(),
            Some(xr_finalize),
        );
    }

    /// Pushes a zero-work driver batch carrying only an OpenXR finalize payload.
    ///
    /// Used when the trailing render submit of the tick was skipped (HMD render bailed out
    /// after acquiring the swapchain image, or `end_frame_if_open` is closing an opened-
    /// but-not-rendered frame). The driver still needs to release the image (when applicable)
    /// and call `xrEndFrame`; the empty submit takes the queue access gate the same way a
    /// real submit would, preserving the OpenXR external-sync contract. Bypasses the
    /// frame-timing helper because there is no rendering work to attribute the GPU time to.
    pub fn submit_finalize_only(&self, mut xr_finalize: crate::gpu::driver_thread::XrFinalizeWork) {
        xr_finalize.set_submit_context(0, 0);
        let batch = crate::gpu::driver_thread::SubmitBatch {
            submit_kind: crate::gpu::driver_thread::DriverSubmitKind::XrFinalize,
            command_buffers: Vec::new(),
            surface_texture: None,
            on_submitted_work_done: Vec::new(),
            frame_timing: None,
            frame_bracket_readback: None,
            wait: None,
            xr_finalize: Some(xr_finalize),
            frame_seq: 0,
        };
        self.submission.driver_thread.submit(batch);
    }

    /// Internal helper that builds the [`crate::gpu::driver_thread::SubmitBatch`] and pushes it
    /// into the driver thread's ring. Blocks when the ring is full -- that block is the frame-pacing
    /// backpressure.
    fn submit_frame_batch_inner(
        &self,
        kind: FrameSubmitKind,
        command_buffers: Vec<wgpu::CommandBuffer>,
        surface_texture: Option<wgpu::SurfaceTexture>,
        wait: Option<crate::gpu::driver_thread::SubmitWait>,
        extra_on_submitted_work_done: Vec<Box<dyn FnOnce() + Send + 'static>>,
    ) {
        self.submit_frame_batch_inner_full(
            kind,
            command_buffers,
            surface_texture,
            wait,
            extra_on_submitted_work_done,
            None,
        );
    }

    /// Variant of [`Self::submit_frame_batch_inner`] that also accepts an OpenXR finalize
    /// payload. Kept private; public callers go through [`Self::submit_frame_batch`],
    /// [`Self::submit_frame_batch_with_callbacks`],
    /// [`Self::submit_frame_batch_with_xr_finalize`], or [`Self::submit_finalize_only`].
    fn submit_frame_batch_inner_full(
        &self,
        kind: FrameSubmitKind,
        mut command_buffers: Vec<wgpu::CommandBuffer>,
        surface_texture: Option<wgpu::SurfaceTexture>,
        wait: Option<crate::gpu::driver_thread::SubmitWait>,
        extra_on_submitted_work_done: Vec<Box<dyn FnOnce() + Send + 'static>>,
        mut xr_finalize: Option<crate::gpu::driver_thread::XrFinalizeWork>,
    ) {
        let has_gpu_work = !command_buffers.is_empty();
        if has_gpu_work && kind.emits_render_submit_frame_mark() {
            crate::profiling::emit_render_submit_frame_mark();
        }
        let track = if kind.tracks_primary_frame_timing() && has_gpu_work {
            let mut ft = self
                .submission
                .frame_timing
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            ft.on_before_tracked_submit()
        } else {
            None
        };
        let frame_timing = track.map(|(generation, seq, frame_start)| {
            crate::gpu::frame_cpu_gpu_timing::FrameTimingTrack {
                handle: Arc::clone(&self.submission.frame_timing),
                generation,
                seq,
                frame_start,
            }
        });
        // Only bracket primary submits with non-empty work -- empty submits (driver flush
        // sentinels) have no GPU time to measure, and non-primary submits have no HUD slot.
        let frame_bracket_readback = if track.is_some()
            && !command_buffers.is_empty()
            && !self.avoid_mapped_buffers_this_frame()
        {
            self.submission.frame_bracket.open_session().map(|session| {
                let begin = session.begin_command_buffer();
                let end = session.end_command_buffer();
                command_buffers.insert(0, begin);
                command_buffers.push(end);
                session.into_readback()
            })
        } else {
            None
        };
        let frame_seq = track.map_or(0, |(_, seq, _)| u64::from(seq));
        if let Some(finalize) = xr_finalize.as_mut() {
            finalize.set_submit_context(frame_seq, command_buffers.len());
        }
        let batch = crate::gpu::driver_thread::SubmitBatch {
            submit_kind: kind.into(),
            command_buffers,
            surface_texture,
            on_submitted_work_done: extra_on_submitted_work_done,
            frame_timing,
            frame_bracket_readback,
            wait,
            xr_finalize,
            frame_seq,
        };
        if let Some(token) = self.submission.driver_thread.submit(batch) {
            self.submission
                .last_frame_submit_token
                .store(token.raw(), Ordering::Release);
        }
    }

    /// Drains any driver-thread error captured since the last check, leaving the slot empty.
    ///
    /// Call once per tick from the frame epilogue; route the returned error through the
    /// existing device-recovery path (same as a swapchain `SurfaceError::Lost`).
    pub fn take_driver_error(&self) -> Option<crate::gpu::driver_thread::DriverError> {
        self.submission.driver_thread.take_pending_error()
    }

    /// Snapshot of the driver-thread submit counters: `(pushed, done)`. The instantaneous
    /// gap is the number of batches still owed by the driver and is suitable for a Tracy
    /// plot. Cheap (two atomic loads); call once per tick from the frame epilogue.
    pub fn driver_submit_backlog(&self) -> u64 {
        let (pushed, done) = self.submission.driver_thread.submit_counter_snapshot();
        pushed.saturating_sub(done)
    }

    /// Blocks until the driver thread has processed every previously-submitted batch.
    ///
    /// Used by the headless readback path to establish ordering between the rendered
    /// frame's submit (which runs on the driver thread) and the readback copy (which
    /// runs on the main thread). Most code paths never need this.
    pub fn flush_driver(&self) {
        self.submission.driver_thread.flush();
    }

    /// Blocks only until the most recently submitted surface-carrying batch has reached
    /// [`wgpu::SurfaceTexture::present`] on the driver thread.
    ///
    /// Call this right before [`wgpu::Surface::get_current_texture`] to honour wgpu's
    /// "only one outstanding surface texture" rule without flushing the whole ring.
    /// Unlike [`Self::flush_driver`] this permits non-surface work (submits without a
    /// swapchain texture, [`wgpu::Queue::on_submitted_work_done`] callbacks) to remain
    /// pipelined alongside the next frame's CPU recording.
    pub fn wait_for_previous_present(&self) {
        let start = Instant::now();
        self.submission.driver_thread.wait_for_previous_present();
        let wait = start.elapsed();
        self.record_frame_timing_excluded_wait(wait);
        crate::profiling::plot_surface_previous_present_wait_ms(wait);
    }
}

#[cfg(test)]
mod tests {
    use super::FrameSubmitKind;

    #[test]
    fn render_submit_frame_mark_only_tracks_primary_frame_work() {
        assert!(FrameSubmitKind::PrimaryRender.emits_render_submit_frame_mark());
        assert!(FrameSubmitKind::PrimaryClear.emits_render_submit_frame_mark());
        assert!(!FrameSubmitKind::BackgroundGpuWork.emits_render_submit_frame_mark());
        assert!(!FrameSubmitKind::Presentation.emits_render_submit_frame_mark());
        assert!(!FrameSubmitKind::XrFinalize.emits_render_submit_frame_mark());
    }

    #[test]
    fn driver_submit_kind_labels_match_frame_submit_kind() {
        use crate::gpu::driver_thread::DriverSubmitKind;

        assert_eq!(
            DriverSubmitKind::from(FrameSubmitKind::PrimaryRender).label(),
            "primary_render"
        );
        assert_eq!(
            DriverSubmitKind::from(FrameSubmitKind::PrimaryClear).label(),
            "primary_clear"
        );
        assert_eq!(
            DriverSubmitKind::from(FrameSubmitKind::BackgroundGpuWork).label(),
            "background_gpu_work"
        );
        assert_eq!(
            DriverSubmitKind::from(FrameSubmitKind::Presentation).label(),
            "presentation"
        );
        assert_eq!(
            DriverSubmitKind::from(FrameSubmitKind::XrFinalize).label(),
            "xr_finalize"
        );
    }
}
