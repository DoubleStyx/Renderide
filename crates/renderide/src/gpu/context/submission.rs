//! Submission facade methods on [`GpuContext`].
//!
//! All `Queue::submit` and `SurfaceTexture::present` calls flow through the dedicated
//! [`crate::gpu::driver_thread::DriverThread`]; these methods build
//! [`crate::gpu::driver_thread::SubmitBatch`] instances and hand them off. The frame-timing
//! track is attached here so the driver thread can update CPU/GPU intervals asynchronously.

use std::sync::Arc;

use super::GpuContext;

impl GpuContext {
    /// Submits a single command buffer for this frame through the driver thread, tracked for
    /// the debug HUD frame timing HUD. No surface is presented on this path; the older callers
    /// (VR mirror eye-to-staging blit) will migrate to [`Self::submit_frame_batch`] in a
    /// follow-up.
    pub fn submit_tracked_frame_commands(&self, cmd: wgpu::CommandBuffer) {
        self.submit_frame_batch_inner(vec![cmd], None, None, Vec::new());
    }

    /// Same as [`Self::submit_tracked_frame_commands`] but accepts an externally-held
    /// [`wgpu::Queue`] reference. Retained for API compatibility with the pre-driver-thread
    /// call sites -- the reference is ignored because submit now always runs on the driver
    /// thread with its own cloned [`Arc<wgpu::Queue>`].
    pub fn submit_tracked_frame_commands_with_queue(
        &self,
        _queue: &wgpu::Queue,
        cmd: wgpu::CommandBuffer,
    ) {
        self.submit_frame_batch_inner(vec![cmd], None, None, Vec::new());
    }

    /// Submits multiple command buffers through the driver thread in a single
    /// [`wgpu::Queue::submit`] call, tracked for frame timing. No surface is presented on
    /// this path -- for swapchain frames use [`Self::submit_frame_batch`] with a
    /// [`wgpu::SurfaceTexture`].
    ///
    /// All `Queue::write_buffer` calls on the main thread must have occurred before this
    /// call so they are visible to GPU commands in the same submit (wgpu guarantees this
    /// ordering regardless of which thread performs the submit).
    pub fn submit_tracked_frame_commands_batch(
        &self,
        _queue: &wgpu::Queue,
        cmds: impl IntoIterator<Item = wgpu::CommandBuffer>,
    ) {
        self.submit_frame_batch_inner(cmds.into_iter().collect(), None, None, Vec::new());
    }

    /// Hands a finished frame off to the driver thread for submit + present.
    ///
    /// The surface texture is optional: pass `Some` for the main swapchain frame (the
    /// driver calls [`wgpu::SurfaceTexture::present`] after submit), `None` for frames
    /// that render to an offscreen target only. `wait` is an opaque oneshot used by
    /// synchronous callers (headless tests) that need to block until the driver has
    /// finished with this batch.
    pub fn submit_frame_batch(
        &self,
        cmds: Vec<wgpu::CommandBuffer>,
        surface_texture: Option<wgpu::SurfaceTexture>,
        wait: Option<crate::gpu::driver_thread::SubmitWait>,
    ) {
        self.submit_frame_batch_inner(cmds, surface_texture, wait, Vec::new());
    }

    /// Same as [`Self::submit_frame_batch`] but attaches extra `on_submitted_work_done`
    /// callbacks that fire after the driver has submitted this batch to the queue.
    ///
    /// Use this to schedule main-thread work (e.g. `map_async` for Hi-Z readback) that
    /// depends on the submit having completed without paying a driver-ring flush.
    pub fn submit_frame_batch_with_callbacks(
        &self,
        cmds: Vec<wgpu::CommandBuffer>,
        surface_texture: Option<wgpu::SurfaceTexture>,
        wait: Option<crate::gpu::driver_thread::SubmitWait>,
        extra_on_submitted_work_done: Vec<Box<dyn FnOnce() + Send + 'static>>,
    ) {
        self.submit_frame_batch_inner(cmds, surface_texture, wait, extra_on_submitted_work_done);
    }

    /// Internal helper that builds the [`crate::gpu::driver_thread::SubmitBatch`] (including the
    /// frame-timing track and an optional frame-bracket timestamp readback) and pushes it into
    /// the driver thread's ring. Blocks when the ring is full -- that block is the frame-pacing
    /// backpressure.
    fn submit_frame_batch_inner(
        &self,
        mut command_buffers: Vec<wgpu::CommandBuffer>,
        surface_texture: Option<wgpu::SurfaceTexture>,
        wait: Option<crate::gpu::driver_thread::SubmitWait>,
        extra_on_submitted_work_done: Vec<Box<dyn FnOnce() + Send + 'static>>,
    ) {
        if !command_buffers.is_empty() {
            crate::profiling::emit_render_submit_frame_mark();
        }
        let track = {
            let mut ft = self
                .submission
                .frame_timing
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            ft.on_before_tracked_submit()
        };
        let frame_timing = track.map(|(generation, seq, frame_start)| {
            crate::gpu::frame_cpu_gpu_timing::FrameTimingTrack {
                handle: Arc::clone(&self.submission.frame_timing),
                generation,
                seq,
                frame_start,
            }
        });
        // Only bracket tracked submits with non-empty work -- empty submits (driver flush
        // sentinels) have no GPU time to measure, and untracked submits have no HUD slot.
        let frame_bracket_readback = if track.is_some() && !command_buffers.is_empty() {
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
        let batch = crate::gpu::driver_thread::SubmitBatch {
            command_buffers,
            surface_texture,
            on_submitted_work_done: extra_on_submitted_work_done,
            frame_timing,
            frame_bracket_readback,
            wait,
            frame_seq,
        };
        self.submission.driver_thread.submit(batch);
    }

    /// Drains any driver-thread error captured since the last check, leaving the slot empty.
    ///
    /// Call once per tick from the frame epilogue; route the returned error through the
    /// existing device-recovery path (same as a swapchain `SurfaceError::Lost`).
    pub fn take_driver_error(&self) -> Option<crate::gpu::driver_thread::DriverError> {
        self.submission.driver_thread.take_pending_error()
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
        self.submission.driver_thread.wait_for_previous_present();
    }
}
