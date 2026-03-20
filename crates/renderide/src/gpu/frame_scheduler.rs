//! Tracks in-flight queue submissions so uniform ring buffer slots are not reused too early.
//!
//! Each render graph submit that records draws using [`super::pipeline::ring_buffer`] data must
//! pair a [`GpuFrameScheduler::acquire_frame_index`] with [`GpuFrameScheduler::record_submission`].
//! When more than [`super::pipeline::NUM_FRAMES_IN_FLIGHT`] submissions are in flight, the
//! scheduler blocks on the oldest submission before returning the next frame index.

use std::collections::VecDeque;

use super::pipeline::NUM_FRAMES_IN_FLIGHT;

/// Serializes ring-buffer slot reuse against GPU completion for all render-graph submits on a queue.
#[derive(Debug, Default)]
pub struct GpuFrameScheduler {
    /// Monotonic index assigned to the next render submission (passed to passes as `frame_index`).
    next_frame_index: u64,
    /// FIFO queue of `(submission, frame_index)` for work that may still read ring-buffer regions.
    in_flight: VecDeque<(wgpu::SubmissionIndex, u64)>,
}

impl GpuFrameScheduler {
    /// Creates an empty scheduler (no submissions in flight).
    pub fn new() -> Self {
        Self {
            next_frame_index: 0,
            in_flight: VecDeque::new(),
        }
    }

    /// Waits until a ring slot is available, then returns the frame index to use while recording.
    ///
    /// On WebGPU, [`wgpu::Device::poll`] may not block; the host should still call
    /// [`Self::record_submission`] after each matching submit to keep bookkeeping consistent.
    pub fn acquire_frame_index(&mut self, device: &wgpu::Device) -> u64 {
        // Drain any already-completed submissions with a non-blocking poll first.
        // This avoids a blocking Wait when the GPU has already finished the oldest frames.
        let _ = device.poll(wgpu::PollType::Poll);

        while self.in_flight.len() >= NUM_FRAMES_IN_FLIGHT {
            let Some((oldest, _)) = self.in_flight.front() else {
                break;
            };
            let oldest = oldest.clone();
            let _ = device.poll(wgpu::PollType::Wait {
                submission_index: Some(oldest),
                timeout: None,
            });
            let _ = self.in_flight.pop_front();
        }

        let idx = self.next_frame_index;
        self.next_frame_index = self.next_frame_index.wrapping_add(1);
        idx
    }

    /// Records a completed submit that used `frame_index` from [`Self::acquire_frame_index`].
    pub fn record_submission(&mut self, submission: wgpu::SubmissionIndex, frame_index: u64) {
        self.in_flight.push_back((submission, frame_index));
    }

    /// Returns how many submits are still assumed in flight (for tests and diagnostics).
    #[cfg(test)]
    pub(crate) fn in_flight_len(&self) -> usize {
        self.in_flight.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acquire_without_record_never_blocks_first_three() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: None,
            force_fallback_adapter: true,
        }));
        let Ok(adapter) = adapter else {
            return;
        };
        let Ok((device, queue)) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("frame_scheduler_test"),
                ..Default::default()
            }))
        else {
            return;
        };

        let mut sched = GpuFrameScheduler::new();
        let a = sched.acquire_frame_index(&device);
        let b = sched.acquire_frame_index(&device);
        let c = sched.acquire_frame_index(&device);
        assert_eq!(a, 0);
        assert_eq!(b, 1);
        assert_eq!(c, 2);

        let s0 = queue.submit(Vec::<wgpu::CommandBuffer>::new());
        let s1 = queue.submit(Vec::<wgpu::CommandBuffer>::new());
        let s2 = queue.submit(Vec::<wgpu::CommandBuffer>::new());
        sched.record_submission(s0, a);
        sched.record_submission(s1, b);
        sched.record_submission(s2, c);
        assert_eq!(sched.in_flight_len(), 3);

        let d = sched.acquire_frame_index(&device);
        assert_eq!(d, 3);
        // Oldest submission was waited on and retired so a fourth slot could be acquired.
        assert_eq!(sched.in_flight_len(), 2);
    }
}
