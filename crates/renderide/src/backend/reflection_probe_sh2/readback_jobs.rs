//! Keyed GPU readback lifecycle for reflection-probe SH2 projection jobs.

use std::collections::HashMap;

use crossbeam_channel as mpsc;
use glam::Vec3;

use super::{Sh2SourceKey, MAX_PENDING_JOB_AGE_FRAMES};
use crate::shared::RenderSH2;

/// GPU resources that must stay alive until an SH2 projection readback completes.
pub(super) struct SubmittedGpuSh2Job {
    /// Staging buffer copied from the compute output.
    pub(super) staging: wgpu::Buffer,
    /// Compute output buffer kept alive until readback finishes.
    pub(super) output: wgpu::Buffer,
    /// Bind group kept alive until the queued command has completed.
    pub(super) bind_group: wgpu::BindGroup,
    /// Uniform/parameter buffers kept alive until the queued command has completed.
    pub(super) buffers: Vec<wgpu::Buffer>,
}

/// Pure state transitions for one pending readback job.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ReadbackJobLifecycle {
    /// Whether the submit-done callback has fired.
    submit_done: bool,
    /// Whether a main-thread `map_async` request has been started.
    map_started: bool,
    /// Age in renderer ticks.
    age_frames: u32,
}

impl ReadbackJobLifecycle {
    /// Marks the job as submitted by the driver-thread callback.
    fn mark_submit_done(&mut self) {
        self.submit_done = true;
    }

    /// Returns true when the job can start a main-thread `map_async`.
    fn should_start_map(&self) -> bool {
        self.submit_done && !self.map_started
    }

    /// Marks the job as having an active `map_async` request.
    fn mark_map_started(&mut self) {
        self.map_started = true;
    }

    /// Increments age and returns true when the job has exceeded `max_age`.
    fn advance_age_and_is_expired(&mut self, max_age: u32) -> bool {
        self.age_frames = self.age_frames.saturating_add(1);
        self.age_frames > max_age
    }
}

/// A GPU job whose commands have been submitted and whose readback may complete later.
struct PendingGpuJob {
    /// Staging buffer copied from the compute output.
    staging: wgpu::Buffer,
    /// Compute output buffer kept alive until readback finishes.
    _output: wgpu::Buffer,
    /// Bind group kept alive until the queued command has completed.
    _bind_group: wgpu::BindGroup,
    /// Uniform/parameter buffers kept alive until the queued command has completed.
    _buffers: Vec<wgpu::Buffer>,
    /// Pure submit/map/age state.
    lifecycle: ReadbackJobLifecycle,
    /// Pending `map_async` result receiver.
    map_recv: Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
}

impl From<SubmittedGpuSh2Job> for PendingGpuJob {
    fn from(job: SubmittedGpuSh2Job) -> Self {
        Self {
            staging: job.staging,
            _output: job.output,
            _bind_group: job.bind_group,
            _buffers: job.buffers,
            lifecycle: ReadbackJobLifecycle::default(),
            map_recv: None,
        }
    }
}

/// Completed and failed readbacks drained during one maintenance tick.
pub(super) struct Sh2ReadbackOutcomes {
    /// Successfully mapped SH2 results.
    pub(super) completed: Vec<(Sh2SourceKey, RenderSH2)>,
    /// Source keys whose job failed, expired, or disconnected.
    pub(super) failed: Vec<Sh2SourceKey>,
}

/// Owns all in-flight SH2 readback jobs plus their submit-done notification channel.
pub(super) struct Sh2ReadbackJobs {
    /// In-flight GPU jobs keyed by source identity.
    pending: HashMap<Sh2SourceKey, PendingGpuJob>,
    /// Submit-done channel sender captured by queue callbacks.
    submit_done_tx: mpsc::Sender<Sh2SourceKey>,
    /// Submit-done channel receiver drained on the main thread.
    submit_done_rx: mpsc::Receiver<Sh2SourceKey>,
}

impl Default for Sh2ReadbackJobs {
    fn default() -> Self {
        Self::new()
    }
}

impl Sh2ReadbackJobs {
    /// Creates an empty readback job owner.
    pub(super) fn new() -> Self {
        let (submit_done_tx, submit_done_rx) = mpsc::unbounded();
        Self {
            pending: HashMap::new(),
            submit_done_tx,
            submit_done_rx,
        }
    }

    /// Returns a sender that queue-submit callbacks can use to mark jobs done.
    pub(super) fn submit_done_sender(&self) -> mpsc::Sender<Sh2SourceKey> {
        self.submit_done_tx.clone()
    }

    /// Returns the number of currently pending readbacks.
    pub(super) fn len(&self) -> usize {
        self.pending.len()
    }

    /// Returns true when `key` is already pending.
    pub(super) fn contains_key(&self, key: &Sh2SourceKey) -> bool {
        self.pending.contains_key(key)
    }

    /// Inserts a newly submitted GPU readback job.
    pub(super) fn insert(&mut self, key: Sh2SourceKey, job: SubmittedGpuSh2Job) {
        self.pending.insert(key, job.into());
    }

    /// Advances submit notifications, mapping, completion, and age/failure handling.
    pub(super) fn maintain(&mut self) -> Sh2ReadbackOutcomes {
        self.drain_submit_done();
        self.start_ready_maps();
        let mut outcomes = self.drain_completed_maps();
        outcomes.failed.extend(self.age_pending_jobs());
        outcomes
    }

    /// Marks jobs whose queue submit has completed.
    fn drain_submit_done(&mut self) {
        while let Ok(key) = self.submit_done_rx.try_recv() {
            if let Some(job) = self.pending.get_mut(&key) {
                job.lifecycle.mark_submit_done();
            }
        }
    }

    /// Starts `map_async` for submitted jobs on the main thread.
    fn start_ready_maps(&mut self) {
        for job in self.pending.values_mut() {
            if !job.lifecycle.should_start_map() {
                continue;
            }
            let slice = job.staging.slice(..);
            let (tx, rx) = mpsc::bounded::<Result<(), wgpu::BufferAsyncError>>(1);
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            job.map_recv = Some(rx);
            job.lifecycle.mark_map_started();
        }
    }

    /// Moves completed mapped buffers into an outcome batch.
    fn drain_completed_maps(&mut self) -> Sh2ReadbackOutcomes {
        let mut completed = Vec::new();
        let mut failed = Vec::new();
        for (key, job) in &mut self.pending {
            let Some(recv) = job.map_recv.as_ref() else {
                continue;
            };
            match recv.try_recv() {
                Ok(Ok(())) => match read_sh2_from_staging(&job.staging) {
                    Some(sh) => completed.push((key.clone(), sh)),
                    None => failed.push(key.clone()),
                },
                Ok(Err(_)) => failed.push(key.clone()),
                Err(mpsc::TryRecvError::Empty) => {}
                Err(mpsc::TryRecvError::Disconnected) => failed.push(key.clone()),
            }
        }
        for (key, _) in &completed {
            self.pending.remove(key);
        }
        for key in &failed {
            if let Some(job) = self.pending.remove(key) {
                job.staging.unmap();
            }
        }
        Sh2ReadbackOutcomes { completed, failed }
    }

    /// Ages in-flight jobs and returns sources that never mapped back.
    fn age_pending_jobs(&mut self) -> Vec<Sh2SourceKey> {
        let mut expired = Vec::new();
        for (key, job) in &mut self.pending {
            if job
                .lifecycle
                .advance_age_and_is_expired(MAX_PENDING_JOB_AGE_FRAMES)
            {
                expired.push(key.clone());
            }
        }
        for key in &expired {
            if let Some(job) = self.pending.remove(key) {
                job.staging.unmap();
            }
        }
        expired
    }
}

/// Reads a mapped staging buffer into a RenderSH2 payload and always unmaps it.
fn read_sh2_from_staging(staging: &wgpu::Buffer) -> Option<RenderSH2> {
    let mapped = staging.slice(..).get_mapped_range();
    let result = parse_sh2_bytes(&mapped);
    drop(mapped);
    staging.unmap();
    result
}

/// Parses nine packed `vec4<f32>` SH rows from GPU readback bytes.
fn parse_sh2_bytes(bytes: &[u8]) -> Option<RenderSH2> {
    let mut coeffs = [[0.0f32; 4]; 9];
    for (i, chunk) in bytes.chunks_exact(16).take(9).enumerate() {
        coeffs[i] = [
            f32::from_le_bytes(chunk.get(0..4)?.try_into().ok()?),
            f32::from_le_bytes(chunk.get(4..8)?.try_into().ok()?),
            f32::from_le_bytes(chunk.get(8..12)?.try_into().ok()?),
            f32::from_le_bytes(chunk.get(12..16)?.try_into().ok()?),
        ];
    }
    Some(RenderSH2 {
        sh0: Vec3::new(coeffs[0][0], coeffs[0][1], coeffs[0][2]),
        sh1: Vec3::new(coeffs[1][0], coeffs[1][1], coeffs[1][2]),
        sh2: Vec3::new(coeffs[2][0], coeffs[2][1], coeffs[2][2]),
        sh3: Vec3::new(coeffs[3][0], coeffs[3][1], coeffs[3][2]),
        sh4: Vec3::new(coeffs[4][0], coeffs[4][1], coeffs[4][2]),
        sh5: Vec3::new(coeffs[5][0], coeffs[5][1], coeffs[5][2]),
        sh6: Vec3::new(coeffs[6][0], coeffs[6][1], coeffs[6][2]),
        sh7: Vec3::new(coeffs[7][0], coeffs[7][1], coeffs[7][2]),
        sh8: Vec3::new(coeffs[8][0], coeffs[8][1], coeffs[8][2]),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lifecycle_starts_map_only_after_submit_done() {
        let mut lifecycle = ReadbackJobLifecycle::default();
        assert!(!lifecycle.should_start_map());
        lifecycle.mark_submit_done();
        assert!(lifecycle.should_start_map());
        lifecycle.mark_map_started();
        assert!(!lifecycle.should_start_map());
    }

    #[test]
    fn lifecycle_expires_after_max_age() {
        let mut lifecycle = ReadbackJobLifecycle::default();
        for _ in 0..MAX_PENDING_JOB_AGE_FRAMES {
            assert!(!lifecycle.advance_age_and_is_expired(MAX_PENDING_JOB_AGE_FRAMES));
        }
        assert!(lifecycle.advance_age_and_is_expired(MAX_PENDING_JOB_AGE_FRAMES));
    }

    #[test]
    fn parse_sh2_bytes_reads_nine_rgb_rows() {
        let mut bytes = vec![0u8; 9 * 16];
        for row in 0..9 {
            for channel in 0..3 {
                let value = row as f32 + channel as f32 * 0.25;
                let offset = row * 16 + channel * 4;
                bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            }
        }

        let sh = parse_sh2_bytes(&bytes).unwrap();
        assert_eq!(sh.sh0, Vec3::new(0.0, 0.25, 0.5));
        assert_eq!(sh.sh8, Vec3::new(8.0, 8.25, 8.5));
    }
}
