//! Main-thread readback lifecycle for submitted GPU jobs.

use std::collections::HashMap;
use std::hash::Hash;

use crossbeam_channel as mpsc;

use super::GpuJobResources;

/// Pure state transitions for one pending GPU readback job.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ReadbackJobLifecycle {
    /// Whether the submit-complete callback has fired.
    submit_done: bool,
    /// Whether a main-thread `map_async` request has been started.
    map_started: bool,
    /// Age in renderer maintenance ticks.
    age_frames: u32,
}

impl ReadbackJobLifecycle {
    /// Marks the job as submitted by the driver-thread callback.
    pub(crate) fn mark_submit_done(&mut self) {
        self.submit_done = true;
    }

    /// Returns true when the job can start a main-thread `map_async`.
    pub(crate) fn should_start_map(&self) -> bool {
        self.submit_done && !self.map_started
    }

    /// Marks the job as having an active `map_async` request.
    pub(crate) fn mark_map_started(&mut self) {
        self.map_started = true;
    }

    /// Increments age and returns true when the job has exceeded `max_age`.
    pub(crate) fn advance_age_and_is_expired(&mut self, max_age: u32) -> bool {
        self.age_frames = self.age_frames.saturating_add(1);
        self.age_frames > max_age
    }
}

/// GPU resources and staging buffer for a submitted readback job.
pub(crate) struct SubmittedReadbackJob {
    /// Staging buffer copied from the GPU job output.
    pub(crate) staging: wgpu::Buffer,
    /// Resources retained until the readback is finished or abandoned.
    pub(crate) resources: GpuJobResources,
}

/// A submitted readback whose buffer may complete later.
struct PendingReadbackJob {
    /// Staging buffer copied from the GPU job output.
    staging: wgpu::Buffer,
    /// Resources retained until this job leaves the pending set.
    _resources: GpuJobResources,
    /// Submit/map/age state.
    lifecycle: ReadbackJobLifecycle,
    /// Pending `map_async` result receiver.
    map_recv: Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
}

impl From<SubmittedReadbackJob> for PendingReadbackJob {
    fn from(job: SubmittedReadbackJob) -> Self {
        Self {
            staging: job.staging,
            _resources: job.resources,
            lifecycle: ReadbackJobLifecycle::default(),
            map_recv: None,
        }
    }
}

/// Completed and failed readbacks drained during one maintenance tick.
pub(crate) struct GpuReadbackOutcomes<K, T> {
    /// Successfully mapped and parsed readback results.
    pub(crate) completed: Vec<(K, T)>,
    /// Keys whose job failed, expired, or disconnected.
    pub(crate) failed: Vec<K>,
}

/// Owns in-flight GPU readback jobs plus their submit-done notification channel.
pub(crate) struct GpuReadbackJobs<K, T>
where
    K: Clone + Eq + Hash,
{
    /// In-flight GPU jobs keyed by source identity.
    pending: HashMap<K, PendingReadbackJob>,
    /// Submit-done channel sender captured by queue callbacks.
    submit_done_tx: mpsc::Sender<K>,
    /// Submit-done channel receiver drained on the main thread.
    submit_done_rx: mpsc::Receiver<K>,
    /// Maximum maintenance ticks before a pending readback is treated as failed.
    max_age_frames: u32,
    /// Parses mapped staging bytes into the typed readback payload.
    parse: fn(&[u8]) -> Option<T>,
}

impl<K, T> GpuReadbackJobs<K, T>
where
    K: Clone + Eq + Hash,
{
    /// Creates an empty readback job owner.
    pub(crate) fn new(max_age_frames: u32, parse: fn(&[u8]) -> Option<T>) -> Self {
        let (submit_done_tx, submit_done_rx) = mpsc::unbounded();
        Self {
            pending: HashMap::new(),
            submit_done_tx,
            submit_done_rx,
            max_age_frames,
            parse,
        }
    }

    /// Returns a sender that queue-submit callbacks can use to mark jobs done.
    pub(crate) fn submit_done_sender(&self) -> mpsc::Sender<K> {
        self.submit_done_tx.clone()
    }

    /// Returns the number of currently pending readbacks.
    pub(crate) fn len(&self) -> usize {
        self.pending.len()
    }

    /// Returns true when `key` is already pending.
    pub(crate) fn contains_key(&self, key: &K) -> bool {
        self.pending.contains_key(key)
    }

    /// Inserts a newly submitted GPU readback job.
    pub(crate) fn insert(&mut self, key: K, job: SubmittedReadbackJob) {
        self.pending.insert(key, job.into());
    }

    /// Advances submit notifications, mapping, completion, and age/failure handling.
    pub(crate) fn maintain(&mut self) -> GpuReadbackOutcomes<K, T> {
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
    fn drain_completed_maps(&mut self) -> GpuReadbackOutcomes<K, T> {
        let mut completed = Vec::new();
        let mut failed = Vec::new();
        for (key, job) in &mut self.pending {
            let Some(recv) = job.map_recv.as_ref() else {
                continue;
            };
            match recv.try_recv() {
                Ok(Ok(())) => match read_from_staging(&job.staging, self.parse) {
                    Some(result) => completed.push((key.clone(), result)),
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
        GpuReadbackOutcomes { completed, failed }
    }

    /// Ages in-flight jobs and returns sources that never mapped back.
    fn age_pending_jobs(&mut self) -> Vec<K> {
        let mut expired = Vec::new();
        for (key, job) in &mut self.pending {
            if job
                .lifecycle
                .advance_age_and_is_expired(self.max_age_frames)
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

/// Reads a mapped staging buffer into a typed payload and always unmaps it.
fn read_from_staging<T>(staging: &wgpu::Buffer, parse: fn(&[u8]) -> Option<T>) -> Option<T> {
    let mapped = staging.slice(..).get_mapped_range();
    let result = parse(&mapped);
    drop(mapped);
    staging.unmap();
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies readback jobs do not map before queue submit completion.
    #[test]
    fn lifecycle_starts_map_only_after_submit_done() {
        let mut lifecycle = ReadbackJobLifecycle::default();
        assert!(!lifecycle.should_start_map());
        lifecycle.mark_submit_done();
        assert!(lifecycle.should_start_map());
        lifecycle.mark_map_started();
        assert!(!lifecycle.should_start_map());
    }

    /// Verifies age tracking expires only after the configured cap.
    #[test]
    fn lifecycle_expires_after_max_age() {
        let mut lifecycle = ReadbackJobLifecycle::default();
        for _ in 0..3 {
            assert!(!lifecycle.advance_age_and_is_expired(3));
        }
        assert!(lifecycle.advance_age_and_is_expired(3));
    }
}
