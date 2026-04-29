//! Completion tracking for submitted non-readback GPU jobs.

use std::collections::HashMap;
use std::hash::Hash;

use crossbeam_channel as mpsc;

use super::GpuJobResources;

/// Pure state transitions for one pending submit-only GPU job.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SubmitJobLifecycle {
    /// Whether the submit-complete callback has fired.
    submit_done: bool,
    /// Age in renderer maintenance ticks.
    age_frames: u32,
}

impl SubmitJobLifecycle {
    /// Marks the job as completed by the driver-thread callback.
    pub(crate) fn mark_submit_done(&mut self) {
        self.submit_done = true;
    }

    /// Returns true after the queue callback has run.
    pub(crate) fn is_submit_done(&self) -> bool {
        self.submit_done
    }

    /// Increments age and returns true when the job has exceeded `max_age`.
    pub(crate) fn advance_age_and_is_expired(&mut self, max_age: u32) -> bool {
        self.age_frames = self.age_frames.saturating_add(1);
        self.age_frames > max_age
    }
}

/// GPU resources for a job that only needs submit-completion notification.
pub(crate) struct SubmittedGpuJob {
    /// Resources retained until the submit callback fires or the job expires.
    pub(crate) resources: GpuJobResources,
}

/// Submitted job retained until completion or timeout.
struct PendingSubmitJob {
    /// Resources retained until this job leaves the pending set.
    _resources: GpuJobResources,
    /// Submit/age state.
    lifecycle: SubmitJobLifecycle,
}

impl From<SubmittedGpuJob> for PendingSubmitJob {
    fn from(job: SubmittedGpuJob) -> Self {
        Self {
            _resources: job.resources,
            lifecycle: SubmitJobLifecycle::default(),
        }
    }
}

/// Completed and failed submit-only jobs drained during one maintenance tick.
pub(crate) struct GpuSubmitOutcomes<K> {
    /// Keys whose submit-complete callbacks fired.
    pub(crate) completed: Vec<K>,
    /// Keys whose callbacks never arrived before timeout.
    pub(crate) failed: Vec<K>,
}

/// Tracks submit-completion callbacks for keyed backend GPU jobs.
pub(crate) struct GpuSubmitJobTracker<K>
where
    K: Clone + Eq + Hash,
{
    /// In-flight GPU jobs keyed by source identity.
    pending: HashMap<K, PendingSubmitJob>,
    /// Submit-done channel sender captured by queue callbacks.
    submit_done_tx: mpsc::Sender<K>,
    /// Submit-done channel receiver drained on the main thread.
    submit_done_rx: mpsc::Receiver<K>,
    /// Maximum maintenance ticks before a pending submit is treated as failed.
    max_age_frames: u32,
}

impl<K> GpuSubmitJobTracker<K>
where
    K: Clone + Eq + Hash,
{
    /// Creates an empty submit tracker.
    pub(crate) fn new(max_age_frames: u32) -> Self {
        let (submit_done_tx, submit_done_rx) = mpsc::unbounded();
        Self {
            pending: HashMap::new(),
            submit_done_tx,
            submit_done_rx,
            max_age_frames,
        }
    }

    /// Returns a sender that queue-submit callbacks can use to mark jobs done.
    pub(crate) fn submit_done_sender(&self) -> mpsc::Sender<K> {
        self.submit_done_tx.clone()
    }

    /// Returns the number of currently pending jobs.
    pub(crate) fn len(&self) -> usize {
        self.pending.len()
    }

    /// Returns true when `key` is already pending.
    pub(crate) fn contains_key(&self, key: &K) -> bool {
        self.pending.contains_key(key)
    }

    /// Inserts a newly submitted GPU job.
    pub(crate) fn insert(&mut self, key: K, job: SubmittedGpuJob) {
        self.pending.insert(key, job.into());
    }

    /// Advances submit notifications and timeout handling.
    pub(crate) fn maintain(&mut self) -> GpuSubmitOutcomes<K> {
        self.drain_submit_done();
        let completed = self.drain_completed_jobs();
        let failed = self.age_pending_jobs();
        GpuSubmitOutcomes { completed, failed }
    }

    /// Marks jobs whose queue submit has completed.
    fn drain_submit_done(&mut self) {
        while let Ok(key) = self.submit_done_rx.try_recv() {
            if let Some(job) = self.pending.get_mut(&key) {
                job.lifecycle.mark_submit_done();
            }
        }
    }

    /// Removes jobs whose completion callback has fired.
    fn drain_completed_jobs(&mut self) -> Vec<K> {
        let completed = self
            .pending
            .iter()
            .filter_map(|(key, job)| job.lifecycle.is_submit_done().then_some(key.clone()))
            .collect::<Vec<_>>();
        for key in &completed {
            self.pending.remove(key);
        }
        completed
    }

    /// Ages in-flight jobs and returns keys whose completion callback did not arrive in time.
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
            self.pending.remove(key);
        }
        expired
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies submit-only jobs become visible after the callback state changes.
    #[test]
    fn lifecycle_completes_after_submit_done() {
        let mut lifecycle = SubmitJobLifecycle::default();
        assert!(!lifecycle.is_submit_done());
        lifecycle.mark_submit_done();
        assert!(lifecycle.is_submit_done());
    }

    /// Verifies submit-only jobs expire only after the configured cap.
    #[test]
    fn lifecycle_expires_after_max_age() {
        let mut lifecycle = SubmitJobLifecycle::default();
        for _ in 0..3 {
            assert!(!lifecycle.advance_age_and_is_expired(3));
        }
        assert!(lifecycle.advance_age_and_is_expired(3));
    }
}
