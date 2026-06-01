//! Dedicated bounded CPU worker pool for asset preparation jobs.
//!
//! Asset decode and derived-stream preparation can be large enough to interfere with frame-critical
//! Rayon work. This pool keeps that traffic on named secondary threads with a fixed queue and an
//! inline fallback when the queue is saturated.

use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::thread::{self, JoinHandle};

use crossbeam_channel::{Receiver, Sender, TrySendError};

const ASSET_WORKER_MAX_THREADS: usize = 4;
const ASSET_WORKER_QUEUE_CAPACITY: usize = 256;

/// Result of dispatching a job to the asset worker pool.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AssetWorkerDispatch {
    /// The job was queued for an asset worker thread.
    Queued,
    /// The bounded queue was unavailable, so the caller executed the job immediately.
    Inline,
}

/// Snapshot of asset-worker pressure and throughput.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct AssetWorkerDiagnosticSnapshot {
    /// Jobs waiting in the bounded queue.
    pub(crate) queued: usize,
    /// Jobs currently executing on asset worker threads.
    pub(crate) running: usize,
    /// Maximum queued depth observed since startup.
    pub(crate) max_queued: usize,
    /// Jobs accepted by the dispatch path.
    pub(crate) spawned: u64,
    /// Jobs completed on asset worker threads.
    pub(crate) completed: u64,
    /// Jobs executed inline because the worker queue could not accept them.
    pub(crate) inline_executed: u64,
    /// Queue-full events that forced inline execution.
    pub(crate) saturated: u64,
}

/// Dispatches `work` to the global asset worker pool.
pub(crate) fn spawn_asset_job(work: impl FnOnce() + Send + 'static) -> AssetWorkerDispatch {
    match global_asset_worker() {
        Ok(worker) => worker.spawn(Box::new(work)),
        Err(err) => {
            logger::warn!("asset worker unavailable; running asset job inline: {err}");
            work();
            AssetWorkerDispatch::Inline
        }
    }
}

/// Returns the current global asset-worker diagnostics.
pub(crate) fn diagnostic_snapshot() -> AssetWorkerDiagnosticSnapshot {
    match global_asset_worker() {
        Ok(worker) => worker.diagnostic_snapshot(),
        Err(_) => AssetWorkerDiagnosticSnapshot::default(),
    }
}

struct AssetWorker {
    sender: Option<Sender<AssetWorkerJob>>,
    stats: Arc<AssetWorkerStats>,
    threads: Vec<JoinHandle<()>>,
}

impl AssetWorker {
    fn new(
        thread_name_prefix: &'static str,
        worker_count: usize,
        queue_capacity: usize,
    ) -> Result<Self, String> {
        let (tx, rx) = crossbeam_channel::bounded(queue_capacity.max(1));
        let stats = Arc::new(AssetWorkerStats::default());
        let mut threads = Vec::with_capacity(worker_count);
        for index in 0..worker_count.max(1) {
            let rx = rx.clone();
            let stats = Arc::clone(&stats);
            let name = format!("{thread_name_prefix}-{index}");
            let handle = thread::Builder::new()
                .name(name)
                .spawn(move || worker_loop(rx, stats))
                .map_err(|e| format!("asset worker thread creation failed: {e}"))?;
            threads.push(handle);
        }
        Ok(Self {
            sender: Some(tx),
            stats,
            threads,
        })
    }

    fn global() -> Result<Self, String> {
        Self::new(
            "asset-worker",
            default_asset_worker_count(),
            ASSET_WORKER_QUEUE_CAPACITY,
        )
    }

    fn spawn(&self, job: AssetJob) -> AssetWorkerDispatch {
        self.stats.spawned.fetch_add(1, Ordering::Relaxed);
        let worker_job = AssetWorkerJob { job };
        let Some(sender) = self.sender.as_ref() else {
            self.run_inline(worker_job);
            return AssetWorkerDispatch::Inline;
        };
        self.stats.reserve_queue_slot();
        match sender.try_send(worker_job) {
            Ok(()) => {
                self.stats.record_queued_peak();
                AssetWorkerDispatch::Queued
            }
            Err(TrySendError::Full(worker_job)) => {
                self.stats.cancel_queue_slot();
                self.stats.saturated.fetch_add(1, Ordering::Relaxed);
                self.run_inline(worker_job);
                AssetWorkerDispatch::Inline
            }
            Err(TrySendError::Disconnected(worker_job)) => {
                self.stats.cancel_queue_slot();
                self.run_inline(worker_job);
                AssetWorkerDispatch::Inline
            }
        }
    }

    fn run_inline(&self, worker_job: AssetWorkerJob) {
        profiling::scope!("asset_worker::inline");
        self.stats.inline_executed.fetch_add(1, Ordering::Relaxed);
        worker_job.run();
    }

    fn diagnostic_snapshot(&self) -> AssetWorkerDiagnosticSnapshot {
        self.stats.diagnostic_snapshot()
    }
}

impl Drop for AssetWorker {
    fn drop(&mut self) {
        self.sender.take();
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
    }
}

type AssetJob = Box<dyn FnOnce() + Send + 'static>;

struct AssetWorkerJob {
    job: AssetJob,
}

impl AssetWorkerJob {
    fn run(self) {
        (self.job)();
    }
}

#[derive(Default)]
struct AssetWorkerStats {
    queued: AtomicUsize,
    running: AtomicUsize,
    max_queued: AtomicUsize,
    spawned: AtomicU64,
    completed: AtomicU64,
    inline_executed: AtomicU64,
    saturated: AtomicU64,
}

impl AssetWorkerStats {
    fn reserve_queue_slot(&self) {
        self.queued.fetch_add(1, Ordering::Relaxed);
    }

    fn cancel_queue_slot(&self) {
        self.queued.fetch_sub(1, Ordering::Relaxed);
    }

    fn record_queued_peak(&self) {
        let queued = self.queued.load(Ordering::Relaxed);
        let mut observed = self.max_queued.load(Ordering::Relaxed);
        while queued > observed {
            match self.max_queued.compare_exchange_weak(
                observed,
                queued,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(next) => observed = next,
            }
        }
    }

    fn note_worker_start(&self) {
        self.queued.fetch_sub(1, Ordering::Relaxed);
        self.running.fetch_add(1, Ordering::Relaxed);
    }

    fn note_worker_done(&self) {
        self.running.fetch_sub(1, Ordering::Relaxed);
        self.completed.fetch_add(1, Ordering::Relaxed);
    }

    fn diagnostic_snapshot(&self) -> AssetWorkerDiagnosticSnapshot {
        AssetWorkerDiagnosticSnapshot {
            queued: self.queued.load(Ordering::Relaxed),
            running: self.running.load(Ordering::Relaxed),
            max_queued: self.max_queued.load(Ordering::Relaxed),
            spawned: self.spawned.load(Ordering::Relaxed),
            completed: self.completed.load(Ordering::Relaxed),
            inline_executed: self.inline_executed.load(Ordering::Relaxed),
            saturated: self.saturated.load(Ordering::Relaxed),
        }
    }
}

fn global_asset_worker() -> Result<&'static AssetWorker, &'static str> {
    static WORKER: OnceLock<Result<AssetWorker, String>> = OnceLock::new();
    WORKER
        .get_or_init(AssetWorker::global)
        .as_ref()
        .map_err(String::as_str)
}

fn worker_loop(rx: Receiver<AssetWorkerJob>, stats: Arc<AssetWorkerStats>) {
    while let Ok(job) = rx.recv() {
        profiling::scope!("asset_worker::job");
        stats.note_worker_start();
        job.run();
        stats.note_worker_done();
    }
}

fn default_asset_worker_count() -> usize {
    thread::available_parallelism()
        .map_or(1, |threads| asset_worker_count_for_threads(threads.get()))
}

fn asset_worker_count_for_threads(thread_count: usize) -> usize {
    thread_count.div_ceil(4).clamp(1, ASSET_WORKER_MAX_THREADS)
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    use super::{AssetWorker, AssetWorkerDispatch, asset_worker_count_for_threads};

    #[test]
    fn worker_count_is_bounded() {
        assert_eq!(asset_worker_count_for_threads(0), 1);
        assert_eq!(asset_worker_count_for_threads(1), 1);
        assert_eq!(asset_worker_count_for_threads(8), 2);
        assert_eq!(asset_worker_count_for_threads(128), 4);
    }

    #[test]
    fn queued_job_runs_on_named_asset_worker() {
        let worker = AssetWorker::new("asset-worker-test", 1, 4).unwrap();
        let (tx, rx) = mpsc::channel();

        assert_eq!(
            worker.spawn(Box::new(move || {
                let name = thread::current().name().unwrap_or("").to_string();
                tx.send(name).unwrap();
            })),
            AssetWorkerDispatch::Queued
        );

        let thread_name = rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert!(thread_name.starts_with("asset-worker-test-"));
        wait_for_completed_jobs(&worker, 1);
    }

    #[test]
    fn saturated_queue_runs_job_inline() {
        let worker = AssetWorker::new("asset-worker-saturation-test", 1, 1).unwrap();
        let (worker_started_tx, worker_started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        assert_eq!(
            worker.spawn(Box::new(move || {
                worker_started_tx.send(()).unwrap();
                release_rx.recv().unwrap();
            })),
            AssetWorkerDispatch::Queued
        );
        worker_started_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap();

        let (queued_tx, queued_rx) = mpsc::channel();
        assert_eq!(
            worker.spawn(Box::new(move || {
                queued_tx.send(()).unwrap();
            })),
            AssetWorkerDispatch::Queued
        );

        let inline_thread = thread::current().id();
        let (inline_tx, inline_rx) = mpsc::channel();
        assert_eq!(
            worker.spawn(Box::new(move || {
                inline_tx.send(thread::current().id()).unwrap();
            })),
            AssetWorkerDispatch::Inline
        );
        assert_eq!(
            inline_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
            inline_thread
        );

        release_tx.send(()).unwrap();
        queued_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        let stats = worker.diagnostic_snapshot();
        assert_eq!(stats.inline_executed, 1);
        assert_eq!(stats.saturated, 1);
    }

    fn wait_for_completed_jobs(worker: &AssetWorker, expected: u64) {
        for _ in 0..100 {
            if worker.diagnostic_snapshot().completed >= expected {
                return;
            }
            thread::sleep(Duration::from_millis(1));
        }
        panic!("asset worker did not complete {expected} job(s) before timeout");
    }
}
