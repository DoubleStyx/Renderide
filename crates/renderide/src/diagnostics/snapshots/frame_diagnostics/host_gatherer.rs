//! Throttled `sysinfo` sampling for the debug HUD host CPU / RAM section.

use super::host::HostCpuMemoryHud;
use std::sync::mpsc::{Receiver, SyncSender, channel, sync_channel};
use std::thread;
use sysinfo::{
    CpuRefreshKind, MemoryRefreshKind, Pid, ProcessRefreshKind, ProcessesToUpdate, RefreshKind,
    System,
};

/// Reads cached host metrics while a worker performs throttled `sysinfo` sampling. -xlinka
pub struct HostHudGatherer {
    worker: Option<HostHudWorker>,
    frame_counter: u64,
    cached: HostCpuMemoryHud,
}

const REFRESH_INTERVAL_FRAMES: u64 = 30;

struct HostHudWorker {
    request_tx: SyncSender<()>,
    result_rx: Receiver<HostCpuMemoryHud>,
}

impl HostHudWorker {
    fn spawn() -> Option<Self> {
        let (request_tx, request_rx) = sync_channel(1);
        let (result_tx, result_rx) = channel();
        thread::Builder::new()
            .name("renderide-host-hud".to_string())
            .spawn(move || {
                let pid = sysinfo::get_current_pid().ok();
                let mut system = System::new_with_specifics(
                    RefreshKind::nothing()
                        .with_cpu(CpuRefreshKind::everything())
                        .with_memory(MemoryRefreshKind::everything()),
                );
                while request_rx.recv().is_ok() {
                    let snapshot = refresh_host_hud(&mut system, pid);
                    if result_tx.send(snapshot).is_err() {
                        break;
                    }
                }
            })
            .ok()?;
        Some(Self {
            request_tx,
            result_rx,
        })
    }

    fn request_refresh(&self) {
        let _ = self.request_tx.try_send(());
    }

    fn take_latest(&self) -> Option<HostCpuMemoryHud> {
        self.result_rx.try_iter().last()
    }
}

impl HostHudGatherer {
    /// Creates a gatherer backed by a nonblocking sampling worker. -xlinka
    pub fn new() -> Self {
        let supported = sysinfo::IS_SUPPORTED_SYSTEM;
        Self {
            worker: supported.then(HostHudWorker::spawn).flatten(),
            frame_counter: 0,
            cached: HostCpuMemoryHud {
                cpu_model: if supported {
                    String::new()
                } else {
                    "unsupported platform".to_string()
                },
                ..Default::default()
            },
        }
    }

    /// Returns host CPU/RAM plus this process RAM for the current frame (cached between refreshes).
    pub fn snapshot(&mut self) -> HostCpuMemoryHud {
        if let Some(worker) = &self.worker
            && let Some(snapshot) = worker.take_latest()
        {
            self.cached = snapshot;
        }

        self.frame_counter = self.frame_counter.wrapping_add(1);
        if self.frame_counter % REFRESH_INTERVAL_FRAMES == 1
            && let Some(worker) = &self.worker
        {
            worker.request_refresh();
        }

        self.cached.clone()
    }
}

fn refresh_host_hud(system: &mut System, pid: Option<Pid>) -> HostCpuMemoryHud {
    system.refresh_cpu_usage();
    system.refresh_memory();
    let process_ram = pid.and_then(|pid| {
        system.refresh_processes_specifics(
            ProcessesToUpdate::Some(&[pid]),
            true,
            ProcessRefreshKind::nothing().with_memory(),
        );
        system.process(pid).map(sysinfo::Process::memory)
    });
    HostCpuMemoryHud {
        cpu_model: system
            .cpus()
            .first()
            .map(|cpu| cpu.brand().to_string())
            .unwrap_or_default(),
        logical_cpus: system.cpus().len(),
        cpu_usage_percent: system.global_cpu_usage(),
        ram_total_bytes: system.total_memory(),
        ram_used_bytes: system.used_memory(),
        process_ram_bytes: process_ram,
    }
}

impl Default for HostHudGatherer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_requests_refresh_without_waiting_for_a_result() {
        let (request_tx, request_rx) = sync_channel(1);
        let (_result_tx, result_rx) = channel();
        let mut gatherer = HostHudGatherer {
            worker: Some(HostHudWorker {
                request_tx,
                result_rx,
            }),
            frame_counter: 0,
            cached: HostCpuMemoryHud::default(),
        };

        assert_eq!(gatherer.snapshot().ram_total_bytes, 0);
        assert!(request_rx.try_recv().is_ok());
    }

    #[test]
    fn snapshot_adopts_the_latest_completed_worker_result() {
        let (request_tx, _request_rx) = sync_channel(1);
        let (result_tx, result_rx) = channel();
        let mut gatherer = HostHudGatherer {
            worker: Some(HostHudWorker {
                request_tx,
                result_rx,
            }),
            frame_counter: 1,
            cached: HostCpuMemoryHud::default(),
        };
        result_tx
            .send(HostCpuMemoryHud {
                ram_total_bytes: 123,
                ram_used_bytes: 45,
                ..Default::default()
            })
            .unwrap();

        let snapshot = gatherer.snapshot();

        assert_eq!(snapshot.ram_total_bytes, 123);
        assert_eq!(snapshot.ram_used_bytes, 45);
    }
}
