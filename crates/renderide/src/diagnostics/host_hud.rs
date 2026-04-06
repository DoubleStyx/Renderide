//! Throttled `sysinfo` sampling for the debug HUD host CPU / RAM section.

use super::frame_diagnostics_snapshot::HostCpuMemoryHud;
use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};

/// Owns a lazily allocated [`System`] and samples [`HostCpuMemoryHud`] every
/// [`REFRESH_INTERVAL_FRAMES`] frames to limit `sysinfo` work.
pub struct HostHudGatherer {
    system: Option<System>,
    frame_counter: u64,
}

const REFRESH_INTERVAL_FRAMES: u64 = 30;

impl HostHudGatherer {
    /// Creates a gatherer; the first [`Self::snapshot`] may allocate the [`System`].
    pub fn new() -> Self {
        Self {
            system: None,
            frame_counter: 0,
        }
    }

    /// Returns host CPU/RAM for the current frame (reusing cached values between refreshes).
    pub fn snapshot(&mut self) -> HostCpuMemoryHud {
        self.frame_counter = self.frame_counter.wrapping_add(1);

        if self.system.is_none() && sysinfo::IS_SUPPORTED_SYSTEM {
            self.system = Some(System::new_with_specifics(
                RefreshKind::nothing()
                    .with_cpu(CpuRefreshKind::everything())
                    .with_memory(MemoryRefreshKind::everything()),
            ));
        }

        let Some(ref mut sys) = self.system else {
            return HostCpuMemoryHud {
                cpu_model: if sysinfo::IS_SUPPORTED_SYSTEM {
                    String::new()
                } else {
                    "unsupported platform".to_string()
                },
                ..Default::default()
            };
        };

        if self.frame_counter % REFRESH_INTERVAL_FRAMES == 1 {
            sys.refresh_cpu_usage();
            sys.refresh_memory();
        }

        let cpu_model = sys
            .cpus()
            .first()
            .map(|c| c.brand().to_string())
            .unwrap_or_default();

        HostCpuMemoryHud {
            cpu_model,
            logical_cpus: sys.cpus().len(),
            cpu_usage_percent: sys.global_cpu_usage(),
            ram_total_bytes: sys.total_memory(),
            ram_used_bytes: sys.used_memory(),
        }
    }
}

impl Default for HostHudGatherer {
    fn default() -> Self {
        Self::new()
    }
}
