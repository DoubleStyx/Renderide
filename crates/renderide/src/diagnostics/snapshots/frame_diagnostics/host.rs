//! Host CPU model and memory metrics fragment of [`super::FrameDiagnosticsSnapshot`].
//!
//! [`HostCpuMemoryHud`] doubles as the capture-input type (filled by
//! [`crate::diagnostics::HostHudGatherer`]) and the snapshot fragment storage; capture is the
//! identity transform.

/// Host CPU model and memory usage (from `sysinfo`, refreshed periodically).
#[derive(Clone, Debug, Default)]
pub struct HostCpuMemoryHud {
    /// Reported CPU model name (first logical CPU brand string).
    pub cpu_model: String,
    /// Number of logical CPUs.
    pub logical_cpus: usize,
    /// Global CPU usage percentage (0-100).
    pub cpu_usage_percent: f32,
    /// Installed RAM in bytes.
    pub ram_total_bytes: u64,
    /// Used RAM in bytes (OS-defined).
    pub ram_used_bytes: u64,
    /// Resident memory of the renderer process in bytes (OS-defined; `None` when unavailable).
    pub process_ram_bytes: Option<u64>,
}
