//! Tracks fatal GPU device-loss events reported by wgpu.

use std::sync::atomic::{AtomicU64, Ordering};

/// Shared generation counter for device-loss events.
///
/// A lost `wgpu::Device` cannot be repaired by resetting mapped staging buffers or
/// reconfiguring the window surface. The renderer reads this counter at frame boundaries and
/// switches into shutdown before any more GPU work is recorded on the invalid device.
#[derive(Debug, Default)]
pub(crate) struct GpuDeviceHealth {
    /// Monotonic count of device-loss notifications reported for the active GPU device.
    lost_generation: AtomicU64,
}

impl GpuDeviceHealth {
    /// Creates a device-health tracker with no device loss recorded.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Records a newly reported device-loss event and returns its generation.
    pub(crate) fn mark_lost(&self, reason: impl AsRef<str>) -> u64 {
        let generation = self.lost_generation.fetch_add(1, Ordering::AcqRel) + 1;
        let reason = reason.as_ref();
        logger::error!("GPU device loss recorded: generation={generation} reason={reason}");
        generation
    }

    /// Current device-loss generation.
    pub(crate) fn lost_generation(&self) -> u64 {
        self.lost_generation.load(Ordering::Acquire)
    }

    /// Whether wgpu has reported this device as lost.
    pub(crate) fn is_lost(&self) -> bool {
        self.lost_generation() != 0
    }
}

#[cfg(test)]
mod tests {
    use super::GpuDeviceHealth;

    #[test]
    fn lost_generation_advances_on_device_loss() {
        let health = GpuDeviceHealth::new();

        assert_eq!(health.lost_generation(), 0);
        assert!(!health.is_lost());

        assert_eq!(health.mark_lost("first"), 1);
        assert_eq!(health.mark_lost("second"), 2);
        assert_eq!(health.lost_generation(), 2);
        assert!(health.is_lost());
    }
}
