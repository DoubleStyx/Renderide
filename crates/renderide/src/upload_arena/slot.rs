//! Persistent upload arena slot type, lifecycle state, and slot-level helpers.

use std::sync::atomic::{AtomicU64, Ordering};

pub(super) const UPLOAD_ARENA_SLOTS: usize = 3;
const DEFAULT_SLOT_BYTES: u64 = 1024 * 1024;
static OVERSIZED_UPLOAD_LOG_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Lifecycle of one persistent upload arena slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum UploadArenaSlotState {
    /// No buffer has been allocated for this slot yet.
    Empty,
    /// Buffer is mapped and available for a new frame's writes.
    Free,
    /// Buffer is currently being filled on the main thread.
    Writing { generation: u64 },
    /// Buffer is referenced by submitted GPU work.
    InFlight { generation: u64 },
    /// GPU work completed; the buffer is waiting for `map_async` to finish.
    Remapping { generation: u64 },
}

impl UploadArenaSlotState {
    pub(super) fn can_write(self) -> bool {
        matches!(self, Self::Empty | Self::Free)
    }
}

/// One buffer slot in the persistent upload arena.
pub(super) struct UploadArenaSlot {
    pub(super) buffer: Option<wgpu::Buffer>,
    pub(super) capacity: u64,
    pub(super) state: UploadArenaSlotState,
}

impl UploadArenaSlot {
    pub(super) fn empty() -> Self {
        Self {
            buffer: None,
            capacity: 0,
            state: UploadArenaSlotState::Empty,
        }
    }
}

pub(super) fn select_writable_slot(
    slots: &[UploadArenaSlot; UPLOAD_ARENA_SLOTS],
    required: u64,
) -> Option<usize> {
    slots
        .iter()
        .position(|slot| slot.state.can_write() && slot.capacity >= required)
        .or_else(|| {
            slots
                .iter()
                .position(|slot| matches!(slot.state, UploadArenaSlotState::Empty))
        })
        .or_else(|| {
            slots
                .iter()
                .position(|slot| matches!(slot.state, UploadArenaSlotState::Free))
        })
}

pub(super) fn next_slot_capacity(required: u64, current: u64, max_buffer_size: u64) -> Option<u64> {
    if required == 0 || required > max_buffer_size {
        return None;
    }
    let doubled = current.saturating_mul(2);
    let target = DEFAULT_SLOT_BYTES.max(required).max(doubled);
    let rounded = target
        .checked_next_power_of_two()
        .unwrap_or(max_buffer_size);
    Some(rounded.min(max_buffer_size).max(required))
}

pub(super) fn create_persistent_slot_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    profiling::scope!("frame_upload_arena::create_persistent_slot");
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("frame_upload_arena_slot"),
        size,
        usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: true,
    });
    crate::profiling::note_resource_churn!(Buffer, "render_graph::frame_upload_arena_slot");
    buffer
}

pub(super) fn create_temporary_staging_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    profiling::scope!("frame_upload_arena::create_temporary_staging");
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("frame_upload_temporary_staging"),
        size,
        usage: wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: true,
    });
    crate::profiling::note_resource_churn!(Buffer, "render_graph::frame_upload_temporary_staging");
    buffer
}

pub(super) fn log_oversized_upload(required: u64, max_buffer_size: u64) {
    let count = OVERSIZED_UPLOAD_LOG_COUNTER.fetch_add(1, Ordering::Relaxed) + 1;
    if count <= 5 || count.is_multiple_of(120) {
        logger::warn!(
            "frame upload arena: staging bytes {required} exceed max_buffer_size {max_buffer_size}; falling back to queue writes"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_slots() -> [UploadArenaSlot; UPLOAD_ARENA_SLOTS] {
        std::array::from_fn(|_| UploadArenaSlot::empty())
    }

    #[test]
    fn slot_capacity_grows_to_power_of_two_with_default_floor() {
        assert_eq!(
            next_slot_capacity(64, 0, 8 * 1024 * 1024),
            Some(1024 * 1024)
        );
        assert_eq!(
            next_slot_capacity(2 * 1024 * 1024 + 1, 1024 * 1024, 8 * 1024 * 1024),
            Some(4 * 1024 * 1024)
        );
    }

    #[test]
    fn slot_capacity_clamps_to_device_max_without_underallocating() {
        assert_eq!(next_slot_capacity(900, 0, 1000), Some(1000));
        assert_eq!(next_slot_capacity(1001, 0, 1000), None);
    }

    #[test]
    fn select_writable_slot_prefers_existing_capacity() {
        let mut slots = empty_slots();
        slots[0].state = UploadArenaSlotState::Free;
        slots[0].capacity = 64;
        slots[1].state = UploadArenaSlotState::Free;
        slots[1].capacity = 1024;

        assert_eq!(select_writable_slot(&slots, 512), Some(1));
    }

    #[test]
    fn select_writable_slot_uses_empty_before_growing_small_free_slot() {
        let mut slots = empty_slots();
        slots[0].state = UploadArenaSlotState::Free;
        slots[0].capacity = 64;

        assert_eq!(select_writable_slot(&slots, 512), Some(1));
    }

    #[test]
    fn select_writable_slot_returns_none_when_all_slots_are_busy() {
        let mut slots = empty_slots();
        for (i, slot) in slots.iter_mut().enumerate() {
            slot.state = UploadArenaSlotState::InFlight {
                generation: i as u64 + 1,
            };
        }

        assert_eq!(select_writable_slot(&slots, 512), None);
    }
}
