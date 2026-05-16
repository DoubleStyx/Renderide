//! `PersistentUploadArena`: coordinator across persistent slots, the GPU completion channel, and
//! the per-frame staging handout.

use std::sync::mpsc;

use super::slot::{
    UPLOAD_ARENA_SLOTS, UploadArenaSlot, UploadArenaSlotState, create_persistent_slot_buffer,
    create_temporary_staging_buffer, log_oversized_upload, next_slot_capacity,
    select_writable_slot,
};
use super::staging::{
    PreparedUploadStaging, UploadArenaAcquireStats, UploadArenaPressure, UploadStagingSource,
};

/// One completed upload slot event delivered from a wgpu callback to the main thread.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum UploadArenaCompletion {
    /// Submitted GPU work that references this slot has completed.
    Submitted { slot: usize, generation: u64 },
    /// A post-submit remap request has completed.
    Remapped {
        slot: usize,
        generation: u64,
        success: bool,
    },
}

/// Persistent triple-buffered upload staging arena for render-graph buffer writes.
pub(crate) struct PersistentUploadArena {
    slots: [UploadArenaSlot; UPLOAD_ARENA_SLOTS],
    completion_tx: mpsc::Sender<UploadArenaCompletion>,
    completion_rx: mpsc::Receiver<UploadArenaCompletion>,
    next_generation: u64,
}

impl PersistentUploadArena {
    /// Creates an empty arena. Slots allocate lazily on first use.
    pub(crate) fn new() -> Self {
        let (completion_tx, completion_rx) = mpsc::channel();
        Self {
            slots: std::array::from_fn(|_| UploadArenaSlot::empty()),
            completion_tx,
            completion_rx,
            next_generation: 1,
        }
    }

    /// Drains submitted-work and remap callbacks, then polls once to advance pending remaps.
    pub(crate) fn maintain(&mut self, device: &wgpu::Device) {
        profiling::scope!("frame_upload_arena::maintain");
        self.drain_completions();
        let _ = device.poll(wgpu::PollType::Poll);
        self.drain_completions();
    }

    /// Drops all retained staging slots and ignores stale completion callbacks from older slots.
    pub(crate) fn reset(&mut self) {
        profiling::scope!("frame_upload_arena::reset");
        self.drain_pending_completion_events();
        self.next_generation = self.next_generation.saturating_add(1).max(1);

        let released_capacity = self
            .slots
            .iter()
            .fold(0u64, |total, slot| total.saturating_add(slot.capacity));
        for slot in &mut self.slots {
            slot.buffer = None;
            slot.capacity = 0;
            slot.state = UploadArenaSlotState::Empty;
        }

        logger::debug!(
            "frame upload arena: reset persistent slots released_capacity_bytes={released_capacity}"
        );
    }

    /// Prepares staging storage for `required` aligned bytes.
    pub(crate) fn prepare_staging_buffer(
        &mut self,
        device: &wgpu::Device,
        max_buffer_size: u64,
        required: u64,
        staged_writes: usize,
    ) -> PreparedUploadStaging {
        profiling::scope!("frame_upload_arena::prepare_staging");
        if required == 0 {
            return PreparedUploadStaging {
                buffer: None,
                source: UploadStagingSource::None,
                size: 0,
                acquire_stats: UploadArenaAcquireStats::default(),
            };
        }
        if required > max_buffer_size {
            log_oversized_upload(required, max_buffer_size);
            return PreparedUploadStaging {
                buffer: None,
                source: UploadStagingSource::QueueFallbackOversized,
                size: required,
                acquire_stats: UploadArenaAcquireStats {
                    oversized_queue_fallback_writes: staged_writes,
                    ..UploadArenaAcquireStats::default()
                },
            };
        }

        if let Some(slot) = select_writable_slot(&self.slots, required) {
            return self.prepare_persistent_slot(device, max_buffer_size, required, slot);
        }

        logger::debug!(
            "frame upload arena: no persistent slot available; using temporary staging buffer bytes={required}"
        );
        PreparedUploadStaging {
            buffer: Some(create_temporary_staging_buffer(device, required)),
            source: UploadStagingSource::Temporary,
            size: required,
            acquire_stats: UploadArenaAcquireStats {
                temporary_staging_bytes: required,
                temporary_staging_fallbacks: 1,
                ..UploadArenaAcquireStats::default()
            },
        }
    }

    /// Current pressure/capacity sample for diagnostics.
    pub(crate) fn pressure(&self) -> UploadArenaPressure {
        let mut pressure = UploadArenaPressure::default();
        for slot in &self.slots {
            pressure.capacity_bytes = pressure.capacity_bytes.saturating_add(slot.capacity);
            match slot.state {
                UploadArenaSlotState::Free => {
                    pressure.free_slots = pressure.free_slots.saturating_add(1);
                }
                UploadArenaSlotState::InFlight { .. } => {
                    pressure.in_flight_slots = pressure.in_flight_slots.saturating_add(1);
                }
                UploadArenaSlotState::Remapping { .. } => {
                    pressure.remapping_slots = pressure.remapping_slots.saturating_add(1);
                }
                UploadArenaSlotState::Empty | UploadArenaSlotState::Writing { .. } => {}
            }
        }
        pressure
    }

    /// Returns the next slot generation for reset-policy unit tests.
    #[cfg(test)]
    pub(crate) fn next_generation_for_tests(&self) -> u64 {
        self.next_generation
    }

    fn prepare_persistent_slot(
        &mut self,
        device: &wgpu::Device,
        max_buffer_size: u64,
        required: u64,
        slot: usize,
    ) -> PreparedUploadStaging {
        let generation = self.next_generation;
        self.next_generation = self.next_generation.saturating_add(1).max(1);

        let Some(arena_slot) = self.slots.get_mut(slot) else {
            return PreparedUploadStaging {
                buffer: Some(create_temporary_staging_buffer(device, required)),
                source: UploadStagingSource::Temporary,
                size: required,
                acquire_stats: UploadArenaAcquireStats {
                    temporary_staging_bytes: required,
                    temporary_staging_fallbacks: 1,
                    ..UploadArenaAcquireStats::default()
                },
            };
        };

        let mut acquire_stats = UploadArenaAcquireStats {
            persistent_staging_bytes: required,
            ..UploadArenaAcquireStats::default()
        };
        if arena_slot.buffer.is_none() || arena_slot.capacity < required {
            let Some(capacity) = next_slot_capacity(required, arena_slot.capacity, max_buffer_size)
            else {
                return PreparedUploadStaging {
                    buffer: None,
                    source: UploadStagingSource::QueueFallbackOversized,
                    size: required,
                    acquire_stats: UploadArenaAcquireStats {
                        oversized_queue_fallback_writes: 1,
                        ..UploadArenaAcquireStats::default()
                    },
                };
            };
            arena_slot.buffer = Some(create_persistent_slot_buffer(device, capacity));
            arena_slot.capacity = capacity;
            acquire_stats.persistent_slot_grows = 1;
        } else {
            acquire_stats.persistent_slot_reuses = 1;
        }
        arena_slot.state = UploadArenaSlotState::Writing { generation };
        PreparedUploadStaging {
            buffer: arena_slot.buffer.clone(),
            source: UploadStagingSource::Persistent { slot, generation },
            size: required,
            acquire_stats,
        }
    }

    pub(super) fn finish_persistent_write(
        &mut self,
        slot: usize,
        generation: u64,
    ) -> Option<Box<dyn FnOnce() + Send + 'static>> {
        let arena_slot = self.slots.get_mut(slot)?;
        if arena_slot.state != (UploadArenaSlotState::Writing { generation }) {
            return None;
        }
        let buffer = arena_slot.buffer.as_ref()?;
        buffer.unmap();
        arena_slot.state = UploadArenaSlotState::InFlight { generation };
        let tx = self.completion_tx.clone();
        Some(Box::new(move || {
            let _ = tx.send(UploadArenaCompletion::Submitted { slot, generation });
        }))
    }

    fn drain_completions(&mut self) {
        while let Ok(completion) = self.completion_rx.try_recv() {
            match completion {
                UploadArenaCompletion::Submitted { slot, generation } => {
                    self.start_remap(slot, generation);
                }
                UploadArenaCompletion::Remapped {
                    slot,
                    generation,
                    success,
                } => self.finish_remap(slot, generation, success),
            }
        }
    }

    fn drain_pending_completion_events(&self) {
        while self.completion_rx.try_recv().is_ok() {}
    }

    fn start_remap(&mut self, slot: usize, generation: u64) {
        let Some(arena_slot) = self.slots.get_mut(slot) else {
            return;
        };
        if arena_slot.state != (UploadArenaSlotState::InFlight { generation }) {
            return;
        }
        let Some(buffer) = arena_slot.buffer.clone() else {
            arena_slot.state = UploadArenaSlotState::Empty;
            arena_slot.capacity = 0;
            return;
        };
        arena_slot.state = UploadArenaSlotState::Remapping { generation };
        let tx = self.completion_tx.clone();
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Write, move |result| {
                let _ = tx.send(UploadArenaCompletion::Remapped {
                    slot,
                    generation,
                    success: result.is_ok(),
                });
            });
    }

    fn finish_remap(&mut self, slot: usize, generation: u64, success: bool) {
        let Some(arena_slot) = self.slots.get_mut(slot) else {
            return;
        };
        if arena_slot.state != (UploadArenaSlotState::Remapping { generation }) {
            return;
        }
        if success {
            arena_slot.state = UploadArenaSlotState::Free;
            return;
        }
        logger::warn!("frame upload arena: persistent slot remap failed; dropping slot");
        arena_slot.buffer = None;
        arena_slot.capacity = 0;
        arena_slot.state = UploadArenaSlotState::Empty;
    }
}

impl Default for PersistentUploadArena {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stale_completion_does_not_free_newer_generation() {
        let mut arena = PersistentUploadArena::new();
        arena.slots[0].state = UploadArenaSlotState::InFlight { generation: 2 };

        arena.start_remap(0, 1);

        assert_eq!(
            arena.slots[0].state,
            UploadArenaSlotState::InFlight { generation: 2 }
        );
    }

    #[test]
    fn successful_remap_returns_slot_to_free_state() {
        let mut arena = PersistentUploadArena::new();
        arena.slots[0].state = UploadArenaSlotState::Remapping { generation: 4 };

        arena.finish_remap(0, 4, true);

        assert_eq!(arena.slots[0].state, UploadArenaSlotState::Free);
    }

    #[test]
    fn failed_remap_drops_slot() {
        let mut arena = PersistentUploadArena::new();
        arena.slots[0].state = UploadArenaSlotState::Remapping { generation: 4 };
        arena.slots[0].capacity = 1024;

        arena.finish_remap(0, 4, false);

        assert_eq!(arena.slots[0].state, UploadArenaSlotState::Empty);
        assert_eq!(arena.slots[0].capacity, 0);
    }

    #[test]
    fn reset_drops_slots_and_ignores_stale_completions() {
        let mut arena = PersistentUploadArena::new();
        arena.slots[0].state = UploadArenaSlotState::Free;
        arena.slots[0].capacity = 64;
        arena.slots[1].state = UploadArenaSlotState::InFlight { generation: 7 };
        arena.slots[1].capacity = 128;
        arena.slots[2].state = UploadArenaSlotState::Remapping { generation: 8 };
        arena.slots[2].capacity = 256;

        arena.reset();
        let next_generation_after_reset = arena.next_generation;
        let _ = arena.completion_tx.send(UploadArenaCompletion::Remapped {
            slot: 2,
            generation: 8,
            success: true,
        });
        arena.drain_completions();

        for slot in &arena.slots {
            assert_eq!(slot.state, UploadArenaSlotState::Empty);
            assert_eq!(slot.capacity, 0);
        }
        assert!(
            arena.next_generation > 1,
            "reset must advance generations away from stale callbacks"
        );
        assert_eq!(arena.next_generation, next_generation_after_reset);
    }

    #[test]
    fn pressure_counts_in_flight_and_remapping_slots() {
        let mut arena = PersistentUploadArena::new();
        arena.slots[0].state = UploadArenaSlotState::Free;
        arena.slots[0].capacity = 64;
        arena.slots[1].state = UploadArenaSlotState::InFlight { generation: 1 };
        arena.slots[1].capacity = 128;
        arena.slots[2].state = UploadArenaSlotState::Remapping { generation: 2 };
        arena.slots[2].capacity = 256;

        let pressure = arena.pressure();

        assert_eq!(pressure.capacity_bytes, 448);
        assert_eq!(pressure.free_slots, 1);
        assert_eq!(pressure.in_flight_slots, 1);
        assert_eq!(pressure.remapping_slots, 1);
    }
}
