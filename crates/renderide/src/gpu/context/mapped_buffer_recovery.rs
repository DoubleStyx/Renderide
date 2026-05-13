//! [`GpuMappedBufferRecovery`] sub-handle for CPU-mapped GPU buffer recovery policy.
//!
//! The renderer marks mapped staging/readback buffers as "invalid" when wgpu validation flags an
//! interrupted map or device-loss-style event. A small per-frame state machine then short-circuits
//! mapped-buffer use for the next [`MAPPED_BUFFER_RECOVERY_FRAMES`] frames so callers can rebuild
//! their pools.

use std::sync::Arc;

use crate::gpu::sync::mapped_buffer_health::GpuMappedBufferHealth;

const MAPPED_BUFFER_RECOVERY_FRAMES: u8 = 2;

/// Result of beginning one mapped-buffer recovery frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct MappedBufferRecoveryFrame {
    /// Current invalidation generation after observing the shared health counter.
    pub(crate) generation: u64,
    /// Whether a new invalidation generation was observed at this frame boundary.
    pub(crate) invalidated: bool,
    /// Whether this frame must avoid CPU-mapped staging/readback buffers.
    pub(crate) avoid_mapped_buffers: bool,
}

/// Per-frame mapped-buffer recovery state.
#[derive(Debug)]
pub(crate) struct GpuMappedBufferRecovery {
    /// Shared mapped-buffer invalidation generation set by wgpu and surface error paths.
    health: Arc<GpuMappedBufferHealth>,
    /// Last mapped-buffer invalidation generation handled by frame setup.
    seen_generation: u64,
    /// Number of upcoming frames that must avoid CPU-mapped staging/readback buffers.
    frames_remaining: u8,
    /// Whether the current frame must avoid CPU-mapped staging/readback buffers.
    avoid_this_frame: bool,
}

impl GpuMappedBufferRecovery {
    /// Builds an idle recovery handle that shares `health` with the rest of the GPU stack.
    pub(crate) fn new(health: Arc<GpuMappedBufferHealth>) -> Self {
        Self {
            health,
            seen_generation: 0,
            frames_remaining: 0,
            avoid_this_frame: false,
        }
    }

    /// Records that mapped staging/readback buffers should be discarded before reuse.
    pub(crate) fn mark_mapped_buffers_invalid(&self, reason: impl AsRef<str>) {
        self.health.mark_invalid(reason);
    }

    /// Shared mapped-buffer invalidation generation for GPU owners outside [`GpuContext`].
    pub(crate) fn health(&self) -> Arc<GpuMappedBufferHealth> {
        Arc::clone(&self.health)
    }

    /// Begins mapped-buffer recovery bookkeeping for a render frame.
    pub(crate) fn begin_mapped_buffer_recovery_frame(&mut self) -> MappedBufferRecoveryFrame {
        let generation = self.health.generation();
        let invalidated = self.observe_generation(generation);

        self.avoid_this_frame = self.frames_remaining > 0;
        if self.frames_remaining > 0 {
            self.frames_remaining = self.frames_remaining.saturating_sub(1);
        }

        MappedBufferRecoveryFrame {
            generation,
            invalidated,
            avoid_mapped_buffers: self.avoid_this_frame,
        }
    }

    /// Observes invalidations reported by wgpu while the current frame is already running.
    pub(crate) fn observe_mapped_buffer_invalidation_during_frame(&mut self) -> bool {
        let generation = self.health.generation();
        let invalidated = self.observe_generation(generation);
        if invalidated {
            self.avoid_this_frame = true;
        }
        invalidated
    }

    /// Whether this frame should avoid CPU-mapped staging/readback buffers.
    pub(crate) fn avoid_mapped_buffers_this_frame(&self) -> bool {
        self.avoid_this_frame
    }

    /// Current mapped-buffer invalidation generation.
    pub(crate) fn mapped_buffer_invalidation_generation(&self) -> u64 {
        self.health.generation()
    }

    fn observe_generation(&mut self, generation: u64) -> bool {
        if generation == self.seen_generation {
            return false;
        }
        self.seen_generation = generation;
        self.frames_remaining = MAPPED_BUFFER_RECOVERY_FRAMES;
        true
    }
}
