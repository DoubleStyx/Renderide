//! Tracks GPU events that can invalidate CPU-mapped staging/readback buffers.

use std::sync::atomic::{AtomicU64, Ordering};

/// Shared generation counter for mapped-buffer invalidation events.
///
/// The renderer has several long-lived mapped-buffer paths: frame uploads, Hi-Z readback, and
/// frame-bracket timestamp readback. Surface validation failures and device-loss events can leave
/// those buffers invalid even though higher-level owners still hold handles. This counter gives
/// frame setup a cheap way to observe that all such state needs to be discarded before another
/// mapped range is requested.
#[derive(Debug, Default)]
pub(crate) struct GpuMappedBufferHealth {
    generation: AtomicU64,
}

impl GpuMappedBufferHealth {
    /// Creates a health tracker with no invalidation events recorded.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Records that mapped staging/readback buffers should be considered invalid.
    pub(crate) fn mark_invalid(&self, reason: impl AsRef<str>) -> u64 {
        let generation = self.generation.fetch_add(1, Ordering::AcqRel) + 1;
        let reason = reason.as_ref();
        if generation <= 5 || generation.is_multiple_of(120) {
            logger::warn!(
                "GPU mapped-buffer recovery requested: generation={generation} reason={reason}"
            );
        } else {
            logger::debug!(
                "GPU mapped-buffer recovery requested: generation={generation} reason={reason}"
            );
        }
        generation
    }

    /// Current invalidation generation.
    pub(crate) fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }
}

/// Returns true when a wgpu validation report is likely about a mapped-buffer owner that must be
/// discarded before the next frame touches it.
pub(crate) fn validation_mentions_mapped_buffer_invalidation(
    description: &str,
    source: &str,
) -> bool {
    let buffer_related = description.contains("Buffer") || source.contains("Buffer");
    let invalid_resource = source.contains("has been destroyed") || source.contains("is invalid");
    let mapped_operation = source.contains("Buffer::buffer_unmap")
        || source.contains("Buffer::map_async")
        || source.contains("Buffer::get_mapped_range")
        || source.contains("frame_upload_arena_slot")
        || source.contains("hi_z_staging")
        || source.contains("frame_bracket_readback");
    buffer_related && invalid_resource && mapped_operation
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_advances_on_invalidations() {
        let health = GpuMappedBufferHealth::new();

        assert_eq!(health.generation(), 0);
        assert_eq!(health.mark_invalid("first"), 1);
        assert_eq!(health.mark_invalid("second"), 2);
        assert_eq!(health.generation(), 2);
    }

    #[test]
    fn validation_filter_matches_destroyed_mapped_buffers() {
        assert!(validation_mentions_mapped_buffer_invalidation(
            "Validation Error",
            "In Buffer::buffer_unmap Buffer with 'hi_z_staging_r_0' label has been destroyed",
        ));
        assert!(validation_mentions_mapped_buffer_invalidation(
            "Validation Error",
            "In Buffer::map_async Buffer with 'frame_bracket_readback' label is invalid",
        ));
        assert!(validation_mentions_mapped_buffer_invalidation(
            "Validation Error",
            "In Buffer::get_mapped_range Buffer with 'mesh 12 vertices' label is invalid",
        ));
        assert!(!validation_mentions_mapped_buffer_invalidation(
            "Validation Error",
            "Texture view format mismatch",
        ));
    }
}
