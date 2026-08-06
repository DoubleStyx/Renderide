//! Front-to-back granularity for opaque draws in the packed sort prefix.
//!
//! The opaque depth bucket sits above the batch-key hash in
//! [`super::pack_sort_prefix`], so it is the dominant ordering term for opaque draws. Every extra
//! bit of depth resolution splits instance runs that would otherwise merge: the same mesh and
//! material landing in two buckets becomes two draw groups.
//!
//! Measured in a heavy world: 4988 draws collapsed to only 1522 instance batches, 3.3 draws per
//! batch, against ~529 unique meshes. Most of that fragmentation is depth bucketing, not material
//! diversity.
//!
//! The forward pass runs behind a depth prepass and GPU occlusion culling, so front-to-back order
//! inside the forward pass is worth far less here than it would be in a single-pass renderer.
//! Trading depth resolution for batching is therefore usually a win on the CPU (fewer draw
//! submissions, cheaper `CommandEncoder::finish`) and on the GPU (fewer state changes).
//!
//! Default keeps the historical 8 bits so behavior is unchanged until the knob is set.

use std::sync::LazyLock;

/// Process environment override for opaque depth-bucket resolution.
const OPAQUE_DEPTH_BUCKET_BITS_ENV: &str = "RENDERIDE_OPAQUE_DEPTH_BUCKET_BITS";

/// Historical resolution: 256 logarithmic front-to-back buckets.
const DEFAULT_OPAQUE_DEPTH_BUCKET_BITS: u32 = 8;
/// Widest resolution the prefix layout can carry.
const MAX_OPAQUE_DEPTH_BUCKET_BITS: u32 = 8;

/// Bits of opaque depth resolution kept in the sort prefix.
///
/// `0` sorts opaque draws purely by batch key, maximizing instancing and abandoning front-to-back
/// order. `8` is the historical behavior. Values in between coarsen the buckets: `n` bits keeps the
/// high `n` bits of the log2 distance bucket, so `3` gives 8 coarse shells that still draw roughly
/// near-to-far while letting each shell batch properly.
pub(crate) fn opaque_depth_bucket_bits() -> u32 {
    static SELECTED: LazyLock<u32> = LazyLock::new(|| {
        let Some(raw) = std::env::var(OPAQUE_DEPTH_BUCKET_BITS_ENV).ok() else {
            return DEFAULT_OPAQUE_DEPTH_BUCKET_BITS;
        };
        match raw.trim().parse::<u32>() {
            Ok(bits) if bits <= MAX_OPAQUE_DEPTH_BUCKET_BITS => bits,
            _ => {
                logger::warn!(
                    "Invalid {OPAQUE_DEPTH_BUCKET_BITS_ENV}={raw:?}; expected 0..={MAX_OPAQUE_DEPTH_BUCKET_BITS}, using {DEFAULT_OPAQUE_DEPTH_BUCKET_BITS}"
                );
                DEFAULT_OPAQUE_DEPTH_BUCKET_BITS
            }
        }
    });
    *SELECTED
}

/// Reduces a full-resolution depth bucket to the configured granularity.
///
/// Shifting off the low bits merges neighbouring shells, so draws that differ only in fine depth
/// end up with an identical prefix and can share one instanced draw.
#[inline]
pub(crate) fn coarsen_opaque_depth_bucket(bucket: u16, bits: u32) -> u64 {
    if bits == 0 {
        return 0;
    }
    let clamped = u64::from(bucket.min((1u16 << MAX_OPAQUE_DEPTH_BUCKET_BITS) - 1));
    clamped >> (MAX_OPAQUE_DEPTH_BUCKET_BITS - bits)
}

/// Emits the startup banner for the selected granularity.
pub(crate) fn log_opaque_depth_bucket_policy() {
    let bits = opaque_depth_bucket_bits();
    logger::info!(
        "Opaque depth-bucket sort resolution: bits={bits} buckets={} (0 maximizes instancing, {DEFAULT_OPAQUE_DEPTH_BUCKET_BITS} is front-to-back)",
        1u32 << bits,
    );
}

#[cfg(test)]
mod tests {
    use super::{MAX_OPAQUE_DEPTH_BUCKET_BITS, coarsen_opaque_depth_bucket};

    #[test]
    fn zero_bits_collapses_every_bucket() {
        for bucket in [0u16, 1, 37, 128, 255] {
            assert_eq!(coarsen_opaque_depth_bucket(bucket, 0), 0);
        }
    }

    #[test]
    fn full_bits_preserve_the_bucket() {
        for bucket in [0u16, 1, 37, 128, 255] {
            assert_eq!(
                coarsen_opaque_depth_bucket(bucket, MAX_OPAQUE_DEPTH_BUCKET_BITS),
                u64::from(bucket)
            );
        }
    }

    #[test]
    fn coarsening_merges_neighbours_but_keeps_far_apart_shells_ordered() {
        // 3 bits keeps 8 shells, so buckets 32 apart stay distinct while adjacent ones merge
        let near = coarsen_opaque_depth_bucket(64, 3);
        let also_near = coarsen_opaque_depth_bucket(95, 3);
        let far = coarsen_opaque_depth_bucket(200, 3);

        assert_eq!(
            near, also_near,
            "adjacent shells must merge to batch together"
        );
        assert!(near < far, "coarse shells still order near before far");
    }

    #[test]
    fn coarsening_is_monotonic() {
        let mut previous = 0;
        for bucket in 0u16..=255 {
            let coarse = coarsen_opaque_depth_bucket(bucket, 4);
            assert!(coarse >= previous);
            previous = coarse;
        }
    }
}
