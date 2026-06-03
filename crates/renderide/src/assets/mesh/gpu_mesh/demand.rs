//! Derived vertex-stream demand and dirty-state policy for mesh uploads.

use crate::materials::EmbeddedTangentFallbackMode;
use crate::shared::MeshUploadData;

/// Bit mask identifying derived mesh streams that may be uploaded separately from the host's
/// interleaved vertex buffer.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub(crate) struct MeshDerivedStreamMask(u16);

impl MeshDerivedStreamMask {
    /// No derived streams.
    pub(crate) const EMPTY: Self = Self(0);
    /// Decomposed position stream.
    pub(crate) const POSITION: Self = Self(1 << 0);
    /// Decomposed normal stream.
    pub(crate) const NORMAL: Self = Self(1 << 1);
    /// Compact UV0 stream.
    pub(crate) const UV0: Self = Self(1 << 2);
    /// Vertex color stream.
    pub(crate) const COLOR: Self = Self(1 << 3);
    /// Sanitized geometric tangent stream.
    pub(crate) const TANGENT: Self = Self(1 << 4);
    /// Raw tangent payload stream.
    pub(crate) const RAW_TANGENT: Self = Self(1 << 5);
    /// Compact UV1 stream.
    pub(crate) const UV1: Self = Self(1 << 6);
    /// Compact UV2 stream.
    pub(crate) const UV2: Self = Self(1 << 7);
    /// Compact UV3 stream.
    pub(crate) const UV3: Self = Self(1 << 8);
    /// Packed UV0-UV3 stream for wide low UV inputs.
    pub(crate) const WIDE_UV_LOW: Self = Self(1 << 9);
    /// Packed UV4-UV7 stream for high UV inputs.
    pub(crate) const WIDE_UV_HIGH: Self = Self(1 << 10);
    /// Streams affected when interleaved vertex data changes.
    pub(crate) const VERTEX_DERIVED: Self = Self(
        Self::POSITION.0
            | Self::NORMAL.0
            | Self::UV0.0
            | Self::COLOR.0
            | Self::TANGENT.0
            | Self::RAW_TANGENT.0
            | Self::UV1.0
            | Self::UV2.0
            | Self::UV3.0
            | Self::WIDE_UV_LOW.0
            | Self::WIDE_UV_HIGH.0,
    );
    /// Streams affected when index order changes.
    pub(crate) const INDEX_DERIVED: Self = Self(Self::TANGENT.0);
    /// Primary streams every drawable world mesh currently needs for vertex binding.
    pub(crate) const DRAWABLE_PRIMARY: Self = Self(Self::POSITION.0 | Self::NORMAL.0);
    /// Streams produced by generated particle meshes in steady state.
    pub(crate) const GENERATED_PARTICLE: Self =
        Self(Self::POSITION.0 | Self::NORMAL.0 | Self::UV0.0 | Self::COLOR.0);

    /// Returns the raw bit representation.
    #[inline]
    pub(crate) const fn bits(self) -> u16 {
        self.0
    }

    /// Returns true when the mask has no streams set.
    #[inline]
    pub(crate) const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Returns true when every bit in `other` is present.
    #[inline]
    pub(crate) const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Returns true when any bit in `other` is present.
    #[inline]
    pub(crate) const fn intersects(self, other: Self) -> bool {
        (self.0 & other.0) != 0
    }

    /// Returns `self` with every bit in `other` added.
    #[inline]
    pub(crate) const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Returns `self` without any bits in `other`.
    #[inline]
    pub(crate) const fn without(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }

    /// Returns the intersection of both masks.
    #[inline]
    pub(crate) const fn intersection(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    /// Runtime-required streams independent of material reflection.
    pub(crate) fn runtime_required(data: &MeshUploadData) -> Self {
        if data.vertex_count > 0 {
            Self::DRAWABLE_PRIMARY
        } else {
            Self::EMPTY
        }
    }
}

impl std::ops::BitOr for MeshDerivedStreamMask {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        self.union(rhs)
    }
}

impl std::ops::BitOrAssign for MeshDerivedStreamMask {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = self.union(rhs);
    }
}

impl std::ops::BitAnd for MeshDerivedStreamMask {
    type Output = Self;

    #[inline]
    fn bitand(self, rhs: Self) -> Self::Output {
        self.intersection(rhs)
    }
}

impl std::ops::Sub for MeshDerivedStreamMask {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        self.without(rhs)
    }
}

/// Material-driven stream demand for one mesh.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct MeshDerivedStreamDemand {
    /// Requested derived streams.
    pub(crate) mask: MeshDerivedStreamMask,
    /// Strongest requested tangent fallback policy.
    pub(crate) tangent_fallback_mode: EmbeddedTangentFallbackMode,
}

impl MeshDerivedStreamDemand {
    /// Empty demand.
    pub(crate) const EMPTY: Self = Self {
        mask: MeshDerivedStreamMask::EMPTY,
        tangent_fallback_mode: EmbeddedTangentFallbackMode::PreserveHostOrDefault,
    };

    /// Demand for generated particle meshes.
    pub(crate) const GENERATED_PARTICLE: Self = Self {
        mask: MeshDerivedStreamMask::GENERATED_PARTICLE,
        tangent_fallback_mode: EmbeddedTangentFallbackMode::PreserveHostOrDefault,
    };

    /// Returns demand with runtime-required streams added.
    #[inline]
    pub(crate) fn with_runtime_required(self, data: &MeshUploadData) -> Self {
        Self {
            mask: self.mask | MeshDerivedStreamMask::runtime_required(data),
            tangent_fallback_mode: self.tangent_fallback_mode,
        }
    }

    /// Merges another demand into this demand.
    #[inline]
    pub(crate) fn merge(&mut self, other: Self) {
        self.mask |= other.mask;
        self.tangent_fallback_mode = self.tangent_fallback_mode.max(other.tangent_fallback_mode);
    }
}

/// Per-mesh derived-stream state used to decide which skipped streams need a later rebuild.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct MeshDerivedStreamState {
    /// Streams requested by currently known materials or runtime mesh processing.
    pub(crate) demand_mask: MeshDerivedStreamMask,
    /// Streams whose retained source is newer than the resident derived buffer, or whose buffer was
    /// skipped and can be built lazily later.
    pub(crate) dirty_mask: MeshDerivedStreamMask,
}

impl MeshDerivedStreamState {
    /// Creates state for a full upload.
    pub(crate) fn after_full_upload(
        demand: MeshDerivedStreamDemand,
        available_mask: MeshDerivedStreamMask,
        rebuildable_mask: MeshDerivedStreamMask,
    ) -> Self {
        let dirty_mask = rebuildable_mask.without(available_mask);
        Self {
            demand_mask: demand.mask,
            dirty_mask,
        }
    }

    /// Returns updated state after an in-place upload touched a subset of derived sources.
    pub(crate) fn after_in_place_update(
        self,
        demand: MeshDerivedStreamDemand,
        changed_mask: MeshDerivedStreamMask,
        rebuildable_mask: MeshDerivedStreamMask,
    ) -> Self {
        let written_now = changed_mask.intersection(demand.mask);
        let skipped_now = changed_mask
            .without(demand.mask)
            .intersection(rebuildable_mask);
        Self {
            demand_mask: demand.mask,
            dirty_mask: self.dirty_mask.without(written_now).union(skipped_now),
        }
    }

    /// Records new material demand and returns whether it changed this state.
    pub(crate) fn record_demand(&mut self, demand: MeshDerivedStreamDemand) -> bool {
        let before = *self;
        self.demand_mask |= demand.mask;
        *self != before
    }

    /// Marks a stream rebuilt from current retained source.
    #[inline]
    pub(crate) fn mark_clean(&mut self, mask: MeshDerivedStreamMask) {
        self.dirty_mask = self.dirty_mask.without(mask);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_required_includes_primary_streams_for_non_empty_meshes() {
        let data = MeshUploadData {
            vertex_count: 1,
            ..Default::default()
        };

        let demand = MeshDerivedStreamDemand::EMPTY.with_runtime_required(&data);

        assert!(
            demand
                .mask
                .contains(MeshDerivedStreamMask::DRAWABLE_PRIMARY)
        );
    }

    #[test]
    fn runtime_required_keeps_empty_meshes_streamless() {
        let data = MeshUploadData {
            vertex_count: 0,
            ..Default::default()
        };

        let demand = MeshDerivedStreamDemand::EMPTY.with_runtime_required(&data);

        assert!(demand.mask.is_empty());
    }

    #[test]
    fn demand_merge_keeps_all_bits_and_strongest_tangent_mode() {
        let mut demand = MeshDerivedStreamDemand {
            mask: MeshDerivedStreamMask::UV1,
            tangent_fallback_mode: EmbeddedTangentFallbackMode::PreserveHostOrDefault,
        };

        demand.merge(MeshDerivedStreamDemand {
            mask: MeshDerivedStreamMask::TANGENT,
            tangent_fallback_mode: EmbeddedTangentFallbackMode::GenerateMissing,
        });

        assert!(demand.mask.contains(MeshDerivedStreamMask::UV1));
        assert!(demand.mask.contains(MeshDerivedStreamMask::TANGENT));
        assert_eq!(
            demand.tangent_fallback_mode,
            EmbeddedTangentFallbackMode::GenerateMissing
        );
    }

    #[test]
    fn full_upload_marks_rebuildable_skipped_streams_dirty() {
        let state = MeshDerivedStreamState::after_full_upload(
            MeshDerivedStreamDemand {
                mask: MeshDerivedStreamMask::DRAWABLE_PRIMARY,
                tangent_fallback_mode: EmbeddedTangentFallbackMode::default(),
            },
            MeshDerivedStreamMask::DRAWABLE_PRIMARY,
            MeshDerivedStreamMask::DRAWABLE_PRIMARY | MeshDerivedStreamMask::COLOR,
        );

        assert!(state.dirty_mask.contains(MeshDerivedStreamMask::COLOR));
        assert!(!state.dirty_mask.contains(MeshDerivedStreamMask::NORMAL));
    }

    #[test]
    fn in_place_update_marks_skipped_stream_dirty_and_cleans_written_stream() {
        let initial = MeshDerivedStreamState {
            demand_mask: MeshDerivedStreamMask::COLOR,
            dirty_mask: MeshDerivedStreamMask::COLOR,
        };

        let state = initial.after_in_place_update(
            MeshDerivedStreamDemand {
                mask: MeshDerivedStreamMask::UV0,
                tangent_fallback_mode: EmbeddedTangentFallbackMode::default(),
            },
            MeshDerivedStreamMask::UV0 | MeshDerivedStreamMask::COLOR,
            MeshDerivedStreamMask::UV0 | MeshDerivedStreamMask::COLOR,
        );

        assert!(state.dirty_mask.contains(MeshDerivedStreamMask::COLOR));
        assert!(!state.dirty_mask.contains(MeshDerivedStreamMask::UV0));
    }

    #[test]
    fn mark_clean_removes_rebuilt_streams() {
        let mut state = MeshDerivedStreamState {
            demand_mask: MeshDerivedStreamMask::COLOR,
            dirty_mask: MeshDerivedStreamMask::COLOR | MeshDerivedStreamMask::UV1,
        };

        state.mark_clean(MeshDerivedStreamMask::COLOR);

        assert!(!state.dirty_mask.contains(MeshDerivedStreamMask::COLOR));
        assert!(state.dirty_mask.contains(MeshDerivedStreamMask::UV1));
    }
}
