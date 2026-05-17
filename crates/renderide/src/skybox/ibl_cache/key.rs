//! IBL bake keys and scalar helper math.

use std::hash::{Hash, Hasher};

use crate::gpu::GpuLimits;
use crate::skybox::specular::SkyboxIblSource;

/// Compute workgroup edge used by every mip-0 producer and the GGX convolve.
const IBL_WORKGROUP_EDGE: u32 = 8;
/// Base GGX importance sample count for mip 1; doubles per mip up to [`IBL_MAX_SAMPLES`].
const IBL_BASE_SAMPLE_COUNT: u32 = 64;
/// Cap on GGX importance sample count for the highest-roughness mips.
const IBL_MAX_SAMPLES: u32 = 1024;

/// Clamps the configured cube face size against the device texture limit.
pub(crate) fn clamp_face_size(face_size: u32, limits: &GpuLimits) -> u32 {
    face_size.min(limits.max_texture_dimension_2d()).max(1)
}

/// Identity for one IBL bake.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub(crate) enum SkyboxIblKey {
    /// Host-uploaded cubemap identity.
    Cubemap {
        /// Material asset id when this source came from a material, or `-1` for direct probe sources.
        material_asset_id: i32,
        /// Material property generation when this source came from a material.
        material_generation: u64,
        /// Stable hash of the shader route stem when this source came from a material.
        route_hash: u64,
        /// Source cubemap asset id.
        asset_id: i32,
        /// Source GPU allocation generation.
        allocation_generation: u64,
        /// Source resident mip count; growth re-bakes once more mips arrive.
        mip_levels_resident: u32,
        /// Source content generation; re-uploading the same mips re-bakes.
        content_generation: u64,
        /// Storage V-flip flag for the source cube.
        storage_v_inverted: bool,
        /// Destination cube face edge.
        face_size: u32,
    },
    /// Constant-color reflection-probe identity.
    SolidColor {
        /// Renderer-side identity for this color source.
        identity: u64,
        /// Linear RGBA color bit hash.
        color_hash: u64,
        /// Destination cube face edge.
        face_size: u32,
    },
    /// Renderer-captured OnChanges reflection-probe cubemap identity.
    RuntimeCubemap {
        /// Render space that owns the captured probe.
        render_space_id: i32,
        /// Dense reflection-probe renderable index.
        renderable_index: i32,
        /// Monotonic renderer-side capture generation.
        generation: u64,
        /// Source mip count resident on the captured cubemap.
        mip_levels: u32,
        /// Storage V-flip flag for the captured source cube.
        storage_v_inverted: bool,
        /// Destination cube face edge.
        face_size: u32,
    },
}

impl SkyboxIblKey {
    /// Returns the destination face size for this bake.
    pub(super) fn face_size(&self) -> u32 {
        match *self {
            Self::Cubemap { face_size, .. }
            | Self::SolidColor { face_size, .. }
            | Self::RuntimeCubemap { face_size, .. } => face_size,
        }
    }
}

/// Builds a cache key for an active source using an already-clamped destination face size.
pub(crate) fn build_key(source: &SkyboxIblSource, face_size: u32) -> SkyboxIblKey {
    match source {
        SkyboxIblSource::Cubemap(src) => SkyboxIblKey::Cubemap {
            material_asset_id: src.material_asset_id,
            material_generation: src.material_generation,
            route_hash: src.route_hash,
            asset_id: src.asset_id,
            allocation_generation: src.allocation_generation,
            mip_levels_resident: src.mip_levels_resident,
            content_generation: src.content_generation,
            storage_v_inverted: src.storage_v_inverted,
            face_size,
        },
        SkyboxIblSource::SolidColor(src) => SkyboxIblKey::SolidColor {
            identity: src.identity,
            color_hash: hash_float4(&src.color),
            face_size,
        },
        SkyboxIblSource::RuntimeCubemap(src) => SkyboxIblKey::RuntimeCubemap {
            render_space_id: src.render_space_id,
            renderable_index: src.renderable_index,
            generation: src.generation,
            mip_levels: src.mip_levels,
            storage_v_inverted: src.storage_v_inverted,
            face_size,
        },
    }
}

/// Hashes four `f32`s by their bit patterns.
fn hash_float4(values: &[f32; 4]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for v in values {
        v.to_bits().hash(&mut hasher);
    }
    hasher.finish()
}

/// Returns the full mip count for a cube face edge.
pub(crate) fn mip_levels_for_edge(edge: u32) -> u32 {
    u32::BITS - edge.max(1).leading_zeros()
}

/// Returns the dispatch group count along one 8x8 compute dimension.
pub(super) fn dispatch_groups(size: u32) -> u32 {
    size.max(1).div_ceil(IBL_WORKGROUP_EDGE)
}

/// Returns a mip edge clamped to one texel.
pub(crate) fn mip_extent(base: u32, mip: u32) -> u32 {
    (base >> mip).max(1)
}

/// Returns the highest source mip LOD available to filtered importance sampling.
pub(super) fn source_max_lod(mip_levels: u32) -> f32 {
    mip_levels.saturating_sub(1) as f32
}

/// Returns the GGX importance sample count for the given convolve mip.
pub(super) fn convolve_sample_count(mip_index: u32) -> u32 {
    if mip_index == 0 {
        return 1;
    }
    let exponent = (mip_index - 1).min(4);
    (IBL_BASE_SAMPLE_COUNT << exponent).min(IBL_MAX_SAMPLES)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip: applying the runtime parabolic LOD then the inverse returns the input.
    #[test]
    fn roughness_lod_round_trip() {
        for i in 0..=20u32 {
            let r = i as f32 / 20.0;
            let lod = r * (2.0 - r);
            let r_back = 1.0 - (1.0 - lod).max(0.0).sqrt();
            assert!((r - r_back).abs() < 1e-6, "r={r} r_back={r_back}");
        }
    }

    /// Mip count includes mip 0 through the one-texel mip.
    #[test]
    fn mip_levels_for_edge_includes_tail_mip() {
        assert_eq!(mip_levels_for_edge(1), 1);
        assert_eq!(mip_levels_for_edge(2), 2);
        assert_eq!(mip_levels_for_edge(128), 8);
        assert_eq!(mip_levels_for_edge(256), 9);
    }

    /// Source-LOD clamping exposes every generated source mip to filtered importance sampling.
    #[test]
    fn source_max_lod_tracks_last_generated_mip() {
        assert_eq!(source_max_lod(0), 0.0);
        assert_eq!(source_max_lod(1), 0.0);
        assert_eq!(source_max_lod(8), 7.0);
    }

    /// Per-mip sample count clamps to the documented base/cap envelope.
    #[test]
    fn convolve_sample_count_envelope() {
        assert_eq!(convolve_sample_count(0), 1);
        assert_eq!(convolve_sample_count(1), 64);
        assert_eq!(convolve_sample_count(2), 128);
        assert_eq!(convolve_sample_count(3), 256);
        assert_eq!(convolve_sample_count(4), 512);
        assert_eq!(convolve_sample_count(5), 1024);
        assert_eq!(convolve_sample_count(8), 1024);
    }

    /// Runtime-capture storage orientation is part of the bake identity.
    #[test]
    fn runtime_cubemap_key_invalidates_on_storage_orientation() {
        let base = runtime_cubemap_key(false);
        let flipped = runtime_cubemap_key(true);

        assert_ne!(base, flipped);
    }

    /// Cubemap key invariants: residency growth and face size resize both invalidate.
    #[test]
    fn cubemap_key_invalidates_on_residency_or_face_change() {
        let a = cubemap_key(1, 1, 0, 1, 256);
        let b = cubemap_key(1, 1, 0, 4, 256);
        let c = cubemap_key(1, 1, 0, 1, 128);
        assert_ne!(a, b);
        assert_ne!(a, c);
        let d = cubemap_key(1, 2, 0, 1, 256);
        assert_ne!(a, d);
    }

    /// Cubemap allocation and material identity invalidate same-id sources.
    #[test]
    fn cubemap_key_invalidates_on_allocation_or_material_change() {
        let base = cubemap_key(1, 1, 5, 1, 256);
        let reallocated_same_upload_generation = cubemap_key(2, 1, 5, 1, 256);
        let material_changed = cubemap_key(1, 1, 6, 1, 256);

        assert_ne!(base, reallocated_same_upload_generation);
        assert_ne!(base, material_changed);
    }

    fn cubemap_key(
        allocation_generation: u64,
        content_generation: u64,
        material_generation: u64,
        mip_levels_resident: u32,
        face_size: u32,
    ) -> SkyboxIblKey {
        SkyboxIblKey::Cubemap {
            material_asset_id: 21,
            material_generation,
            route_hash: 99,
            asset_id: 7,
            allocation_generation,
            mip_levels_resident,
            content_generation,
            storage_v_inverted: false,
            face_size,
        }
    }

    fn runtime_cubemap_key(storage_v_inverted: bool) -> SkyboxIblKey {
        SkyboxIblKey::RuntimeCubemap {
            render_space_id: 7,
            renderable_index: 2,
            generation: 5,
            mip_levels: 1,
            storage_v_inverted,
            face_size: 128,
        }
    }
}
