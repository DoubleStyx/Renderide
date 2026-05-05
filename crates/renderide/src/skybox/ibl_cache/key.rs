//! IBL bake keys and scalar helper math.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::skybox::specular::SkyboxIblSource;

/// Compute workgroup edge used by every mip-0 producer and the GGX convolve.
const IBL_WORKGROUP_EDGE: u32 = 8;
/// Base GGX importance sample count for mip 1; doubles per mip up to [`IBL_MAX_SAMPLES`].
const IBL_BASE_SAMPLE_COUNT: u32 = 64;
/// Cap on GGX importance sample count for the highest-roughness mips.
const IBL_MAX_SAMPLES: u32 = 1024;

/// Identity for one IBL bake.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub(crate) enum SkyboxIblKey {
    /// Analytic procedural / gradient skybox material identity.
    Analytic {
        /// Active skybox material asset id.
        material_asset_id: i32,
        /// Material property generation; invalidates when host edits material props.
        material_generation: u64,
        /// Stable hash of the shader route stem.
        route_hash: u64,
        /// Destination cube face edge (clamped to device limits).
        face_size: u32,
    },
    /// Host-uploaded cubemap material identity.
    Cubemap {
        /// Source cubemap asset id.
        asset_id: i32,
        /// Source resident mip count; growth re-bakes once more mips arrive.
        mip_levels_resident: u32,
        /// Storage V-flip flag for the source cube.
        storage_v_inverted: bool,
        /// Destination cube face edge.
        face_size: u32,
    },
    /// Host-uploaded equirect Texture2D material identity.
    Equirect {
        /// Source Texture2D asset id.
        asset_id: i32,
        /// Source resident mip count.
        mip_levels_resident: u32,
        /// Storage V-flip flag for the source texture.
        storage_v_inverted: bool,
        /// Bit-stable hash of `_FOV` material parameters.
        fov_hash: u64,
        /// Bit-stable hash of `_MainTex_ST` material parameters.
        st_hash: u64,
        /// Destination cube face edge.
        face_size: u32,
    },
}

impl SkyboxIblKey {
    /// Returns the destination face size for this bake.
    pub(super) fn face_size(&self) -> u32 {
        match *self {
            Self::Analytic { face_size, .. }
            | Self::Cubemap { face_size, .. }
            | Self::Equirect { face_size, .. } => face_size,
        }
    }

    /// Returns a stable renderer-side identity hash for the frame-global binding key.
    pub(super) fn source_hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }
}

/// Builds a cache key for an active source using an already-clamped destination face size.
pub(super) fn build_key(source: &SkyboxIblSource, face_size: u32) -> SkyboxIblKey {
    match source {
        SkyboxIblSource::Analytic(src) => SkyboxIblKey::Analytic {
            material_asset_id: src.material_asset_id,
            material_generation: src.material_generation,
            route_hash: src.route_hash,
            face_size,
        },
        SkyboxIblSource::Cubemap(src) => SkyboxIblKey::Cubemap {
            asset_id: src.asset_id,
            mip_levels_resident: src.mip_levels_resident,
            storage_v_inverted: src.storage_v_inverted,
            face_size,
        },
        SkyboxIblSource::Equirect(src) => SkyboxIblKey::Equirect {
            asset_id: src.asset_id,
            mip_levels_resident: src.mip_levels_resident,
            storage_v_inverted: src.storage_v_inverted,
            fov_hash: hash_float4(&src.equirect_fov),
            st_hash: hash_float4(&src.equirect_st),
            face_size,
        },
    }
}

/// Hashes four `f32`s by their bit patterns.
pub(super) fn hash_float4(values: &[f32; 4]) -> u64 {
    let mut hasher = DefaultHasher::new();
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
pub(super) fn mip_extent(base: u32, mip: u32) -> u32 {
    (base >> mip).max(1)
}

/// Returns the GGX importance sample count for the given convolve mip.
pub(super) fn convolve_sample_count(mip_index: u32) -> u32 {
    if mip_index == 0 {
        return 1;
    }
    let exponent = (mip_index - 1).min(4);
    (IBL_BASE_SAMPLE_COUNT << exponent).min(IBL_MAX_SAMPLES)
}
