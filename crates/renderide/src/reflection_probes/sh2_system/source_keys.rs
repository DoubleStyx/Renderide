//! Source-key types, projection parameter aliases, and shared constants.

use hashbrown::HashSet;

use crate::scene::RenderSpaceId;
use crate::skybox::params::{SkyboxEvaluatorParams, SkyboxParamMode};

/// Skybox projection sample resolution per cube face.
pub(in crate::reflection_probes) const DEFAULT_SAMPLE_SIZE: u32 =
    crate::skybox::params::DEFAULT_SKYBOX_SAMPLE_SIZE;
/// Number of renderer ticks before a pending GPU readback is treated as failed.
pub(in crate::reflection_probes) const MAX_PENDING_JOB_AGE_FRAMES: u32 = 120;
/// Bytes copied back from the compute output buffer.
pub(in crate::reflection_probes) const SH2_OUTPUT_BYTES: u64 = (9 * 16) as u64;

/// Uniform payload shared by SH2 projection compute kernels.
pub(in crate::reflection_probes) type Sh2ProjectParams = SkyboxEvaluatorParams;
/// Parameter mode used when building cubemap projection uniforms.
pub(in crate::reflection_probes) type SkyParamMode = SkyboxParamMode;

/// Hashable description of the source projected into SH2.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub(in crate::reflection_probes) enum Sh2SourceKey {
    /// Analytic constant-color source.
    ConstantColor {
        /// Render-space id that owns the probe.
        render_space_id: i32,
        /// RGBA color bit pattern.
        color_bits: [u32; 4],
    },
    /// Resident cubemap source.
    Cubemap {
        /// Render-space id that owns the probe.
        render_space_id: i32,
        /// Skybox material asset id when this source came from a material, or `-1` for direct probe sources.
        material_asset_id: i32,
        /// Host material generation mixed into skybox sources.
        material_generation: u64,
        /// Stable hash of the shader route stem when this source came from a material.
        route_hash: u64,
        /// Cubemap asset id.
        asset_id: i32,
        /// Source GPU allocation generation.
        allocation_generation: u64,
        /// Face size.
        size: u32,
        /// Contiguous resident mip count.
        resident_mips: u32,
        /// Source cubemap content generation.
        content_generation: u64,
        /// Source cubemap storage orientation.
        storage_v_inverted: bool,
        /// Projection sample grid edge per cube face.
        sample_size: u32,
    },
    /// Renderer-captured dynamic cubemap source.
    RuntimeCubemap {
        /// Render space that owns the probe.
        render_space_id: i32,
        /// Dense reflection-probe renderable index.
        renderable_index: i32,
        /// Renderer-side capture generation.
        generation: u64,
        /// Face size.
        size: u32,
        /// Projection sample grid edge per cube face.
        sample_size: u32,
    },
}

impl Sh2SourceKey {
    /// Builds a cubemap source key from the source's material identity and residency snapshot.
    pub(in crate::reflection_probes) fn cubemap(
        render_space_id: i32,
        material: CubemapSourceMaterialIdentity,
        asset_id: i32,
        residency: CubemapResidency,
    ) -> Self {
        Self::Cubemap {
            render_space_id,
            material_asset_id: material.material_asset_id,
            material_generation: material.material_generation,
            route_hash: material.route_hash,
            asset_id,
            allocation_generation: residency.allocation_generation,
            size: residency.size,
            resident_mips: residency.resident_mips,
            content_generation: residency.content_generation,
            storage_v_inverted: residency.storage_v_inverted,
            sample_size: DEFAULT_SAMPLE_SIZE,
        }
    }

    /// Render space that owns this SH2 source.
    pub(in crate::reflection_probes) fn render_space_id(&self) -> i32 {
        match *self {
            Self::ConstantColor {
                render_space_id, ..
            }
            | Self::Cubemap {
                render_space_id, ..
            }
            | Self::RuntimeCubemap {
                render_space_id, ..
            } => render_space_id,
        }
    }
}

/// Material identity fields packed into a cubemap source key.
#[derive(Clone, Copy, Debug)]
pub(in crate::reflection_probes) struct CubemapSourceMaterialIdentity {
    /// Skybox material asset id, or `-1` for direct probe sources.
    pub material_asset_id: i32,
    /// Host material generation mixed into skybox sources.
    pub material_generation: u64,
    /// Stable hash of the shader route stem when this source came from a material.
    pub route_hash: u64,
}

impl CubemapSourceMaterialIdentity {
    /// Identity for cubemap sources read directly off a reflection probe (no skybox material).
    pub(in crate::reflection_probes) const DIRECT_PROBE: Self = Self {
        material_asset_id: -1,
        material_generation: 0,
        route_hash: 0,
    };
}

/// Cubemap residency snapshot captured at key-build time.
#[derive(Clone, Copy, Debug, Default)]
pub(in crate::reflection_probes) struct CubemapResidency {
    /// Source GPU allocation generation.
    pub allocation_generation: u64,
    /// Face size in texels.
    pub size: u32,
    /// Contiguous resident mip count.
    pub resident_mips: u32,
    /// Source cubemap content generation.
    pub content_generation: u64,
    /// Source cubemap storage orientation.
    pub storage_v_inverted: bool,
}

/// Returns true when `key` belongs to one of the closed render spaces.
pub(in crate::reflection_probes) fn sh2_key_matches_closed_spaces(
    key: &Sh2SourceKey,
    spaces: &HashSet<RenderSpaceId>,
) -> bool {
    spaces.contains(&RenderSpaceId(key.render_space_id()))
}
