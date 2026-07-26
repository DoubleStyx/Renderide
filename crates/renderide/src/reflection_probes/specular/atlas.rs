use std::sync::Arc;

use crate::gpu::GpuLimits;
use crate::scene::RenderSpaceId;
use crate::skybox::ibl_cache::SkyboxIblKey;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) struct AtlasProbeIdentity {
    pub(super) space_id: RenderSpaceId,
    pub(super) renderable_index: i32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct AtlasResidentProbe {
    pub(super) identity: AtlasProbeIdentity,
    pub(super) key: SkyboxIblKey,
    pub(super) mip_levels: u32,
}

#[derive(Clone)]
pub(super) struct ReflectionProbeAtlas {
    pub(super) texture: Arc<wgpu::Texture>,
    pub(super) face_size: u32,
    pub(super) mip_levels: u32,
    /// Number of unique filtered cubemap slots allocated in the texture array.
    pub(super) capacity: u16,
    /// Number of per-probe rows allocated in the metadata storage buffer.
    pub(super) metadata_capacity: u16,
    pub(super) slots: Vec<Option<AtlasResidentProbe>>,
}

pub(super) struct AtlasCopyJob {
    pub(super) slot: u16,
    pub(super) completed_cube: Option<Arc<wgpu::Texture>>,
    pub(super) resident_slot: u16,
    pub(super) mip_levels: u32,
}

pub(super) fn max_atlas_slots(limits: &GpuLimits) -> u16 {
    (limits.max_texture_array_layers() / 6)
        .min(u32::from(u16::MAX))
        .max(1) as u16
}
