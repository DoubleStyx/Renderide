//! Cubemap asset view needed by reflection-probe baking and SH projection.

use std::sync::Arc;

/// Resident cubemap fields consumed by reflection-probe systems.
#[derive(Clone)]
pub(crate) struct ReflectionProbeCubemapAsset {
    /// Upload allocation generation for cache keys.
    pub(crate) allocation_generation: u64,
    /// Cubemap face size in pixels.
    pub(crate) size: u32,
    /// Number of resident mip levels available for sampling.
    pub(crate) mip_levels_resident: u32,
    /// Content generation used to invalidate cached projections.
    pub(crate) content_generation: u64,
    /// Whether storage uses the inverted cubemap V convention.
    pub(crate) storage_v_inverted: bool,
    /// Primary cubemap view.
    pub(crate) view: Arc<wgpu::TextureView>,
    /// Array view containing all cubemap faces and resident mip levels.
    pub(crate) array_view: Arc<wgpu::TextureView>,
}

/// Minimal cubemap asset lookup required by reflection-probe systems.
pub(crate) trait ReflectionProbeCubemapAssets {
    /// Returns the resident cubemap asset view for `asset_id`.
    fn reflection_probe_cubemap(&self, asset_id: i32) -> Option<ReflectionProbeCubemapAsset>;
}
