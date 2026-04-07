//! Host shader asset → raster material family (mesh draws). All paths currently resolve to the debug
//! mesh family until per-shader `match` arms are added.

use super::MaterialFamilyId;
use crate::pipelines::raster::DEBUG_WORLD_NORMALS_FAMILY_ID;

/// Resolves the material family used for **mesh rasterization** for a host shader asset id.
///
/// The router’s per-shader table is ignored here so every draw uses the same debug path until real
/// shader routing is implemented.
pub fn resolve_raster_family(_shader_asset_id: i32) -> MaterialFamilyId {
    DEBUG_WORLD_NORMALS_FAMILY_ID
}
