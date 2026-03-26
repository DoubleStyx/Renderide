//! Per-draw material identity for [`MaterialPropertyStore`] merged lookup (material + mesh property block).

use crate::assets::MaterialPropertyLookupIds;

/// Resolved property lookup for one mesh draw (native UI WGSL and similar paths).
#[derive(Clone, Copy, Debug)]
pub(super) struct MaterialDrawContext {
    /// Material asset id and optional per-submesh `MaterialPropertyBlock` for merged property reads.
    pub(super) property_lookup: MaterialPropertyLookupIds,
}

impl MaterialDrawContext {
    /// Builds merged lookup for a non-skinned draw: host material id plus optional property block
    /// for that submesh (after multi-material fan-out, typically the active submesh’s block).
    pub(super) fn for_non_skinned_draw(
        material_asset_id: i32,
        mesh_property_block_id: Option<i32>,
    ) -> Self {
        Self {
            property_lookup: MaterialPropertyLookupIds {
                material_asset_id,
                mesh_property_block_slot0: mesh_property_block_id,
            },
        }
    }
}
