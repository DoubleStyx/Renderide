//! Mesh material-slot sentinel handling for world-mesh draw preparation.

/// Host material id used when a mesh renderer has a missing material slot.
pub(crate) const MISSING_MATERIAL_ASSET_ID: i32 = -1;

/// Returns whether a material id can produce a world-mesh draw.
#[inline]
pub(crate) fn material_slot_is_drawable(material_asset_id: i32) -> bool {
    material_asset_id >= MISSING_MATERIAL_ASSET_ID
}

/// Returns the normalized material/property-block pair for draw and cache resolution.
#[inline]
pub(crate) fn normalized_material_slot(
    material_asset_id: i32,
    property_block_id: Option<i32>,
) -> Option<(i32, Option<i32>)> {
    if !material_slot_is_drawable(material_asset_id) {
        return None;
    }
    Some((
        material_asset_id,
        normalized_property_block_for_material(material_asset_id, property_block_id),
    ))
}

/// Clears property-block state for missing-material slots so Null fallback state stays fixed.
#[inline]
pub(crate) fn normalized_property_block_for_material(
    material_asset_id: i32,
    property_block_id: Option<i32>,
) -> Option<i32> {
    if material_asset_id == MISSING_MATERIAL_ASSET_ID {
        None
    } else {
        property_block_id
    }
}
