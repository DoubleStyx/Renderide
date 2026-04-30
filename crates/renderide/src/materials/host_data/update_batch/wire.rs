//! Wire-format target routing for [`super::parse_materials_update_batch_into_store`].
//!
//! `SelectTarget` opcodes in the IPC stream alternate between materials (the first
//! `material_update_count` rows) and property blocks (everything after). [`MaterialBatchTarget`]
//! captures the active selection so payload opcodes route to the correct
//! [`MaterialPropertyStore`] entry point.

use super::super::properties::{MaterialPropertyStore, MaterialPropertyValue};

/// Host material vs property-block target for one `select_target` row.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum MaterialBatchTarget {
    /// Active target is a material asset id.
    Material(i32),
    /// Active target is a property block asset id.
    PropertyBlock(i32),
}

impl MaterialBatchTarget {
    /// True when this target routes to a property block.
    pub(super) fn is_property_block(self) -> bool {
        matches!(self, MaterialBatchTarget::PropertyBlock(_))
    }
}

/// Routes a parsed [`MaterialPropertyValue`] to the active material or property block.
pub(super) fn set_property_on_batch_target(
    store: &mut MaterialPropertyStore,
    target: MaterialBatchTarget,
    property_id: i32,
    value: MaterialPropertyValue,
) {
    match target {
        MaterialBatchTarget::Material(id) => store.set_material(id, property_id, value),
        MaterialBatchTarget::PropertyBlock(id) => store.set_property_block(id, property_id, value),
    }
}

/// Decides whether the next `SelectTarget` row binds a material or a property block.
///
/// The wire encodes "first `material_update_count` `SelectTarget` rows are materials, the rest are
/// property blocks" implicitly by ordering rather than by an explicit tag, so the index counter is
/// authoritative.
pub(super) fn select_target_kind(
    property_id: i32,
    select_target_index: &mut usize,
    material_update_count: usize,
) -> MaterialBatchTarget {
    let is_material = *select_target_index < material_update_count;
    *select_target_index += 1;
    if is_material {
        MaterialBatchTarget::Material(property_id)
    } else {
        MaterialBatchTarget::PropertyBlock(property_id)
    }
}
