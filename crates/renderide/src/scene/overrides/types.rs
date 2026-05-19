//! Types for host `RenderTransformOverride*` / `RenderMaterialOverride*` mirror state.

use glam::{Quat, Vec3};

use crate::shared::RenderingContext;

const MATERIAL_RENDERER_TYPE_SHIFT: u32 = 30;
const MATERIAL_RENDERER_ID_MASK: i32 = 0x3fff_ffff;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MeshRendererOverrideTarget {
    Static(i32),
    Skinned(i32),
    #[default]
    Unknown,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MaterialOverrideBinding {
    /// Submesh / material slot index on the host renderer.
    pub material_slot_index: i32,
    /// Replacement material asset id.
    pub material_asset_id: i32,
}

#[derive(Debug, Clone, Default)]
pub struct RenderTransformOverrideEntry {
    /// Scene node this override applies to.
    pub node_id: i32,
    /// User vs external render context.
    pub context: RenderingContext,
    /// Optional world-space position override for `node_id`.
    pub position_override: Option<Vec3>,
    /// Optional rotation override.
    pub rotation_override: Option<Quat>,
    /// Optional non-uniform scale override.
    pub scale_override: Option<Vec3>,
    /// Skinned mesh renderable indices receiving the override (host ids).
    pub skinned_mesh_renderer_indices: Vec<i32>,
}

#[derive(Debug, Clone, Default)]
pub struct RenderMaterialOverrideEntry {
    /// Scene node owning the mesh renderer.
    pub node_id: i32,
    /// User vs external render context.
    pub context: RenderingContext,
    /// Static vs skinned mesh target for material swaps.
    pub target: MeshRendererOverrideTarget,
    /// Per-slot material replacements.
    pub material_overrides: Vec<MaterialOverrideBinding>,
}

/// Decodes host-packed mesh renderer target for material override rows.
pub(in crate::scene) fn decode_packed_mesh_renderer_target(
    packed: i32,
) -> MeshRendererOverrideTarget {
    if packed < 0 {
        return MeshRendererOverrideTarget::Unknown;
    }
    let kind = (packed as u32) >> MATERIAL_RENDERER_TYPE_SHIFT;
    let id = packed & MATERIAL_RENDERER_ID_MASK;
    match kind {
        0 => MeshRendererOverrideTarget::Static(id),
        1 => MeshRendererOverrideTarget::Skinned(id),
        _ => MeshRendererOverrideTarget::Unknown,
    }
}
