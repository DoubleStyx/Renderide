//! Render-context override state mirrored from host `RenderTransformOverride*` / `RenderMaterialOverride*`.

use glam::{Quat, Vec3};

use crate::ipc::SharedMemoryAccessor;
use crate::shared::{
    MaterialOverrideState, RenderMaterialOverrideState, RenderMaterialOverridesUpdate,
    RenderTransform, RenderTransformOverrideState, RenderTransformOverridesUpdate,
    RenderingContext, RENDER_MATERIAL_OVERRIDE_STATE_HOST_ROW_BYTES,
    RENDER_TRANSFORM_OVERRIDE_STATE_HOST_ROW_BYTES,
};

use super::error::SceneError;
use super::render_space::RenderSpaceState;
use super::transforms_apply::TransformRemovalEvent;
use super::world::fixup_transform_id;

const MATERIAL_RENDERER_TYPE_SHIFT: u32 = 30;
const MATERIAL_RENDERER_ID_MASK: i32 = 0x3fff_ffff;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshRendererOverrideTarget {
    Static(i32),
    Skinned(i32),
    Unknown,
}

impl Default for MeshRendererOverrideTarget {
    fn default() -> Self {
        Self::Unknown
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MaterialOverrideBinding {
    pub material_slot_index: i32,
    pub material_asset_id: i32,
}

#[derive(Debug, Clone, Default)]
pub struct RenderTransformOverrideEntry {
    pub node_id: i32,
    pub context: RenderingContext,
    pub position_override: Option<Vec3>,
    pub rotation_override: Option<Quat>,
    pub scale_override: Option<Vec3>,
    pub skinned_mesh_renderer_indices: Vec<i32>,
}

#[derive(Debug, Clone, Default)]
pub struct RenderMaterialOverrideEntry {
    pub node_id: i32,
    pub context: RenderingContext,
    pub target: MeshRendererOverrideTarget,
    pub material_overrides: Vec<MaterialOverrideBinding>,
}

impl RenderSpaceState {
    pub fn main_render_context(&self) -> RenderingContext {
        if self.view_position_is_external {
            RenderingContext::external_view
        } else {
            RenderingContext::user_view
        }
    }

    pub fn has_transform_overrides_in_context(&self, context: RenderingContext) -> bool {
        self.render_transform_overrides
            .iter()
            .any(|entry| entry.context == context && entry.node_id >= 0)
    }

    pub fn overridden_local_transform(
        &self,
        node_id: i32,
        context: RenderingContext,
    ) -> Option<RenderTransform> {
        let base = *self.nodes.get(node_id as usize)?;
        let mut local = base;
        let mut matched = false;
        for entry in self
            .render_transform_overrides
            .iter()
            .filter(|entry| entry.node_id == node_id && entry.context == context)
        {
            if let Some(position) = entry.position_override {
                local.position = position;
            }
            if let Some(rotation) = entry.rotation_override {
                local.rotation = rotation;
            }
            if let Some(scale) = entry.scale_override {
                local.scale = scale;
            }
            matched = true;
        }
        matched.then_some(local)
    }

    pub fn overridden_material_asset_id(
        &self,
        context: RenderingContext,
        target: MeshRendererOverrideTarget,
        slot_index: usize,
    ) -> Option<i32> {
        let mut replacement = None;
        for entry in self
            .render_material_overrides
            .iter()
            .filter(|entry| entry.context == context && entry.target == target)
        {
            for material in &entry.material_overrides {
                if material.material_slot_index == slot_index as i32 {
                    replacement = Some(material.material_asset_id);
                }
            }
        }
        replacement
    }
}

pub(crate) fn apply_render_transform_overrides_update(
    space: &mut RenderSpaceState,
    shm: &mut SharedMemoryAccessor,
    update: &RenderTransformOverridesUpdate,
    scene_id: i32,
    transform_removals: &[TransformRemovalEvent],
) -> Result<(), SceneError> {
    fixup_transform_override_nodes_for_transform_removals(space, transform_removals);

    if update.removals.length > 0 {
        let ctx = format!("render transform override removals scene_id={scene_id}");
        let removals = shm
            .access_copy_diagnostic_with_context::<i32>(&update.removals, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?;
        apply_dense_removals(&mut space.render_transform_overrides, removals.as_slice());
    }

    if update.additions.length > 0 {
        let ctx = format!("render transform override additions scene_id={scene_id}");
        let additions = shm
            .access_copy_diagnostic_with_context::<i32>(&update.additions, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?;
        for &node_id in additions.iter().take_while(|&&id| id >= 0) {
            space
                .render_transform_overrides
                .push(RenderTransformOverrideEntry {
                    node_id,
                    ..Default::default()
                });
        }
    }

    if update.states.length > 0 {
        let ctx = format!("render transform override states scene_id={scene_id}");
        let states = shm
            .access_copy_memory_packable_rows::<RenderTransformOverrideState>(
                &update.states,
                RENDER_TRANSFORM_OVERRIDE_STATE_HOST_ROW_BYTES,
                Some(&ctx),
            )
            .map_err(SceneError::SharedMemoryAccess)?;
        let skinned_indices = if update.skinned_mesh_renderers_indexes.length > 0 {
            let ctx = format!("render transform override skinned mesh indexes scene_id={scene_id}");
            shm.access_copy_diagnostic_with_context::<i32>(
                &update.skinned_mesh_renderers_indexes,
                Some(&ctx),
            )
            .map_err(SceneError::SharedMemoryAccess)?
        } else {
            Vec::new()
        };
        let mut skinned_cursor = 0usize;
        for state in states {
            if state.renderable_index < 0 {
                break;
            }
            let idx = state.renderable_index as usize;
            let Some(entry) = space.render_transform_overrides.get_mut(idx) else {
                continue;
            };
            entry.context = state.context;
            entry.position_override =
                ((state.override_flags & 0b001) != 0).then_some(state.position_override);
            entry.rotation_override =
                ((state.override_flags & 0b010) != 0).then_some(state.rotation_override);
            entry.scale_override =
                ((state.override_flags & 0b100) != 0).then_some(state.scale_override);
            let count = state.skinned_mesh_renderer_count.max(0) as usize;
            entry.skinned_mesh_renderer_indices.clear();
            if count > 0 {
                let end = skinned_cursor
                    .saturating_add(count)
                    .min(skinned_indices.len());
                entry
                    .skinned_mesh_renderer_indices
                    .extend_from_slice(&skinned_indices[skinned_cursor..end]);
                skinned_cursor = end;
            }
        }
    }

    Ok(())
}

pub(crate) fn apply_render_material_overrides_update(
    space: &mut RenderSpaceState,
    shm: &mut SharedMemoryAccessor,
    update: &RenderMaterialOverridesUpdate,
    scene_id: i32,
    transform_removals: &[TransformRemovalEvent],
) -> Result<(), SceneError> {
    fixup_material_override_nodes_for_transform_removals(space, transform_removals);

    if update.removals.length > 0 {
        let ctx = format!("render material override removals scene_id={scene_id}");
        let removals = shm
            .access_copy_diagnostic_with_context::<i32>(&update.removals, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?;
        apply_dense_removals(&mut space.render_material_overrides, removals.as_slice());
    }

    if update.additions.length > 0 {
        let ctx = format!("render material override additions scene_id={scene_id}");
        let additions = shm
            .access_copy_diagnostic_with_context::<i32>(&update.additions, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?;
        for &node_id in additions.iter().take_while(|&&id| id >= 0) {
            space
                .render_material_overrides
                .push(RenderMaterialOverrideEntry {
                    node_id,
                    ..Default::default()
                });
        }
    }

    if update.states.length > 0 {
        let ctx = format!("render material override states scene_id={scene_id}");
        let states = shm
            .access_copy_memory_packable_rows::<RenderMaterialOverrideState>(
                &update.states,
                RENDER_MATERIAL_OVERRIDE_STATE_HOST_ROW_BYTES,
                Some(&ctx),
            )
            .map_err(SceneError::SharedMemoryAccess)?;
        let materials = if update.material_override_states.length > 0 {
            let ctx = format!("render material override rows scene_id={scene_id}");
            shm.access_copy_diagnostic_with_context::<MaterialOverrideState>(
                &update.material_override_states,
                Some(&ctx),
            )
            .map_err(SceneError::SharedMemoryAccess)?
        } else {
            Vec::new()
        };
        let mut material_cursor = 0usize;
        for state in states {
            if state.renderable_index < 0 {
                break;
            }
            let idx = state.renderable_index as usize;
            let Some(entry) = space.render_material_overrides.get_mut(idx) else {
                continue;
            };
            entry.context = state.context;
            entry.target = decode_packed_mesh_renderer_target(state.packed_mesh_renderer_index);
            let count = state.materrial_override_count.max(0) as usize;
            entry.material_overrides.clear();
            if count > 0 {
                let end = material_cursor.saturating_add(count).min(materials.len());
                entry
                    .material_overrides
                    .extend(materials[material_cursor..end].iter().map(|row| {
                        MaterialOverrideBinding {
                            material_slot_index: row.material_slot_index,
                            material_asset_id: row.material_asset_id,
                        }
                    }));
                material_cursor = end;
            }
        }
    }

    Ok(())
}

fn apply_dense_removals<T>(entries: &mut Vec<T>, removals: &[i32]) {
    for &raw in removals.iter().take_while(|&&idx| idx >= 0) {
        let idx = raw as usize;
        if idx < entries.len() {
            entries.swap_remove(idx);
        }
    }
}

fn fixup_transform_override_nodes_for_transform_removals(
    space: &mut RenderSpaceState,
    removals: &[TransformRemovalEvent],
) {
    for removal in removals {
        for entry in &mut space.render_transform_overrides {
            entry.node_id = fixup_transform_id(
                entry.node_id,
                removal.removed_index,
                removal.last_index_before_swap,
            );
        }
    }
}

fn fixup_material_override_nodes_for_transform_removals(
    space: &mut RenderSpaceState,
    removals: &[TransformRemovalEvent],
) {
    for removal in removals {
        for entry in &mut space.render_material_overrides {
            entry.node_id = fixup_transform_id(
                entry.node_id,
                removal.removed_index,
                removal.last_index_before_swap,
            );
        }
    }
}

fn decode_packed_mesh_renderer_target(packed: i32) -> MeshRendererOverrideTarget {
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

#[cfg(test)]
mod tests {
    use glam::{Quat, Vec3};

    use super::*;
    use crate::shared::RenderTransform;

    #[test]
    fn decode_packed_mesh_renderer_target_matches_shared_packer_layout() {
        assert_eq!(
            decode_packed_mesh_renderer_target(7),
            MeshRendererOverrideTarget::Static(7)
        );
        assert_eq!(
            decode_packed_mesh_renderer_target((1 << 30) | 11),
            MeshRendererOverrideTarget::Skinned(11)
        );
    }

    #[test]
    fn main_render_context_uses_external_flag() {
        let mut space = RenderSpaceState::default();
        assert_eq!(space.main_render_context(), RenderingContext::user_view);
        space.view_position_is_external = true;
        assert_eq!(space.main_render_context(), RenderingContext::external_view);
    }

    #[test]
    fn overridden_local_transform_replaces_requested_components_only() {
        let mut space = RenderSpaceState::default();
        space.nodes.push(RenderTransform {
            position: Vec3::new(1.0, 2.0, 3.0),
            rotation: Quat::IDENTITY,
            scale: Vec3::splat(2.0),
        });
        space
            .render_transform_overrides
            .push(RenderTransformOverrideEntry {
                node_id: 0,
                context: RenderingContext::user_view,
                position_override: Some(Vec3::new(10.0, 20.0, 30.0)),
                rotation_override: None,
                scale_override: Some(Vec3::ONE),
                skinned_mesh_renderer_indices: Vec::new(),
            });

        let local = space
            .overridden_local_transform(0, RenderingContext::user_view)
            .expect("override");
        assert_eq!(local.position, Vec3::new(10.0, 20.0, 30.0));
        assert_eq!(local.rotation, Quat::IDENTITY);
        assert_eq!(local.scale, Vec3::ONE);
    }

    #[test]
    fn overridden_material_asset_id_matches_context_target_and_slot() {
        let mut space = RenderSpaceState::default();
        space
            .render_material_overrides
            .push(RenderMaterialOverrideEntry {
                node_id: 0,
                context: RenderingContext::user_view,
                target: MeshRendererOverrideTarget::Static(4),
                material_overrides: vec![MaterialOverrideBinding {
                    material_slot_index: 1,
                    material_asset_id: 99,
                }],
            });

        assert_eq!(
            space.overridden_material_asset_id(
                RenderingContext::user_view,
                MeshRendererOverrideTarget::Static(4),
                1,
            ),
            Some(99)
        );
        assert_eq!(
            space.overridden_material_asset_id(
                RenderingContext::external_view,
                MeshRendererOverrideTarget::Static(4),
                1,
            ),
            None
        );
    }
}
