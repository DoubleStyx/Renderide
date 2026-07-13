//! Render-world dirty tracking for scene apply updates.

use crate::shared::RenderSpaceUpdate;

use super::super::ids::RenderSpaceId;
use super::super::overrides::{MeshRendererOverrideTarget, decode_packed_mesh_renderer_target};
use super::super::render_space::RenderSpaceState;
use super::apply::ExtractedRenderSpaceUpdate;
use super::reports::{RenderWorldRendererKind, SceneApplyReport};

/// Returns whether render-space header fields changed retained render-world routing.
pub(in crate::scene::coordinator) fn render_world_header_changed(
    space: Option<&RenderSpaceState>,
    update: &RenderSpaceUpdate,
) -> bool {
    let Some(space) = space else {
        return true;
    };
    space.is_active != update.is_active
        || space.is_overlay != update.is_overlay
        || space.view_position_is_external != update.view_position_is_external
}

/// Returns whether an extracted update can affect retained renderer templates.
#[cfg(test)]
pub(in crate::scene::coordinator) fn extracted_update_affects_render_world(
    update: &ExtractedRenderSpaceUpdate,
) -> bool {
    update.transforms.is_some()
        || update.meshes.is_some()
        || update.skinned_meshes.is_some()
        || update.layers.is_some()
        || update.lod_groups.is_some()
        || update.transform_overrides.is_some()
        || update.material_overrides.is_some()
        || update.billboard_render_buffers.is_some()
        || update.mesh_render_buffers.is_some()
        || update.trail_render_buffers.is_some()
}

/// Returns whether an extracted update can affect reflection-probe source or spatial state.
#[cfg(test)]
pub(in crate::scene::coordinator) fn extracted_update_affects_reflection_probes(
    update: &ExtractedRenderSpaceUpdate,
) -> bool {
    update.reflection_probes.is_some()
        || update.transforms.is_some()
        || update.transform_overrides.is_some()
}

pub(in crate::scene::coordinator) fn extracted_update_changes_render_world(
    update: &ExtractedRenderSpaceUpdate,
    transforms_changed: bool,
) -> bool {
    transforms_changed
        || update
            .meshes
            .as_ref()
            .is_some_and(static_mesh_update_has_work)
        || update
            .skinned_meshes
            .as_ref()
            .is_some_and(skinned_mesh_update_has_work)
        || update.layers.as_ref().is_some_and(layer_update_has_work)
        || update
            .lod_groups
            .as_ref()
            .is_some_and(lod_group_update_has_work)
        || update
            .transform_overrides
            .as_ref()
            .is_some_and(transform_override_update_has_work)
        || update
            .material_overrides
            .as_ref()
            .is_some_and(material_override_update_has_work)
        || update
            .billboard_render_buffers
            .as_ref()
            .is_some_and(billboard_update_has_work)
        || update
            .mesh_render_buffers
            .as_ref()
            .is_some_and(mesh_render_buffer_update_has_work)
        || update
            .trail_render_buffers
            .as_ref()
            .is_some_and(trail_update_has_work)
}

pub(in crate::scene::coordinator) fn extracted_update_changes_reflection_probes(
    update: &ExtractedRenderSpaceUpdate,
    transforms_changed: bool,
) -> bool {
    update
        .reflection_probes
        .as_ref()
        .is_some_and(reflection_probe_update_has_work)
        || transforms_changed
        || update
            .transform_overrides
            .as_ref()
            .is_some_and(transform_override_update_has_work)
}

pub(in crate::scene::coordinator) fn transform_update_changes_space(
    space: &RenderSpaceState,
    transforms: &super::super::transforms::ExtractedTransformsUpdate,
) -> bool {
    if has_active_dense_indices(&transforms.removals)
        || (transforms.target_transform_count >= 0
            && transforms.target_transform_count as usize != space.nodes.len())
    {
        return true;
    }

    let parent_changed = transforms
        .parent_updates
        .iter()
        .take_while(|parent| parent.transform_id >= 0)
        .any(|parent| {
            space
                .node_parents
                .get(parent.transform_id as usize)
                .is_some_and(|current| *current != parent.new_parent_id)
        });
    parent_changed
        || transforms
            .pose_updates
            .iter()
            .take_while(|pose| pose.transform_id >= 0)
            .any(|pose| {
                space
                    .nodes
                    .get(pose.transform_id as usize)
                    .is_some_and(|current| transform_pose_changes(current, &pose.pose))
            })
}

/// Records fine-grained render-world dirty events for one extracted render-space update.
pub(in crate::scene::coordinator) fn note_render_world_dirty_for_extracted_update(
    report: &mut SceneApplyReport,
    space_id: RenderSpaceId,
    header_dirty: bool,
    current_node_count: usize,
    current_space: Option<&RenderSpaceState>,
    update: &ExtractedRenderSpaceUpdate,
) {
    if header_dirty {
        report.render_world_dirty.note_full_space(space_id);
    }
    if let Some(ref transforms) = update.transforms {
        note_transform_update_render_world_dirty(
            report,
            space_id,
            current_node_count,
            current_space,
            transforms,
        );
    }
    if let Some(ref meshes) = update.meshes {
        note_mesh_update_render_world_dirty(
            report,
            space_id,
            RenderWorldRendererKind::Static,
            &meshes.removals,
            &meshes.additions,
            meshes
                .mesh_states
                .iter()
                .map(|state| state.renderable_index),
        );
    }
    if let Some(ref skinned_meshes) = update.skinned_meshes {
        note_skinned_mesh_update_render_world_dirty(report, space_id, skinned_meshes);
    }
    if update.layers.as_ref().is_some_and(layer_update_has_work)
        || update
            .transform_overrides
            .as_ref()
            .is_some_and(transform_override_update_has_work)
    {
        report.render_world_dirty.note_full_space(space_id);
    }
    if update
        .lod_groups
        .as_ref()
        .is_some_and(lod_group_update_has_work)
    {
        report.render_world_dirty.note_full_space(space_id);
    }
    if update
        .billboard_render_buffers
        .as_ref()
        .is_some_and(billboard_update_has_work)
        || update
            .mesh_render_buffers
            .as_ref()
            .is_some_and(mesh_render_buffer_update_has_work)
        || update
            .trail_render_buffers
            .as_ref()
            .is_some_and(trail_update_has_work)
    {
        report.render_world_dirty.note_full_space(space_id);
    }
    if let Some(material_overrides) = update
        .material_overrides
        .as_ref()
        .filter(|update| material_override_update_has_work(update))
    {
        note_material_override_update_render_world_dirty(
            report,
            space_id,
            current_space,
            material_overrides,
        );
    }
}

/// Returns whether a sentinel-terminated dense-index array contains at least one active row. -xlinka
fn has_active_dense_indices(values: &[i32]) -> bool {
    values
        .iter()
        .take_while(|&&value| value >= 0)
        .next()
        .is_some()
}

fn static_mesh_update_has_work(
    update: &super::super::meshes::ExtractedMeshRenderablesUpdate,
) -> bool {
    has_active_dense_indices(&update.removals)
        || has_active_dense_indices(&update.additions)
        || update
            .mesh_states
            .first()
            .is_some_and(|state| state.renderable_index >= 0)
}

fn skinned_mesh_update_has_work(
    update: &super::super::meshes::ExtractedSkinnedMeshRenderablesUpdate,
) -> bool {
    has_active_dense_indices(&update.removals)
        || has_active_dense_indices(&update.additions)
        || update
            .mesh_states
            .first()
            .is_some_and(|state| state.renderable_index >= 0)
        || update
            .bone_assignments
            .first()
            .is_some_and(|state| state.renderable_index >= 0)
        || update
            .blendshape_update_batches
            .first()
            .is_some_and(|state| state.renderable_index >= 0)
        || update
            .bounds_updates
            .first()
            .is_some_and(|state| state.renderable_index >= 0)
}

fn layer_update_has_work(update: &super::super::layer::ExtractedLayerUpdate) -> bool {
    has_active_dense_indices(&update.removals) || has_active_dense_indices(&update.additions)
}

fn lod_group_update_has_work(
    update: &super::super::lod_groups::ExtractedLodGroupRenderablesUpdate,
) -> bool {
    has_active_dense_indices(&update.removals)
        || has_active_dense_indices(&update.additions)
        || update
            .states
            .first()
            .is_some_and(|state| state.renderable_index >= 0)
}

fn transform_override_update_has_work(
    update: &super::super::overrides::ExtractedRenderTransformOverridesUpdate,
) -> bool {
    has_active_dense_indices(&update.removals)
        || has_active_dense_indices(&update.additions)
        || update
            .states
            .first()
            .is_some_and(|state| state.renderable_index >= 0)
}

fn material_override_update_has_work(
    update: &super::super::overrides::ExtractedRenderMaterialOverridesUpdate,
) -> bool {
    has_active_dense_indices(&update.removals)
        || has_active_dense_indices(&update.additions)
        || update
            .states
            .first()
            .is_some_and(|state| state.renderable_index >= 0)
}

fn billboard_update_has_work(
    update: &super::super::render_buffers::ExtractedBillboardRenderBufferUpdate,
) -> bool {
    has_active_dense_indices(&update.removals)
        || has_active_dense_indices(&update.additions)
        || update
            .states
            .first()
            .is_some_and(|state| state.renderable_index >= 0)
}

fn mesh_render_buffer_update_has_work(
    update: &super::super::render_buffers::ExtractedMeshRenderBufferUpdate,
) -> bool {
    has_active_dense_indices(&update.removals)
        || has_active_dense_indices(&update.additions)
        || update
            .states
            .first()
            .is_some_and(|state| state.renderable_index >= 0)
}

fn trail_update_has_work(
    update: &super::super::render_buffers::ExtractedTrailRendererUpdate,
) -> bool {
    has_active_dense_indices(&update.removals)
        || has_active_dense_indices(&update.additions)
        || update
            .states
            .first()
            .is_some_and(|state| state.renderable_index >= 0)
}

fn reflection_probe_update_has_work(
    update: &super::super::reflection_probe::ExtractedReflectionProbeRenderablesUpdate,
) -> bool {
    has_active_dense_indices(&update.removals)
        || has_active_dense_indices(&update.additions)
        || update
            .states
            .first()
            .is_some_and(|state| state.renderable_index >= 0)
        || update
            .changed_probes_to_render
            .first()
            .is_some_and(|state| state.renderable_index >= 0)
}

/// Records dirty retained-template rows for a static or skinned mesh renderer update.
fn note_mesh_update_render_world_dirty<I>(
    report: &mut SceneApplyReport,
    space_id: RenderSpaceId,
    kind: RenderWorldRendererKind,
    removals: &[i32],
    additions: &[i32],
    renderable_indices: I,
) where
    I: IntoIterator<Item = i32>,
{
    if has_active_dense_indices(removals) || has_active_dense_indices(additions) {
        report.render_world_dirty.note_full_space(space_id);
        return;
    }
    for renderable_index in renderable_indices {
        if renderable_index < 0 {
            break;
        }
        report
            .render_world_dirty
            .note_renderer(space_id, kind, renderable_index as usize);
    }
}

/// Records all skinned renderer rows affected by mesh, bone, blendshape, or bounds updates.
fn note_skinned_mesh_update_render_world_dirty(
    report: &mut SceneApplyReport,
    space_id: RenderSpaceId,
    skinned_meshes: &super::super::meshes::ExtractedSkinnedMeshRenderablesUpdate,
) {
    if has_active_dense_indices(&skinned_meshes.removals)
        || has_active_dense_indices(&skinned_meshes.additions)
    {
        report.render_world_dirty.note_full_space(space_id);
        return;
    }
    let kind = RenderWorldRendererKind::Skinned;
    for state in &skinned_meshes.mesh_states {
        if state.renderable_index < 0 {
            break;
        }
        report
            .render_world_dirty
            .note_renderer(space_id, kind, state.renderable_index as usize);
    }
    for assignment in &skinned_meshes.bone_assignments {
        if assignment.renderable_index < 0 {
            break;
        }
        report.render_world_dirty.note_renderer(
            space_id,
            kind,
            assignment.renderable_index as usize,
        );
    }
    for batch in &skinned_meshes.blendshape_update_batches {
        if batch.renderable_index < 0 {
            break;
        }
        report
            .render_world_dirty
            .note_renderer(space_id, kind, batch.renderable_index as usize);
    }
    for bounds in &skinned_meshes.bounds_updates {
        if bounds.renderable_index < 0 {
            break;
        }
        report
            .render_world_dirty
            .note_bounds(space_id, kind, bounds.renderable_index as usize);
    }
}

/// Records transform roots whose descendants may need retained-template refresh.
fn note_transform_update_render_world_dirty(
    report: &mut SceneApplyReport,
    space_id: RenderSpaceId,
    current_node_count: usize,
    current_space: Option<&RenderSpaceState>,
    transforms: &super::super::transforms::ExtractedTransformsUpdate,
) {
    if has_active_dense_indices(&transforms.removals)
        || (transforms.target_transform_count >= 0
            && transforms.target_transform_count as usize != current_node_count)
    {
        report.render_world_dirty.note_full_space(space_id);
        return;
    }
    let pose_roots = transforms
        .pose_updates
        .iter()
        .take_while(|pose| pose.transform_id >= 0)
        .filter(|pose| {
            current_space.is_none_or(|space| {
                space
                    .nodes
                    .get(pose.transform_id as usize)
                    .is_some_and(|current| transform_pose_changes(current, &pose.pose))
            })
        })
        .map(|pose| pose.transform_id);
    let parent_roots = transforms
        .parent_updates
        .iter()
        .take_while(|parent| parent.transform_id >= 0)
        .filter(|parent| {
            current_space.is_none_or(|space| {
                space
                    .node_parents
                    .get(parent.transform_id as usize)
                    .is_some_and(|current| *current != parent.new_parent_id)
            })
        })
        .map(|parent| parent.transform_id);
    report
        .render_world_dirty
        .note_transform_roots(space_id, pose_roots.chain(parent_roots));
}

fn render_transform_matches(
    current: &crate::shared::RenderTransform,
    incoming: &crate::shared::RenderTransform,
) -> bool {
    current.position == incoming.position
        && current.scale == incoming.scale
        && current.rotation == incoming.rotation
}

fn transform_pose_changes(
    current: &crate::shared::RenderTransform,
    incoming: &crate::shared::RenderTransform,
) -> bool {
    if render_transform_matches(current, incoming) {
        return false;
    }
    let repaired = crate::scene::pose::repair_render_transform(incoming, current);
    !render_transform_matches(current, &repaired)
}

/// Records material override targets that can refresh retained templates without a full rebuild.
fn note_material_override_update_render_world_dirty(
    report: &mut SceneApplyReport,
    space_id: RenderSpaceId,
    current_space: Option<&RenderSpaceState>,
    material_overrides: &super::super::overrides::ExtractedRenderMaterialOverridesUpdate,
) {
    if has_active_dense_indices(&material_overrides.removals)
        || has_active_dense_indices(&material_overrides.additions)
    {
        report.render_world_dirty.note_full_space(space_id);
        return;
    }
    for state in &material_overrides.states {
        if state.renderable_index < 0 {
            break;
        }
        let previous_target_requires_full_space = match current_space
            .and_then(|space| {
                space
                    .render_material_overrides
                    .get(state.renderable_index as usize)
            })
            .filter(|entry| entry.node_id >= 0)
        {
            Some(previous) => note_material_override_target_dirty(
                report,
                space_id,
                previous.context,
                previous.target,
            ),
            None => false,
        };
        if previous_target_requires_full_space {
            return;
        }
        let target = decode_packed_mesh_renderer_target(state.packed_mesh_renderer_index);
        if note_material_override_target_dirty(report, space_id, state.context, target) {
            return;
        }
    }
}

fn note_material_override_target_dirty(
    report: &mut SceneApplyReport,
    space_id: RenderSpaceId,
    context: crate::shared::RenderingContext,
    target: MeshRendererOverrideTarget,
) -> bool {
    match target {
        MeshRendererOverrideTarget::Static(_) | MeshRendererOverrideTarget::Skinned(_) => {
            report
                .render_world_dirty
                .note_material_override(space_id, context, target);
            false
        }
        MeshRendererOverrideTarget::Unknown => {
            report.render_world_dirty.note_full_space(space_id);
            true
        }
    }
}
