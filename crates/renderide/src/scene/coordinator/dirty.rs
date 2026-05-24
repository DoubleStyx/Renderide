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
pub(in crate::scene::coordinator) fn extracted_update_affects_render_world(
    update: &ExtractedRenderSpaceUpdate,
) -> bool {
    update.transforms.is_some()
        || update.meshes.is_some()
        || update.skinned_meshes.is_some()
        || update.layers.is_some()
        || update.transform_overrides.is_some()
        || update.material_overrides.is_some()
        || update.billboard_render_buffers.is_some()
        || update.mesh_render_buffers.is_some()
        || update.trail_render_buffers.is_some()
}

/// Records fine-grained render-world dirty events for one extracted render-space update.
pub(in crate::scene::coordinator) fn note_render_world_dirty_for_extracted_update(
    report: &mut SceneApplyReport,
    space_id: RenderSpaceId,
    header_dirty: bool,
    current_node_count: usize,
    update: &ExtractedRenderSpaceUpdate,
) {
    if header_dirty {
        report.render_world_dirty.note_full_space(space_id);
    }
    if let Some(ref transforms) = update.transforms {
        note_transform_update_render_world_dirty(report, space_id, current_node_count, transforms);
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
    if update.layers.is_some() || update.transform_overrides.is_some() {
        report.render_world_dirty.note_full_space(space_id);
    }
    if update.billboard_render_buffers.is_some()
        || update.mesh_render_buffers.is_some()
        || update.trail_render_buffers.is_some()
    {
        report.render_world_dirty.note_full_space(space_id);
    }
    if let Some(ref material_overrides) = update.material_overrides {
        note_material_override_update_render_world_dirty(report, space_id, material_overrides);
    }
}

/// Returns whether a sentinel-terminated dense-index array contains at least one active row.
fn has_active_dense_indices(values: &[i32]) -> bool {
    values
        .iter()
        .take_while(|&&value| value >= 0)
        .next()
        .is_some()
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
            .note_renderer(space_id, kind, bounds.renderable_index as usize);
    }
}

/// Records transform roots whose descendants may need retained-template refresh.
fn note_transform_update_render_world_dirty(
    report: &mut SceneApplyReport,
    space_id: RenderSpaceId,
    current_node_count: usize,
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
        .map(|pose| pose.transform_id);
    let parent_roots = transforms
        .parent_updates
        .iter()
        .take_while(|parent| parent.transform_id >= 0)
        .map(|parent| parent.transform_id);
    report
        .render_world_dirty
        .note_transform_roots(space_id, pose_roots.chain(parent_roots));
}

/// Records material override targets that can refresh retained templates without a full rebuild.
fn note_material_override_update_render_world_dirty(
    report: &mut SceneApplyReport,
    space_id: RenderSpaceId,
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
        let target = decode_packed_mesh_renderer_target(state.packed_mesh_renderer_index);
        match target {
            MeshRendererOverrideTarget::Static(_) | MeshRendererOverrideTarget::Skinned(_) => {
                report
                    .render_world_dirty
                    .note_material_override(space_id, state.context, target);
            }
            MeshRendererOverrideTarget::Unknown => {
                report.render_world_dirty.note_full_space(space_id);
                return;
            }
        }
    }
}
