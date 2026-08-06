//! Render-world dirty tracking for scene apply updates.

use hashbrown::HashMap;

use crate::shared::RenderSpaceUpdate;

use super::super::ids::RenderSpaceId;
use super::super::meshes::types::{MeshRendererStateApplyPlan, decode_mesh_renderer_state_plan};
use super::super::overrides::{MeshRendererOverrideTarget, decode_packed_mesh_renderer_target};
use super::super::render_space::RenderSpaceState;
use super::apply::ExtractedRenderSpaceUpdate;
use super::reports::{RenderWorldParticleRendererKind, RenderWorldRendererKind, SceneApplyReport};

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
    report.note_render_world_classified_space(space_id);
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
        note_static_mesh_update_render_world_dirty(report, space_id, current_space, meshes);
    }
    if let Some(ref skinned_meshes) = update.skinned_meshes {
        note_skinned_mesh_update_render_world_dirty(
            report,
            space_id,
            current_space,
            skinned_meshes,
        );
    }
    if update.layers.as_ref().is_some_and(layer_update_has_work) {
        report.render_world_dirty.note_full_space(space_id);
    }
    if let Some(transform_overrides) = update
        .transform_overrides
        .as_ref()
        .filter(|update| transform_override_update_has_work(update))
    {
        note_transform_override_update_render_world_dirty(
            report,
            space_id,
            current_space,
            transform_overrides,
        );
    }
    if update
        .lod_groups
        .as_ref()
        .is_some_and(lod_group_update_has_work)
    {
        report.render_world_dirty.note_full_space(space_id);
    }
    if let Some(update) = update
        .billboard_render_buffers
        .as_ref()
        .filter(|update| billboard_update_has_work(update))
    {
        note_particle_update_render_world_dirty(
            report,
            space_id,
            RenderWorldParticleRendererKind::Billboard,
            &update.removals,
            &update.additions,
            update.states.iter().map(|state| state.renderable_index),
        );
    }
    if let Some(update) = update
        .mesh_render_buffers
        .as_ref()
        .filter(|update| mesh_render_buffer_update_has_work(update))
    {
        note_particle_update_render_world_dirty(
            report,
            space_id,
            RenderWorldParticleRendererKind::Mesh,
            &update.removals,
            &update.additions,
            update.states.iter().map(|state| state.renderable_index),
        );
    }
    if let Some(update) = update
        .trail_render_buffers
        .as_ref()
        .filter(|update| trail_update_has_work(update))
    {
        note_particle_update_render_world_dirty(
            report,
            space_id,
            RenderWorldParticleRendererKind::Trail,
            &update.removals,
            &update.additions,
            update.states.iter().map(|state| state.renderable_index),
        );
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

/// Merges static mesh-state rows and reports only final retained draw-state changes.
fn note_static_mesh_update_render_world_dirty(
    report: &mut SceneApplyReport,
    space_id: RenderSpaceId,
    current_space: Option<&RenderSpaceState>,
    update: &super::super::meshes::ExtractedMeshRenderablesUpdate,
) {
    if has_active_dense_indices(&update.removals) || has_active_dense_indices(&update.additions) {
        report.render_world_dirty.note_full_space(space_id);
        return;
    }

    let packed = update.mesh_materials_and_property_blocks.as_deref();
    let mut packed_cursor = 0usize;
    let mut plans: HashMap<usize, MeshRendererStateApplyPlan> =
        HashMap::with_capacity(update.mesh_states.len());
    for state in &update.mesh_states {
        if state.renderable_index < 0 {
            break;
        }
        let plan = decode_mesh_renderer_state_plan(state, packed, &mut packed_cursor);
        let renderable_index = state.renderable_index as usize;
        if current_space.is_some_and(|space| renderable_index >= space.static_mesh_renderers.len())
        {
            continue;
        }
        if let Some(existing) = plans.get_mut(&renderable_index) {
            existing.merge_later_row(plan);
        } else {
            plans.insert(renderable_index, plan);
        }
    }

    for (renderable_index, plan) in plans {
        let changed = current_space
            .and_then(|space| space.static_mesh_renderers.get(renderable_index))
            .is_none_or(|renderer| plan.changes_render_world_state(renderer));
        if changed {
            report.render_world_dirty.note_renderer(
                space_id,
                RenderWorldRendererKind::Static,
                renderable_index,
            );
        }
    }
}

/// Records generated-renderer dirties without invalidating static/skinned retained templates.
///
/// Dense-table membership operations use swap-remove and can change more than one renderer
/// identity, so those refresh the complete particle suffix for the space. State-only updates keep
/// stable dense indices and can patch exactly the affected prepared renderer ranges.
fn note_particle_update_render_world_dirty<I>(
    report: &mut SceneApplyReport,
    space_id: RenderSpaceId,
    kind: RenderWorldParticleRendererKind,
    removals: &[i32],
    additions: &[i32],
    renderable_indices: I,
) where
    I: IntoIterator<Item = i32>,
{
    if has_active_dense_indices(removals) || has_active_dense_indices(additions) {
        report.render_world_dirty.note_particle_space(space_id);
        return;
    }
    for renderable_index in renderable_indices {
        if renderable_index < 0 {
            break;
        }
        report
            .render_world_dirty
            .note_particle_renderer(space_id, kind, renderable_index as usize);
    }
}

/// Routes transform-override changes exclusively to their lightweight context overlays.
fn note_transform_override_update_render_world_dirty(
    report: &mut SceneApplyReport,
    space_id: RenderSpaceId,
    current_space: Option<&RenderSpaceState>,
    update: &super::super::overrides::ExtractedRenderTransformOverridesUpdate,
) {
    let membership_changed =
        has_active_dense_indices(&update.removals) || has_active_dense_indices(&update.additions);
    if membership_changed {
        if let Some(space) = current_space {
            for entry in &space.render_transform_overrides {
                if entry.node_id >= 0 {
                    report
                        .render_world_dirty
                        .note_context_override(space_id, entry.context);
                }
            }
        }
        for state in update
            .states
            .iter()
            .take_while(|state| state.renderable_index >= 0)
        {
            report
                .render_world_dirty
                .note_context_override(space_id, state.context);
        }
        if has_active_dense_indices(&update.additions) && update.states.is_empty() {
            report
                .render_world_dirty
                .note_context_override(space_id, crate::shared::RenderingContext::UserView);
        }
        return;
    }

    for state in update
        .states
        .iter()
        .take_while(|state| state.renderable_index >= 0)
    {
        if let Some(previous) = current_space.and_then(|space| {
            space
                .render_transform_overrides
                .get(state.renderable_index as usize)
        }) {
            report
                .render_world_dirty
                .note_context_override(space_id, previous.context);
        }
        report
            .render_world_dirty
            .note_context_override(space_id, state.context);
    }
}

/// Classifies final skinned-renderer effects against the pre-apply scene mirror.
///
/// Membership operations retain the full-space fallback because dense swap-removes can move
/// renderer identities. Stable-table updates are merged with the same cursor and duplicate-row
/// semantics as the apply path before any dirty event is emitted.
fn note_skinned_mesh_update_render_world_dirty(
    report: &mut SceneApplyReport,
    space_id: RenderSpaceId,
    current_space: Option<&RenderSpaceState>,
    skinned_meshes: &super::super::meshes::ExtractedSkinnedMeshRenderablesUpdate,
) {
    if has_active_dense_indices(&skinned_meshes.removals)
        || has_active_dense_indices(&skinned_meshes.additions)
    {
        report.render_world_dirty.note_full_space(space_id);
        return;
    }

    note_changed_skinned_mesh_state_plans(report, space_id, current_space, skinned_meshes);
    note_changed_skinned_bone_plans(report, space_id, current_space, skinned_meshes);
    note_changed_skinned_blendshape_weights(report, space_id, current_space, skinned_meshes);
    note_changed_skinned_bounds(report, space_id, current_space, skinned_meshes);
}

/// Merges state rows by renderer and reports only plans that alter retained draw fields.
fn note_changed_skinned_mesh_state_plans(
    report: &mut SceneApplyReport,
    space_id: RenderSpaceId,
    current_space: Option<&RenderSpaceState>,
    update: &super::super::meshes::ExtractedSkinnedMeshRenderablesUpdate,
) {
    let packed = update.mesh_materials_and_property_blocks.as_deref();
    let mut packed_cursor = 0usize;
    let mut plans: HashMap<usize, MeshRendererStateApplyPlan> =
        HashMap::with_capacity(update.mesh_states.len());
    for state in &update.mesh_states {
        if state.renderable_index < 0 {
            break;
        }
        let plan = decode_mesh_renderer_state_plan(state, packed, &mut packed_cursor);
        let index = state.renderable_index as usize;
        if current_space.is_some_and(|space| index >= space.skinned_mesh_renderers.len()) {
            continue;
        }
        if let Some(existing) = plans.get_mut(&index) {
            existing.merge_later_row(plan);
        } else {
            plans.insert(index, plan);
        }
    }

    for (renderable_index, plan) in plans {
        let changed = current_space
            .and_then(|space| space.skinned_mesh_renderers.get(renderable_index))
            .is_none_or(|renderer| plan.changes_render_world_state(&renderer.base));
        if changed {
            report.render_world_dirty.note_renderer(
                space_id,
                RenderWorldRendererKind::Skinned,
                renderable_index,
            );
        }
    }
}

/// Final accepted bone assignment for one renderer.
struct ClassifiedBonePlan {
    root: Option<i32>,
}

/// Mirrors bone-slab cursor handling and reports only cull-root changes.
///
/// Palette index values are read directly by mesh deformation and cannot change the current
/// prepared skin-eligibility predicate, which depends only on receiving `Some(slice)`.
fn note_changed_skinned_bone_plans(
    report: &mut SceneApplyReport,
    space_id: RenderSpaceId,
    current_space: Option<&RenderSpaceState>,
    update: &super::super::meshes::ExtractedSkinnedMeshRenderablesUpdate,
) {
    let mut index_offset = 0usize;
    let mut plans: HashMap<usize, ClassifiedBonePlan> =
        HashMap::with_capacity(update.bone_assignments.len());
    for assignment in &update.bone_assignments {
        if assignment.renderable_index < 0 {
            break;
        }
        let renderable_index = assignment.renderable_index as usize;
        let bone_count = assignment.bone_count.max(0) as usize;
        let Some(end) = index_offset.checked_add(bone_count) else {
            break;
        };
        let target_is_valid =
            current_space.is_none_or(|space| renderable_index < space.skinned_mesh_renderers.len());
        if target_is_valid {
            let root = (assignment.root_bone_transform_id >= 0)
                .then_some(assignment.root_bone_transform_id);
            if bone_count == 0 {
                plans.insert(renderable_index, ClassifiedBonePlan { root });
            } else if end <= update.bone_transform_indexes.len() {
                plans.insert(renderable_index, ClassifiedBonePlan { root });
            }
        }
        // The slab cursor advances even for out-of-range targets and rejected/truncated rows.
        index_offset = end;
    }

    for (renderable_index, plan) in plans {
        let current =
            current_space.and_then(|space| space.skinned_mesh_renderers.get(renderable_index));
        if current.is_none_or(|renderer| renderer.root_bone_transform_id != plan.root) {
            report.render_world_dirty.note_bounds(
                space_id,
                RenderWorldRendererKind::Skinned,
                renderable_index,
            );
        }
    }
}

/// Blendshape indices at or above this cap are ignored by the scene apply path.
const MAX_CLASSIFIED_BLENDSHAPE_INDEX: usize = 4096;

/// Applies one accepted blendshape slice using the scene mutation path's exact index semantics.
fn apply_classified_blendshape_slice(
    weights: &mut Vec<f32>,
    updates: &[crate::shared::BlendshapeUpdate],
) {
    for update in updates {
        let blendshape_index = update.blendshape_index.max(0) as usize;
        if blendshape_index >= MAX_CLASSIFIED_BLENDSHAPE_INDEX {
            continue;
        }
        if weights.len() <= blendshape_index {
            weights.resize(blendshape_index + 1, 0.0);
        }
        weights[blendshape_index] = update.weight;
    }
}

/// Returns whether two stored values have the same effective deformation contribution.
///
/// Missing, signed-zero, and non-finite weights all select no sparse blendshape coefficient.
fn effective_blendshape_weight_matches(left: f32, right: f32) -> bool {
    let left_inactive = !left.is_finite() || left == 0.0;
    let right_inactive = !right.is_finite() || right == 0.0;
    (left_inactive && right_inactive) || left == right
}

/// Compares blendshape vectors with missing trailing entries interpreted as zero.
fn effective_blendshape_weights_match(left: &[f32], right: &[f32]) -> bool {
    let count = left.len().max(right.len());
    (0..count).all(|index| {
        effective_blendshape_weight_matches(
            left.get(index).copied().unwrap_or(0.0),
            right.get(index).copied().unwrap_or(0.0),
        )
    })
}

/// Reports only renderers whose final accepted blendshape stream changes effective weights.
fn note_changed_skinned_blendshape_weights(
    report: &mut SceneApplyReport,
    space_id: RenderSpaceId,
    current_space: Option<&RenderSpaceState>,
    update: &super::super::meshes::ExtractedSkinnedMeshRenderablesUpdate,
) {
    let mut candidates: HashMap<usize, Vec<f32>> =
        HashMap::with_capacity(update.blendshape_update_batches.len());
    let mut update_offset = 0usize;
    for batch in &update.blendshape_update_batches {
        if batch.renderable_index < 0 {
            break;
        }
        let renderable_index = batch.renderable_index as usize;
        let update_count = batch.blendshape_update_count.max(0) as usize;
        let Some(end) = update_offset.checked_add(update_count) else {
            break;
        };
        let target_is_valid =
            current_space.is_none_or(|space| renderable_index < space.skinned_mesh_renderers.len());
        if target_is_valid && end <= update.blendshape_updates.len() {
            let weights = candidates.entry(renderable_index).or_insert_with(|| {
                current_space
                    .and_then(|space| space.skinned_mesh_renderers.get(renderable_index))
                    .map_or_else(Vec::new, |renderer| {
                        renderer.base.blend_shape_weights.clone()
                    })
            });
            apply_classified_blendshape_slice(
                weights,
                &update.blendshape_updates[update_offset..end],
            );
        }
        // Rejected and out-of-range batches still consume their declared slab segment.
        update_offset = end;
    }

    for (renderable_index, candidate) in candidates {
        let current = current_space
            .and_then(|space| space.skinned_mesh_renderers.get(renderable_index))
            .map(|renderer| renderer.base.blend_shape_weights.as_slice())
            .unwrap_or_default();
        if !effective_blendshape_weights_match(current, &candidate) {
            report.render_world_dirty.note_deform_renderer(
                space_id,
                RenderWorldRendererKind::Skinned,
                renderable_index,
            );
        }
    }
}

/// Compares host bounds by component bit pattern so an identical NaN payload is idempotent.
fn render_bounds_match(
    left: &crate::shared::RenderBoundingBox,
    right: &crate::shared::RenderBoundingBox,
) -> bool {
    left.center
        .to_array()
        .into_iter()
        .chain(left.extents.to_array())
        .zip(
            right
                .center
                .to_array()
                .into_iter()
                .chain(right.extents.to_array()),
        )
        .all(|(left, right)| left.to_bits() == right.to_bits())
}

/// Coalesces posed-bounds rows by renderer and suppresses identical final values.
fn note_changed_skinned_bounds(
    report: &mut SceneApplyReport,
    space_id: RenderSpaceId,
    current_space: Option<&RenderSpaceState>,
    update: &super::super::meshes::ExtractedSkinnedMeshRenderablesUpdate,
) {
    let mut final_bounds = HashMap::with_capacity(update.bounds_updates.len());
    for row in &update.bounds_updates {
        if row.renderable_index < 0 {
            break;
        }
        let renderable_index = row.renderable_index as usize;
        if current_space.is_some_and(|space| renderable_index >= space.skinned_mesh_renderers.len())
        {
            continue;
        }
        final_bounds.insert(renderable_index, row.local_bounds);
    }

    for (renderable_index, bounds) in final_bounds {
        let unchanged = current_space
            .and_then(|space| space.skinned_mesh_renderers.get(renderable_index))
            .and_then(|renderer| renderer.posed_object_bounds.as_ref())
            .is_some_and(|current| render_bounds_match(current, &bounds));
        if !unchanged {
            report.render_world_dirty.note_bounds(
                space_id,
                RenderWorldRendererKind::Skinned,
                renderable_index,
            );
        }
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
        if let Some(space) = current_space {
            for entry in &space.render_material_overrides {
                if entry.node_id >= 0 {
                    report
                        .render_world_dirty
                        .note_context_override(space_id, entry.context);
                }
            }
        }
        for state in material_overrides
            .states
            .iter()
            .take_while(|state| state.renderable_index >= 0)
        {
            report
                .render_world_dirty
                .note_context_override(space_id, state.context);
        }
        if has_active_dense_indices(&material_overrides.additions)
            && material_overrides.states.is_empty()
        {
            report
                .render_world_dirty
                .note_context_override(space_id, crate::shared::RenderingContext::UserView);
        }
        return;
    }
    for state in &material_overrides.states {
        if state.renderable_index < 0 {
            break;
        }
        if let Some(previous) = current_space
            .and_then(|space| {
                space
                    .render_material_overrides
                    .get(state.renderable_index as usize)
            })
            .filter(|entry| entry.node_id >= 0)
        {
            note_material_override_target_dirty(
                report,
                space_id,
                previous.context,
                previous.target,
            );
        }
        let target = decode_packed_mesh_renderer_target(state.packed_mesh_renderer_index);
        note_material_override_target_dirty(report, space_id, state.context, target);
    }
}

fn note_material_override_target_dirty(
    report: &mut SceneApplyReport,
    space_id: RenderSpaceId,
    context: crate::shared::RenderingContext,
    target: MeshRendererOverrideTarget,
) {
    match target {
        MeshRendererOverrideTarget::Static(_) | MeshRendererOverrideTarget::Skinned(_) => {
            report
                .render_world_dirty
                .note_material_override(space_id, context, target);
        }
        MeshRendererOverrideTarget::Unknown => {
            report
                .render_world_dirty
                .note_context_override(space_id, context);
        }
    }
}
