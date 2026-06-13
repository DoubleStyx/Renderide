//! Phase B parallelism policy: per-domain work-unit estimates and admission decisions.
//!
//! Work units approximate extracted row counts so sparse ticks with many tiny spaces stay on the
//! serial path while dense multi-space ticks fan out across rayon workers.

use crate::cpu_parallelism::{
    ParallelAdmission, VISIBILITY_CULL_CHUNK_ITEMS, has_visibility_parallel_work,
};
use crate::scene::blit_to_display::ExtractedBlitToDisplayUpdate;
use crate::scene::camera::ExtractedCameraRenderablesUpdate;
use crate::scene::camera_portal::ExtractedCameraPortalRenderablesUpdate;
use crate::scene::layer::ExtractedLayerUpdate;
use crate::scene::lod_groups::ExtractedLodGroupRenderablesUpdate;
use crate::scene::meshes::{ExtractedMeshRenderablesUpdate, ExtractedSkinnedMeshRenderablesUpdate};
use crate::scene::overrides::{
    ExtractedRenderMaterialOverridesUpdate, ExtractedRenderTransformOverridesUpdate,
};
use crate::scene::reflection_probe::ExtractedReflectionProbeRenderablesUpdate;
use crate::scene::render_buffers::{
    ExtractedBillboardRenderBufferUpdate, ExtractedMeshRenderBufferUpdate,
    ExtractedTrailRendererUpdate,
};
use crate::scene::transforms::ExtractedTransformsUpdate;

use super::super::ApplyWorkSlot;
use super::extract::ExtractedRenderSpaceUpdate;

/// Render-space apply slots assigned to one mutation worker.
pub(super) const APPLY_PARALLEL_CHUNK_SPACES: usize = 1;
/// Minimum independent render-space slots before Phase B may use rayon workers.
const MIN_SPACES_FOR_PARALLEL_APPLY: usize = APPLY_PARALLEL_CHUNK_SPACES * 2;

/// Minimum extracted row count before Phase B may fan out across render-space slots.
pub(super) const MIN_APPLY_PARALLEL_WORK_UNITS: usize = VISIBILITY_CULL_CHUNK_ITEMS * 2;
/// Dominant-slot ratio numerator for switching to intra-space work splitting.
const SPACE_SPLIT_DOMINANCE_NUMERATOR: usize = 2;

/// Returns `true` when [`ExtractedRenderSpaceUpdate`] carries no body work for this tick (every
/// per-update payload is `None`).
///
/// Header fields ([`crate::scene::render_space::RenderSpaceState::apply_update_header`]) are
/// applied during Phase A regardless, so an "empty" extracted update means Phase B has no work
/// for this space and can skip the lift/reinsert pair entirely. Common on ticks where only
/// camera matrices changed.
#[inline]
pub(super) fn is_extracted_empty(e: &ExtractedRenderSpaceUpdate) -> bool {
    e.cameras.is_none()
        && e.camera_portals.is_none()
        && e.reflection_probes.is_none()
        && e.transforms.is_none()
        && e.meshes.is_none()
        && e.skinned_meshes.is_none()
        && e.layers.is_none()
        && e.lod_groups.is_none()
        && e.transform_overrides.is_none()
        && e.material_overrides.is_none()
        && e.blit_to_displays.is_none()
        && e.billboard_render_buffers.is_none()
        && e.mesh_render_buffers.is_none()
        && e.trail_render_buffers.is_none()
}

/// Returns the row count that approximates how much Phase B work a space update carries.
///
/// The estimate intentionally uses extracted payload lengths rather than render-space size so sparse
/// ticks with many tiny spaces stay on the serial path, while dense multi-space ticks still fan out.
pub(super) fn extracted_apply_work_units(e: &ExtractedRenderSpaceUpdate) -> usize {
    let mut units = 0usize;
    if let Some(update) = &e.cameras {
        units += camera_update_work_units(update);
    }
    if let Some(update) = &e.camera_portals {
        units += camera_portal_update_work_units(update);
    }
    if let Some(update) = &e.reflection_probes {
        units += reflection_probe_update_work_units(update);
    }
    if let Some(update) = &e.transforms {
        units += transform_update_work_units(update);
    }
    if let Some(update) = &e.meshes {
        units += mesh_update_work_units(update);
    }
    if let Some(update) = &e.skinned_meshes {
        units += skinned_mesh_update_work_units(update);
    }
    if let Some(update) = &e.layers {
        units += layer_update_work_units(update);
    }
    if let Some(update) = &e.lod_groups {
        units += lod_group_update_work_units(update);
    }
    if let Some(update) = &e.transform_overrides {
        units += transform_override_update_work_units(update);
    }
    if let Some(update) = &e.material_overrides {
        units += material_override_update_work_units(update);
    }
    if let Some(update) = &e.blit_to_displays {
        units += blit_to_display_update_work_units(update);
    }
    if let Some(update) = &e.billboard_render_buffers {
        units += billboard_render_buffer_update_work_units(update);
    }
    if let Some(update) = &e.mesh_render_buffers {
        units += mesh_render_buffer_update_work_units(update);
    }
    if let Some(update) = &e.trail_render_buffers {
        units += trail_renderer_update_work_units(update);
    }
    units
}

/// Returns the apply admission decision for a known worker count.
#[inline]
pub(super) fn apply_parallel_admission_with_workers(
    slot_count: usize,
    work_units: usize,
    worker_count: usize,
) -> ParallelAdmission {
    if slot_count >= MIN_SPACES_FOR_PARALLEL_APPLY
        && work_units >= MIN_APPLY_PARALLEL_WORK_UNITS
        && has_visibility_parallel_work(work_units, worker_count)
    {
        ParallelAdmission::Parallel {
            chunk_size: APPLY_PARALLEL_CHUNK_SPACES,
        }
    } else {
        ParallelAdmission::Serial
    }
}

/// Returns the total Phase B row estimate for all lifted work slots.
#[inline]
pub(super) fn apply_work_units(work: &[ApplyWorkSlot]) -> usize {
    work.iter().map(|slot| slot.work_units).sum()
}

/// Returns the largest estimated work carried by one render-space slot.
#[inline]
pub(super) fn dominant_slot_work_units(work: &[ApplyWorkSlot]) -> usize {
    work.iter().map(|slot| slot.work_units).max().unwrap_or(0)
}

/// Returns whether a dominant render-space slot should avoid the outer per-space fan-out.
#[inline]
pub(super) fn space_split_apply_preferred(
    slot_count: usize,
    total_work_units: usize,
    dominant_work_units: usize,
    worker_count: usize,
) -> bool {
    worker_count > 1
        && slot_count > 0
        && slot_count < worker_count
        && dominant_work_units >= MIN_APPLY_PARALLEL_WORK_UNITS
        && dominant_work_units.saturating_mul(SPACE_SPLIT_DOMINANCE_NUMERATOR) >= total_work_units
}

/// Counts extracted camera rows and side slabs.
#[inline]
fn camera_update_work_units(update: &ExtractedCameraRenderablesUpdate) -> usize {
    update.removals.len()
        + update.additions.len()
        + update.states.len()
        + update.transform_ids.as_ref().map_or(0, Vec::len)
}

/// Counts extracted camera-portal rows.
#[inline]
fn camera_portal_update_work_units(update: &ExtractedCameraPortalRenderablesUpdate) -> usize {
    update.removals.len() + update.additions.len() + update.states.len()
}

/// Counts extracted reflection-probe rows.
#[inline]
fn reflection_probe_update_work_units(update: &ExtractedReflectionProbeRenderablesUpdate) -> usize {
    update.removals.len()
        + update.additions.len()
        + update.states.len()
        + update.changed_probes_to_render.len()
}

/// Counts extracted transform rows.
#[inline]
fn transform_update_work_units(update: &ExtractedTransformsUpdate) -> usize {
    update.removals.len() + update.parent_updates.len() + update.pose_updates.len()
}

/// Counts extracted static mesh renderer rows and material slabs.
#[inline]
fn mesh_update_work_units(update: &ExtractedMeshRenderablesUpdate) -> usize {
    update.removals.len()
        + update.additions.len()
        + update.mesh_states.len()
        + update
            .mesh_materials_and_property_blocks
            .as_ref()
            .map_or(0, Vec::len)
}

/// Counts extracted skinned mesh renderer rows and dependent slabs.
#[inline]
fn skinned_mesh_update_work_units(update: &ExtractedSkinnedMeshRenderablesUpdate) -> usize {
    update.removals.len()
        + update.additions.len()
        + update.mesh_states.len()
        + update
            .mesh_materials_and_property_blocks
            .as_ref()
            .map_or(0, Vec::len)
        + update.bone_assignments.len()
        + update.bone_transform_indexes.len()
        + update.blendshape_update_batches.len()
        + update.blendshape_updates.len()
        + update.bounds_updates.len()
}

/// Counts extracted layer assignment rows.
#[inline]
fn layer_update_work_units(update: &ExtractedLayerUpdate) -> usize {
    update.removals.len() + update.additions.len() + update.layer_assignments.len()
}

/// Counts extracted LOD group rows and dependent renderer slabs.
#[inline]
fn lod_group_update_work_units(update: &ExtractedLodGroupRenderablesUpdate) -> usize {
    update.removals.len()
        + update.additions.len()
        + update.states.len()
        + update.lod_states.len()
        + update.packed_mesh_renderer_ids.len()
}

/// Counts extracted render-transform override rows.
#[inline]
fn transform_override_update_work_units(update: &ExtractedRenderTransformOverridesUpdate) -> usize {
    update.removals.len()
        + update.additions.len()
        + update.states.len()
        + update.skinned_mesh_renderers_indexes.len()
}

/// Counts extracted render-material override rows.
#[inline]
fn material_override_update_work_units(update: &ExtractedRenderMaterialOverridesUpdate) -> usize {
    update.removals.len()
        + update.additions.len()
        + update.states.len()
        + update.material_override_states.len()
}

/// Counts extracted `BlitToDisplay` rows.
#[inline]
fn blit_to_display_update_work_units(update: &ExtractedBlitToDisplayUpdate) -> usize {
    update.removals.len() + update.additions.len() + update.states.len()
}

/// Counts extracted billboard render-buffer rows.
#[inline]
fn billboard_render_buffer_update_work_units(
    update: &ExtractedBillboardRenderBufferUpdate,
) -> usize {
    update.removals.len() + update.additions.len() + update.states.len()
}

/// Counts extracted mesh render-buffer rows.
#[inline]
fn mesh_render_buffer_update_work_units(update: &ExtractedMeshRenderBufferUpdate) -> usize {
    update.removals.len() + update.additions.len() + update.states.len()
}

/// Counts extracted trail renderer rows.
#[inline]
fn trail_renderer_update_work_units(update: &ExtractedTrailRendererUpdate) -> usize {
    update.removals.len() + update.additions.len() + update.states.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::ids::RenderSpaceId;

    fn empty_extracted(space_id: i32) -> ExtractedRenderSpaceUpdate {
        ExtractedRenderSpaceUpdate {
            space_id: RenderSpaceId(space_id),
            cameras: None,
            camera_portals: None,
            reflection_probes: None,
            transforms: None,
            meshes: None,
            skinned_meshes: None,
            layers: None,
            lod_groups: None,
            transform_overrides: None,
            material_overrides: None,
            blit_to_displays: None,
            billboard_render_buffers: None,
            mesh_render_buffers: None,
            trail_render_buffers: None,
        }
    }

    #[test]
    fn apply_work_units_count_extracted_rows() {
        let mut extracted = empty_extracted(7);
        extracted.cameras = Some(ExtractedCameraRenderablesUpdate {
            removals: vec![0, -1],
            additions: vec![11, -1],
            transform_ids: Some(vec![2, 3, 4]),
            ..Default::default()
        });
        extracted.layers = Some(ExtractedLayerUpdate {
            removals: vec![1, -1],
            additions: vec![5, 6, -1],
            ..Default::default()
        });

        assert_eq!(extracted_apply_work_units(&extracted), 12);
    }

    #[test]
    fn apply_parallelism_requires_multiple_slots_and_enough_work() {
        assert!(
            !apply_parallel_admission_with_workers(1, MIN_APPLY_PARALLEL_WORK_UNITS, 4)
                .is_parallel()
        );
        assert!(
            !apply_parallel_admission_with_workers(
                MIN_SPACES_FOR_PARALLEL_APPLY,
                MIN_APPLY_PARALLEL_WORK_UNITS - 1,
                4
            )
            .is_parallel()
        );
        assert!(
            apply_parallel_admission_with_workers(
                MIN_SPACES_FOR_PARALLEL_APPLY,
                MIN_APPLY_PARALLEL_WORK_UNITS,
                4
            )
            .is_parallel()
        );
        assert!(
            !apply_parallel_admission_with_workers(
                MIN_SPACES_FOR_PARALLEL_APPLY,
                MIN_APPLY_PARALLEL_WORK_UNITS,
                1
            )
            .is_parallel()
        );
    }

    #[test]
    fn space_split_apply_prefers_dominant_underfilled_slots() {
        assert!(space_split_apply_preferred(
            2,
            MIN_APPLY_PARALLEL_WORK_UNITS + 64,
            MIN_APPLY_PARALLEL_WORK_UNITS,
            8
        ));
        assert!(!space_split_apply_preferred(
            2,
            MIN_APPLY_PARALLEL_WORK_UNITS * 2,
            MIN_APPLY_PARALLEL_WORK_UNITS - 1,
            8
        ));
        assert!(!space_split_apply_preferred(
            8,
            MIN_APPLY_PARALLEL_WORK_UNITS * 8,
            MIN_APPLY_PARALLEL_WORK_UNITS * 2,
            8
        ));
        assert!(!space_split_apply_preferred(
            2,
            MIN_APPLY_PARALLEL_WORK_UNITS,
            MIN_APPLY_PARALLEL_WORK_UNITS,
            1
        ));
    }
}
