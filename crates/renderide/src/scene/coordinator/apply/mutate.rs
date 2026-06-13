//! Phase B: parallel per-space mutation from owned extracted payloads.
//!
//! [`apply_extracted_render_space_update`] mutates one render space's
//! [`crate::scene::RenderSpaceState`] and [`crate::scene::WorldTransformCache`] using only the
//! owned payloads produced by [`super::extract`]. Distinct render spaces own disjoint state, so
//! these calls can fan out across rayon workers.

use crate::scene::blit_to_display::apply_blit_to_display_update_extracted;
use crate::scene::camera::fixup_cameras_for_transform_removals;
use crate::scene::camera_portal::{
    apply_camera_portal_renderables_update_extracted, fixup_camera_portals_for_transform_removals,
};
use crate::scene::layer::apply_layer_update_extracted;
use crate::scene::lod_groups::apply_lod_group_renderables_update_extracted;
use crate::scene::meshes::{
    apply_mesh_renderables_update_extracted, apply_skinned_mesh_renderables_update_extracted,
    fixup_static_meshes_for_transform_removals,
};
use crate::scene::overrides::{
    apply_render_material_overrides_update_extracted,
    apply_render_transform_overrides_update_extracted,
};
use crate::scene::reflection_probe::{
    apply_reflection_probe_renderables_update_extracted,
    fixup_reflection_probes_for_transform_removals,
};
use crate::scene::render_buffers::{
    apply_billboard_render_buffer_update_extracted, apply_mesh_render_buffer_update_extracted,
    apply_trail_renderer_update_extracted, fixup_render_buffers_for_transform_removals,
};
use crate::scene::transforms::{TransformRemovalEvent, apply_transforms_update_extracted};

use super::super::ApplyWorkSlot;
use super::extract::ExtractedRenderSpaceUpdate;

/// Per-space mutable inputs threaded through [`apply_extracted_render_space_update`].
///
/// Bundles the per-space [`crate::scene::RenderSpaceState`] and
/// [`crate::scene::WorldTransformCache`] alongside a scratch buffer for transform removal events
/// (cleared at the start of each call).
pub(in crate::scene::coordinator) struct PerSpaceApplyInputs<'a> {
    /// Per-space scene state (cameras, mesh renderables, layer assignments, overrides).
    pub space: &'a mut crate::scene::render_space::RenderSpaceState,
    /// Per-space world matrix cache (resized + invalidated to match [`Self::space`]).
    pub cache: &'a mut crate::scene::world::WorldTransformCache,
    /// Reused buffer for [`TransformRemovalEvent`]s emitted by transform removals.
    pub removal_events: &'a mut Vec<TransformRemovalEvent>,
}

/// Runs [`apply_extracted_render_space_update`] against one lifted work slot, recording the
/// resulting world-cache dirty mark on the slot.
pub(super) fn apply_work_slot_mutation(slot: &mut ApplyWorkSlot) {
    slot.world_dirty = apply_extracted_render_space_update(
        &slot.extracted,
        PerSpaceApplyInputs {
            space: &mut slot.space,
            cache: &mut slot.cache,
            removal_events: &mut slot.removal_events,
        },
    );
}

/// Applies one [`ExtractedRenderSpaceUpdate`] against pre-borrowed per-space state.
///
/// Returns `true` when the world cache for this space needs to be re-flushed (mirrors the
/// historical `world_dirty` insert performed by [`crate::scene::SceneCoordinator::apply_frame_submit`]).
///
/// Safe to call concurrently across distinct render spaces because all mutated state is
/// reachable only through the per-space [`PerSpaceApplyInputs`] borrow.
pub(in crate::scene::coordinator) fn apply_extracted_render_space_update(
    extracted: &ExtractedRenderSpaceUpdate,
    inputs: PerSpaceApplyInputs<'_>,
) -> bool {
    profiling::scope!("scene::apply_render_space_chunk");
    let scene_id = extracted.space_id.0;
    let PerSpaceApplyInputs {
        space,
        cache,
        removal_events,
    } = inputs;

    let world_dirty = apply_render_space_transform_phase(extracted, space, cache, removal_events);
    let transform_removals: &[TransformRemovalEvent] = removal_events;
    apply_render_space_geometry_phase(extracted, space, transform_removals, scene_id);
    apply_render_space_layer_phase(extracted, space, transform_removals);
    apply_render_space_context_phase(extracted, space, transform_removals);
    world_dirty
}

fn apply_render_space_transform_phase(
    extracted: &ExtractedRenderSpaceUpdate,
    space: &mut crate::scene::render_space::RenderSpaceState,
    cache: &mut crate::scene::world::WorldTransformCache,
    removal_events: &mut Vec<TransformRemovalEvent>,
) -> bool {
    if let Some(ref tu) = extracted.transforms {
        profiling::scope!("scene::apply_render_space_chunk::transforms");
        apply_transforms_update_extracted(space, cache, extracted.space_id, tu, removal_events)
    } else {
        removal_events.clear();
        false
    }
}

fn apply_render_space_geometry_phase(
    extracted: &ExtractedRenderSpaceUpdate,
    space: &mut crate::scene::render_space::RenderSpaceState,
    transform_removals: &[TransformRemovalEvent],
    scene_id: i32,
) {
    let has_transform_removals = !transform_removals.is_empty();

    // Roll pre-existing cameras' transform ids forward through this frame's swap-removes before
    // applying the extracted camera update (whose addition indices are post-swap from the host).
    if has_transform_removals {
        profiling::scope!("scene::apply_render_space_chunk::fixup_cameras");
        fixup_cameras_for_transform_removals(space, transform_removals);
    }
    if let Some(ref cu) = extracted.cameras {
        profiling::scope!("scene::apply_render_space_chunk::cameras");
        crate::scene::camera::apply_camera_renderables_update_extracted(space, cu);
    }
    if has_transform_removals {
        profiling::scope!("scene::apply_render_space_chunk::fixup_reflection_probes");
        fixup_reflection_probes_for_transform_removals(space, transform_removals);
    }
    if let Some(ref rpu) = extracted.reflection_probes {
        profiling::scope!("scene::apply_render_space_chunk::reflection_probes");
        apply_reflection_probe_renderables_update_extracted(space, rpu);
    } else {
        space.pending_reflection_probe_render_changes.clear();
    }

    if has_transform_removals {
        profiling::scope!("scene::apply_render_space_chunk::fixup_meshes");
        fixup_static_meshes_for_transform_removals(space, transform_removals);
    }
    if let Some(ref mu) = extracted.meshes {
        profiling::scope!("scene::apply_render_space_chunk::meshes");
        apply_mesh_renderables_update_extracted(space, mu, scene_id);
    }
    if let Some(ref su) = extracted.skinned_meshes {
        profiling::scope!("scene::apply_render_space_chunk::skinned_meshes");
        apply_skinned_mesh_renderables_update_extracted(space, su, transform_removals, scene_id);
    }
    if has_transform_removals {
        profiling::scope!("scene::apply_render_space_chunk::fixup_camera_portals");
        fixup_camera_portals_for_transform_removals(space, transform_removals);
    }
    if let Some(ref cpu) = extracted.camera_portals {
        profiling::scope!("scene::apply_render_space_chunk::camera_portals");
        apply_camera_portal_renderables_update_extracted(space, cpu);
    }
    if let Some(ref lgu) = extracted.lod_groups {
        profiling::scope!("scene::apply_render_space_chunk::lod_groups");
        apply_lod_group_renderables_update_extracted(space, lgu, scene_id);
    }
}

fn apply_render_space_layer_phase(
    extracted: &ExtractedRenderSpaceUpdate,
    space: &mut crate::scene::render_space::RenderSpaceState,
    transform_removals: &[TransformRemovalEvent],
) {
    let has_transform_removals = !transform_removals.is_empty();
    let mesh_membership_or_nodes_changed =
        extracted.meshes.is_some() || extracted.skinned_meshes.is_some();
    let layer_inputs_changed =
        has_transform_removals || extracted.layers.is_some() || space.hierarchy_dirty;
    if layer_inputs_changed || mesh_membership_or_nodes_changed || space.layer_index_dirty {
        profiling::scope!("scene::layers");
        if has_transform_removals {
            crate::scene::layer::fixup_layer_assignments_for_transform_removals(
                space,
                transform_removals,
            );
        }
        if let Some(ref lu) = extracted.layers {
            apply_layer_update_extracted(space, lu);
        }
        crate::scene::layer::resolve_mesh_layers_from_assignments(space);
    }
}

fn apply_render_space_context_phase(
    extracted: &ExtractedRenderSpaceUpdate,
    space: &mut crate::scene::render_space::RenderSpaceState,
    transform_removals: &[TransformRemovalEvent],
) {
    let has_transform_removals = !transform_removals.is_empty();
    if let Some(ref rtu) = extracted.transform_overrides {
        profiling::scope!("scene::apply_render_space_chunk::transform_overrides");
        apply_render_transform_overrides_update_extracted(space, rtu, transform_removals);
    }
    if let Some(ref rmu) = extracted.material_overrides {
        profiling::scope!("scene::apply_render_space_chunk::material_overrides");
        apply_render_material_overrides_update_extracted(space, rmu, transform_removals);
    }
    if let Some(ref btd) = extracted.blit_to_displays {
        profiling::scope!("scene::apply_render_space_chunk::blit_to_displays");
        apply_blit_to_display_update_extracted(space, btd);
    }
    if has_transform_removals {
        profiling::scope!("scene::apply_render_space_chunk::fixup_render_buffers");
        fixup_render_buffers_for_transform_removals(space, transform_removals);
    }
    if let Some(ref bu) = extracted.billboard_render_buffers {
        profiling::scope!("scene::apply_render_space_chunk::billboard_render_buffers");
        apply_billboard_render_buffer_update_extracted(space, bu);
    }
    if let Some(ref mu) = extracted.mesh_render_buffers {
        profiling::scope!("scene::apply_render_space_chunk::mesh_render_buffers");
        apply_mesh_render_buffer_update_extracted(space, mu);
    }
    if let Some(ref tu) = extracted.trail_render_buffers {
        profiling::scope!("scene::apply_render_space_chunk::trail_renderers");
        apply_trail_renderer_update_extracted(space, tu);
    }
}
