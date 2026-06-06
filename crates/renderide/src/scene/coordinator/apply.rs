//! Two-phase per-tick scene apply: serial shared-memory pre-extract + parallel per-space mutation.
//!
//! `apply_frame_submit` historically iterated [`crate::shared::RenderSpaceUpdate`] chunks serially
//! because every per-space helper takes `&mut SharedMemoryAccessor`. This module splits that work
//! into:
//!
//! 1. **Phase A (serial):** [`extract_render_space_update`] reads every shared-memory descriptor
//!    referenced by the host update for one render space into owned [`Vec`]s held in
//!    [`ExtractedRenderSpaceUpdate`].
//! 2. **Phase B (parallel):** [`apply_extracted_render_space_update`] mutates the per-space
//!    [`crate::scene::RenderSpaceState`] and [`crate::scene::WorldTransformCache`] using only the
//!    owned payloads. Distinct render spaces own disjoint state, so the apply step can fan out
//!    across rayon workers.
//!
//! Light updates ([`crate::scene::LightCache`]) target shared state and stay serial in a Phase C
//! pass after the parallel apply completes, matching the plan's "simpler first cut".

use crate::ipc::SharedMemoryAccessor;
use crate::shared::{LightRenderablesUpdate, LightsBufferRendererUpdate, RenderSpaceUpdate};

use super::{ApplyWorkSlot, SceneCoordinator};

use super::super::blit_to_display::{
    ExtractedBlitToDisplayUpdate, apply_blit_to_display_update_extracted,
    extract_blit_to_display_update,
};
use super::super::camera::{
    ExtractedCameraRenderablesUpdate, extract_camera_renderables_update,
    fixup_cameras_for_transform_removals,
};
use super::super::camera_portal::{
    ExtractedCameraPortalRenderablesUpdate, apply_camera_portal_renderables_update_extracted,
    extract_camera_portal_renderables_update, fixup_camera_portals_for_transform_removals,
};
use super::super::error::SceneError;
use super::super::ids::RenderSpaceId;
use super::super::layer::{
    ExtractedLayerUpdate, apply_layer_update_extracted, extract_layer_update,
};
use super::super::lod_groups::{
    ExtractedLodGroupRenderablesUpdate, apply_lod_group_renderables_update_extracted,
    extract_lod_group_renderables_update,
};
use super::super::meshes::{
    ExtractedMeshRenderablesUpdate, ExtractedSkinnedMeshRenderablesUpdate,
    apply_mesh_renderables_update_extracted, apply_skinned_mesh_renderables_update_extracted,
    extract_mesh_renderables_update, extract_skinned_mesh_renderables_update,
    fixup_static_meshes_for_transform_removals,
};
use super::super::overrides::{
    ExtractedRenderMaterialOverridesUpdate, ExtractedRenderTransformOverridesUpdate,
    apply_render_material_overrides_update_extracted,
    apply_render_transform_overrides_update_extracted, extract_render_material_overrides_update,
    extract_render_transform_overrides_update,
};
use super::super::reflection_probe::{
    ExtractedReflectionProbeRenderablesUpdate, apply_reflection_probe_renderables_update_extracted,
    extract_reflection_probe_renderables_update, fixup_reflection_probes_for_transform_removals,
};
use super::super::render_buffers::{
    ExtractedBillboardRenderBufferUpdate, ExtractedMeshRenderBufferUpdate,
    ExtractedTrailRendererUpdate, apply_billboard_render_buffer_update_extracted,
    apply_mesh_render_buffer_update_extracted, apply_trail_renderer_update_extracted,
    extract_billboard_render_buffer_update, extract_mesh_render_buffer_update,
    extract_trail_renderer_update, fixup_render_buffers_for_transform_removals,
};
use super::super::transforms::{
    ExtractedTransformsUpdate, TransformRemovalEvent, apply_transforms_update_extracted,
    extract_transforms_update,
};
use crate::cpu_parallelism::{
    ParallelAdmission, VISIBILITY_CULL_CHUNK_ITEMS, current_reference_worker_count,
    has_visibility_parallel_work, record_parallel_admission,
};

/// Render-space apply slots assigned to one mutation worker.
const APPLY_PARALLEL_CHUNK_SPACES: usize = 1;
/// Minimum independent render-space slots before Phase B may use rayon workers.
const MIN_SPACES_FOR_PARALLEL_APPLY: usize = APPLY_PARALLEL_CHUNK_SPACES * 2;

/// Minimum extracted row count before Phase B may fan out across render-space slots.
const MIN_APPLY_PARALLEL_WORK_UNITS: usize = VISIBILITY_CULL_CHUNK_ITEMS * 2;
/// Dominant-slot ratio numerator for switching to intra-space work splitting.
const SPACE_SPLIT_DOMINANCE_NUMERATOR: usize = 2;

macro_rules! extract_optional_render_space_update {
    ($shm:expr, $update:expr, $field:ident, $scope:literal, $extract:path $(, $extra:expr)* $(,)?) => {{
        match ($update).$field.as_ref() {
            Some(payload) => {
                profiling::scope!($scope);
                Some($extract($shm, payload, $($extra,)* ($update).id)?)
            }
            None => None,
        }
    }};
}

/// Returns `true` when [`ExtractedRenderSpaceUpdate`] carries no body work for this tick (every
/// per-update payload is `None`).
///
/// Header fields ([`crate::scene::render_space::RenderSpaceState::apply_update_header`]) are
/// applied during Phase A regardless, so an "empty" extracted update means Phase B has no work
/// for this space and can skip the lift/reinsert pair entirely. Common on ticks where only
/// camera matrices changed.
#[inline]
pub(in crate::scene::coordinator) fn is_extracted_empty(e: &ExtractedRenderSpaceUpdate) -> bool {
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
fn extracted_apply_work_units(e: &ExtractedRenderSpaceUpdate) -> usize {
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
fn apply_parallel_admission_with_workers(
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
fn apply_work_units(work: &[ApplyWorkSlot]) -> usize {
    work.iter().map(|slot| slot.work_units).sum()
}

/// Returns the largest estimated work carried by one render-space slot.
#[inline]
fn dominant_slot_work_units(work: &[ApplyWorkSlot]) -> usize {
    work.iter().map(|slot| slot.work_units).max().unwrap_or(0)
}

/// Returns whether a dominant render-space slot should avoid the outer per-space fan-out.
#[inline]
fn space_split_apply_preferred(
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

fn apply_work_slot_mutation(slot: &mut ApplyWorkSlot) {
    slot.world_dirty = apply_extracted_render_space_update(
        &slot.extracted,
        PerSpaceApplyInputs {
            space: &mut slot.space,
            cache: &mut slot.cache,
            removal_events: &mut slot.removal_events,
        },
    );
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

/// Owned per-space payload bundle: every shared-memory buffer referenced by one
/// [`RenderSpaceUpdate`] pre-read into [`Vec`]s, ready for parallel apply.
///
/// Each `Option<...>` field mirrors the corresponding `Option<...>` on [`RenderSpaceUpdate`] and is
/// `None` when the host omitted that update kind for this tick.
pub(in crate::scene::coordinator) struct ExtractedRenderSpaceUpdate {
    /// Render space identity for this chunk (mirrors [`RenderSpaceUpdate::id`]).
    pub space_id: RenderSpaceId,
    /// Camera-renderable update payload.
    pub cameras: Option<ExtractedCameraRenderablesUpdate>,
    /// Camera-portal renderable update payload.
    pub camera_portals: Option<ExtractedCameraPortalRenderablesUpdate>,
    /// Reflection-probe renderable update payload.
    pub reflection_probes: Option<ExtractedReflectionProbeRenderablesUpdate>,
    /// Dense transform-table update payload.
    pub transforms: Option<ExtractedTransformsUpdate>,
    /// Static mesh-renderable update payload.
    pub meshes: Option<ExtractedMeshRenderablesUpdate>,
    /// Skinned mesh-renderable update payload (state, bones, blendshapes).
    pub skinned_meshes: Option<ExtractedSkinnedMeshRenderablesUpdate>,
    /// Layer-assignment update payload.
    pub layers: Option<ExtractedLayerUpdate>,
    /// LOD group update payload.
    pub lod_groups: Option<ExtractedLodGroupRenderablesUpdate>,
    /// Render-context transform-override update payload.
    pub transform_overrides: Option<ExtractedRenderTransformOverridesUpdate>,
    /// Render-context material-override update payload.
    pub material_overrides: Option<ExtractedRenderMaterialOverridesUpdate>,
    /// `BlitToDisplay` renderables update payload.
    pub blit_to_displays: Option<ExtractedBlitToDisplayUpdate>,
    /// PhotonDust billboard renderer update payload.
    pub billboard_render_buffers: Option<ExtractedBillboardRenderBufferUpdate>,
    /// PhotonDust mesh-particle renderer update payload.
    pub mesh_render_buffers: Option<ExtractedMeshRenderBufferUpdate>,
    /// PhotonDust trail renderer update payload.
    pub trail_render_buffers: Option<ExtractedTrailRendererUpdate>,
}

/// Extracted renderer payloads tied directly to scene geometry and visibility.
struct ExtractedGeometryRenderSpaceUpdates {
    /// Camera renderer update payload.
    cameras: Option<ExtractedCameraRenderablesUpdate>,
    /// Camera-portal renderer update payload.
    camera_portals: Option<ExtractedCameraPortalRenderablesUpdate>,
    /// Reflection-probe renderer update payload.
    reflection_probes: Option<ExtractedReflectionProbeRenderablesUpdate>,
    /// Transform update payload.
    transforms: Option<ExtractedTransformsUpdate>,
    /// Static mesh renderer update payload.
    meshes: Option<ExtractedMeshRenderablesUpdate>,
    /// Skinned mesh renderer update payload.
    skinned_meshes: Option<ExtractedSkinnedMeshRenderablesUpdate>,
    /// Layer update payload.
    layers: Option<ExtractedLayerUpdate>,
    /// LOD-group renderer update payload.
    lod_groups: Option<ExtractedLodGroupRenderablesUpdate>,
}

/// Extracted renderer payloads for render-context state and generated particle renderers.
struct ExtractedContextRenderSpaceUpdates {
    /// Render-context transform-override update payload.
    transform_overrides: Option<ExtractedRenderTransformOverridesUpdate>,
    /// Render-context material-override update payload.
    material_overrides: Option<ExtractedRenderMaterialOverridesUpdate>,
    /// `BlitToDisplay` renderables update payload.
    blit_to_displays: Option<ExtractedBlitToDisplayUpdate>,
    /// PhotonDust billboard renderer update payload.
    billboard_render_buffers: Option<ExtractedBillboardRenderBufferUpdate>,
    /// PhotonDust mesh-particle renderer update payload.
    mesh_render_buffers: Option<ExtractedMeshRenderBufferUpdate>,
    /// PhotonDust trail renderer update payload.
    trail_render_buffers: Option<ExtractedTrailRendererUpdate>,
}

/// Reads every shared-memory buffer referenced by `update` into owned vectors.
///
/// Light updates are intentionally **not** extracted here: their apply step mutates the shared
/// [`crate::scene::LightCache`] and is handled in a separate serial pass (see
/// [`light_updates_view`]).
pub(in crate::scene::coordinator) fn extract_render_space_update(
    shm: &mut SharedMemoryAccessor,
    update: &RenderSpaceUpdate,
    frame_index: i32,
) -> Result<ExtractedRenderSpaceUpdate, SceneError> {
    profiling::scope!("scene::extract_render_space");
    let space_id = RenderSpaceId(update.id);
    let geometry = extract_geometry_render_space_updates(shm, update, frame_index)?;
    let context = extract_context_render_space_updates(shm, update)?;
    Ok(ExtractedRenderSpaceUpdate {
        space_id,
        cameras: geometry.cameras,
        camera_portals: geometry.camera_portals,
        reflection_probes: geometry.reflection_probes,
        transforms: geometry.transforms,
        meshes: geometry.meshes,
        skinned_meshes: geometry.skinned_meshes,
        layers: geometry.layers,
        lod_groups: geometry.lod_groups,
        transform_overrides: context.transform_overrides,
        material_overrides: context.material_overrides,
        blit_to_displays: context.blit_to_displays,
        billboard_render_buffers: context.billboard_render_buffers,
        mesh_render_buffers: context.mesh_render_buffers,
        trail_render_buffers: context.trail_render_buffers,
    })
}

/// Extracts scene-geometry update payloads referenced by `update`.
fn extract_geometry_render_space_updates(
    shm: &mut SharedMemoryAccessor,
    update: &RenderSpaceUpdate,
    frame_index: i32,
) -> Result<ExtractedGeometryRenderSpaceUpdates, SceneError> {
    Ok(ExtractedGeometryRenderSpaceUpdates {
        cameras: extract_optional_render_space_update!(
            shm,
            update,
            cameras_update,
            "scene::extract_render_space::cameras",
            extract_camera_renderables_update,
        ),
        camera_portals: extract_optional_render_space_update!(
            shm,
            update,
            camera_portals_update,
            "scene::extract_render_space::camera_portals",
            extract_camera_portal_renderables_update,
        ),
        reflection_probes: extract_optional_render_space_update!(
            shm,
            update,
            reflection_probes_update,
            "scene::extract_render_space::reflection_probes",
            extract_reflection_probe_renderables_update,
        ),
        transforms: extract_optional_render_space_update!(
            shm,
            update,
            transforms_update,
            "scene::extract_render_space::transforms",
            extract_transforms_update,
            frame_index,
        ),
        meshes: extract_optional_render_space_update!(
            shm,
            update,
            mesh_renderers_update,
            "scene::extract_render_space::meshes",
            extract_mesh_renderables_update,
        ),
        skinned_meshes: extract_optional_render_space_update!(
            shm,
            update,
            skinned_mesh_renderers_update,
            "scene::extract_render_space::skinned_meshes",
            extract_skinned_mesh_renderables_update,
        ),
        layers: extract_optional_render_space_update!(
            shm,
            update,
            layers_update,
            "scene::extract_render_space::layers",
            extract_layer_update,
        ),
        lod_groups: extract_optional_render_space_update!(
            shm,
            update,
            lod_group_update,
            "scene::extract_render_space::lod_groups",
            extract_lod_group_renderables_update,
        ),
    })
}

/// Extracts render-context and generated-particle update payloads referenced by `update`.
fn extract_context_render_space_updates(
    shm: &mut SharedMemoryAccessor,
    update: &RenderSpaceUpdate,
) -> Result<ExtractedContextRenderSpaceUpdates, SceneError> {
    Ok(ExtractedContextRenderSpaceUpdates {
        transform_overrides: extract_optional_render_space_update!(
            shm,
            update,
            render_transform_overrides_update,
            "scene::extract_render_space::transform_overrides",
            extract_render_transform_overrides_update,
        ),
        material_overrides: extract_optional_render_space_update!(
            shm,
            update,
            render_material_overrides_update,
            "scene::extract_render_space::material_overrides",
            extract_render_material_overrides_update,
        ),
        blit_to_displays: extract_optional_render_space_update!(
            shm,
            update,
            blit_to_displays_update,
            "scene::extract_render_space::blit_to_displays",
            extract_blit_to_display_update,
        ),
        billboard_render_buffers: extract_optional_render_space_update!(
            shm,
            update,
            billboard_buffers_update,
            "scene::extract_render_space::billboard_render_buffers",
            extract_billboard_render_buffer_update,
        ),
        mesh_render_buffers: extract_optional_render_space_update!(
            shm,
            update,
            mesh_render_buffers_update,
            "scene::extract_render_space::mesh_render_buffers",
            extract_mesh_render_buffer_update,
        ),
        trail_render_buffers: extract_optional_render_space_update!(
            shm,
            update,
            trail_renderers_update,
            "scene::extract_render_space::trail_renderers",
            extract_trail_renderer_update,
        ),
    })
}

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
        super::super::camera::apply_camera_renderables_update_extracted(space, cu);
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
            super::super::layer::fixup_layer_assignments_for_transform_removals(
                space,
                transform_removals,
            );
        }
        if let Some(ref lu) = extracted.layers {
            apply_layer_update_extracted(space, lu);
        }
        super::super::layer::resolve_mesh_layers_from_assignments(space);
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

/// Borrowed view of the still-serial light-update payloads for a [`RenderSpaceUpdate`].
///
/// Carried alongside the parallel-applied per-space payloads so the post-parallel light pass can
/// re-walk the host updates without re-scanning [`crate::shared::FrameSubmitData::render_spaces`].
pub(in crate::scene::coordinator) struct LightUpdateView<'a> {
    /// Render space identity (mirrors [`RenderSpaceUpdate::id`]).
    pub space_id: i32,
    /// Optional [`crate::shared::LightRenderablesUpdate`] payload (regular [`crate::shared::LightState`] rows).
    pub lights_update: Option<&'a LightRenderablesUpdate>,
    /// Optional [`crate::shared::LightsBufferRendererUpdate`] payload (buffer-based lights).
    pub lights_buffer_renderers_update: Option<&'a LightsBufferRendererUpdate>,
}

/// Borrows the still-serial light update fields from a [`RenderSpaceUpdate`].
pub(in crate::scene::coordinator) fn light_updates_view(
    update: &RenderSpaceUpdate,
) -> LightUpdateView<'_> {
    LightUpdateView {
        space_id: update.id,
        lights_update: update.lights_update.as_ref(),
        lights_buffer_renderers_update: update.lights_buffer_renderers_update.as_ref(),
    }
}

impl SceneCoordinator {
    /// Drains per-space state, runs Phase B (parallel where it pays), and re-inserts the results.
    ///
    /// Drives the rayon fan-out used by [`SceneCoordinator::apply_frame_submit`]. For one or zero
    /// entries we stay serial to skip rayon dispatch overhead. Per-space dirty cache marks are
    /// merged into [`SceneCoordinator::world_dirty`] on the main thread before reinsert. The
    /// caller's `extracted_per_space` is drained in place so the backing [`Vec`] capacity persists
    /// across frames.
    pub(super) fn apply_extracted_per_space(
        &mut self,
        extracted_per_space: &mut Vec<ExtractedRenderSpaceUpdate>,
    ) -> Result<(), SceneError> {
        if extracted_per_space.is_empty() {
            return Ok(());
        }
        profiling::scope!("scene::apply_frame_submit::apply");

        // Hoist the per-frame work buffer out of `self` so we can fill it while still mutating
        // `self.spaces` / `self.world_caches`. `mem::take` preserves the underlying allocation
        // so steady-state apply does not allocate.
        let mut work = std::mem::take(&mut self.apply_work_scratch);
        debug_assert!(work.is_empty());
        {
            profiling::scope!("scene::apply::lift");
            for extracted in extracted_per_space.drain(..) {
                let id = extracted.space_id;
                // Header fields were already applied in Phase A. If the host sent no body
                // payloads for this space this tick, skip the lift/reinsert pair entirely so the
                // steady-state path never moves a [`RenderSpaceState`].
                if is_extracted_empty(&extracted) {
                    continue;
                }
                let work_units = extracted_apply_work_units(&extracted);
                let Some(space) = self.spaces.remove(&id) else {
                    continue;
                };
                let cache = self.world_caches.remove(&id).unwrap_or_default();
                let removal_events = self
                    .transform_removals_by_space
                    .remove(&id)
                    .unwrap_or_default();
                work.push(ApplyWorkSlot {
                    id,
                    space,
                    cache,
                    extracted,
                    work_units,
                    removal_events,
                    world_dirty: false,
                });
            }
        }

        let work_units = {
            profiling::scope!("scene::apply::estimate_parallel_work");
            apply_work_units(&work)
        };
        let worker_count = current_reference_worker_count();
        let admission = apply_parallel_admission_with_workers(work.len(), work_units, worker_count);
        record_parallel_admission("scene_apply", work_units, work.len(), admission);
        if space_split_apply_preferred(
            work.len(),
            work_units,
            dominant_slot_work_units(&work),
            worker_count,
        ) {
            profiling::scope!("scene::apply::mutate::space_split");
            for slot in &mut work {
                if slot.work_units >= MIN_APPLY_PARALLEL_WORK_UNITS {
                    profiling::scope!("scene::apply::mutate::space_split_slot");
                    apply_work_slot_mutation(slot);
                } else {
                    apply_work_slot_mutation(slot);
                }
            }
            self.reinsert_applied_work_slots(&mut work);
            self.apply_work_scratch = work;
            return Ok(());
        }
        if !admission.is_parallel() {
            profiling::scope!("scene::apply::mutate::serial_small_batch");
            for slot in &mut work {
                profiling::scope!("scene::apply::mutate::serial_slot");
                apply_work_slot_mutation(slot);
            }
            self.reinsert_applied_work_slots(&mut work);
            self.apply_work_scratch = work;
            return Ok(());
        }

        use rayon::prelude::*;
        {
            profiling::scope!("scene::apply::mutate");
            work.par_iter_mut()
                .with_min_len(APPLY_PARALLEL_CHUNK_SPACES)
                .for_each(|slot| {
                    profiling::scope!("scene::apply::mutate::worker_slot");
                    apply_work_slot_mutation(slot);
                });
        };
        self.reinsert_applied_work_slots(&mut work);
        self.apply_work_scratch = work;
        Ok(())
    }

    fn reinsert_applied_work_slots(&mut self, work: &mut Vec<ApplyWorkSlot>) {
        profiling::scope!("scene::apply::reinsert");
        for slot in work.drain(..) {
            if slot.world_dirty {
                self.world_dirty.insert(slot.id);
            }
            self.spaces.insert(slot.id, slot.space);
            self.world_caches.insert(slot.id, slot.cache);
            self.stash_transform_removals(slot.id, slot.removal_events);
        }
    }

    /// Moves a per-space transform-removal buffer back into
    /// [`SceneCoordinator::transform_removals_by_space`] so Phase C can read it.
    fn stash_transform_removals(
        &mut self,
        id: RenderSpaceId,
        removals: Vec<TransformRemovalEvent>,
    ) {
        self.transform_removals_by_space.insert(id, removals);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
