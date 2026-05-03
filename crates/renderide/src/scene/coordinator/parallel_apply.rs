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

use super::SceneCoordinator;

use super::super::camera_apply::{
    ExtractedCameraRenderablesUpdate, extract_camera_renderables_update,
    fixup_cameras_for_transform_removals,
};
use super::super::error::SceneError;
use super::super::ids::RenderSpaceId;
use super::super::layer_apply::{
    ExtractedLayerUpdate, apply_layer_update_extracted, extract_layer_update,
};
use super::super::mesh_apply::{
    ExtractedMeshRenderablesUpdate, ExtractedSkinnedMeshRenderablesUpdate,
    apply_mesh_renderables_update_extracted, apply_skinned_mesh_renderables_update_extracted,
    extract_mesh_renderables_update, extract_skinned_mesh_renderables_update,
    fixup_static_meshes_for_transform_removals,
};
use super::super::reflection_probe::{
    ExtractedReflectionProbeRenderablesUpdate, apply_reflection_probe_renderables_update_extracted,
    extract_reflection_probe_renderables_update, fixup_reflection_probes_for_transform_removals,
};
use super::super::render_overrides::{
    ExtractedRenderMaterialOverridesUpdate, ExtractedRenderTransformOverridesUpdate,
    apply_render_material_overrides_update_extracted,
    apply_render_transform_overrides_update_extracted, extract_render_material_overrides_update,
    extract_render_transform_overrides_update,
};
use super::super::transforms_apply::{
    ExtractedTransformsUpdate, TransformRemovalEvent, apply_transforms_update_extracted,
    extract_transforms_update,
};

/// Returns `true` when [`ExtractedRenderSpaceUpdate`] carries no body work for this tick (every
/// per-update payload is `None`).
///
/// Header fields ([`crate::scene::render_space::RenderSpaceState::apply_update_header`]) are
/// applied during Phase A regardless, so an "empty" extracted update means Phase B has no work
/// for this space and can skip the lift/reinsert pair entirely. Common on ticks where only
/// camera matrices changed.
#[inline]
pub fn is_extracted_empty(e: &ExtractedRenderSpaceUpdate) -> bool {
    e.cameras.is_none()
        && e.reflection_probes.is_none()
        && e.transforms.is_none()
        && e.meshes.is_none()
        && e.skinned_meshes.is_none()
        && e.layers.is_none()
        && e.transform_overrides.is_none()
        && e.material_overrides.is_none()
}

/// Owned per-space payload bundle: every shared-memory buffer referenced by one
/// [`RenderSpaceUpdate`] pre-read into [`Vec`]s, ready for parallel apply.
///
/// Each `Option<...>` field mirrors the corresponding `Option<...>` on [`RenderSpaceUpdate`] and is
/// `None` when the host omitted that update kind for this tick.
pub struct ExtractedRenderSpaceUpdate {
    /// Render space identity for this chunk (mirrors [`RenderSpaceUpdate::id`]).
    pub space_id: RenderSpaceId,
    /// Camera-renderable update payload.
    pub cameras: Option<ExtractedCameraRenderablesUpdate>,
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
    /// Render-context transform-override update payload.
    pub transform_overrides: Option<ExtractedRenderTransformOverridesUpdate>,
    /// Render-context material-override update payload.
    pub material_overrides: Option<ExtractedRenderMaterialOverridesUpdate>,
}

/// Reads every shared-memory buffer referenced by `update` into owned vectors.
///
/// Light updates are intentionally **not** extracted here: their apply step mutates the shared
/// [`crate::scene::LightCache`] and is handled in a separate serial pass (see
/// [`light_updates_view`]).
pub fn extract_render_space_update(
    shm: &mut SharedMemoryAccessor,
    update: &RenderSpaceUpdate,
    frame_index: i32,
) -> Result<ExtractedRenderSpaceUpdate, SceneError> {
    profiling::scope!("scene::extract_render_space");
    let space_id = RenderSpaceId(update.id);
    let cameras = match update.cameras_update.as_ref() {
        Some(cu) => Some(extract_camera_renderables_update(shm, cu, update.id)?),
        None => None,
    };
    let reflection_probes = match update.reflection_probes_update.as_ref() {
        Some(rpu) => Some(extract_reflection_probe_renderables_update(
            shm, rpu, update.id,
        )?),
        None => None,
    };
    let transforms = match update.transforms_update.as_ref() {
        Some(tu) => Some(extract_transforms_update(shm, tu, frame_index, update.id)?),
        None => None,
    };
    let meshes = match update.mesh_renderers_update.as_ref() {
        Some(mu) => Some(extract_mesh_renderables_update(shm, mu, update.id)?),
        None => None,
    };
    let skinned_meshes = match update.skinned_mesh_renderers_update.as_ref() {
        Some(su) => Some(extract_skinned_mesh_renderables_update(shm, su, update.id)?),
        None => None,
    };
    let layers = match update.layers_update.as_ref() {
        Some(lu) => Some(extract_layer_update(shm, lu, update.id)?),
        None => None,
    };
    let transform_overrides = match update.render_transform_overrides_update.as_ref() {
        Some(rtu) => Some(extract_render_transform_overrides_update(
            shm, rtu, update.id,
        )?),
        None => None,
    };
    let material_overrides = match update.render_material_overrides_update.as_ref() {
        Some(rmu) => Some(extract_render_material_overrides_update(
            shm, rmu, update.id,
        )?),
        None => None,
    };
    Ok(ExtractedRenderSpaceUpdate {
        space_id,
        cameras,
        reflection_probes,
        transforms,
        meshes,
        skinned_meshes,
        layers,
        transform_overrides,
        material_overrides,
    })
}

/// Per-space mutable inputs threaded through [`apply_extracted_render_space_update`].
///
/// Bundles the per-space [`crate::scene::RenderSpaceState`] and
/// [`crate::scene::WorldTransformCache`] alongside a scratch buffer for transform removal events
/// (cleared at the start of each call).
pub struct PerSpaceApplyInputs<'a> {
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
pub fn apply_extracted_render_space_update(
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

    let mut world_dirty = false;
    if let Some(ref tu) = extracted.transforms {
        if apply_transforms_update_extracted(space, cache, extracted.space_id, tu, removal_events) {
            world_dirty = true;
        }
    } else {
        removal_events.clear();
    }
    let transform_removals: &[TransformRemovalEvent] = removal_events;

    // Roll pre-existing cameras' transform ids forward through this frame's swap-removes before
    // applying the extracted camera update (whose addition indices are post-swap from the host).
    fixup_cameras_for_transform_removals(space, transform_removals);
    if let Some(ref cu) = extracted.cameras {
        super::super::camera_apply::apply_camera_renderables_update_extracted(space, cu);
    }
    fixup_reflection_probes_for_transform_removals(space, transform_removals);
    if let Some(ref rpu) = extracted.reflection_probes {
        apply_reflection_probe_renderables_update_extracted(space, rpu);
    } else {
        space.pending_reflection_probe_render_changes.clear();
    }

    fixup_static_meshes_for_transform_removals(space, transform_removals);
    if let Some(ref mu) = extracted.meshes {
        apply_mesh_renderables_update_extracted(space, mu, scene_id);
    }
    if let Some(ref su) = extracted.skinned_meshes {
        apply_skinned_mesh_renderables_update_extracted(space, su, transform_removals, scene_id);
    }
    {
        profiling::scope!("scene::layers");
        super::super::layer_apply::fixup_layer_assignments_for_transform_removals(
            space,
            transform_removals,
        );
        if let Some(ref lu) = extracted.layers {
            apply_layer_update_extracted(space, lu);
        }
        super::super::layer_apply::resolve_mesh_layers_from_assignments(space);
    };
    if let Some(ref rtu) = extracted.transform_overrides {
        apply_render_transform_overrides_update_extracted(space, rtu, transform_removals);
    }
    if let Some(ref rmu) = extracted.material_overrides {
        apply_render_material_overrides_update_extracted(space, rmu, transform_removals);
    }
    world_dirty
}

/// Borrowed view of the still-serial light-update payloads for a [`RenderSpaceUpdate`].
///
/// Carried alongside the parallel-applied per-space payloads so the post-parallel light pass can
/// re-walk the host updates without re-scanning [`crate::shared::FrameSubmitData::render_spaces`].
pub struct LightUpdateView<'a> {
    /// Render space identity (mirrors [`RenderSpaceUpdate::id`]).
    pub space_id: i32,
    /// Optional [`crate::shared::LightRenderablesUpdate`] payload (regular [`crate::shared::LightState`] rows).
    pub lights_update: Option<&'a LightRenderablesUpdate>,
    /// Optional [`crate::shared::LightsBufferRendererUpdate`] payload (buffer-based lights).
    pub lights_buffer_renderers_update: Option<&'a LightsBufferRendererUpdate>,
}

/// Borrows the still-serial light update fields from a [`RenderSpaceUpdate`].
pub fn light_updates_view(update: &RenderSpaceUpdate) -> LightUpdateView<'_> {
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
                let Some(space) = self.spaces.remove(&id) else {
                    continue;
                };
                let cache = self.world_caches.remove(&id).unwrap_or_default();
                work.push((id, space, cache, extracted, Vec::new()));
            }
        }

        // Stay on the serial path for a single space; two or more independent spaces can use the
        // worker pool under the aggressive early parallelism policy.
        const MIN_SPACES_FOR_PARALLEL_APPLY: usize = 2;
        if work.len() < MIN_SPACES_FOR_PARALLEL_APPLY {
            profiling::scope!("scene::apply::serial_inner");
            for (id, mut space, mut cache, extracted, mut removal_events) in work.drain(..) {
                let dirty = apply_extracted_render_space_update(
                    &extracted,
                    PerSpaceApplyInputs {
                        space: &mut space,
                        cache: &mut cache,
                        removal_events: &mut removal_events,
                    },
                );
                if dirty {
                    self.world_dirty.insert(id);
                }
                self.spaces.insert(id, space);
                self.world_caches.insert(id, cache);
                self.stash_transform_removals(id, removal_events);
            }
            self.apply_work_scratch = work;
            return Ok(());
        }

        use rayon::iter::ParallelDrainRange;
        use rayon::prelude::*;
        // `par_drain(..)` empties `work` while keeping its allocation, mirroring the standard
        // `Vec::drain` semantics so the buffer can be reused next frame.
        let processed: Vec<(
            RenderSpaceId,
            crate::scene::render_space::RenderSpaceState,
            crate::scene::world::WorldTransformCache,
            bool,
            Vec<TransformRemovalEvent>,
        )> = {
            profiling::scope!("scene::apply::mutate");
            work.par_drain(..)
                .map(
                    |(id, mut space, mut cache, extracted, mut removal_events)| {
                        let dirty = apply_extracted_render_space_update(
                            &extracted,
                            PerSpaceApplyInputs {
                                space: &mut space,
                                cache: &mut cache,
                                removal_events: &mut removal_events,
                            },
                        );
                        (id, space, cache, dirty, removal_events)
                    },
                )
                .collect()
        };
        {
            profiling::scope!("scene::apply::reinsert");
            for (id, space, cache, dirty, removal_events) in processed {
                if dirty {
                    self.world_dirty.insert(id);
                }
                self.spaces.insert(id, space);
                self.world_caches.insert(id, cache);
                self.stash_transform_removals(id, removal_events);
            }
        }
        self.apply_work_scratch = work;
        Ok(())
    }

    /// Moves a per-space transform-removal buffer into [`SceneCoordinator::transform_removals_by_space`]
    /// so Phase C can read it. Reuses the pre-allocated entry when present so the steady-state
    /// path swaps `Vec` contents instead of reallocating.
    fn stash_transform_removals(
        &mut self,
        id: RenderSpaceId,
        mut removals: Vec<TransformRemovalEvent>,
    ) {
        let slot = self.transform_removals_by_space.entry(id).or_default();
        slot.clear();
        slot.append(&mut removals);
    }
}
