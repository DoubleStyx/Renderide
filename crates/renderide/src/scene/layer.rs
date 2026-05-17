//! Host layer assignment ingestion and inherited mesh layer resolution.

use hashbrown::HashMap;

use crate::ipc::SharedMemoryAccessor;
use crate::shared::{LayerType, LayerUpdate};

use super::error::SceneError;
use super::render_space::{LayerAssignmentEntry, RenderSpaceState};
use super::transforms::TransformRemovalEvent;
use super::world::fixup_transform_id;

/// Owned per-space layer-update payload extracted from shared memory.
///
/// Produced by [`extract_layer_update`] in the serial pre-extract phase so the per-space apply
/// step can run on a rayon worker without holding a mutable [`SharedMemoryAccessor`] borrow.
#[derive(Default, Debug)]
pub struct ExtractedLayerUpdate {
    /// Dense layer-assignment removal indices (terminated by `< 0`).
    pub removals: Vec<i32>,
    /// New layer-assignment node ids (terminated by `< 0`).
    pub additions: Vec<i32>,
    /// Per-entry [`LayerType`] rows for assignments added by the same update.
    pub layer_assignments: Vec<LayerType>,
}

/// Reads every shared-memory buffer referenced by [`LayerUpdate`] into owned vectors.
pub(crate) fn extract_layer_update(
    shm: &mut SharedMemoryAccessor,
    update: &LayerUpdate,
    scene_id: i32,
) -> Result<ExtractedLayerUpdate, SceneError> {
    let mut out = ExtractedLayerUpdate::default();
    if update.removals.length > 0 {
        let ctx = format!("layer removals scene_id={scene_id}");
        out.removals = shm
            .access_copy_diagnostic_with_context::<i32>(&update.removals, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?;
    }
    if update.additions.length > 0 {
        let ctx = format!("layer additions scene_id={scene_id}");
        out.additions = shm
            .access_copy_diagnostic_with_context::<i32>(&update.additions, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?;
    }
    if update.layer_assignments.length > 0 {
        let ctx = format!("layer assignments scene_id={scene_id}");
        out.layer_assignments = shm
            .access_copy_memory_packable_rows::<LayerType>(
                &update.layer_assignments,
                size_of::<LayerType>(),
                Some(&ctx),
            )
            .map_err(SceneError::SharedMemoryAccess)?;
    }
    Ok(out)
}

/// Mutates [`RenderSpaceState::layer_assignments`] using a pre-extracted [`ExtractedLayerUpdate`].
pub(crate) fn apply_layer_update_extracted(
    space: &mut RenderSpaceState,
    extracted: &ExtractedLayerUpdate,
) {
    profiling::scope!("scene::apply_layers");
    let mut mutated = false;
    for &raw in extracted.removals.iter().take_while(|&&idx| idx >= 0) {
        let idx = raw as usize;
        if idx < space.layer_assignments.len() {
            space.layer_assignments.swap_remove(idx);
            mutated = true;
        }
    }
    let additions_start = space.layer_assignments.len();
    for &node_id in extracted.additions.iter().take_while(|&&id| id >= 0) {
        space.layer_assignments.push(LayerAssignmentEntry {
            node_id,
            layer: LayerType::Hidden,
        });
        mutated = true;
    }
    for (idx, layer) in extracted.layer_assignments.iter().copied().enumerate() {
        let Some(entry) = space.layer_assignments.get_mut(additions_start + idx) else {
            continue;
        };
        entry.layer = layer;
        mutated = true;
    }
    if mutated {
        space.layer_index_dirty = true;
    }
}

/// Combined renderer count above which the per-renderer layer resolve fans out to the rayon pool.
///
/// Each call to [`resolve_layer_for_node`] walks the parent chain and scans `layer_assignments`,
/// so per-renderer cost scales with scene depth x assignment count. Above this threshold the
/// parallel dispatch pays for itself; below it the serial path avoids rayon overhead.
const LAYER_RESOLVE_PARALLEL_MIN: usize = 128;

pub(crate) fn resolve_mesh_layers_from_assignments(space: &mut RenderSpaceState) {
    profiling::scope!("scene::apply::layer_resolve");

    // Rebuild `space.layer_index` only when an apply path mutated `layer_assignments` since the
    // last resolve. The forward `insert` matches the prior `iter().rev().find()` "latest
    // assignment wins" semantics: when the host re-emits an entry for an existing node_id, the
    // later push overwrites. The index is reused across frames otherwise.
    let layer_index_changed = space.layer_index_dirty;
    if space.layer_index_dirty {
        profiling::scope!("scene::apply::layer_resolve::build_index");
        space.layer_index.clear();
        space.layer_index.reserve(space.layer_assignments.len());
        for entry in &space.layer_assignments {
            if entry.node_id >= 0 {
                space.layer_index.insert(entry.node_id, entry.layer);
            }
        }
        space.layer_index_dirty = false;
    }

    // Cross-frame resolution cache is valid as long as neither the layer assignments nor the
    // node-parent hierarchy have mutated since it was last populated. Either signal forces a
    // full repopulate.
    if layer_index_changed || space.hierarchy_dirty {
        profiling::scope!("scene::apply::layer_resolve::invalidate_cache");
        space.resolved_layer_cache.clear();
        space.hierarchy_dirty = false;
    }

    // Phase 1 (serial): ensure every renderer's `node_id` has a resolution recorded in the
    // cross-frame cache, including `None` for nodes whose ancestry carries no layer assignment.
    // Steady-state hits are a single hashbrown probe per unique renderer node id; misses do the
    // same parent walk the previous code did, but only once per node id per cache lifetime.
    {
        profiling::scope!("scene::apply::layer_resolve::ensure_cache");
        let RenderSpaceState {
            node_parents,
            layer_index,
            resolved_layer_cache,
            static_mesh_renderers,
            skinned_mesh_renderers,
            layer_resolve_seen_scratch,
            ..
        } = space;
        layer_resolve_seen_scratch.clear();
        layer_resolve_seen_scratch
            .reserve(static_mesh_renderers.len() + skinned_mesh_renderers.len());
        for r in static_mesh_renderers.iter() {
            ensure_resolved_cache_entry(
                node_parents,
                layer_index,
                resolved_layer_cache,
                layer_resolve_seen_scratch,
                r.node_id,
            );
        }
        for r in skinned_mesh_renderers.iter() {
            ensure_resolved_cache_entry(
                node_parents,
                layer_index,
                resolved_layer_cache,
                layer_resolve_seen_scratch,
                r.base.node_id,
            );
        }
    }

    let total = space.static_mesh_renderers.len() + space.skinned_mesh_renderers.len();
    let resolved_cache = &space.resolved_layer_cache;

    // Phase 2 (parallel above the threshold): walk renderers and consume the cache. Reads are
    // safe to share across rayon workers because `hashbrown::HashMap` is `Sync` for `&` access.
    if total >= LAYER_RESOLVE_PARALLEL_MIN {
        profiling::scope!("scene::apply::layer_resolve::apply_cached_parallel");
        use rayon::prelude::*;
        space.static_mesh_renderers.par_iter_mut().for_each(|r| {
            apply_cached_layer(resolved_cache, &mut r.layer, r.node_id);
        });
        space.skinned_mesh_renderers.par_iter_mut().for_each(|r| {
            apply_cached_layer(resolved_cache, &mut r.base.layer, r.base.node_id);
        });
    } else {
        profiling::scope!("scene::apply::layer_resolve::apply_cached_serial");
        for renderer in &mut space.static_mesh_renderers {
            apply_cached_layer(resolved_cache, &mut renderer.layer, renderer.node_id);
        }
        for renderer in &mut space.skinned_mesh_renderers {
            apply_cached_layer(
                resolved_cache,
                &mut renderer.base.layer,
                renderer.base.node_id,
            );
        }
    }
}

/// Populates [`RenderSpaceState::resolved_layer_cache`] for `node_id` if it is not already
/// recorded. Records `Some(layer)` on a successful parent-chain walk and `None` otherwise so the
/// fallback path also avoids the walk on later frames.
fn ensure_resolved_cache_entry(
    node_parents: &[i32],
    layer_for_node: &HashMap<i32, LayerType>,
    cache: &mut HashMap<i32, Option<LayerType>>,
    seen: &mut hashbrown::HashSet<i32>,
    node_id: i32,
) {
    if node_id < 0 {
        return;
    }
    if !seen.insert(node_id) {
        return;
    }
    if cache.contains_key(&node_id) {
        return;
    }
    let resolved = resolve_layer_for_node(node_parents, layer_for_node, node_id);
    cache.insert(node_id, resolved);
}

/// Reads the cached resolution for `node_id` and applies it to `out_layer`. Falls back to the
/// default (and records the node id for one-shot warning) when no entry is present or the cached
/// resolution is `None`.
#[inline]
fn apply_cached_layer(
    cache: &HashMap<i32, Option<LayerType>>,
    out_layer: &mut LayerType,
    node_id: i32,
) {
    match cache.get(&node_id).copied() {
        Some(Some(layer)) => *out_layer = layer,
        Some(None) | None => {
            *out_layer = LayerType::default();
        }
    }
}

fn resolve_layer_for_node(
    node_parents: &[i32],
    layer_for_node: &HashMap<i32, LayerType>,
    node_id: i32,
) -> Option<LayerType> {
    if node_id < 0 {
        return None;
    }

    let mut cursor = node_id;
    for _ in 0..node_parents.len() {
        if let Some(layer) = layer_for_node.get(&cursor) {
            return Some(*layer);
        }
        let parent = *node_parents.get(cursor as usize)?;
        if parent < 0 || parent == cursor || parent as usize >= node_parents.len() {
            return None;
        }
        cursor = parent;
    }
    None
}

/// Layer-assignment count above which the per-removal fixup sweep fans out to the rayon pool.
///
/// Each entry's `fixup_transform_id` is a trivial branch, but removals x assignments can grow
/// into the tens of thousands during bulky scene teardowns.
const LAYER_FIXUP_PARALLEL_MIN: usize = 128;

pub(crate) fn fixup_layer_assignments_for_transform_removals(
    space: &mut RenderSpaceState,
    removals: &[TransformRemovalEvent],
) {
    if removals.is_empty() {
        return;
    }
    for removal in removals {
        if space.layer_assignments.len() >= LAYER_FIXUP_PARALLEL_MIN {
            use rayon::prelude::*;
            space.layer_assignments.par_iter_mut().for_each(|entry| {
                entry.node_id = fixup_transform_id(
                    entry.node_id,
                    removal.removed_index,
                    removal.last_index_before_swap,
                );
            });
        } else {
            for entry in &mut space.layer_assignments {
                entry.node_id = fixup_transform_id(
                    entry.node_id,
                    removal.removed_index,
                    removal.last_index_before_swap,
                );
            }
        }
        space.layer_assignments.retain(|entry| entry.node_id >= 0);
    }
    space.layer_index_dirty = true;
}

#[cfg(test)]
mod tests {
    use glam::{Quat, Vec3};

    use crate::scene::StaticMeshRenderer;
    use crate::scene::render_space::{LayerAssignmentEntry, RenderSpaceState};
    use crate::shared::{LayerType, RenderTransform};

    use super::{
        ExtractedLayerUpdate, apply_layer_update_extracted, resolve_mesh_layers_from_assignments,
    };

    /// Builds an identity transform without relying on the wire default's zero scale.
    fn identity_transform() -> RenderTransform {
        RenderTransform {
            position: Vec3::ZERO,
            scale: Vec3::ONE,
            rotation: Quat::IDENTITY,
        }
    }

    #[test]
    fn resolves_layer_from_nearest_ancestor_assignment() {
        let mut space = RenderSpaceState {
            node_parents: vec![-1, 0, 1],
            layer_assignments: vec![
                LayerAssignmentEntry {
                    node_id: 0,
                    layer: LayerType::Overlay,
                },
                LayerAssignmentEntry {
                    node_id: 1,
                    layer: LayerType::Hidden,
                },
            ],
            static_mesh_renderers: vec![StaticMeshRenderer {
                node_id: 2,
                layer: LayerType::Overlay,
                ..Default::default()
            }],
            ..Default::default()
        };

        resolve_mesh_layers_from_assignments(&mut space);

        assert_eq!(space.static_mesh_renderers[0].layer, LayerType::Hidden);
    }

    #[test]
    fn inherited_overlay_applies_to_static_and_skinned_children() {
        let mut space = RenderSpaceState {
            node_parents: vec![-1, 0, 1],
            layer_assignments: vec![LayerAssignmentEntry {
                node_id: 1,
                layer: LayerType::Overlay,
            }],
            static_mesh_renderers: vec![StaticMeshRenderer {
                node_id: 2,
                ..Default::default()
            }],
            skinned_mesh_renderers: vec![crate::scene::SkinnedMeshRenderer {
                base: StaticMeshRenderer {
                    node_id: 2,
                    ..Default::default()
                },
                ..Default::default()
            }],
            ..Default::default()
        };

        resolve_mesh_layers_from_assignments(&mut space);

        assert_eq!(space.static_mesh_renderers[0].layer, LayerType::Overlay);
        assert_eq!(
            space.skinned_mesh_renderers[0].base.layer,
            LayerType::Overlay
        );
    }

    /// Distinguishing "renderer is under a Hidden ancestor" from "renderer has no layer
    /// ancestor at all" via `transform_special_layer`. The renderer's `layer` field defaults to
    /// `LayerType::Hidden` (because `LayerType::default() == Hidden`), so it cannot itself be
    /// used to tell the two cases apart. The per-renderer cull in
    /// `world_mesh::draw_prep::collect::scene_walk::per_renderer` must therefore route through
    /// `SceneCoordinator::transform_special_layer`, which returns `Option<LayerType>` --
    /// `Some(Hidden)` means actually under Hidden, `None` means no layer ancestor at all (the
    /// regular world mesh path that must keep rendering).
    #[test]
    fn special_layer_distinguishes_hidden_subtree_from_world_meshes() {
        use crate::scene::SceneCoordinator;
        use crate::scene::ids::RenderSpaceId;

        let id = RenderSpaceId(42);
        let mut scene = SceneCoordinator::new();
        scene.test_seed_space_identity_worlds(
            id,
            vec![
                // 0: root (no layer)
                identity_transform(),
                // 1: Render slot (Hidden layer assignment)
                identity_transform(),
                // 2: inner UI canvas (descendant of Render -> resolves to Hidden)
                identity_transform(),
                // 3: regular world mesh slot (no layer ancestor)
                identity_transform(),
            ],
            vec![-1, 0, 1, 0],
        );
        // Inject the Hidden assignment on node 1 (the "Render" equivalent).
        scene.test_push_layer_assignment(id, 1, LayerType::Hidden);

        assert_eq!(
            scene.transform_special_layer(id, 2),
            Some(LayerType::Hidden),
            "node 2 lives under the Hidden ancestor -> must report Hidden so the main view culls it",
        );
        assert_eq!(
            scene.transform_special_layer(id, 3),
            None,
            "node 3 is a regular world mesh with no layer ancestor -> must report None so the \
             main view keeps rendering it",
        );
        // Sanity: the Hidden anchor itself reports Hidden.
        assert_eq!(
            scene.transform_special_layer(id, 1),
            Some(LayerType::Hidden)
        );
    }

    /// Builds a scene mirroring the FrooxEngine `UserspaceRadiantDash` + `OverlayManager` slot
    /// layout in desktop mode and verifies the resolved `renderer.layer` for the dash's inner UI
    /// (which lives under `LayerType::Hidden` -> only the dash camera sees it) and for the
    /// reparented curved planes (under `LayerType::Overlay` -> drawn screen-anchored).
    ///
    /// Regression guard for the "ghost dash" double-render: if this test ever fails, the main
    /// view's Hidden-layer cull in `world_mesh::draw_prep::collect::scene_walk::per_renderer`
    /// will see the inner UI as `LayerType::Default` and render it at world position alongside
    /// the dash camera's RT.
    #[test]
    fn dash_layout_resolves_inner_ui_to_hidden_and_curved_planes_to_overlay() {
        // Node layout (parent indices in `node_parents`):
        //   0  Userspace root
        //   1  UserspaceRadiantDash.Slot   (parent 0)
        //   2  OverlayManager.Slot         (parent 0)
        //   3  OverlayManager.OverlayRoot  (parent 2, LayerType::Overlay)
        //   4  Dash.Slot (under UserspaceRadiantDash)            (parent 1)
        //   5  Dash.Render (HiddenLayer)   (parent 4, LayerType::Hidden)
        //   6  Dash inner UI canvas        (parent 5) -- renderer R1 hangs off this
        //   7  Dash.VisualsRoot (reparented under OverlayRoot in desktop) (parent 3)
        //   8  Curved plane Screen         (parent 7) -- renderer R2 hangs off this
        let mut space = RenderSpaceState {
            node_parents: vec![-1, 0, 0, 2, 1, 4, 5, 3, 7],
            layer_assignments: vec![
                LayerAssignmentEntry {
                    node_id: 3,
                    layer: LayerType::Overlay,
                },
                LayerAssignmentEntry {
                    node_id: 5,
                    layer: LayerType::Hidden,
                },
            ],
            static_mesh_renderers: vec![
                StaticMeshRenderer {
                    node_id: 6, // inner UI canvas, descendant of Render (Hidden)
                    layer: LayerType::default(),
                    ..Default::default()
                },
                StaticMeshRenderer {
                    node_id: 8, // curved plane, descendant of OverlayRoot (Overlay)
                    layer: LayerType::default(),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        space.layer_index_dirty = true;
        space.hierarchy_dirty = true;

        resolve_mesh_layers_from_assignments(&mut space);

        assert_eq!(
            space.static_mesh_renderers[0].layer,
            LayerType::Hidden,
            "inner UI under Dash.Render must resolve to Hidden so the main view's per-renderer \
             filter culls it (otherwise the dash chrome ghosts at world position)",
        );
        assert_eq!(
            space.static_mesh_renderers[1].layer,
            LayerType::Overlay,
            "curved plane under OverlayRoot must resolve to Overlay so it renders through the \
             screen-anchored overlay-ortho path, not the world camera",
        );
    }

    #[test]
    fn layer_rows_only_apply_to_new_assignments() {
        let mut space = RenderSpaceState {
            layer_assignments: vec![LayerAssignmentEntry {
                node_id: 2,
                layer: LayerType::Overlay,
            }],
            ..Default::default()
        };
        let extracted = ExtractedLayerUpdate {
            additions: vec![11],
            layer_assignments: vec![LayerType::Hidden],
            ..Default::default()
        };

        apply_layer_update_extracted(&mut space, &extracted);

        assert_eq!(space.layer_assignments.len(), 2);
        assert_eq!(space.layer_assignments[0].node_id, 2);
        assert_eq!(space.layer_assignments[0].layer, LayerType::Overlay);
        assert_eq!(space.layer_assignments[1].node_id, 11);
        assert_eq!(space.layer_assignments[1].layer, LayerType::Hidden);
    }
}
