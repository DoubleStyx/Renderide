//! Host layer assignment ingestion and inherited mesh layer resolution.

use std::collections::HashSet;
use std::sync::LazyLock;

use hashbrown::HashMap;
use parking_lot::Mutex;

use crate::ipc::SharedMemoryAccessor;
use crate::shared::{LayerType, LayerUpdate};

use super::error::SceneError;
use super::render_space::{LayerAssignmentEntry, RenderSpaceState};
use super::transforms_apply::TransformRemovalEvent;
use super::world::fixup_transform_id;

/// One-shot dedup for [`resolve_mesh_layers_from_assignments`] fallback warnings, keyed by node id.
///
/// When a renderable's node has no [`LayerAssignmentEntry`] up its parent chain,
/// [`resolve_layer_for_node`] returns `None` and the renderable falls through to
/// [`LayerType::default`] (= [`LayerType::Hidden`]). The host re-emits soon enough that this
/// self-corrects, but the fallback is a possible co-symptom of the broader instance-changed
/// host-renderer drift, so log once per node id to make it diagnosable without spamming.
static LAYER_FALLBACK_WARNED_NODES: LazyLock<Mutex<HashSet<i32>>> =
    LazyLock::new(|| Mutex::new(HashSet::new()));

fn record_layer_fallback(node_id: i32) {
    if node_id < 0 {
        return;
    }
    let mut w = LAYER_FALLBACK_WARNED_NODES.lock();
    if w.insert(node_id) {
        logger::trace!(
            "layer resolve: no LayerAssignmentEntry for node_id={node_id} or any ancestor; falling back to Hidden. Subsequent occurrences for this node are suppressed."
        );
    }
}

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
    /// Per-entry [`LayerType`] rows, indexed positionally into [`RenderSpaceState::layer_assignments`].
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
                std::mem::size_of::<LayerType>(),
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
    for &raw in extracted.removals.iter().take_while(|&&idx| idx >= 0) {
        let idx = raw as usize;
        if idx < space.layer_assignments.len() {
            space.layer_assignments.swap_remove(idx);
        }
    }
    for &node_id in extracted.additions.iter().take_while(|&&id| id >= 0) {
        space.layer_assignments.push(LayerAssignmentEntry {
            node_id,
            layer: LayerType::Hidden,
        });
    }
    for (idx, layer) in extracted.layer_assignments.iter().copied().enumerate() {
        let Some(entry) = space.layer_assignments.get_mut(idx) else {
            continue;
        };
        entry.layer = layer;
    }
}

/// Combined renderer count above which the per-renderer layer resolve fans out to the rayon pool.
///
/// Each call to [`resolve_layer_for_node`] walks the parent chain and scans `layer_assignments`,
/// so per-renderer cost scales with scene depth × assignment count. Above this threshold the
/// parallel dispatch pays for itself; below it the serial path avoids rayon overhead.
const LAYER_RESOLVE_PARALLEL_MIN: usize = 256;

pub(crate) fn resolve_mesh_layers_from_assignments(space: &mut RenderSpaceState) {
    profiling::scope!("scene::apply::layer_resolve");
    let node_parents = &space.node_parents;

    // Build a node_id → layer index for this resolve. The previous code did
    // `layer_assignments.iter().rev().find()` per ancestor walk, which made each renderable
    // O(scene_depth × assignment_count). Constructing the index up front collapses the per-
    // renderable cost to O(scene_depth) regardless of assignment count. The forward `insert`
    // matches the prior `iter().rev().find()` "latest assignment wins" semantics: when the
    // host re-emits a LayerAssignmentEntry for an existing node_id, the later push overwrites.
    let layer_for_node: HashMap<i32, LayerType> = {
        profiling::scope!("scene::apply::layer_resolve::build_index");
        let mut map = HashMap::with_capacity(space.layer_assignments.len());
        for entry in &space.layer_assignments {
            if entry.node_id >= 0 {
                map.insert(entry.node_id, entry.layer);
            }
        }
        map
    };

    let total = space.static_mesh_renderers.len() + space.skinned_mesh_renderers.len();

    // Collect node ids whose layer resolution falls through to the default; logged after the
    // borrow on `space` ends. The parallel branch funnels its missing-node ids through a
    // mutex-guarded vec; the cost is negligible vs. the per-renderable parent walk above it.
    let fallback_log: Mutex<Vec<i32>> = Mutex::new(Vec::new());

    if total >= LAYER_RESOLVE_PARALLEL_MIN {
        use rayon::prelude::*;
        space.static_mesh_renderers.par_iter_mut().for_each(|r| {
            match resolve_layer_for_node(node_parents, &layer_for_node, r.node_id) {
                Some(layer) => r.layer = layer,
                None => {
                    r.layer = LayerType::default();
                    fallback_log.lock().push(r.node_id);
                }
            }
        });
        space.skinned_mesh_renderers.par_iter_mut().for_each(|r| {
            match resolve_layer_for_node(node_parents, &layer_for_node, r.base.node_id) {
                Some(layer) => r.base.layer = layer,
                None => {
                    r.base.layer = LayerType::default();
                    fallback_log.lock().push(r.base.node_id);
                }
            }
        });
    } else {
        for renderer in &mut space.static_mesh_renderers {
            match resolve_layer_for_node(node_parents, &layer_for_node, renderer.node_id) {
                Some(layer) => renderer.layer = layer,
                None => {
                    renderer.layer = LayerType::default();
                    fallback_log.lock().push(renderer.node_id);
                }
            }
        }
        for renderer in &mut space.skinned_mesh_renderers {
            match resolve_layer_for_node(node_parents, &layer_for_node, renderer.base.node_id) {
                Some(layer) => renderer.base.layer = layer,
                None => {
                    renderer.base.layer = LayerType::default();
                    fallback_log.lock().push(renderer.base.node_id);
                }
            }
        }
    }

    for node_id in fallback_log.into_inner() {
        record_layer_fallback(node_id);
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
/// Each entry's `fixup_transform_id` is a trivial branch, but removals × assignments can grow
/// into the tens of thousands during bulky scene teardowns.
const LAYER_FIXUP_PARALLEL_MIN: usize = 128;

pub(crate) fn fixup_layer_assignments_for_transform_removals(
    space: &mut RenderSpaceState,
    removals: &[TransformRemovalEvent],
) {
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
}

#[cfg(test)]
mod tests {
    use crate::scene::{LayerAssignmentEntry, RenderSpaceState, StaticMeshRenderer};
    use crate::shared::LayerType;

    use super::resolve_mesh_layers_from_assignments;

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
}
