//! Per-view world-mesh forward-pass preparation: prefetched draws, helper requirements,
//! pipeline state, and the blackboard slots that move them between graph passes.

use crate::materials::{MaterialPipelineDesc, ShaderPermutation};
use crate::render_graph::MaterialBatchPacket;
use crate::render_graph::blackboard::BlackboardSlot;
use crate::skybox::PreparedSkybox;
use crate::world_mesh::cull::WorldMeshCullProjParams;
use crate::world_mesh::draw_prep::{InstancePlan, WorldMeshDrawCollection, WorldMeshDrawItem};

/// Pipeline state resolved during world-mesh forward preparation.
pub struct WorldMeshForwardPipelineState {
    /// Whether this view records multiview raster passes.
    pub use_multiview: bool,
    /// Material pipeline descriptor for this view's color/depth/sample state.
    pub pass_desc: MaterialPipelineDesc,
    /// Shader permutation used by material pipeline lookup.
    pub shader_perm: ShaderPermutation,
}

/// Snapshot-dependent helper work required by a prefetched world-mesh view.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct WorldMeshHelperNeeds {
    /// Whether any draw in the view samples the scene-depth snapshot for the intersection subpass.
    pub depth_snapshot: bool,
    /// Whether any draw in the view samples the scene-color snapshot for the grab-pass subpass.
    pub color_snapshot: bool,
}

impl WorldMeshHelperNeeds {
    /// Derives helper-pass requirements from the material flags on a collected draw list.
    pub fn from_collection(collection: &WorldMeshDrawCollection) -> Self {
        let mut needs = Self::default();
        for item in &collection.items {
            needs.depth_snapshot |= item.batch_key.embedded_uses_scene_depth_snapshot;
            needs.color_snapshot |= item.batch_key.embedded_uses_scene_color_snapshot;
            if needs.depth_snapshot && needs.color_snapshot {
                break;
            }
        }
        needs
    }
}

/// Per-view prefetched world-mesh data seeded before graph execution.
#[derive(Clone, Debug)]
pub struct PrefetchedWorldMeshViewDraws {
    /// Draw items and culling statistics collected for the view.
    pub collection: WorldMeshDrawCollection,
    /// Projection state used during culling, reused when capturing Hi-Z temporal feedback.
    pub cull_proj: Option<WorldMeshCullProjParams>,
    /// Helper snapshots and tail passes required by this view's collected materials.
    pub helper_needs: WorldMeshHelperNeeds,
}

impl PrefetchedWorldMeshViewDraws {
    /// Builds a prefetched view packet and derives helper-pass requirements from `collection`.
    pub fn new(
        collection: WorldMeshDrawCollection,
        cull_proj: Option<WorldMeshCullProjParams>,
    ) -> Self {
        let helper_needs = WorldMeshHelperNeeds::from_collection(&collection);
        Self {
            collection,
            cull_proj,
            helper_needs,
        }
    }

    /// Builds an explicit empty draw packet for views that should skip world-mesh work.
    pub fn empty() -> Self {
        Self::new(WorldMeshDrawCollection::empty(), None)
    }
}

/// Per-view forward-pass preparation shared by future split graph nodes.
pub struct PreparedWorldMeshForwardFrame {
    /// Sorted world mesh draw items for this view.
    pub draws: Vec<WorldMeshDrawItem>,
    /// Per-view [`InstancePlan`]: per-draw slab layout plus regular and intersection
    /// [`crate::render_graph::DrawGroup`]s that the forward pass turns into one `draw_indexed` each.
    ///
    /// Replaces the older `regular_indices` / `intersect_indices: Vec<usize>` pair.
    /// Decouples the per-draw slab layout from the sorted-draw order so that same-mesh
    /// instances merge regardless of where the sort placed individual members.
    pub plan: InstancePlan,
    /// Pipeline format/sample/multiview state.
    pub pipeline: WorldMeshForwardPipelineState,
    /// Scene snapshot helper work needed by the prepared draw list.
    pub helper_needs: WorldMeshHelperNeeds,
    /// Whether indexed draws may use base instance.
    pub supports_base_instance: bool,
    /// Whether the opaque/clear forward subpass was already recorded by a split graph node.
    pub opaque_recorded: bool,
    /// Whether the scene-depth snapshot for intersection draws was already recorded by a split graph node.
    pub depth_snapshot_recorded: bool,
    /// Whether the intersection/color-resolve tail raster was already recorded by a split graph node.
    pub tail_raster_recorded: bool,
    /// Per-batch resolved pipelines and bind groups, pre-computed by the prepare pass in parallel.
    ///
    /// One entry per unique `MaterialDrawBatchKey` run in `draws`, covering `[first_draw_idx,
    /// last_draw_idx]` (inclusive). Both raster sub-passes (opaque and intersect) share this
    /// list; each sub-pass only reads entries whose draw-index range overlaps its own index slice.
    pub precomputed_batches: Vec<MaterialBatchPacket>,
    /// Optional background draw prepared for the opaque subpass.
    pub skybox: Option<PreparedSkybox>,
}

/// Blackboard slot key for the per-view world-mesh forward plan.
///
/// Populated by [`crate::render_graph::passes::WorldMeshForwardPreparePass`] and consumed by the
/// four downstream forward passes (opaque, depth snapshot, intersect, depth resolve).
pub struct WorldMeshForwardPlanSlot;
impl BlackboardSlot for WorldMeshForwardPlanSlot {
    type Value = PreparedWorldMeshForwardFrame;
}

/// Blackboard slot key for pre-collected world-mesh draws (secondary cameras / prefetch path).
///
/// When set before the graph executes, [`crate::render_graph::passes::WorldMeshForwardPreparePass`]
/// skips draw collection and uses this list instead.
pub struct PrefetchedWorldMeshDrawsSlot;
impl BlackboardSlot for PrefetchedWorldMeshDrawsSlot {
    type Value = PrefetchedWorldMeshViewDraws;
}
