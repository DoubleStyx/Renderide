//! Forward-pass state and blackboard slots.

use crate::materials::{MaterialPipelineDesc, ShaderPermutation};
use crate::render_graph::blackboard::blackboard_slot;
use crate::skybox::PreparedSkybox;
use crate::world_mesh::{InstancePlan, WorldMeshDrawItem, WorldMeshHelperNeeds};

use super::MaterialBatchPacket;

/// Tracks whether the imported single-sample depth target matches the MSAA forward depth target.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct DepthFreshness {
    /// Whether the single-sample frame depth contains the latest MSAA forward depth contents.
    single_sample_matches_msaa: bool,
}

impl DepthFreshness {
    /// Marks the single-sample depth target as matching the current MSAA forward depth target.
    pub(crate) fn mark_resolved(&mut self) {
        self.single_sample_matches_msaa = true;
    }

    /// Marks the single-sample depth target as stale relative to the MSAA forward depth target.
    pub(crate) fn mark_dirty(&mut self) {
        self.single_sample_matches_msaa = false;
    }

    /// Returns whether the single-sample depth target already matches MSAA forward depth.
    pub(crate) fn is_current(self) -> bool {
        self.single_sample_matches_msaa
    }
}

/// Pipeline state resolved during world-mesh forward preparation.
pub(crate) struct WorldMeshForwardPipelineState {
    /// Whether this view records multiview raster passes.
    pub use_multiview: bool,
    /// Material pipeline descriptor for this view's color/depth/sample state.
    pub pass_desc: MaterialPipelineDesc,
    /// Shader permutation used by material pipeline lookup.
    pub shader_perm: ShaderPermutation,
}

/// Per-view forward-pass preparation shared by split graph nodes.
pub(crate) struct PreparedWorldMeshForwardFrame {
    /// Sorted world mesh draw items for this view.
    pub draws: Vec<WorldMeshDrawItem>,
    /// Per-view [`InstancePlan`]: per-draw slab layout plus phase-grouped draw submissions.
    pub plan: InstancePlan,
    /// Pipeline format/sample/multiview state.
    pub pipeline: WorldMeshForwardPipelineState,
    /// Scene snapshot helper work needed by the prepared draw list.
    pub helper_needs: WorldMeshHelperNeeds,
    /// Whether indexed draws may use base instance.
    pub supports_base_instance: bool,
    /// Whether the opaque/clear forward subpass was already recorded by a split graph node.
    pub opaque_recorded: bool,
    /// Whether the scene-depth snapshot was already recorded by a split graph node.
    pub depth_snapshot_recorded: bool,
    /// Whether tail raster work was already recorded by a split graph node.
    pub tail_raster_recorded: bool,
    /// Freshness state for the single-sample depth target when MSAA rendering is active.
    pub depth_freshness: DepthFreshness,
    /// Per-batch resolved pipelines and bind groups, pre-computed by backend frame planning.
    pub precomputed_batches: Vec<MaterialBatchPacket>,
    /// Optional background draw prepared for the opaque subpass.
    pub skybox: Option<PreparedSkybox>,
    /// Overlay view-projection used to project per-draw `_Rect` corners into screen space for
    /// the GPU scissor optimisation in [`super::encode::draw_subset`].
    ///
    /// Identity view is folded in: overlay draws use identity view in
    /// [`super::vp::compute_per_draw_vp_matrices`] (the model already encodes screen-space-relative
    /// position via [`crate::scene::SceneCoordinator::overlay_layer_model_matrix_for_context`]),
    /// so this is just the active overlay projection.
    pub overlay_view_proj: glam::Mat4,
    /// Main surface extent in pixels for the GPU scissor optimisation.
    pub viewport_px: (u32, u32),
}

blackboard_slot! {
    /// Blackboard slot key for the per-view world-mesh forward plan.
    pub(crate) WorldMeshForwardPlanSlot => PreparedWorldMeshForwardFrame,
}

#[cfg(test)]
mod tests {
    use super::DepthFreshness;

    #[test]
    fn depth_freshness_tracks_resolve_and_tail_writes() {
        let mut freshness = DepthFreshness::default();

        assert!(!freshness.is_current());
        freshness.mark_resolved();
        assert!(freshness.is_current());
        freshness.mark_dirty();
        assert!(!freshness.is_current());
    }
}
