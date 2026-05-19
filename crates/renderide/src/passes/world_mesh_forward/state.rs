//! Forward-pass state and blackboard slots.

use crate::materials::{MaterialPipelineDesc, ShaderPermutation};
use crate::render_graph::blackboard::BlackboardSlot;
use crate::skybox::PreparedSkybox;
use crate::world_mesh::{InstancePlan, WorldMeshDrawItem, WorldMeshDrawPlan, WorldMeshHelperNeeds};

use super::DepthPrepassPhase;
use super::MaterialBatchPacket;

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
    /// Per-view [`InstancePlan`]: per-draw slab layout plus regular and intersection draw groups.
    pub plan: InstancePlan,
    /// Prepared safe-opaque depth prepass replay phase.
    pub depth_prepass_phase: DepthPrepassPhase,
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

/// Blackboard slot key for the per-view world-mesh forward plan.
pub(crate) struct WorldMeshForwardPlanSlot;
impl BlackboardSlot for WorldMeshForwardPlanSlot {
    type Value = PreparedWorldMeshForwardFrame;
}

/// Blackboard slot carrying prefetched shadow-caster draws for the view's shadow-map pass.
pub(crate) struct ShadowCasterDrawPlanSlot;
impl BlackboardSlot for ShadowCasterDrawPlanSlot {
    type Value = WorldMeshDrawPlan;
}
