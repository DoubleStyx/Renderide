//! Backend-owned world-mesh forward frame planning.

use std::sync::Arc;

use crate::diagnostics::PerViewHudOutputs;
use crate::gpu::GpuLimits;
use crate::graph_inputs::{GraphPassFrame, PerViewFramePlan};
use crate::passes::{
    PreparedWorldMeshForwardFrame, WorldMeshForwardInstancePlanCache,
    WorldMeshForwardInstancePlanCacheStats, WorldMeshForwardPrepareContext,
    WorldMeshForwardSkyboxRenderer, prepare_world_mesh_forward_frame,
};
use crate::render_graph::blackboard::{
    Blackboard, GraphCommandStats, GraphCommandStatsSlot, blackboard_slot,
};
use crate::render_graph::frame_upload_batch::GraphUploadSink;
use crate::world_mesh::{PrefetchedWorldMeshViewDraws, WorldMeshDrawPlan};

blackboard_slot! {
    /// Blackboard slot carrying the world-mesh draw plan into backend graph preparation.
    pub WorldMeshDrawPlanSlot => WorldMeshDrawPlan,
}

/// Backend-owned world-mesh forward preparation caches.
pub(crate) struct BackendWorldMeshFramePlanner {
    /// Skybox/background preparation cache shared across frame plans.
    skybox: WorldMeshForwardSkyboxRenderer,
    /// Retained forward instance plans keyed by draw and resolved material submission identity.
    instance_plan_cache: WorldMeshForwardInstancePlanCache,
}

/// Per-view world-mesh packet prepared before graph pass recording.
pub(crate) struct WorldMeshPreparedView {
    /// Forward draw state consumed by graph raster and helper passes.
    pub(crate) prepared: Option<PreparedWorldMeshForwardFrame>,
    /// Optional HUD output produced while building this view's draw packet.
    pub(crate) hud_outputs: Option<PerViewHudOutputs>,
}

impl BackendWorldMeshFramePlanner {
    /// Creates an empty world-mesh frame planner.
    pub(crate) fn new() -> Self {
        Self {
            skybox: WorldMeshForwardSkyboxRenderer::default(),
            instance_plan_cache: WorldMeshForwardInstancePlanCache::default(),
        }
    }

    /// Releases view-scoped cached planning resources.
    pub(crate) fn release_view_resources(&self, retired_views: &[crate::camera::ViewId]) {
        self.skybox.release_view_resources(retired_views);
    }

    /// Retained forward instance-plan cache counters for diagnostics.
    pub(crate) fn instance_plan_cache_stats(&self) -> WorldMeshForwardInstancePlanCacheStats {
        self.instance_plan_cache.stats()
    }

    /// Builds one per-view world-mesh packet from an explicit draw plan.
    pub(crate) fn prepare_view(
        &self,
        device: &wgpu::Device,
        uploads: GraphUploadSink<'_>,
        gpu_limits: &GpuLimits,
        frame: &GraphPassFrame<'_>,
        frame_plan: &PerViewFramePlan,
        draw_plan: WorldMeshDrawPlan,
    ) -> WorldMeshPreparedView {
        let frame_plan = PerViewFramePlan {
            frame_bind_group: Arc::clone(&frame_plan.frame_bind_group),
            frame_uniform_buffer: frame_plan.frame_uniform_buffer.clone(),
            view_idx: frame_plan.view_idx,
        };
        let prefetched = match draw_plan {
            WorldMeshDrawPlan::Prefetched(draws) => *draws,
            WorldMeshDrawPlan::Empty => PrefetchedWorldMeshViewDraws::empty(),
        };
        let prepared = prepare_world_mesh_forward_frame(
            WorldMeshForwardPrepareContext {
                device,
                uploads,
                gpu_limits,
                frame,
                frame_plan: &frame_plan,
                skybox_renderer: &self.skybox,
                instance_plan_cache: &self.instance_plan_cache,
            },
            prefetched,
        );
        WorldMeshPreparedView {
            prepared: prepared.prepared,
            hud_outputs: prepared.hud_outputs,
        }
    }
}

impl Default for BackendWorldMeshFramePlanner {
    fn default() -> Self {
        Self::new()
    }
}

/// Consumes a world-mesh draw-plan slot and inserts the forward plan slots used by graph passes.
pub(crate) fn prepare_world_mesh_view_blackboard(
    planner: &BackendWorldMeshFramePlanner,
    device: &wgpu::Device,
    uploads: GraphUploadSink<'_>,
    gpu_limits: &GpuLimits,
    frame: &GraphPassFrame<'_>,
    frame_plan: &PerViewFramePlan,
    blackboard: &mut Blackboard,
) {
    let draw_plan = take_world_mesh_draw_plan(blackboard);
    let prepared = planner.prepare_view(device, uploads, gpu_limits, frame, frame_plan, draw_plan);
    if let Some(forward) = prepared.prepared {
        blackboard.insert::<GraphCommandStatsSlot>(command_stats_from_prepared(&forward));
        blackboard.insert::<crate::passes::WorldMeshForwardPlanSlot>(forward);
    }
    if let Some(hud_outputs) = prepared.hud_outputs {
        blackboard.insert::<crate::diagnostics::PerViewHudOutputsSlot>(hud_outputs);
    }
}

fn take_world_mesh_draw_plan(blackboard: &mut Blackboard) -> WorldMeshDrawPlan {
    blackboard
        .take::<WorldMeshDrawPlanSlot>()
        .unwrap_or(WorldMeshDrawPlan::Empty)
}

fn command_stats_from_prepared(prepared: &PreparedWorldMeshForwardFrame) -> GraphCommandStats {
    let group_pipeline_passes = |group: &crate::world_mesh::DrawGroup| {
        prepared
            .precomputed_batches
            .get(group.material_packet_idx)
            .and_then(|packet| packet.pipelines.as_ref())
            .map_or(0, |pipelines| pipelines.len())
    };
    let pipeline_passes: usize = prepared
        .plan
        .primary_forward_groups()
        .map(group_pipeline_passes)
        .sum();
    GraphCommandStats {
        draw_items: prepared.draws.len(),
        instance_batches: prepared.plan.primary_forward_group_count(),
        pipeline_pass_submits: pipeline_passes,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::{WorldMeshDrawPlanSlot, take_world_mesh_draw_plan};
    use crate::render_graph::blackboard::Blackboard;
    use crate::world_mesh::{
        PrefetchedWorldMeshViewDraws, WorldMeshDrawCollection, WorldMeshDrawPlan,
    };

    #[test]
    fn take_world_mesh_draw_plan_defaults_absent_slot_to_empty() {
        let mut blackboard = Blackboard::new();

        assert!(matches!(
            take_world_mesh_draw_plan(&mut blackboard),
            WorldMeshDrawPlan::Empty
        ));
    }

    #[test]
    fn take_world_mesh_draw_plan_consumes_empty_slot() {
        let mut blackboard = Blackboard::new();
        blackboard.insert::<WorldMeshDrawPlanSlot>(WorldMeshDrawPlan::Empty);

        assert!(matches!(
            take_world_mesh_draw_plan(&mut blackboard),
            WorldMeshDrawPlan::Empty
        ));
        assert!(!blackboard.contains::<WorldMeshDrawPlanSlot>());
    }

    #[test]
    fn take_world_mesh_draw_plan_consumes_prefetched_slot() {
        let mut blackboard = Blackboard::new();
        blackboard.insert::<WorldMeshDrawPlanSlot>(WorldMeshDrawPlan::Prefetched(Box::new(
            PrefetchedWorldMeshViewDraws::new(WorldMeshDrawCollection::empty(), None),
        )));

        let WorldMeshDrawPlan::Prefetched(draws) = take_world_mesh_draw_plan(&mut blackboard)
        else {
            panic!("expected prefetched draw plan");
        };

        assert!(draws.collection.items.is_empty());
        assert!(!blackboard.contains::<WorldMeshDrawPlanSlot>());
    }
}
