//! Backend-owned world-mesh forward frame planning.

use std::sync::Arc;

use hashbrown::HashMap;
use parking_lot::Mutex;

use crate::camera::ViewId;
use crate::diagnostics::PerViewHudOutputs;
use crate::gpu::GpuLimits;
use crate::graph_inputs::{
    FrameSystemsShared, GraphPassFrame, GraphPassFrameView, PerViewFramePlan,
};
use crate::passes::{
    PreparedWorldMeshForwardFrame, WorldMeshForwardInstancePlanCache,
    WorldMeshForwardInstancePlanCacheStats, WorldMeshForwardPrepareCaches,
    WorldMeshForwardPrepareGpu, WorldMeshForwardPrepareInputs, WorldMeshForwardPrepareScratch,
    WorldMeshForwardPrepareView, WorldMeshForwardSkyboxRenderer, prepare_world_mesh_forward_frame,
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

blackboard_slot! {
    /// Blackboard slot carrying the desktop overlay draw plan into backend graph preparation.
    pub WorldMeshOverlayDrawPlanSlot => WorldMeshDrawPlan,
}

/// Backend-owned world-mesh forward preparation caches.
pub(crate) struct BackendWorldMeshFramePlanner {
    /// Skybox/background preparation cache shared across frame plans.
    skybox: WorldMeshForwardSkyboxRenderer,
    /// Retained forward instance plans keyed by draw and resolved material submission identity.
    instance_plan_cache: WorldMeshForwardInstancePlanCache,
    /// Per-view CPU scratch used while building forward draw packets.
    prepare_scratch: Mutex<HashMap<ViewId, Arc<Mutex<WorldMeshForwardPrepareScratch>>>>,
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
            prepare_scratch: Mutex::new(HashMap::new()),
        }
    }

    /// Releases view-scoped cached planning resources.
    pub(crate) fn release_view_resources(&self, retired_views: &[ViewId]) {
        self.skybox.release_view_resources(retired_views);
        if retired_views.is_empty() {
            return;
        }
        let mut scratch = self.prepare_scratch.lock();
        for &view_id in retired_views {
            scratch.remove(&view_id);
        }
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
        let scratch_slot = self.prepare_scratch_for_view(frame.view.view_id);
        let prepared = {
            let mut scratch = scratch_slot.lock();
            prepare_world_mesh_forward_frame(
                WorldMeshForwardPrepareInputs {
                    gpu: WorldMeshForwardPrepareGpu {
                        device,
                        uploads,
                        gpu_limits,
                    },
                    view: WorldMeshForwardPrepareView {
                        systems: &frame.shared,
                        view: &frame.view,
                        frame_plan: &frame_plan,
                    },
                    caches: WorldMeshForwardPrepareCaches {
                        skybox_renderer: &self.skybox,
                        instance_plan_cache: &self.instance_plan_cache,
                    },
                },
                prefetched,
                &mut scratch,
            )
        };
        WorldMeshPreparedView {
            prepared: prepared.prepared,
            hud_outputs: prepared.hud_outputs,
        }
    }

    fn prepare_scratch_for_view(
        &self,
        view_id: ViewId,
    ) -> Arc<Mutex<WorldMeshForwardPrepareScratch>> {
        self.prepare_scratch
            .lock()
            .entry(view_id)
            .or_insert_with(|| Arc::new(Mutex::new(WorldMeshForwardPrepareScratch::default())))
            .clone()
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
    let mut command_stats = GraphCommandStats::default();
    let mut command_stats_present = false;
    let draw_plan = take_world_mesh_draw_plan(blackboard);
    let prepared = planner.prepare_view(device, uploads, gpu_limits, frame, frame_plan, draw_plan);
    if let Some(forward) = prepared.prepared {
        command_stats.add(command_stats_from_prepared(&forward));
        command_stats_present = true;
        blackboard.insert::<crate::passes::WorldMeshForwardPlanSlot>(forward);
    }
    if let Some(hud_outputs) = prepared.hud_outputs {
        blackboard.insert::<crate::diagnostics::PerViewHudOutputsSlot>(hud_outputs);
    }
    if let Some(overlay_draw_plan) = blackboard.take::<WorldMeshOverlayDrawPlanSlot>() {
        prepare_desktop_overlay_blackboard(
            DesktopOverlayBlackboardPrepareCtx {
                planner,
                device,
                uploads,
                gpu_limits,
                frame,
                frame_plan,
                blackboard,
                command_stats: &mut command_stats,
                command_stats_present: &mut command_stats_present,
            },
            overlay_draw_plan,
        );
    }
    if command_stats_present {
        blackboard.insert::<GraphCommandStatsSlot>(command_stats);
    }
}

struct DesktopOverlayBlackboardPrepareCtx<'a, 'uploads, 'frame> {
    planner: &'a BackendWorldMeshFramePlanner,
    device: &'a wgpu::Device,
    uploads: GraphUploadSink<'uploads>,
    gpu_limits: &'a GpuLimits,
    frame: &'a GraphPassFrame<'frame>,
    frame_plan: &'a PerViewFramePlan,
    blackboard: &'a mut Blackboard,
    command_stats: &'a mut GraphCommandStats,
    command_stats_present: &'a mut bool,
}

fn prepare_desktop_overlay_blackboard(
    ctx: DesktopOverlayBlackboardPrepareCtx<'_, '_, '_>,
    draw_plan: WorldMeshDrawPlan,
) {
    let Some((frame_bind_group, frame_uniform_buffer)) = ctx
        .frame
        .shared
        .frame_resources
        .per_view_frame_bind_group_and_buffer(ViewId::MainOverlay)
    else {
        logger::warn!(
            "desktop overlay frame planning skipped: missing MainOverlay frame resources"
        );
        return;
    };
    let overlay_frame_plan = PerViewFramePlan {
        frame_bind_group,
        frame_uniform_buffer,
        view_idx: ctx.frame_plan.view_idx,
    };
    let overlay_frame = desktop_overlay_graph_frame(ctx.frame);
    let prepared = ctx.planner.prepare_view(
        ctx.device,
        ctx.uploads,
        ctx.gpu_limits,
        &overlay_frame,
        &overlay_frame_plan,
        draw_plan,
    );
    if let Some(forward) = prepared.prepared {
        ctx.command_stats.add(command_stats_from_prepared(&forward));
        *ctx.command_stats_present = true;
        ctx.blackboard
            .insert::<crate::passes::WorldMeshOverlayForwardPlanSlot>(forward);
    }
}

fn desktop_overlay_graph_frame<'a>(frame: &GraphPassFrame<'a>) -> GraphPassFrame<'a> {
    GraphPassFrame {
        shared: FrameSystemsShared {
            scene: frame.shared.scene,
            occlusion: frame.shared.occlusion,
            frame_resources: frame.shared.frame_resources,
            materials: frame.shared.materials,
            asset_resources: frame.shared.asset_resources,
            mesh_preprocess: frame.shared.mesh_preprocess,
            mesh_deform_scratch: None,
            mesh_deform_skin_cache: None,
            skin_cache: frame.shared.skin_cache,
            skin_weight_mode: frame.shared.skin_weight_mode,
            debug_hud: frame.shared.debug_hud,
        },
        view: GraphPassFrameView {
            depth_texture: frame.view.depth_texture,
            depth_view: frame.view.depth_view,
            depth_sample_view: frame.view.depth_sample_view.clone(),
            surface_format: frame.view.surface_format,
            scene_color_format: frame.view.surface_format,
            viewport_px: frame.view.viewport_px,
            host_camera: frame.view.host_camera,
            render_context: frame.view.render_context,
            frame_time_seconds: frame.view.frame_time_seconds,
            multiview_stereo: false,
            offscreen_write_target: frame.view.offscreen_write_target,
            view_winding: frame.view.view_winding,
            view_id: ViewId::MainOverlay,
            hi_z_slot: frame
                .shared
                .occlusion
                .ensure_hi_z_state(ViewId::MainOverlay),
            sample_count: 1,
            gpu_limits: frame.view.gpu_limits.clone(),
            msaa_depth_resolve: frame.view.msaa_depth_resolve.clone(),
            clear: frame.view.clear,
            post_processing: frame.view.post_processing,
        },
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
