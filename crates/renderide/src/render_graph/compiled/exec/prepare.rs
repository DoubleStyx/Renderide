//! Pre-record view resource and blackboard preparation for compiled graph execution.

use crate::cpu_parallelism::{FrameCpuWorkload, FrameParallelPolicy, ParallelAdmission};
use crate::diagnostics::PerViewHudConfig;
use crate::gpu::GpuLimits;
use crate::graph_inputs::{FrameSystemsShared, PerViewFramePlan};
use crate::materials::MaterialSystem;
use crate::mesh_deform::{GpuSkinCache, MeshPreprocessPipelines};
use crate::occlusion::OcclusionGraphHook;
use crate::render_graph::execution_backend::{
    GraphAssetResources, GraphFrameResources, GraphViewBlackboardPreparer,
};
use crate::scene::SceneCoordinator;

use super::super::super::error::GraphExecuteError;
use super::super::super::frame_upload_batch::{FrameUploadBatch, GraphUploadSink};
use super::super::{CompiledRenderGraph, FrameView, FrameViewTarget, MultiViewExecutionContext};
use super::elapsed_ms;
use super::types::{PerViewWorkItem, PreparedPerViewFrameInput, PreparedPerViewFrameParams};

/// Per-view pre-record work items assigned to one blackboard-preparation worker.
const PRE_RECORD_VIEW_PREP_PARALLEL_CHUNK_VIEWS: usize = 1;

struct ViewBlackboardPrepareShared<'a> {
    scene: &'a SceneCoordinator,
    device: &'a wgpu::Device,
    gpu_limits: &'a GpuLimits,
    upload_batch: &'a FrameUploadBatch,
    preparer: &'a dyn GraphViewBlackboardPreparer,
    occlusion: &'a dyn OcclusionGraphHook,
    frame_resources: &'a dyn GraphFrameResources,
    materials: &'a MaterialSystem,
    asset_resources: &'a dyn GraphAssetResources,
    mesh_preprocess: Option<&'a MeshPreprocessPipelines>,
    skin_cache: Option<&'a GpuSkinCache>,
    skin_weight_mode: crate::shared::SkinWeightMode,
    debug_hud: PerViewHudConfig,
    scene_color_format: wgpu::TextureFormat,
}

impl CompiledRenderGraph {
    /// Prepares shared frame resources and owned per-view work packets before recording.
    pub(in crate::render_graph::compiled::exec) fn prepare_resources_and_work_items(
        &self,
        mv_ctx: &mut MultiViewExecutionContext<'_>,
        views: &mut [FrameView<'_>],
        upload_batch: &FrameUploadBatch,
    ) -> Result<(Vec<PerViewWorkItem>, f64), GraphExecuteError> {
        let prepare_resources_start = std::time::Instant::now();
        {
            profiling::scope!("graph::prepare_resources_for_views");
            Self::prepare_view_resources_for_views(mv_ctx, views, upload_batch)?;
        }
        let mut per_view_work_items = {
            profiling::scope!("graph::prepare_work_items");
            self.prepare_per_view_work_items(mv_ctx, views)?
        };
        {
            profiling::scope!("graph::prepare_view_blackboards_for_work_items");
            self.prepare_view_blackboards_for_work_items(
                mv_ctx,
                &mut per_view_work_items,
                upload_batch,
            );
        }
        Ok((per_view_work_items, elapsed_ms(prepare_resources_start)))
    }

    /// Lets backend-specific systems enrich per-view blackboards before graph pass recording.
    fn prepare_view_blackboards_for_work_items(
        &self,
        mv_ctx: &MultiViewExecutionContext<'_>,
        work_items: &mut [PerViewWorkItem],
        upload_batch: &FrameUploadBatch,
    ) {
        profiling::scope!("graph::prepare_view_blackboards");
        let backend = &*mv_ctx.backend;
        let total_draw_count = work_items
            .iter()
            .map(|work_item| work_item.estimated_draw_count)
            .sum::<usize>();
        let preparer = backend.view_blackboard_preparer();
        let shared = ViewBlackboardPrepareShared {
            scene: mv_ctx.scene,
            device: mv_ctx.device,
            gpu_limits: mv_ctx.gpu_limits,
            upload_batch,
            preparer: preparer.as_ref(),
            occlusion: backend.occlusion(),
            frame_resources: backend.frame_resources(),
            materials: backend.materials(),
            asset_resources: backend.asset_resources(),
            mesh_preprocess: backend.mesh_preprocess(),
            skin_cache: backend.skin_cache(),
            skin_weight_mode: backend.skin_weight_mode(),
            debug_hud: backend.per_view_hud_config(),
            scene_color_format: backend.scene_color_format_wgpu(),
        };
        let admission = view_blackboard_prepare_admission(
            FrameParallelPolicy::for_current_thread_pool(),
            work_items.len(),
            total_draw_count,
        );
        if admission.is_parallel() {
            profiling::scope!("graph::prepare_view_blackboards::parallel");
            {
                profiling::scope!("graph::prepare_view_blackboards::sort_by_draw_work");
                work_items
                    .sort_by_key(|work_item| std::cmp::Reverse(work_item.estimated_draw_count));
            }
            use rayon::prelude::*;
            work_items
                .par_iter_mut()
                .with_min_len(admission.chunk_size().unwrap_or(1))
                .for_each(|work_item| {
                    self.prepare_one_view_blackboard(&shared, work_item);
                });
            {
                profiling::scope!("graph::prepare_view_blackboards::restore_view_order");
                work_items.sort_by_key(|work_item| work_item.view_idx);
            }
        } else {
            profiling::scope!("graph::prepare_view_blackboards::serial");
            for work_item in work_items.iter_mut() {
                self.prepare_one_view_blackboard(&shared, work_item);
            }
        }
    }

    /// Prepares one view's blackboard before command recording.
    fn prepare_one_view_blackboard(
        &self,
        shared: &ViewBlackboardPrepareShared<'_>,
        work_item: &mut PerViewWorkItem,
    ) {
        profiling::scope!("graph::prepare_view_blackboard");
        let resolved = work_item.resolved.as_resolved();
        let frame_params = work_item.frame_input.frame_params(
            FrameSystemsShared {
                scene: shared.scene,
                occlusion: shared.occlusion,
                frame_resources: shared.frame_resources,
                materials: shared.materials,
                asset_resources: shared.asset_resources,
                mesh_preprocess: shared.mesh_preprocess,
                mesh_deform_scratch: None,
                mesh_deform_skin_cache: None,
                skin_cache: shared.skin_cache,
                skin_weight_mode: shared.skin_weight_mode,
                debug_hud: shared.debug_hud,
            },
            PreparedPerViewFrameParams {
                resolved: &resolved,
                scene_color_format: shared.scene_color_format,
                host_camera: &work_item.host_camera,
                render_context: work_item.render_context,
                frame_time_seconds: work_item.frame_time_seconds,
                clear: work_item.clear,
                post_processing: resolved.post_processing,
            },
        );
        shared.preparer.prepare_view_blackboard(
            shared.device,
            GraphUploadSink::pre_record_view(shared.upload_batch, work_item.view_idx),
            shared.gpu_limits,
            &frame_params,
            &work_item.frame_input.frame_plan,
            &mut work_item.initial_blackboard,
        );
    }

    /// Prepares owned per-view work items on the main thread before serial or parallel recording.
    fn prepare_per_view_work_items(
        &self,
        mv_ctx: &mut MultiViewExecutionContext<'_>,
        views: &mut [FrameView<'_>],
    ) -> Result<Vec<PerViewWorkItem>, GraphExecuteError> {
        profiling::scope!("graph::prepare_per_view_work_items");
        let mut work_items = Vec::with_capacity(views.len());
        for (view_idx, view) in views.iter_mut().enumerate() {
            let view_id = view.view_id();
            let host_camera = view.host_camera;
            let render_context = view.render_context;
            let frame_time_seconds = view.frame_time_seconds;
            let post_processing = view.post_processing();
            let target_is_swapchain = matches!(view.target, FrameViewTarget::Swapchain);
            let resolved = Self::resolve_owned_view_metadata_from_target(
                view_id,
                view.profile,
                &view.host_camera,
                &view.target,
                mv_ctx.gpu,
            )?;
            let Some(per_view_frame_bg_and_buf) = mv_ctx
                .backend
                .frame_resources()
                .per_view_frame_bind_group_and_buffer(view_id)
            else {
                logger::warn!(
                    "graph prepare: missing per-view frame resources for view {view_id:?}"
                );
                return Err(GraphExecuteError::MissingPerViewResources {
                    view_id,
                    resource: "frame",
                });
            };
            let (frame_bind_group, frame_uniform_buffer) = per_view_frame_bg_and_buf;
            let frame_plan = PerViewFramePlan {
                frame_bind_group,
                frame_uniform_buffer,
                view_idx,
            };
            let hi_z_slot = mv_ctx.backend.occlusion().ensure_hi_z_state(view_id);
            let frame_input = {
                let resolved_view = resolved.as_resolved();
                PreparedPerViewFrameInput::from_resolved(
                    &resolved_view,
                    frame_plan,
                    mv_ctx.backend.gpu_limits().cloned(),
                    mv_ctx.backend.msaa_depth_resolve(),
                    hi_z_slot,
                )
            };
            let estimated_draw_count = mv_ctx
                .backend
                .estimate_view_blackboard_prepare_draw_count(&view.initial_blackboard);
            work_items.push(PerViewWorkItem {
                view_idx,
                host_camera,
                render_context,
                frame_time_seconds,
                view_id,
                clear: view.clear,
                post_processing,
                target_is_swapchain,
                initial_blackboard: std::mem::take(&mut view.initial_blackboard),
                resolved,
                frame_input,
                estimated_draw_count,
            });
        }
        Ok(work_items)
    }
}

/// Returns the Rayon admission decision for per-view blackboard preparation.
fn view_blackboard_prepare_admission(
    policy: FrameParallelPolicy,
    view_count: usize,
    total_draw_count: usize,
) -> ParallelAdmission {
    policy.admit_draw_heavy_views(
        FrameCpuWorkload::view_draws(view_count, total_draw_count),
        PRE_RECORD_VIEW_PREP_PARALLEL_CHUNK_VIEWS,
    )
}
