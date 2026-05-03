//! Frame extraction packets between runtime view planning and backend graph execution.
//!
//! This module owns the immutable CPU-side hand-off for one render tick: prepared views,
//! cull snapshots, prefetched draw plans, and the final submit packet. Keeping these types out
//! of `frame_render` makes the render entrypoint an orchestration layer instead of another
//! subsystem owner.

use rayon::prelude::*;

use crate::backend::{ExtractedFrameShared, RenderBackend};
use crate::gpu::GpuContext;
use crate::occlusion::HiZCullData;
use crate::render_graph::{FrameView, GraphExecuteError, WorldMeshDrawPlan};
use crate::world_mesh::{
    DrawCollectionContext, HiZTemporalState, PrefetchedWorldMeshViewDraws, WorldMeshCullInput,
    WorldMeshCullProjParams, WorldMeshDrawCollectParallelism, build_world_mesh_cull_proj_params,
    collect_and_sort_draws_with_parallelism,
};

use super::frame_view_plan::FrameViewPlan;

/// Immutable runtime-owned extraction packet built before per-view draw collection starts.
///
/// Prepared views live beside the backend's read-only draw-prep view so later stages no longer
/// need to reach back into mutable runtime or backend state.
pub(super) struct ExtractedFrame<'views, 'backend> {
    /// Ordered per-frame view plans and any headless output substitution snapshot.
    prepared_views: PreparedViews<'views>,
    /// Backend-owned draw-prep view assembled once for the frame.
    shared: ExtractedFrameShared<'backend>,
}

impl<'views, 'backend> ExtractedFrame<'views, 'backend> {
    /// Builds a frame extraction packet from prepared views and backend shared setup.
    pub(super) fn new(
        prepared_views: PreparedViews<'views>,
        shared: ExtractedFrameShared<'backend>,
    ) -> Self {
        ExtractedFrame {
            prepared_views,
            shared,
        }
    }

    /// Returns `true` when no view should be rendered this tick.
    pub(super) fn is_empty(&self) -> bool {
        self.prepared_views.is_empty()
    }

    /// Collects and packages explicit world-mesh draw plans for each prepared view.
    pub(super) fn prepare_draws(self) -> PreparedDraws<'views> {
        let ExtractedFrame {
            prepared_views,
            shared,
        } = self;
        let cull_snapshots: Vec<Option<ViewCullSnapshot>> = {
            profiling::scope!("render::gather_view_cull_snapshots");
            prepared_views
                .plans()
                .par_iter()
                .map(|prep| cull_snapshot_for_view(&shared, prep))
                .collect()
        };
        let view_draws = collect_view_draws(&shared, prepared_views.plans(), cull_snapshots);
        PreparedDraws {
            prepared_views,
            view_draws,
        }
    }
}

/// Prepared per-frame view list plus any headless swapchain substitution resources needed to
/// turn it into executable graph views.
pub(super) struct PreparedViews<'a> {
    /// Ordered list of planned views for this tick.
    prepared: Vec<FrameViewPlan<'a>>,
    /// Headless main-target replacement captured before backend execution borrows the GPU.
    headless_snapshot: Option<super::frame_view_plan::HeadlessOffscreenSnapshot>,
}

impl<'a> PreparedViews<'a> {
    /// Builds prepared views from the ordered plan and optional headless target snapshot.
    pub(super) fn new(
        prepared: Vec<FrameViewPlan<'a>>,
        headless_snapshot: Option<super::frame_view_plan::HeadlessOffscreenSnapshot>,
    ) -> Self {
        Self {
            prepared,
            headless_snapshot,
        }
    }

    /// Returns `true` when no view should be rendered this tick.
    pub(super) fn is_empty(&self) -> bool {
        self.prepared.is_empty()
    }

    /// Shared slice of the ordered planned views.
    pub(super) fn plans(&self) -> &[FrameViewPlan<'a>] {
        &self.prepared
    }

    /// Builds executable graph views from the prepared plans and collected draw plans.
    fn build_execution_views<'b>(&'b self, draw_plans: Vec<WorldMeshDrawPlan>) -> Vec<FrameView<'b>>
    where
        'a: 'b,
    {
        let mut views: Vec<FrameView<'b>> = self
            .prepared
            .iter()
            .zip(draw_plans)
            .map(|(prep, draws)| prep.to_frame_view(draws))
            .collect();
        if let Some(snapshot) = self.headless_snapshot.as_ref() {
            snapshot.substitute_swapchain_views(&mut views);
        }
        views
    }
}

/// Immutable per-view draw packet built after culling and draw sorting.
pub(super) struct PreparedDraws<'a> {
    /// Ordered per-frame view plans and headless output substitution snapshot.
    prepared_views: PreparedViews<'a>,
    /// Explicit draw plan for every prepared view.
    view_draws: Vec<WorldMeshDrawPlan>,
}

impl<'a> PreparedDraws<'a> {
    /// Promotes prepared views plus explicit draws into the final submit packet.
    pub(super) fn into_submit_frame(self) -> SubmitFrame<'a> {
        SubmitFrame {
            prepared_views: self.prepared_views,
            view_draws: self.view_draws,
        }
    }
}

/// Final immutable runtime packet handed to backend execution for one frame.
pub(super) struct SubmitFrame<'a> {
    /// Ordered per-frame view plans and headless output substitution snapshot.
    prepared_views: PreparedViews<'a>,
    /// Explicit draw plan for every prepared view.
    view_draws: Vec<WorldMeshDrawPlan>,
}

impl SubmitFrame<'_> {
    /// Executes the final submit packet while the prepared view owners are still alive.
    pub(super) fn execute(
        self,
        gpu: &mut GpuContext,
        scene: &crate::scene::SceneCoordinator,
        backend: &mut RenderBackend,
    ) -> Result<(), GraphExecuteError> {
        let mut views = self.prepared_views.build_execution_views(self.view_draws);
        backend.execute_multi_view_frame(gpu, scene, &mut views, true)
    }
}

/// Frustum + Hi-Z cull inputs for one planned view.
struct ViewCullSnapshot {
    /// Projection parameters matching the view's camera/viewport.
    proj: WorldMeshCullProjParams,
    /// CPU-side Hi-Z snapshot for this view's occlusion slot.
    hi_z: Option<HiZCullData>,
    /// Temporal Hi-Z state captured after the prior frame's depth pyramid author pass.
    hi_z_temporal: Option<HiZTemporalState>,
}

/// Collects and sorts world-mesh draws for every prepared view in parallel.
///
/// Returns one explicit [`WorldMeshDrawPlan`] per prepared view, preserving input order so the
/// compiled graph never has to infer whether draws were intentionally omitted or merely missing.
///
/// Takes ownership of `cull_snapshots` so each view moves its `hi_z` / `hi_z_temporal` payloads
/// (already `Arc`-shared internally) into the cull input instead of cloning, avoiding a per-view
/// refcount bump on the heaviest-view path.
fn collect_view_draws(
    setup: &ExtractedFrameShared<'_>,
    prepared: &[FrameViewPlan<'_>],
    cull_snapshots: Vec<Option<ViewCullSnapshot>>,
) -> Vec<WorldMeshDrawPlan> {
    profiling::scope!("render::collect_view_draws");
    // The MaterialDictionary wraps the property store with read-only views; building it once
    // and sharing across views avoids N redundant constructions inside the rayon par_iter.
    let dict = {
        profiling::scope!("collect::shared_dictionary");
        crate::materials::host_data::MaterialDictionary::new(setup.property_store)
    };
    // One view per frame is the desktop common case; rayon scope dispatch + single-task spawn is
    // pure overhead vs a direct serial call. Per-view collection internally still parallelises
    // across renderer-run chunks via `setup.inner_parallelism`.
    let collect_one = |(prep, snap): (&FrameViewPlan<'_>, Option<ViewCullSnapshot>)| {
        let shader_perm = prep.shader_permutation();
        // The backend pre-refreshed one material batch cache per shader permutation in
        // `extract_frame_shared`, so any view's permutation is guaranteed to find a hit. The
        // previous mono-only fast path collapsed into this lookup.
        let material_cache = setup.material_caches.get(&shader_perm);
        let cull_proj = snap.as_ref().map(|s| s.proj);
        let culling = snap.map(|s| WorldMeshCullInput {
            proj: s.proj,
            host_camera: &prep.host_camera,
            hi_z: s.hi_z,
            hi_z_temporal: s.hi_z_temporal,
        });
        let collection = collect_and_sort_draws_with_parallelism(
            &DrawCollectionContext {
                scene: setup.scene,
                mesh_pool: setup.mesh_pool,
                material_dict: &dict,
                material_router: setup.router,
                pipeline_property_ids: &setup.pipeline_property_ids,
                shader_perm,
                render_context: setup.render_context,
                head_output_transform: prep.host_camera.head_output_transform,
                view_origin_world: prep.view_origin_world(),
                culling: culling.as_ref(),
                transform_filter: prep.draw_filter.as_ref(),
                material_cache,
                prepared: Some(setup.prepared_renderables),
            },
            setup.inner_parallelism,
        );
        WorldMeshDrawPlan::Prefetched(Box::new(PrefetchedWorldMeshViewDraws::new(
            collection, cull_proj,
        )))
    };

    let draw_plans: Vec<WorldMeshDrawPlan> = if prepared.len() == 1 {
        let mut snaps = cull_snapshots.into_iter();
        let snap = snaps.next().unwrap_or(None);
        vec![collect_one((&prepared[0], snap))]
    } else {
        prepared
            .par_iter()
            .zip(cull_snapshots.into_par_iter())
            .map(collect_one)
            .collect()
    };
    trace_view_draw_plans(prepared, &draw_plans);
    draw_plans
}

fn trace_view_draw_plans(prepared: &[FrameViewPlan<'_>], draw_plans: &[WorldMeshDrawPlan]) {
    if !logger::enabled(logger::LogLevel::Trace) {
        return;
    }
    for (prep, draw_plan) in prepared.iter().zip(draw_plans.iter()) {
        let Some(collection) = draw_plan.as_prefetched() else {
            logger::trace!(
                "render view draws: view_id={:?} extent={}x{} shader_perm={:?} empty_plan=true",
                prep.view_id,
                prep.viewport_px.0,
                prep.viewport_px.1,
                prep.shader_permutation(),
            );
            continue;
        };
        let helper_needs = draw_plan.helper_needs();
        logger::trace!(
            "render view draws: view_id={:?} extent={}x{} shader_perm={:?} draws={} pre_cull={} frustum_culled={} hi_z_culled={} helper_depth_snapshot={} helper_color_snapshot={}",
            prep.view_id,
            prep.viewport_px.0,
            prep.viewport_px.1,
            prep.shader_permutation(),
            collection.items.len(),
            collection.draws_pre_cull,
            collection.draws_culled,
            collection.draws_hi_z_culled,
            helper_needs.depth_snapshot,
            helper_needs.color_snapshot,
        );
    }
}

/// Selects the per-view inner-walk parallelism tier for a tick based on how many views will
/// collect draws. Keeps rayon from oversubscribing when several views each spawn worker-level
/// parallelism.
pub(super) fn select_inner_parallelism(
    prepared: &[FrameViewPlan<'_>],
) -> WorldMeshDrawCollectParallelism {
    if prepared.len() > 1 {
        WorldMeshDrawCollectParallelism::SerialInnerForNestedBatch
    } else {
        WorldMeshDrawCollectParallelism::Full
    }
}

/// Builds frustum + Hi-Z cull inputs for one prepared view.
///
/// Returns [`None`] when the view has explicitly suppressed temporal occlusion (selective
/// secondary cameras). Safe to call in parallel across views:
/// [`OcclusionSystem`] is `Sync` because its internal readback channel uses `crossbeam_channel`.
fn cull_snapshot_for_view(
    setup: &ExtractedFrameShared<'_>,
    prep: &FrameViewPlan<'_>,
) -> Option<ViewCullSnapshot> {
    if prep.host_camera.suppress_occlusion_temporal {
        return None;
    }
    let proj = build_world_mesh_cull_proj_params(setup.scene, prep.viewport_px, &prep.host_camera);
    let depth_mode = prep.output_depth_mode();
    Some(ViewCullSnapshot {
        proj,
        hi_z: setup.occlusion.hi_z_cull_data(depth_mode, prep.view_id),
        hi_z_temporal: setup.occlusion.hi_z_temporal_snapshot(prep.view_id),
    })
}

#[cfg(test)]
mod tests {
    use crate::camera::{HostCameraFrame, ViewId};
    use crate::render_graph::FrameViewClear;
    use crate::world_mesh::WorldMeshDrawCollectParallelism;

    use super::super::frame_view_plan::{FrameViewPlan, FrameViewPlanTarget};
    use super::*;

    fn main_swapchain_plan() -> FrameViewPlan<'static> {
        FrameViewPlan {
            host_camera: HostCameraFrame::default(),
            draw_filter: None,
            view_id: ViewId::Main,
            viewport_px: (640, 480),
            clear: FrameViewClear::default(),
            target: FrameViewPlanTarget::MainSwapchain,
        }
    }

    #[test]
    fn select_inner_parallelism_uses_full_for_zero_or_one_view() {
        assert_eq!(
            select_inner_parallelism(&[]),
            WorldMeshDrawCollectParallelism::Full
        );
        assert_eq!(
            select_inner_parallelism(&[main_swapchain_plan()]),
            WorldMeshDrawCollectParallelism::Full
        );
    }

    #[test]
    fn select_inner_parallelism_disables_nested_parallelism_for_multiple_views() {
        assert_eq!(
            select_inner_parallelism(&[main_swapchain_plan(), main_swapchain_plan()]),
            WorldMeshDrawCollectParallelism::SerialInnerForNestedBatch
        );
    }
}
