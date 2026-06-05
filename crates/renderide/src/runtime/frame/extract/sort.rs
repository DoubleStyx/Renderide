//! Per-view draw sorting and draw-plan trace diagnostics.

use rayon::prelude::*;

use crate::cpu_parallelism::{FrameCpuWorkload, FrameParallelPolicy};
use crate::world_mesh::{
    WorldMeshCommandCache, WorldMeshDrawArrangeParallelism, WorldMeshDrawPlan,
};

use super::super::view_plan::FrameViewPlan;
use super::ViewWorldMeshDrawPlans;
use super::queue::QueuedViewDraws;

/// Queued view draw packets assigned to one sort worker.
const VIEW_SORT_PARALLEL_CHUNK_VIEWS: usize = 1;

/// Sorts queued draw packets for each view, preserving view order.
pub(super) fn sort_view_draws(
    view_draws: Vec<QueuedViewDraws>,
    arrange_parallelism: WorldMeshDrawArrangeParallelism,
    command_cache: &WorldMeshCommandCache,
) -> Vec<ViewWorldMeshDrawPlans> {
    profiling::scope!("render::sort_view_draws");
    if should_parallelize_view_sort(&view_draws) {
        sort_view_draws_parallel(view_draws, arrange_parallelism, command_cache)
    } else {
        sort_view_draws_serial(view_draws, arrange_parallelism, command_cache)
    }
}

/// Returns whether the queued per-view sort has enough independent work to use Rayon.
fn should_parallelize_view_sort(view_draws: &[QueuedViewDraws]) -> bool {
    let total_draws = view_draws
        .iter()
        .map(QueuedViewDraws::queued_draw_count)
        .sum::<usize>();
    FrameParallelPolicy::for_current_thread_pool()
        .admit_draw_heavy_views(
            FrameCpuWorkload::view_draws(view_draws.len(), total_draws),
            VIEW_SORT_PARALLEL_CHUNK_VIEWS,
        )
        .is_parallel()
}

/// Selects the per-view arrangement tier independently from collection fan-out.
pub(super) fn select_arrange_parallelism(
    view_draws: &[QueuedViewDraws],
) -> WorldMeshDrawArrangeParallelism {
    let total_draws = view_draws
        .iter()
        .map(QueuedViewDraws::queued_draw_count)
        .sum::<usize>();
    select_arrange_parallelism_for_draws_with_policy(
        FrameParallelPolicy::for_current_thread_pool(),
        total_draws,
    )
}

/// Policy-injected implementation for deterministic unit tests.
pub(super) fn select_arrange_parallelism_for_draws_with_policy(
    policy: FrameParallelPolicy,
    total_draws: usize,
) -> WorldMeshDrawArrangeParallelism {
    if policy.is_draw_heavy(total_draws) {
        WorldMeshDrawArrangeParallelism::Full
    } else {
        WorldMeshDrawArrangeParallelism::Serial
    }
}

fn sort_view_draws_serial(
    view_draws: Vec<QueuedViewDraws>,
    arrange_parallelism: WorldMeshDrawArrangeParallelism,
    command_cache: &WorldMeshCommandCache,
) -> Vec<ViewWorldMeshDrawPlans> {
    profiling::scope!("render::sort_view_draws::serial");
    view_draws
        .into_iter()
        .map(|queued| queued.sort_and_package(arrange_parallelism, command_cache))
        .collect()
}

fn sort_view_draws_parallel(
    view_draws: Vec<QueuedViewDraws>,
    arrange_parallelism: WorldMeshDrawArrangeParallelism,
    command_cache: &WorldMeshCommandCache,
) -> Vec<ViewWorldMeshDrawPlans> {
    profiling::scope!("render::sort_view_draws::parallel");
    if view_draws.len() == 2 {
        return sort_two_view_draws_parallel(view_draws, arrange_parallelism, command_cache);
    }
    view_draws
        .into_par_iter()
        .with_min_len(VIEW_SORT_PARALLEL_CHUNK_VIEWS)
        .map(|queued| queued.sort_and_package(arrange_parallelism, command_cache))
        .collect()
}

fn sort_two_view_draws_parallel(
    view_draws: Vec<QueuedViewDraws>,
    arrange_parallelism: WorldMeshDrawArrangeParallelism,
    command_cache: &WorldMeshCommandCache,
) -> Vec<ViewWorldMeshDrawPlans> {
    profiling::scope!("render::sort_view_draws::two_view_join");
    let mut iter = view_draws.into_iter();
    let Some(first) = iter.next() else {
        return Vec::new();
    };
    let Some(second) = iter.next() else {
        return vec![first.sort_and_package(arrange_parallelism, command_cache)];
    };
    debug_assert_eq!(iter.count(), 0);
    let (first, second) = rayon::join(
        || first.sort_and_package(arrange_parallelism, command_cache),
        || second.sort_and_package(arrange_parallelism, command_cache),
    );
    vec![first, second]
}

/// Emits trace-level draw statistics for each prepared view.
pub(super) fn trace_view_draw_plans(
    prepared: &[FrameViewPlan<'_>],
    draw_plans: &[ViewWorldMeshDrawPlans],
) {
    if !logger::enabled(logger::LogLevel::Trace) {
        return;
    }
    for (prep, draw_plan) in prepared.iter().zip(draw_plans.iter()) {
        let overlay = overlay_trace_stats(draw_plan.desktop_overlay.as_ref());
        let Some(collection) = draw_plan.world.as_prefetched() else {
            logger::trace!(
                "render view draws: view_id={:?} extent={}x{} shader_perm={:?} empty_plan=true overlay_plan_present={} overlay_draws={} overlay_pre_cull={} overlay_frustum_culled={} overlay_hi_z_culled={}",
                prep.view_id,
                prep.viewport_px.0,
                prep.viewport_px.1,
                prep.shader_permutation(),
                overlay.plan_present,
                overlay.draws,
                overlay.draws_pre_cull,
                overlay.draws_culled,
                overlay.draws_hi_z_culled,
            );
            continue;
        };
        let helper_needs = draw_plan.world.helper_needs();
        logger::trace!(
            "render view draws: view_id={:?} extent={}x{} shader_perm={:?} draws={} overlay_plan_present={} overlay_draws={} pre_cull={} frustum_culled={} hi_z_culled={} overlay_pre_cull={} overlay_frustum_culled={} overlay_hi_z_culled={} helper_depth_snapshot={} helper_color_snapshot={}",
            prep.view_id,
            prep.viewport_px.0,
            prep.viewport_px.1,
            prep.shader_permutation(),
            collection.items.len(),
            overlay.plan_present,
            overlay.draws,
            collection.draws_pre_cull,
            collection.draws_culled,
            collection.draws_hi_z_culled,
            overlay.draws_pre_cull,
            overlay.draws_culled,
            overlay.draws_hi_z_culled,
            helper_needs.depth_snapshot,
            helper_needs.color_snapshot,
        );
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct OverlayTraceStats {
    /// Whether an overlay draw plan was attached to the view.
    pub(super) plan_present: bool,
    /// Prefetched overlay draw item count.
    pub(super) draws: usize,
    /// Overlay draw count before culling.
    pub(super) draws_pre_cull: usize,
    /// Overlay draw count rejected by frustum culling.
    pub(super) draws_culled: usize,
    /// Overlay draw count rejected by Hi-Z culling.
    pub(super) draws_hi_z_culled: usize,
}

/// Summarizes optional overlay draw-plan counters for trace diagnostics.
pub(super) fn overlay_trace_stats(plan: Option<&WorldMeshDrawPlan>) -> OverlayTraceStats {
    let Some(plan) = plan else {
        return OverlayTraceStats::default();
    };
    let Some(collection) = plan.as_prefetched() else {
        return OverlayTraceStats {
            plan_present: true,
            ..Default::default()
        };
    };
    OverlayTraceStats {
        plan_present: true,
        draws: collection.items.len(),
        draws_pre_cull: collection.draws_pre_cull,
        draws_culled: collection.draws_culled,
        draws_hi_z_culled: collection.draws_hi_z_culled,
    }
}
