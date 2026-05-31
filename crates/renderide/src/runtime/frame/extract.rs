//! Frame extraction packets between runtime view planning and backend graph execution.
//!
//! This module owns the immutable CPU-side hand-off for one render tick: prepared views,
//! cull snapshots, prefetched draw plans, and the final submit packet. Keeping these types out
//! of [`super::render`] makes the render entrypoint an orchestration layer instead of another
//! subsystem owner.

use rayon::prelude::*;

use crate::backend::{ExtractedFrameShared, RenderBackend, WorldMeshDrawPlanSlot};
use crate::cpu_parallelism::{FrameCpuWorkload, FrameParallelPolicy};
use crate::gpu::GpuContext;
use crate::mesh_deform::SkinCacheKey;
use crate::occlusion::HiZCullData;
use crate::render_graph::blackboard::Blackboard;
use crate::render_graph::{
    FrameGlobalView, FrameView, FrameViewResourceHints, GraphExecuteError, OffscreenWriteTarget,
    ViewFamilyGraphRequirements,
};
use crate::world_mesh::QueuedWorldMeshDraws;
use crate::world_mesh::{
    DrawCollectionContext, HiZTemporalState, PrefetchedWorldMeshViewDraws, WorldMeshCullInput,
    WorldMeshCullProjParams, WorldMeshDrawCollectParallelism, WorldMeshDrawPlan,
    build_world_mesh_cull_proj_params, queue_draws_with_parallelism,
    queue_prepared_draws_for_views_with_parallelism,
};

use super::view_plan::{FrameViewPlan, ViewFamilyPlan};

/// Prepared view plans assigned to one cull-snapshot worker.
const CULL_SNAPSHOT_PARALLEL_CHUNK_VIEWS: usize = 1;
/// Prepared view plans assigned to one draw-collection worker.
const VIEW_COLLECTION_PARALLEL_CHUNK_VIEWS: usize = 1;
/// Queued view draw packets assigned to one sort worker.
const VIEW_SORT_PARALLEL_CHUNK_VIEWS: usize = 1;
/// View draw plans assigned to one visible-deform-key scan worker.
const VISIBLE_DEFORM_KEYS_PARALLEL_CHUNK_VIEWS: usize = 1;

/// Immutable runtime-owned extraction packet built before per-view draw collection starts.
///
/// Prepared views live beside the backend's read-only draw-prep view so later stages no longer
/// need to reach back into mutable runtime or backend state.
pub(in crate::runtime) struct ExtractedFrame<'views, 'backend> {
    /// Ordered per-frame view plans and aggregate graph requirements.
    prepared_views: PreparedViews<'views>,
    /// Backend-owned draw-prep view assembled once for the frame.
    shared: ExtractedFrameShared<'backend>,
    /// Mesh LOD bias multiplier for every view in this schedule.
    mesh_lod_bias: f32,
}

impl<'views, 'backend> ExtractedFrame<'views, 'backend> {
    /// Builds a frame extraction packet from prepared views and backend shared setup.
    pub(in crate::runtime) fn new(
        prepared_views: PreparedViews<'views>,
        shared: ExtractedFrameShared<'backend>,
        mesh_lod_bias: f32,
    ) -> Self {
        ExtractedFrame {
            prepared_views,
            shared,
            mesh_lod_bias,
        }
    }

    /// Queues explicit world-mesh draw candidates for each prepared view.
    pub(in crate::runtime) fn queue_draws(self) -> QueuedDraws<'views> {
        let ExtractedFrame {
            prepared_views,
            shared,
            mesh_lod_bias,
        } = self;
        let cull_snapshots: Vec<Option<ViewCullSnapshot>> = {
            profiling::scope!("render::gather_view_cull_snapshots");
            let plans = prepared_views.plans();
            match plans.len() {
                0 => Vec::new(),
                1 => vec![cull_snapshot_for_view(&shared, &plans[0])],
                _ if FrameParallelPolicy::for_current_thread_pool()
                    .admit_independent_items(
                        FrameCpuWorkload::independent_items(plans.len()),
                        CULL_SNAPSHOT_PARALLEL_CHUNK_VIEWS,
                    )
                    .is_parallel() =>
                {
                    plans
                        .par_iter()
                        .with_min_len(CULL_SNAPSHOT_PARALLEL_CHUNK_VIEWS)
                        .map(|prep| cull_snapshot_for_view(&shared, prep))
                        .collect()
                }
                _ => plans
                    .iter()
                    .map(|prep| cull_snapshot_for_view(&shared, prep))
                    .collect(),
            }
        };
        let view_draws = queue_view_draws(
            &shared,
            prepared_views.plans(),
            cull_snapshots,
            mesh_lod_bias,
        );
        QueuedDraws {
            prepared_views,
            view_draws,
            parallelism: shared.inner_parallelism,
        }
    }
}

/// Queued per-view draw candidates built after view planning and before phase sorting.
pub(in crate::runtime) struct QueuedDraws<'a> {
    /// Ordered per-frame view plans and aggregate graph requirements.
    prepared_views: PreparedViews<'a>,
    /// Queued draw candidates for every prepared view.
    view_draws: Vec<QueuedViewDraws>,
    /// Rayon tier to use for strict-order sorting inside each queued view.
    parallelism: WorldMeshDrawCollectParallelism,
}

impl<'a> QueuedDraws<'a> {
    /// Sorts queued draws and promotes them into final per-view draw plans.
    pub(in crate::runtime) fn sort_draws(self) -> PreparedDraws<'a> {
        let view_draws = sort_view_draws(self.view_draws, self.parallelism);
        {
            profiling::scope!("render::sort_view_draws::trace_plans");
            trace_view_draw_plans(self.prepared_views.plans(), &view_draws);
        }
        PreparedDraws {
            prepared_views: self.prepared_views,
            view_draws,
        }
    }
}

/// Sorts queued draw packets for each view, preserving view order.
fn sort_view_draws(
    view_draws: Vec<QueuedViewDraws>,
    parallelism: WorldMeshDrawCollectParallelism,
) -> Vec<WorldMeshDrawPlan> {
    profiling::scope!("render::sort_view_draws");
    if should_parallelize_view_sort(&view_draws) {
        sort_view_draws_parallel(view_draws, parallelism)
    } else {
        sort_view_draws_serial(view_draws, parallelism)
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

fn sort_view_draws_serial(
    view_draws: Vec<QueuedViewDraws>,
    parallelism: WorldMeshDrawCollectParallelism,
) -> Vec<WorldMeshDrawPlan> {
    profiling::scope!("render::sort_view_draws::serial");
    view_draws
        .into_iter()
        .map(|queued| queued.sort_and_package(parallelism))
        .collect()
}

fn sort_view_draws_parallel(
    view_draws: Vec<QueuedViewDraws>,
    parallelism: WorldMeshDrawCollectParallelism,
) -> Vec<WorldMeshDrawPlan> {
    profiling::scope!("render::sort_view_draws::parallel");
    if view_draws.len() == 2 {
        return sort_two_view_draws_parallel(view_draws, parallelism);
    }
    view_draws
        .into_par_iter()
        .with_min_len(VIEW_SORT_PARALLEL_CHUNK_VIEWS)
        .map(|queued| queued.sort_and_package(parallelism))
        .collect()
}

fn sort_two_view_draws_parallel(
    view_draws: Vec<QueuedViewDraws>,
    parallelism: WorldMeshDrawCollectParallelism,
) -> Vec<WorldMeshDrawPlan> {
    profiling::scope!("render::sort_view_draws::two_view_join");
    let mut iter = view_draws.into_iter();
    let Some(first) = iter.next() else {
        return Vec::new();
    };
    let Some(second) = iter.next() else {
        return vec![first.sort_and_package(parallelism)];
    };
    debug_assert_eq!(iter.count(), 0);
    let (first, second) = rayon::join(
        || first.sort_and_package(parallelism),
        || second.sort_and_package(parallelism),
    );
    vec![first, second]
}

/// Prepared per-frame view list plus aggregate graph requirements.
pub(in crate::runtime) struct PreparedViews<'a> {
    /// Ordered view family and aggregate graph requirements for this tick.
    family: ViewFamilyPlan<'a>,
}

impl<'a> PreparedViews<'a> {
    /// Builds prepared views from the ordered plan.
    pub(in crate::runtime) fn new(family: ViewFamilyPlan<'a>) -> Self {
        Self { family }
    }

    /// Returns `true` when no view should be rendered this tick.
    pub(in crate::runtime) fn is_empty(&self) -> bool {
        self.family.is_empty()
    }

    /// Shared slice of the ordered planned views.
    pub(in crate::runtime) fn plans(&self) -> &[FrameViewPlan<'a>] {
        self.family.plans()
    }

    /// Primary-view metadata for frame-global graph passes.
    pub(in crate::runtime) fn frame_global(&self) -> &FrameGlobalView {
        self.family.frame_global()
    }

    /// Aggregate graph-shaping requirements for the ordered views.
    pub(in crate::runtime) fn graph_requirements(&self) -> ViewFamilyGraphRequirements {
        self.family.requirements()
    }

    /// Builds executable graph views from the prepared plans and collected draw plans.
    fn build_execution_views<'b>(&'b self, draw_plans: Vec<WorldMeshDrawPlan>) -> Vec<FrameView<'b>>
    where
        'a: 'b,
    {
        self.family
            .plans()
            .iter()
            .zip(draw_plans)
            .map(|(prep, draws)| {
                let helper_needs = draws.helper_needs();
                let resource_hints = FrameViewResourceHints {
                    needs_depth_snapshot: helper_needs.depth_snapshot,
                    needs_color_snapshot: helper_needs.color_snapshot,
                };
                let mut initial_blackboard = Blackboard::new();
                initial_blackboard.insert::<WorldMeshDrawPlanSlot>(draws);
                prep.to_frame_view(resource_hints, initial_blackboard)
            })
            .collect()
    }
}

/// Immutable per-view draw packet built after culling and draw sorting.
pub(in crate::runtime) struct PreparedDraws<'a> {
    /// Ordered per-frame view plans and aggregate graph requirements.
    prepared_views: PreparedViews<'a>,
    /// Explicit draw plan for every prepared view.
    view_draws: Vec<WorldMeshDrawPlan>,
}

impl<'a> PreparedDraws<'a> {
    /// Promotes prepared views plus explicit draws into the final submit packet.
    pub(in crate::runtime) fn into_submit_frame(self) -> SubmitFrame<'a> {
        SubmitFrame {
            prepared_views: self.prepared_views,
            view_draws: self.view_draws,
        }
    }
}

/// Final immutable runtime packet handed to backend execution for one frame.
pub(in crate::runtime) struct SubmitFrame<'a> {
    /// Ordered per-frame view plans and aggregate graph requirements.
    prepared_views: PreparedViews<'a>,
    /// Explicit draw plan for every prepared view.
    view_draws: Vec<WorldMeshDrawPlan>,
}

impl SubmitFrame<'_> {
    /// Prepares frame resources that depend on the sorted draw list.
    pub(in crate::runtime) fn prepare_resources(
        &self,
        scene: &crate::scene::SceneCoordinator,
        backend: &mut RenderBackend,
    ) {
        backend.prepare_lights_for_views(
            scene,
            self.prepared_views
                .plans()
                .iter()
                .map(FrameViewPlan::light_view_desc),
        );
        let visible_deform_keys = visible_mesh_deform_keys_from_draw_plans(&self.view_draws);
        backend
            .frame_resources_mut()
            .begin_mesh_deform_submission(visible_deform_keys);
    }

    /// Executes the final submit packet after [`Self::prepare_resources`] has run.
    pub(in crate::runtime) fn execute_after_resource_prepare(
        self,
        gpu: &mut GpuContext,
        scene: &crate::scene::SceneCoordinator,
        backend: &mut RenderBackend,
    ) -> Result<(), GraphExecuteError> {
        let requirements = self.prepared_views.graph_requirements();
        let frame_global = *self.prepared_views.frame_global();
        let mut views = self.prepared_views.build_execution_views(self.view_draws);
        backend.execute_multi_view_frame(gpu, scene, &frame_global, &mut views, requirements, true)
    }
}

fn visible_mesh_deform_keys_from_draw_plans(
    draw_plans: &[WorldMeshDrawPlan],
) -> hashbrown::HashSet<SkinCacheKey> {
    if should_parallelize_visible_deform_keys(draw_plans) {
        return visible_mesh_deform_keys_from_draw_plans_parallel(draw_plans);
    }
    visible_mesh_deform_keys_from_draw_plans_serial(draw_plans)
}

fn should_parallelize_visible_deform_keys(draw_plans: &[WorldMeshDrawPlan]) -> bool {
    let total_draws = draw_plans
        .iter()
        .map(WorldMeshDrawPlan::draw_count)
        .sum::<usize>();
    FrameParallelPolicy::for_current_thread_pool()
        .admit_draw_heavy_views(
            FrameCpuWorkload::view_draws(draw_plans.len(), total_draws),
            VISIBLE_DEFORM_KEYS_PARALLEL_CHUNK_VIEWS,
        )
        .is_parallel()
}

fn visible_mesh_deform_keys_from_draw_plans_serial(
    draw_plans: &[WorldMeshDrawPlan],
) -> hashbrown::HashSet<SkinCacheKey> {
    let mut keys = hashbrown::HashSet::new();
    for plan in draw_plans {
        keys.extend(visible_mesh_deform_keys_for_plan(plan));
    }
    keys
}

fn visible_mesh_deform_keys_from_draw_plans_parallel(
    draw_plans: &[WorldMeshDrawPlan],
) -> hashbrown::HashSet<SkinCacheKey> {
    draw_plans
        .par_iter()
        .with_min_len(VISIBLE_DEFORM_KEYS_PARALLEL_CHUNK_VIEWS)
        .map(visible_mesh_deform_keys_for_plan)
        .reduce(hashbrown::HashSet::new, |mut keys, partial| {
            keys.extend(partial);
            keys
        })
}

fn visible_mesh_deform_keys_for_plan(
    draw_plan: &WorldMeshDrawPlan,
) -> hashbrown::HashSet<SkinCacheKey> {
    let mut keys = hashbrown::HashSet::new();
    let Some(collection) = draw_plan.as_prefetched() else {
        return keys;
    };
    for item in &collection.items {
        if item.world_space_deformed || item.blendshape_deformed {
            keys.insert(SkinCacheKey::from_draw_parts(
                item.space_id,
                item.skinned,
                item.instance_id,
            ));
        }
    }
    keys
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

/// Queued draw candidates and cull projection for one view.
struct QueuedViewDraws {
    /// Draw candidates before final phase sorting.
    queued: QueuedWorldMeshDraws,
    /// Projection parameters matching the view's camera/viewport.
    cull_proj: Option<WorldMeshCullProjParams>,
}

impl QueuedViewDraws {
    /// Number of queued draw candidates before final sorting and arrangement.
    fn queued_draw_count(&self) -> usize {
        self.queued.len()
    }

    /// Sorts this view's queued draws and packages the final draw plan.
    fn sort_and_package(self, parallelism: WorldMeshDrawCollectParallelism) -> WorldMeshDrawPlan {
        let collection = self.queued.sort_and_arrange(parallelism);
        WorldMeshDrawPlan::Prefetched(Box::new(PrefetchedWorldMeshViewDraws::new(
            collection,
            self.cull_proj.as_ref(),
        )))
    }
}

/// Queues world-mesh draws for every prepared view in parallel.
///
/// Returns one queued draw packet per prepared view, preserving input order so the compiled graph
/// never has to infer whether draws were intentionally omitted or merely missing after sorting.
///
/// Takes ownership of `cull_snapshots` so each view moves its `hi_z` / `hi_z_temporal` payloads
/// (already `Arc`-shared internally) into the cull input instead of cloning, avoiding a per-view
/// refcount bump on the heaviest-view path.
fn queue_view_draws(
    setup: &ExtractedFrameShared<'_>,
    prepared: &[FrameViewPlan<'_>],
    cull_snapshots: Vec<Option<ViewCullSnapshot>>,
    mesh_lod_bias: f32,
) -> Vec<QueuedViewDraws> {
    profiling::scope!("render::queue_view_draws");
    // The MaterialDictionary wraps the property store with read-only views; building it once
    // and sharing across views avoids N redundant constructions inside the rayon par_iter.
    let dict = {
        profiling::scope!("queue::shared_dictionary");
        crate::materials::host_data::MaterialDictionary::new(setup.property_store)
    };
    let max_prepared_draw_count = prepared
        .iter()
        .filter_map(|prep| setup.prepared_renderables_for(prep.render_context()))
        .map(crate::world_mesh::FramePreparedRenderables::len)
        .max()
        .unwrap_or(0);
    let inner_parallelism = select_inner_parallelism_for_prepared_work(
        prepared.len(),
        max_prepared_draw_count,
        setup.inner_parallelism,
    );
    let parallelize_views =
        should_parallelize_view_collection(prepared.len(), max_prepared_draw_count);
    let mut cull_inputs = Vec::with_capacity(prepared.len());
    let mut cull_projs = Vec::with_capacity(prepared.len());
    {
        profiling::scope!("render::queue_view_draws::build_cull_inputs");
        let mut snapshots = cull_snapshots.into_iter();
        for prep in prepared {
            let snap = snapshots.next().unwrap_or(None);
            let cull_proj = snap.as_ref().map(|s| s.proj);
            let culling = snap.map(|s| WorldMeshCullInput {
                proj: s.proj,
                host_camera: &prep.host_camera,
                hi_z: s.hi_z,
                hi_z_temporal: s.hi_z_temporal,
            });
            cull_projs.push(cull_proj);
            cull_inputs.push(culling);
        }
    }
    let contexts = {
        profiling::scope!("render::queue_view_draws::build_contexts");
        prepared
            .iter()
            .zip(cull_inputs.iter())
            .map(|(prep, culling)| {
                let shader_perm = prep.shader_permutation();
                let render_context = prep.render_context();
                // The backend pre-refreshed one material batch cache per render-context/permutation pair
                // in `extract_frame_shared`, so any prepared view should find a matching cache here.
                let material_cache = {
                    profiling::scope!("render::queue_view_draws::material_cache_lookup");
                    setup.material_cache_for(render_context, shader_perm)
                };
                DrawCollectionContext {
                    scene: setup.scene,
                    mesh_pool: setup.mesh_pool,
                    material_dict: &dict,
                    material_router: setup.router,
                    pipeline_property_ids: &setup.pipeline_property_ids,
                    shader_perm,
                    render_context,
                    head_output_transform: prep.host_camera.head_output_transform,
                    view_origin_world: prep.view_origin_world(),
                    culling: culling.as_ref(),
                    mesh_lod_bias,
                    transform_filter: prep.draw_filter.as_ref(),
                    render_space_filter: prep.render_space_filter,
                    material_cache,
                    reflection_probes: Some(setup.reflection_probes),
                    prepared: setup.prepared_renderables_for(render_context),
                }
            })
            .collect::<Vec<_>>()
    };
    if let Some(queued) =
        queue_prepared_draws_for_views_with_parallelism(&contexts, inner_parallelism)
    {
        return queued
            .into_iter()
            .zip(cull_projs)
            .map(|(queued, cull_proj)| QueuedViewDraws { queued, cull_proj })
            .collect();
    }
    collect_view_draws_with_strategy(&contexts, &cull_projs, parallelize_views, inner_parallelism)
}

/// Queues one view through the general draw-collection path.
fn collect_one_view_draws(
    ctx: &DrawCollectionContext<'_>,
    cull_proj: Option<&WorldMeshCullProjParams>,
    parallelism: WorldMeshDrawCollectParallelism,
) -> QueuedViewDraws {
    profiling::scope!("render::queue_view_draws::queue_one");
    let queued = queue_draws_with_parallelism(ctx, parallelism);
    QueuedViewDraws {
        queued,
        cull_proj: cull_proj.copied(),
    }
}

/// Returns a cull projection reference for `index` when present.
fn cull_proj_or_none(
    cull_projs: &[Option<WorldMeshCullProjParams>],
    index: usize,
) -> Option<&WorldMeshCullProjParams> {
    cull_projs.get(index).and_then(Option::as_ref)
}

/// Dispatches queued draw collection using the selected view-level parallelism strategy.
fn collect_view_draws_with_strategy(
    contexts: &[DrawCollectionContext<'_>],
    cull_projs: &[Option<WorldMeshCullProjParams>],
    parallelize_views: bool,
    parallelism: WorldMeshDrawCollectParallelism,
) -> Vec<QueuedViewDraws> {
    match contexts.len() {
        0 => Vec::new(),
        1 => {
            profiling::scope!("render::queue_view_draws::single_view");
            vec![collect_one_view_draws(
                &contexts[0],
                cull_proj_or_none(cull_projs, 0),
                parallelism,
            )]
        }
        2 if parallelize_views => {
            profiling::scope!("render::queue_view_draws::parallel_views");
            profiling::scope!("render::queue_view_draws::parallel_views::two_view_join");
            let first_proj = cull_proj_or_none(cull_projs, 0);
            let second_proj = cull_proj_or_none(cull_projs, 1);
            let (first, second) = rayon::join(
                || {
                    profiling::scope!(
                        "render::queue_view_draws::parallel_views::two_view_join::left"
                    );
                    collect_one_view_draws(&contexts[0], first_proj, parallelism)
                },
                || {
                    profiling::scope!(
                        "render::queue_view_draws::parallel_views::two_view_join::right"
                    );
                    collect_one_view_draws(&contexts[1], second_proj, parallelism)
                },
            );
            vec![first, second]
        }
        _ if parallelize_views => {
            profiling::scope!("render::queue_view_draws::parallel_views");
            profiling::scope!("render::queue_view_draws::parallel_views::par_iter");
            contexts
                .par_iter()
                .with_min_len(VIEW_COLLECTION_PARALLEL_CHUNK_VIEWS)
                .enumerate()
                .map(|(index, ctx)| {
                    collect_one_view_draws(ctx, cull_proj_or_none(cull_projs, index), parallelism)
                })
                .collect()
        }
        _ => {
            profiling::scope!("render::queue_view_draws::serial_small_views");
            contexts
                .iter()
                .enumerate()
                .map(|(index, ctx)| {
                    collect_one_view_draws(ctx, cull_proj_or_none(cull_projs, index), parallelism)
                })
                .collect()
        }
    }
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
pub(in crate::runtime) fn select_inner_parallelism(
    prepared: &[FrameViewPlan<'_>],
) -> WorldMeshDrawCollectParallelism {
    if prepared.len() > 1 {
        WorldMeshDrawCollectParallelism::SerialInnerForNestedBatch
    } else {
        WorldMeshDrawCollectParallelism::Full
    }
}

/// Refines the frame-level inner parallelism once the backend has built the prepared draw list.
///
/// The early selector only knows view count. At this point we also know whether each view will
/// walk enough prepared draws to justify nested chunk workers for the common two-view case.
fn select_inner_parallelism_for_prepared_work(
    view_count: usize,
    prepared_draw_count: usize,
    default_parallelism: WorldMeshDrawCollectParallelism,
) -> WorldMeshDrawCollectParallelism {
    select_inner_parallelism_for_prepared_work_with_policy(
        FrameParallelPolicy::for_current_thread_pool(),
        view_count,
        prepared_draw_count,
        default_parallelism,
    )
}

/// Policy-injected implementation for deterministic unit tests.
fn select_inner_parallelism_for_prepared_work_with_policy(
    policy: FrameParallelPolicy,
    view_count: usize,
    prepared_draw_count: usize,
    default_parallelism: WorldMeshDrawCollectParallelism,
) -> WorldMeshDrawCollectParallelism {
    if view_count == 2 && policy.is_draw_heavy(view_count.saturating_mul(prepared_draw_count)) {
        WorldMeshDrawCollectParallelism::Full
    } else {
        default_parallelism
    }
}

/// Returns whether multiple views should collect draws through outer view-level rayon work.
fn should_parallelize_view_collection(view_count: usize, max_prepared_draw_count: usize) -> bool {
    should_parallelize_view_collection_with_policy(
        FrameParallelPolicy::for_current_thread_pool(),
        view_count,
        max_prepared_draw_count,
    )
}

/// Policy-injected implementation for deterministic unit tests.
fn should_parallelize_view_collection_with_policy(
    policy: FrameParallelPolicy,
    view_count: usize,
    max_prepared_draw_count: usize,
) -> bool {
    policy
        .admit_draw_heavy_views(
            FrameCpuWorkload::view_draws(
                view_count,
                view_count.saturating_mul(max_prepared_draw_count),
            ),
            VIEW_COLLECTION_PARALLEL_CHUNK_VIEWS,
        )
        .is_parallel()
}

/// Builds frustum + Hi-Z cull inputs for one prepared view.
///
/// Suppressed temporal occlusion still builds frustum inputs, but skips Hi-Z snapshots. Safe to
/// call in parallel across views:
/// [`OcclusionSystem`] is `Sync` because its internal readback channel uses `crossbeam_channel`.
fn cull_snapshot_for_view(
    setup: &ExtractedFrameShared<'_>,
    prep: &FrameViewPlan<'_>,
) -> Option<ViewCullSnapshot> {
    build_cull_snapshot_for_view(setup.scene, setup.occlusion, prep)
}

fn build_cull_snapshot_for_view(
    scene: &crate::scene::SceneCoordinator,
    occlusion: &crate::occlusion::OcclusionSystem,
    prep: &FrameViewPlan<'_>,
) -> Option<ViewCullSnapshot> {
    let camera_proj = build_world_mesh_cull_proj_params(scene, prep.viewport_px, &prep.host_camera);
    let proj = cull_projection_for_write_target(&camera_proj, prep.write_target());
    let depth_mode = prep.output_depth_mode();
    let (hi_z, hi_z_temporal) = if prep.host_camera.suppress_occlusion_temporal {
        (None, None)
    } else {
        (
            occlusion.hi_z_cull_data(depth_mode, prep.view_id),
            occlusion.hi_z_temporal_snapshot(prep.view_id),
        )
    };
    Some(ViewCullSnapshot {
        proj,
        hi_z,
        hi_z_temporal,
    })
}

fn cull_projection_for_write_target(
    proj: &WorldMeshCullProjParams,
    write_target: OffscreenWriteTarget,
) -> WorldMeshCullProjParams {
    proj.map_projection_matrices(|projection| write_target.render_projection(projection))
}

#[cfg(test)]
mod tests {
    use glam::Mat4;

    use crate::camera::{HostCameraFrame, ViewId};
    use crate::mesh_deform::{SkinCacheKey, SkinCacheRendererKind};
    use crate::occlusion::OcclusionSystem;
    use crate::render_graph::{FrameViewClear, RenderPathProfile};
    use crate::scene::SceneCoordinator;
    use crate::world_mesh::WorldMeshDrawCollectParallelism;
    use crate::world_mesh::test_fixtures::{DummyDrawItemSpec, dummy_world_mesh_draw_item};
    use crate::world_mesh::{PrefetchedWorldMeshViewDraws, WorldMeshDrawCollection};

    use super::super::view_plan::{FrameViewPlan, FrameViewPlanParams, FrameViewPlanTarget};
    use super::*;

    fn main_swapchain_plan() -> FrameViewPlan<'static> {
        FrameViewPlan::new(
            &HostCameraFrame::default(),
            FrameViewPlanParams {
                render_context: crate::shared::RenderingContext::UserView,
                frame_time_seconds: 0.0,
                view_id: ViewId::Main,
                viewport_px: (640, 480),
                clear: FrameViewClear::default(),
                profile: RenderPathProfile::desktop_main(),
                target: FrameViewPlanTarget::Swapchain,
            },
        )
    }

    #[test]
    fn suppressed_occlusion_still_builds_frustum_cull_snapshot() {
        let scene = SceneCoordinator::new();
        let occlusion = OcclusionSystem::new();
        let mut plan = main_swapchain_plan();
        plan.host_camera.suppress_occlusion_temporal = true;

        let snapshot =
            build_cull_snapshot_for_view(&scene, &occlusion, &plan).expect("frustum cull snapshot");

        assert!(snapshot.hi_z.is_none());
        assert!(snapshot.hi_z_temporal.is_none());
        assert!(snapshot.proj.vr_stereo.is_none());
    }

    fn asymmetric_cull_projection_bundle() -> WorldMeshCullProjParams {
        WorldMeshCullProjParams {
            world_proj: Mat4::from_cols_array(&[
                1.0, 0.25, 0.0, 0.0, //
                0.5, 2.0, 0.0, 0.0, //
                0.0, 0.0, 3.0, 0.75, //
                0.0, 0.0, 1.0, 1.0,
            ]),
            overlay_proj: Mat4::from_cols_array(&[
                1.5, 0.125, 0.0, 0.0, //
                0.75, 1.25, 0.0, 0.0, //
                0.0, 0.0, 2.5, 0.5, //
                0.0, 0.0, 1.0, 1.0,
            ]),
            vr_stereo: Some((
                Mat4::from_cols_array(&[
                    1.0, 0.0, 0.0, 0.0, //
                    0.1, 1.0, 0.0, 0.0, //
                    0.0, 0.0, 1.0, 0.0, //
                    0.0, 0.0, 0.0, 1.0,
                ]),
                Mat4::from_cols_array(&[
                    1.0, 0.0, 0.0, 0.0, //
                    -0.1, 1.0, 0.0, 0.0, //
                    0.0, 0.0, 1.0, 0.0, //
                    0.0, 0.0, 0.0, 1.0,
                ]),
            )),
        }
    }

    #[test]
    fn primary_cull_projection_preserves_camera_convention() {
        let raw = asymmetric_cull_projection_bundle();
        let adjusted = cull_projection_for_write_target(&raw, OffscreenWriteTarget::None);

        assert_eq!(adjusted.world_proj, raw.world_proj);
        assert_eq!(adjusted.overlay_proj, raw.overlay_proj);
        assert_eq!(adjusted.vr_stereo, raw.vr_stereo);
    }

    #[test]
    fn host_render_texture_cull_projection_uses_offscreen_convention() {
        let raw = asymmetric_cull_projection_bundle();
        let write_target = OffscreenWriteTarget::HostRenderTexture(77);
        let adjusted = cull_projection_for_write_target(&raw, write_target);
        let (left, right) = raw.vr_stereo.expect("stereo pair");

        assert_eq!(
            adjusted.world_proj,
            write_target.render_projection(raw.world_proj)
        );
        assert_eq!(
            adjusted.overlay_proj,
            write_target.render_projection(raw.overlay_proj)
        );
        assert_eq!(
            adjusted.vr_stereo,
            Some((
                write_target.render_projection(left),
                write_target.render_projection(right)
            ))
        );
    }

    #[test]
    fn untracked_offscreen_cull_projection_uses_offscreen_convention() {
        let raw = asymmetric_cull_projection_bundle();
        let write_target = OffscreenWriteTarget::Untracked;
        let adjusted = cull_projection_for_write_target(&raw, write_target);
        let (left, right) = raw.vr_stereo.expect("stereo pair");

        assert_eq!(
            adjusted.world_proj,
            write_target.render_projection(raw.world_proj)
        );
        assert_eq!(
            adjusted.overlay_proj,
            write_target.render_projection(raw.overlay_proj)
        );
        assert_eq!(
            adjusted.vr_stereo,
            Some((
                write_target.render_projection(left),
                write_target.render_projection(right)
            ))
        );
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

    #[test]
    fn prepared_work_selector_reenables_inner_parallelism_for_large_two_view_frames() {
        let policy = FrameParallelPolicy::new(4);
        let draws_per_view = policy.draw_heavy_threshold() / 2;
        assert_eq!(
            select_inner_parallelism_for_prepared_work_with_policy(
                policy,
                2,
                draws_per_view,
                WorldMeshDrawCollectParallelism::SerialInnerForNestedBatch,
            ),
            WorldMeshDrawCollectParallelism::Full
        );
        assert_eq!(
            select_inner_parallelism_for_prepared_work_with_policy(
                policy,
                2,
                draws_per_view - 1,
                WorldMeshDrawCollectParallelism::SerialInnerForNestedBatch,
            ),
            WorldMeshDrawCollectParallelism::SerialInnerForNestedBatch
        );
    }

    #[test]
    fn prepared_work_selector_keeps_three_view_frames_nested_serial() {
        let policy = FrameParallelPolicy::new(4);
        assert_eq!(
            select_inner_parallelism_for_prepared_work_with_policy(
                policy,
                3,
                policy.draw_heavy_threshold(),
                WorldMeshDrawCollectParallelism::SerialInnerForNestedBatch,
            ),
            WorldMeshDrawCollectParallelism::SerialInnerForNestedBatch
        );
    }

    #[test]
    fn view_collection_parallelism_requires_multiple_views_and_enough_work() {
        let policy = FrameParallelPolicy::new(4);
        let draws_per_view = policy.draw_heavy_threshold() / 2;
        assert!(!should_parallelize_view_collection_with_policy(
            policy,
            1,
            policy.draw_heavy_threshold()
        ));
        assert!(!should_parallelize_view_collection_with_policy(
            policy,
            2,
            draws_per_view - 1
        ));
        assert!(should_parallelize_view_collection_with_policy(
            policy,
            2,
            draws_per_view
        ));
    }

    #[test]
    fn visible_deform_keys_include_only_visible_deformed_draws() {
        let mut rigid = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 1,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 10,
            node_id: 0,
            slot_index: 0,
            collect_order: 0,
            alpha_blended: false,
        });
        rigid.world_space_deformed = false;
        rigid.blendshape_deformed = false;

        let mut blend = rigid.clone();
        blend.node_id = 4;
        blend.renderable_index = 4;
        blend.instance_id = crate::scene::MeshRendererInstanceId(5);
        blend.blendshape_deformed = true;

        let mut skinned = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 2,
            property_block: None,
            skinned: true,
            sorting_order: 0,
            mesh_asset_id: 11,
            node_id: 8,
            slot_index: 0,
            collect_order: 1,
            alpha_blended: false,
        });
        skinned.world_space_deformed = true;

        let plans = [WorldMeshDrawPlan::Prefetched(Box::new(
            PrefetchedWorldMeshViewDraws::new(
                WorldMeshDrawCollection {
                    items: vec![rigid, blend.clone(), skinned.clone()],
                    draws_pre_cull: 3,
                    draws_culled: 0,
                    draws_hi_z_culled: 0,
                    visibility: Default::default(),
                    arrangement: Default::default(),
                },
                None,
            ),
        ))];

        let keys = visible_mesh_deform_keys_from_draw_plans(&plans);

        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&SkinCacheKey::new(
            blend.space_id,
            SkinCacheRendererKind::Static,
            blend.instance_id,
        )));
        assert!(keys.contains(&SkinCacheKey::new(
            skinned.space_id,
            SkinCacheRendererKind::Skinned,
            skinned.instance_id,
        )));
    }
}
