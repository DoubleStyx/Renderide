//! Frame extraction packets between runtime view planning and backend graph execution.
//!
//! This module owns the immutable CPU-side hand-off for one render tick: prepared views,
//! cull snapshots, prefetched draw plans, and the final submit packet. Keeping these types out
//! of [`super::render`] makes the render entrypoint an orchestration layer instead of another
//! subsystem owner.

mod cull;
mod queue;
mod sort;
mod visible_deform;

pub(in crate::runtime) use queue::select_inner_parallelism;

use rayon::prelude::*;

use crate::backend::{
    ExtractedFrameShared, FrameLightCullDesc, FrameLightViewDesc, RenderBackend,
    WorldMeshDrawPlanSlot, WorldMeshOverlayDrawPlanSlot,
};
use crate::cpu_parallelism::{FrameCpuWorkload, FrameParallelPolicy};
use crate::gpu::GpuContext;
use crate::render_graph::blackboard::Blackboard;
use crate::render_graph::{
    FrameGlobalView, FrameView, FrameViewResourceHints, GraphExecuteError,
    ViewFamilyGraphRequirements,
};
use crate::world_mesh::{
    WorldMeshCommandCache, WorldMeshDrawArrangeParallelism, WorldMeshDrawPlan,
    build_world_mesh_cull_proj_params,
};

use cull::{ViewCullSnapshot, cull_projection_for_write_target, cull_snapshot_for_view};
use queue::{QueuedViewDraws, queue_view_draws};
use sort::{select_arrange_parallelism, sort_view_draws, trace_view_draw_plans};
use visible_deform::visible_mesh_deform_keys_from_draw_plans;

use super::view_plan::{FrameViewPlan, ViewFamilyPlan};

/// Prepared view plans assigned to one cull-snapshot worker.
const CULL_SNAPSHOT_PARALLEL_CHUNK_VIEWS: usize = 1;

/// Immutable runtime-owned extraction packet built before per-view draw collection starts.
///
/// Holds prepared views and the backend's read-only draw-preparation state.
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
    pub(in crate::runtime) fn queue_draws(self) -> QueuedDraws<'views, 'backend> {
        let ExtractedFrame {
            prepared_views,
            shared,
            mesh_lod_bias,
        } = self;
        let cull_snapshots = gather_view_cull_snapshots(&shared, prepared_views.plans());
        let view_draws = queue_view_draws(
            &shared,
            prepared_views.plans(),
            cull_snapshots,
            mesh_lod_bias,
        );
        let arrange_parallelism = select_arrange_parallelism(&view_draws);
        QueuedDraws {
            prepared_views,
            view_draws,
            arrange_parallelism,
            command_cache: shared.command_cache,
        }
    }

    /// Fingerprint over the scene draw-generation, camera transform, and view set. When it matches
    /// last frame's, the collected per-view plans can be reused (the GPU re-culls live, so only a
    /// scene/camera/LOD change need rebuild). `None` when any view has no prepared render-world
    /// generation this frame (uncacheable).
    fn draw_cache_fingerprint(&self) -> Option<u64> {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        // Version the dependency layout so adding a field can never accidentally alias a plan
        // retained by an older in-process cache implementation.
        2u8.hash(&mut hasher);
        self.mesh_lod_bias.to_bits().hash(&mut hasher);
        self.shared.retain_gpu_static_candidates.hash(&mut hasher);
        self.shared.reflection_probes.generation().hash(&mut hasher);
        hash_scene_draw_view_headers(self.shared.scene, true, &mut hasher);
        let plans = self.prepared_views.plans();
        plans.len().hash(&mut hasher);
        for plan in plans {
            let shader_perm = plan.shader_permutation();
            let dependencies = self
                .shared
                .draw_dependencies_for(plan.render_context, shader_perm)?;
            dependencies.hash(&mut hasher);
            (plan.render_context as u8).hash(&mut hasher);
            shader_perm.hash(&mut hasher);
            self.shared
                .occlusion
                .hi_z_cpu_snapshot_generation(plan.view_id)
                .hash(&mut hasher);
            hash_frame_view_draw_dependencies(plan, &mut hasher);
        }
        Some(hasher.finish())
    }

    /// Fingerprint for rigid shared-static plans whose per-frame camera visibility is evaluated by
    /// the GPU. Camera projection, eye position, viewport, and Hi-Z readback state are deliberately
    /// excluded; the cache admits this key only after verifying that every retained draw is
    /// camera-independent and that no LOD group can alter renderer selection.
    fn gpu_static_draw_cache_fingerprint(&self) -> Option<u64> {
        use std::hash::{Hash, Hasher};

        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        2u8.hash(&mut hasher);
        self.mesh_lod_bias.to_bits().hash(&mut hasher);
        self.shared.retain_gpu_static_candidates.hash(&mut hasher);
        self.shared.reflection_probes.generation().hash(&mut hasher);
        // A render space's resolved view transform is camera state. Rigid opaque/alpha-test
        // packets do not embed it: their current cull projection and Hi-Z history are refreshed
        // on a GPU-static hit below. Keep hashing the root transform because it does participate
        // in retained object world matrices.
        hash_scene_draw_view_headers(self.shared.scene, false, &mut hasher);
        let plans = self.prepared_views.plans();
        plans.len().hash(&mut hasher);
        for plan in plans {
            let shader_perm = plan.shader_permutation();
            self.shared
                .gpu_static_draw_dependencies_for(plan.render_context, shader_perm)?
                .hash(&mut hasher);
            (plan.render_context as u8).hash(&mut hasher);
            shader_perm.hash(&mut hasher);
            hash_gpu_static_frame_view_dependencies(plan, &mut hasher);
        }
        Some(hasher.finish())
    }

    /// Reuses retained per-view plans. Exact matches cover every draw type; all-rigid shared-static
    /// plans may additionally survive camera movement because compute receives refreshed
    /// projection/Hi-Z state and owns their visibility.
    pub(in crate::runtime) fn prepare_draws_cached(
        self,
        sort: impl FnOnce(QueuedDraws<'views, 'backend>) -> PreparedDraws<'views>,
    ) -> PreparedDraws<'views> {
        let exact_fingerprint = self.draw_cache_fingerprint();
        let gpu_static_fingerprint = self.gpu_static_draw_cache_fingerprint();
        let cache = self.shared.draw_plan_cache;
        if let (Some(exact), Some(gpu_static)) = (exact_fingerprint, gpu_static_fingerprint)
            && let Some((mut view_draws, hit_kind)) = cache.try_reuse(exact, gpu_static)
        {
            profiling::scope!("render::draw_plan_cache_reuse");
            // Even an exact collection/sort hit must consume the current history snapshot. The
            // CPU Hi-Z generation can remain unchanged while the GPU temporal authoring state
            // advances to the next ping-pong texture; retaining the prior packet would make GPU
            // occlusion compare against the wrong previous-frame transform.
            let snapshots = gather_view_cull_snapshots(&self.shared, self.prepared_views.plans());
            refresh_cached_view_cull_snapshots(&mut view_draws, snapshots);
            let reused_draws = draw_plan_count(&view_draws);
            crate::profiling::plot_world_mesh_draw_plan_cache(
                crate::profiling::WorldMeshDrawPlanCacheProfileSample {
                    hit: true,
                    gpu_static_hit: hit_kind == WorldMeshDrawPlanCacheHit::GpuStatic,
                    reused_views: view_draws.len(),
                    reused_draws,
                },
            );
            return PreparedDraws {
                prepared_views: self.prepared_views,
                view_draws,
            };
        }
        if exact_fingerprint.is_none() || gpu_static_fingerprint.is_none() {
            cache.note_uncacheable();
        }
        crate::profiling::plot_world_mesh_draw_plan_cache(
            crate::profiling::WorldMeshDrawPlanCacheProfileSample::default(),
        );
        let mesh_pool = self.shared.mesh_pool;
        let retain_gpu_static_candidates = self.shared.retain_gpu_static_candidates;
        let lod_independent = self
            .prepared_views
            .plans()
            .iter()
            .map(|plan| {
                self.shared
                    .prepared_renderables_for(plan.render_context())
                    .is_some_and(|prepared| !prepared.has_lod_groups())
            })
            .collect::<Vec<_>>();
        let mut prepared = sort(self.queue_draws());
        if let (Some(exact), Some(gpu_static)) = (exact_fingerprint, gpu_static_fingerprint) {
            let shadow_eligible = prepared
                .view_draws
                .iter()
                .map(|view| {
                    gpu_static_shadow_plan_is_camera_independent(&view.shadow_casters, mesh_pool)
                })
                .collect::<Vec<_>>();
            // Reuse each retained camera-independent shadow-caster plan even when the view's world
            // plan is ineligible (particle draws), so the shadow indirect/layer caches keep hitting
            // through particle churn on the main view.
            cache.reuse_shadow_plans(gpu_static, &mut prepared.view_draws, &shadow_eligible);
            let camera_independent_gpu_static = draw_plans_are_camera_independent_gpu_static(
                &prepared.view_draws,
                mesh_pool,
                &lod_independent,
            ) && retain_gpu_static_candidates;
            cache.store(
                exact,
                gpu_static,
                camera_independent_gpu_static,
                shadow_eligible,
                &prepared.view_draws,
            );
        }
        prepared
    }
}

fn hash_frame_view_draw_dependencies(
    plan: &FrameViewPlan<'_>,
    hasher: &mut impl std::hash::Hasher,
) {
    use std::hash::Hash;

    plan.view_id.hash(hasher);
    plan.viewport_px.hash(hasher);
    plan.render_shadows.hash(hasher);
    plan.view_winding.hash(hasher);
    plan.profile.hash(hasher);
    plan.write_target().hash(hasher);
    plan.transform_filter_space.hash(hasher);
    hash_render_space_scope(plan.render_space_scope, hasher);
    hash_layer_policy(plan.layer_policy, hasher);
    hash_draw_filter(plan.draw_filter.as_ref(), hasher);
    hash_host_camera(&plan.host_camera, hasher);
    hash_vec3(plan.view_origin_world(), hasher);
}

fn hash_gpu_static_frame_view_dependencies(
    plan: &FrameViewPlan<'_>,
    hasher: &mut impl std::hash::Hasher,
) {
    use std::hash::Hash;

    plan.view_id.hash(hasher);
    plan.render_shadows.hash(hasher);
    plan.view_winding.hash(hasher);
    plan.profile.hash(hasher);
    plan.transform_filter_space.hash(hasher);
    hash_render_space_scope(plan.render_space_scope, hasher);
    hash_layer_policy(plan.layer_policy, hasher);
    hash_draw_filter(plan.draw_filter.as_ref(), hasher);
}

fn hash_render_space_scope(
    scope: crate::world_mesh::ViewRenderSpaceScope,
    hasher: &mut impl std::hash::Hasher,
) {
    use std::hash::Hash;

    match scope {
        crate::world_mesh::ViewRenderSpaceScope::AllActive => 0u8.hash(hasher),
        crate::world_mesh::ViewRenderSpaceScope::Single(space_id) => {
            1u8.hash(hasher);
            space_id.hash(hasher);
        }
    }
}

fn hash_layer_policy(
    policy: crate::world_mesh::ViewLayerPolicy,
    hasher: &mut impl std::hash::Hasher,
) {
    use std::hash::Hash;

    match policy {
        crate::world_mesh::ViewLayerPolicy::MainView => 0u8.hash(hasher),
        crate::world_mesh::ViewLayerPolicy::Camera { render_private_ui } => {
            1u8.hash(hasher);
            render_private_ui.hash(hasher);
        }
        crate::world_mesh::ViewLayerPolicy::DesktopOverlay => 2u8.hash(hasher),
    }
}

fn hash_draw_filter(
    filter: Option<&crate::world_mesh::CameraTransformDrawFilter>,
    hasher: &mut impl std::hash::Hasher,
) {
    use std::hash::Hash;

    let Some(filter) = filter else {
        0u8.hash(hasher);
        return;
    };
    1u8.hash(hasher);
    match filter.only.as_ref() {
        None => 0u8.hash(hasher),
        Some(only) => {
            1u8.hash(hasher);
            let mut ids = only.iter().copied().collect::<Vec<_>>();
            ids.sort_unstable();
            ids.hash(hasher);
        }
    }
    let mut excluded = filter.exclude.iter().copied().collect::<Vec<_>>();
    excluded.sort_unstable();
    excluded.hash(hasher);
}

fn hash_host_camera(camera: &crate::camera::HostCameraFrame, hasher: &mut impl std::hash::Hasher) {
    use std::hash::Hash;

    // `frame_index` deliberately does not participate: it changes every tick but does not alter
    // collection, culling, LOD, sorting, or material routing.
    camera.clip.near.to_bits().hash(hasher);
    camera.clip.far.to_bits().hash(hasher);
    camera.desktop_fov_degrees.to_bits().hash(hasher);
    camera.vr_active.hash(hasher);
    (camera.output_device as i32).hash(hasher);
    camera.projection_kind.hash(hasher);
    match camera.primary_ortho_task {
        None => 0u8.hash(hasher),
        Some(spec) => {
            1u8.hash(hasher);
            spec.half_height.to_bits().hash(hasher);
            spec.clip.near.to_bits().hash(hasher);
            spec.clip.far.to_bits().hash(hasher);
        }
    }
    match camera.stereo.as_ref() {
        None => 0u8.hash(hasher),
        Some(stereo) => {
            1u8.hash(hasher);
            hash_eye_view(&stereo.left, hasher);
            hash_eye_view(&stereo.right, hasher);
        }
    }
    hash_mat4(camera.head_output_transform, hasher);
    match camera.explicit_view.as_ref() {
        None => 0u8.hash(hasher),
        Some(view) => {
            1u8.hash(hasher);
            hash_eye_view(view, hasher);
        }
    }
    match camera.eye_world_position {
        None => 0u8.hash(hasher),
        Some(position) => {
            1u8.hash(hasher);
            hash_vec3(position, hasher);
        }
    }
    camera.suppress_occlusion_temporal.hash(hasher);
}

fn hash_eye_view(view: &crate::camera::EyeView, hasher: &mut impl std::hash::Hasher) {
    hash_mat4(view.view, hasher);
    hash_mat4(view.proj, hasher);
    hash_mat4(view.view_proj, hasher);
    hash_vec3(view.world_position, hasher);
}

fn hash_scene_draw_view_headers(
    scene: &crate::scene::SceneCoordinator,
    include_view_transform: bool,
    hasher: &mut impl std::hash::Hasher,
) {
    use std::hash::Hash;

    let mut space_ids = scene.render_space_ids().collect::<Vec<_>>();
    space_ids.sort_unstable();
    space_ids.len().hash(hasher);
    for space_id in space_ids {
        space_id.hash(hasher);
        let Some(space) = scene.space(space_id) else {
            0u8.hash(hasher);
            continue;
        };
        1u8.hash(hasher);
        space.is_active().hash(hasher);
        space.is_overlay().hash(hasher);
        space.is_private().hash(hasher);
        space.override_view_position().hash(hasher);
        (space.main_render_context() as u8).hash(hasher);
        hash_scene_draw_space_transforms(
            space.root_transform(),
            space.view_transform(),
            include_view_transform,
            hasher,
        );
    }
}

fn hash_scene_draw_space_transforms(
    root_transform: &crate::shared::RenderTransform,
    view_transform: &crate::shared::RenderTransform,
    include_view_transform: bool,
    hasher: &mut impl std::hash::Hasher,
) {
    hash_render_transform(root_transform, hasher);
    if include_view_transform {
        hash_render_transform(view_transform, hasher);
    }
}

fn hash_render_transform(
    transform: &crate::shared::RenderTransform,
    hasher: &mut impl std::hash::Hasher,
) {
    hash_vec3(transform.position, hasher);
    for component in transform.rotation.to_array() {
        hasher.write_u32(component.to_bits());
    }
    hash_vec3(transform.scale, hasher);
}

fn hash_mat4(matrix: glam::Mat4, hasher: &mut impl std::hash::Hasher) {
    for component in matrix.to_cols_array() {
        hasher.write_u32(component.to_bits());
    }
}

fn hash_vec3(vector: glam::Vec3, hasher: &mut impl std::hash::Hasher) {
    for component in vector.to_array() {
        hasher.write_u32(component.to_bits());
    }
}

/// Retained cross-frame cache of collected per-view world-mesh draw plans. Mixed plans require an
/// exact dependency match; verified rigid shared-static plans use a second key that lets current
/// camera/Hi-Z state be refreshed without repeating CPU collection and sorting.
#[derive(Default)]
pub(crate) struct WorldMeshDrawPlanFrameCache {
    inner: parking_lot::Mutex<WorldMeshDrawPlanCacheState>,
}

#[derive(Default)]
struct WorldMeshDrawPlanCacheState {
    cached: Option<CachedDrawFrame>,
    stats: WorldMeshDrawPlanCacheStats,
}

struct CachedDrawFrame {
    exact_fingerprint: u64,
    gpu_static_fingerprint: u64,
    camera_independent_gpu_static: bool,
    /// Per-view flag: whether that view's shadow-caster plan was camera-independent GPU-static at
    /// store time. Lets a view whose `world` plan is ineligible (e.g. carries particle draws) still
    /// reuse its shadow-caster plan across particle churn.
    shadow_eligible: Vec<bool>,
    view_draws: Vec<ViewWorldMeshDrawPlans>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WorldMeshDrawPlanCacheHit {
    Exact,
    GpuStatic,
}

/// Lifetime counters proving whether retained view plans are actually removing CPU queue/sort work.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct WorldMeshDrawPlanCacheStats {
    pub(crate) hits: u64,
    pub(crate) gpu_static_hits: u64,
    pub(crate) misses: u64,
    pub(crate) stores: u64,
    pub(crate) uncacheable: u64,
    pub(crate) reused_views: u64,
    pub(crate) reused_draws: u64,
    /// Shadow-caster plans reused on a world-plan miss (per-plan-type reuse across particle churn).
    pub(crate) shadow_plans_reused: u64,
}

impl WorldMeshDrawPlanFrameCache {
    /// Clears packets after a GPU attachment changes device-relative resources.
    pub(crate) fn clear(&self) {
        self.inner.lock().cached = None;
    }

    /// Returns cached plans for either an exact frame match or a camera-independent GPU-static
    /// match. The latter is safe only for plans whose visibility is fully deferred to compute.
    fn try_reuse(
        &self,
        exact_fingerprint: u64,
        gpu_static_fingerprint: u64,
    ) -> Option<(Vec<ViewWorldMeshDrawPlans>, WorldMeshDrawPlanCacheHit)> {
        let mut guard = self.inner.lock();
        let reused = guard.cached.as_ref().and_then(|cached| {
            if cached.exact_fingerprint == exact_fingerprint {
                Some((cached.view_draws.clone(), WorldMeshDrawPlanCacheHit::Exact))
            } else if cached.camera_independent_gpu_static
                && cached.gpu_static_fingerprint == gpu_static_fingerprint
            {
                Some((
                    cached.view_draws.clone(),
                    WorldMeshDrawPlanCacheHit::GpuStatic,
                ))
            } else {
                None
            }
        });
        if let Some((view_draws, hit_kind)) = reused.as_ref() {
            guard.stats.hits = guard.stats.hits.saturating_add(1);
            if *hit_kind == WorldMeshDrawPlanCacheHit::GpuStatic {
                guard.stats.gpu_static_hits = guard.stats.gpu_static_hits.saturating_add(1);
            }
            guard.stats.reused_views = guard
                .stats
                .reused_views
                .saturating_add(view_draws.len() as u64);
            guard.stats.reused_draws = guard
                .stats
                .reused_draws
                .saturating_add(draw_plan_count(view_draws) as u64);
        } else {
            guard.stats.misses = guard.stats.misses.saturating_add(1);
        }
        reused
    }

    /// Reuses a cached shadow plan on a matching static fingerprint and world-plan miss.
    ///
    /// Both plans must be camera-independent and GPU-static; retaining the cached `Arc` preserves
    /// source identity for downstream shadow caches.
    fn reuse_shadow_plans(
        &self,
        gpu_static_fingerprint: u64,
        fresh: &mut [ViewWorldMeshDrawPlans],
        fresh_shadow_eligible: &[bool],
    ) -> u64 {
        let mut guard = self.inner.lock();
        let Some(cached) = guard.cached.as_ref() else {
            return 0;
        };
        if cached.gpu_static_fingerprint != gpu_static_fingerprint
            || cached.view_draws.len() != fresh.len()
            || cached.shadow_eligible.len() != fresh.len()
        {
            return 0;
        }
        let mut reused = 0;
        for (index, view) in fresh.iter_mut().enumerate() {
            if fresh_shadow_eligible[index] && cached.shadow_eligible[index] {
                view.shadow_casters = cached.view_draws[index].shadow_casters.clone();
                reused += 1;
            }
        }
        guard.stats.shadow_plans_reused = guard.stats.shadow_plans_reused.saturating_add(reused);
        reused
    }

    /// Stores `view_draws` under both exact and camera-independent dependency fingerprints.
    fn store(
        &self,
        exact_fingerprint: u64,
        gpu_static_fingerprint: u64,
        camera_independent_gpu_static: bool,
        shadow_eligible: Vec<bool>,
        view_draws: &[ViewWorldMeshDrawPlans],
    ) {
        let mut guard = self.inner.lock();
        guard.cached = Some(CachedDrawFrame {
            exact_fingerprint,
            gpu_static_fingerprint,
            camera_independent_gpu_static,
            shadow_eligible,
            view_draws: view_draws.to_vec(),
        });
        guard.stats.stores = guard.stats.stores.saturating_add(1);
    }

    fn note_uncacheable(&self) {
        let mut guard = self.inner.lock();
        guard.stats.uncacheable = guard.stats.uncacheable.saturating_add(1);
    }

    #[cfg(test)]
    pub(crate) fn stats(&self) -> WorldMeshDrawPlanCacheStats {
        self.inner.lock().stats
    }
}

fn draw_plan_count(view_draws: &[ViewWorldMeshDrawPlans]) -> usize {
    view_draws
        .iter()
        .map(|view| {
            view.world.draw_count()
                + view.shadow_casters.draw_count()
                + view
                    .desktop_overlay
                    .as_ref()
                    .map_or(0, WorldMeshDrawPlan::draw_count)
        })
        .sum()
}

fn draw_plans_are_camera_independent_gpu_static(
    view_draws: &[ViewWorldMeshDrawPlans],
    mesh_pool: &crate::gpu_pools::MeshPool,
    lod_independent: &[bool],
) -> bool {
    view_draws.len() == lod_independent.len()
        && view_draws
            .iter()
            .zip(lod_independent)
            .all(|(view, &lod_independent)| {
                lod_independent
                    && view
                        .desktop_overlay
                        .as_ref()
                        .is_none_or(|plan| plan.draw_count() == 0)
                    && gpu_static_world_plan_is_camera_independent(&view.world, mesh_pool)
                    && gpu_static_shadow_plan_is_camera_independent(&view.shadow_casters, mesh_pool)
            })
}

fn gpu_static_world_plan_is_camera_independent(
    plan: &WorldMeshDrawPlan,
    mesh_pool: &crate::gpu_pools::MeshPool,
) -> bool {
    let Some(prefetched) = plan.as_prefetched_view_draws() else {
        return true;
    };
    let collection = &prefetched.collection;
    prefetched.cull_proj.is_some()
        && collection.draws_culled == 0
        && collection.draws_hi_z_culled == 0
        && collection.visibility.broadphase_culled_runs == 0
        && collection.visibility.broadphase_culled_draws == 0
        && collection.items.iter().all(|item| {
            !item.is_overlay
                && item.ui_rect_clip_local.is_none()
                && matches!(
                    crate::world_mesh::phase_classification::classify_world_mesh_batch(
                        &item.batch_key,
                    )
                    .phase,
                    crate::world_mesh::WorldMeshPhase::ForwardOpaque
                        | crate::world_mesh::WorldMeshPhase::ForwardAlphaTest
                )
                && gpu_static_item_is_camera_independent(item, mesh_pool)
        })
}

fn gpu_static_shadow_plan_is_camera_independent(
    plan: &WorldMeshDrawPlan,
    mesh_pool: &crate::gpu_pools::MeshPool,
) -> bool {
    let Some(collection) = plan.as_prefetched() else {
        return true;
    };
    collection.draws_culled == 0
        && collection.draws_hi_z_culled == 0
        && collection.visibility.broadphase_culled_runs == 0
        && collection.visibility.broadphase_culled_draws == 0
        && collection
            .items
            .iter()
            .all(|item| !item.is_overlay && gpu_static_item_is_camera_independent(item, mesh_pool))
}

fn gpu_static_item_is_camera_independent(
    item: &crate::world_mesh::WorldMeshDrawItem,
    mesh_pool: &crate::gpu_pools::MeshPool,
) -> bool {
    !item.skinned
        && !item.world_space_deformed
        && !item.blendshape_deformed
        && mesh_pool.get(item.mesh_asset_id).is_some_and(|mesh| {
            !mesh.dynamic_geometry
                && mesh.uses_shared_static_geometry()
                && mesh.has_raster_core_residency()
        })
}

fn refresh_cached_view_cull_snapshots(
    view_draws: &mut [ViewWorldMeshDrawPlans],
    snapshots: Vec<Option<ViewCullSnapshot>>,
) {
    for (view, snapshot) in view_draws.iter_mut().zip(snapshots) {
        let WorldMeshDrawPlan::Prefetched(draws) = &mut view.world else {
            continue;
        };
        let mut refreshed = (**draws).clone();
        refreshed.cull_proj = snapshot.as_ref().map(|snapshot| snapshot.proj);
        refreshed.hi_z_temporal = snapshot.and_then(|snapshot| snapshot.hi_z_temporal);
        *draws = std::sync::Arc::new(refreshed);
    }
}

fn gather_view_cull_snapshots(
    shared: &ExtractedFrameShared<'_>,
    plans: &[FrameViewPlan<'_>],
) -> Vec<Option<ViewCullSnapshot>> {
    profiling::scope!("render::gather_view_cull_snapshots");
    match plans.len() {
        0 => Vec::new(),
        1 => vec![cull_snapshot_for_view(shared, &plans[0])],
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
                .map(|prep| cull_snapshot_for_view(shared, prep))
                .collect()
        }
        _ => plans
            .iter()
            .map(|prep| cull_snapshot_for_view(shared, prep))
            .collect(),
    }
}

/// Queued per-view draw candidates built after view planning and before phase sorting.
pub(in crate::runtime) struct QueuedDraws<'a, 'backend> {
    /// Ordered per-frame view plans and aggregate graph requirements.
    prepared_views: PreparedViews<'a>,
    /// Queued draw candidates for every prepared view.
    view_draws: Vec<QueuedViewDraws>,
    /// Rayon tier to use for final draw arrangement inside each queued view.
    arrange_parallelism: WorldMeshDrawArrangeParallelism,
    /// Persistent arranged draw command-list cache owned by the backend.
    command_cache: &'backend WorldMeshCommandCache,
}

impl<'a, 'backend> QueuedDraws<'a, 'backend> {
    /// Sorts queued draws and promotes them into final per-view draw plans.
    pub(in crate::runtime) fn sort_draws(self) -> PreparedDraws<'a> {
        let view_draws = sort_view_draws(
            self.view_draws,
            self.arrange_parallelism,
            self.command_cache,
        );
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
    fn build_execution_views<'b>(
        &'b self,
        draw_plans: Vec<ViewWorldMeshDrawPlans>,
    ) -> Vec<FrameView<'b>>
    where
        'a: 'b,
    {
        self.family
            .plans()
            .iter()
            .zip(draw_plans)
            .map(|(prep, draws)| {
                let helper_needs = draws.world.helper_needs();
                let resource_hints = FrameViewResourceHints {
                    needs_depth_snapshot: helper_needs.depth_snapshot,
                    needs_per_object_color_snapshot: helper_needs.per_object_color_snapshot,
                    needs_named_color_snapshot: helper_needs.named_color_snapshot,
                };
                let mut initial_blackboard = Blackboard::new();
                initial_blackboard.insert::<WorldMeshDrawPlanSlot>(draws.world);
                if let Some(overlay) = draws.desktop_overlay {
                    initial_blackboard.insert::<WorldMeshOverlayDrawPlanSlot>(overlay);
                }
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
    view_draws: Vec<ViewWorldMeshDrawPlans>,
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
    view_draws: Vec<ViewWorldMeshDrawPlans>,
}

impl SubmitFrame<'_> {
    /// Prepares frame resources that depend on the sorted draw list.
    pub(in crate::runtime) fn prepare_resources(
        &self,
        scene: &crate::scene::SceneCoordinator,
        backend: &mut RenderBackend,
    ) {
        backend.release_arena_resident_static_mesh_sources();
        backend.prepare_lights_for_views(
            scene,
            self.prepared_views.plans().iter().flat_map(|view| {
                let desc = light_view_desc_with_cull(scene, view);
                let overlay = view.desktop_overlay_resource_view_id().map(|view_id| {
                    let mut desc = desc;
                    desc.view_id = view_id;
                    desc
                });
                std::iter::once(desc).chain(overlay)
            }),
        );
        backend.set_shadow_camera_fits(self.prepared_views.plans().iter().filter_map(|view| {
            crate::backend::ShadowCameraFit::from_scene_camera(
                scene,
                view.viewport_px,
                &view.host_camera,
            )
            .map(|fit| (view.view_id, fit))
        }));
        backend.prepare_shadow_frame_for_views(
            self.prepared_views
                .plans()
                .iter()
                .zip(self.view_draws.iter())
                .map(|(view, draws)| (view.view_id, &draws.shadow_casters)),
        );
        backend.prepare_geometry_arena_for_views(
            self.view_draws.iter().flat_map(|draws| {
                std::iter::once(&draws.world).chain(draws.desktop_overlay.as_ref())
            }),
        );
        let mut visible_deform_keys = visible_mesh_deform_keys_from_draw_plans(&self.view_draws);
        backend
            .frame_resources()
            .extend_shadow_mesh_deform_keys(&mut visible_deform_keys);
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

fn light_view_desc_with_cull(
    scene: &crate::scene::SceneCoordinator,
    view: &FrameViewPlan<'_>,
) -> FrameLightViewDesc {
    let mut desc = view.light_view_desc();
    let raw_proj = build_world_mesh_cull_proj_params(scene, view.viewport_px, &view.host_camera);
    desc.cull = Some(FrameLightCullDesc {
        host_camera: view.host_camera,
        proj: cull_projection_for_write_target(&raw_proj, view.write_target()),
    });
    desc
}

/// Sorted draw plans for one executable view.
#[derive(Clone)]
pub(in crate::runtime) struct ViewWorldMeshDrawPlans {
    /// Regular camera-world draw plan consumed by the main world-mesh pass stack.
    pub(super) world: WorldMeshDrawPlan,
    /// Shadow-caster draw plan consumed by shadow-map preparation.
    pub(super) shadow_casters: WorldMeshDrawPlan,
    /// Desktop overlay draw plan consumed by the post-compose overlay pass.
    pub(super) desktop_overlay: Option<WorldMeshDrawPlan>,
}

impl ViewWorldMeshDrawPlans {
    /// Total draw count represented by all draw plans for this view.
    pub(super) fn draw_count(&self) -> usize {
        self.world.draw_count()
            + self.shadow_casters.draw_count()
            + self
                .desktop_overlay
                .as_ref()
                .map_or(0, WorldMeshDrawPlan::draw_count)
    }
}

#[cfg(test)]
mod tests {
    use std::hash::Hasher;
    use std::sync::Arc;

    use hashbrown::HashSet;

    use glam::Mat4;

    use crate::camera::{HostCameraFrame, ViewId};
    use crate::mesh_deform::{SkinCacheKey, SkinCacheRendererKind};
    use crate::occlusion::OcclusionSystem;
    use crate::render_graph::{FrameViewClear, OffscreenWriteTarget, RenderPathProfile};
    use crate::scene::{RenderSpaceId, SceneCoordinator};
    use crate::shared::{RenderTransform, RenderingContext};
    use crate::world_mesh::CameraTransformDrawFilter;
    use crate::world_mesh::test_fixtures::{DummyDrawItemSpec, dummy_world_mesh_draw_item};
    use crate::world_mesh::{
        PrefetchedWorldMeshViewDraws, ViewLayerPolicy, ViewRenderSpaceScope, WorldMeshCullInput,
        WorldMeshCullProjParams, WorldMeshDrawArrangeParallelism, WorldMeshDrawCollectParallelism,
        WorldMeshDrawCollection, WorldMeshDrawPlan,
    };

    use super::super::view_plan::{FrameViewPlan, FrameViewPlanParams, FrameViewPlanTarget};
    use super::cull::{build_cull_snapshot_for_view, cull_projection_for_write_target};
    use super::queue::{
        desktop_overlay_view_inputs, select_inner_parallelism_for_prepared_work_with_policy,
        shadow_caster_view_inputs,
    };
    use super::sort::{
        OverlayTraceStats, overlay_trace_stats, select_arrange_parallelism_for_draws_with_policy,
    };
    use super::visible_deform::visible_mesh_deform_keys_from_draw_plans;
    use super::*;

    fn main_swapchain_plan() -> FrameViewPlan<'static> {
        FrameViewPlan::new(
            &HostCameraFrame::default(),
            FrameViewPlanParams {
                render_context: RenderingContext::UserView,
                frame_time_seconds: 0.0,
                view_id: ViewId::Main,
                viewport_px: (640, 480),
                clear: FrameViewClear::default(),
                profile: RenderPathProfile::desktop_main(),
                target: FrameViewPlanTarget::Swapchain,
            },
        )
    }

    fn cached_draw_plan(
        draw_count: usize,
        mesh_asset_id: i32,
    ) -> (WorldMeshDrawPlan, Arc<PrefetchedWorldMeshViewDraws>) {
        let items = (0..draw_count)
            .map(|index| {
                dummy_world_mesh_draw_item(DummyDrawItemSpec {
                    material_asset_id: 1,
                    property_block: None,
                    skinned: false,
                    sorting_order: 0,
                    mesh_asset_id,
                    node_id: index as i32,
                    slot_index: 0,
                    collect_order: index,
                    alpha_blended: false,
                })
            })
            .collect::<Vec<_>>();
        let draws = Arc::new(PrefetchedWorldMeshViewDraws::new(
            WorldMeshDrawCollection {
                draws_pre_cull: items.len(),
                items: items.into(),
                draws_culled: 0,
                draws_hi_z_culled: 0,
                visibility: Default::default(),
                arrangement: Default::default(),
            },
            None,
        ));
        (WorldMeshDrawPlan::Prefetched(Arc::clone(&draws)), draws)
    }

    fn draw_filter_fingerprint(filter: Option<&CameraTransformDrawFilter>) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        hash_draw_filter(filter, &mut hasher);
        hasher.finish()
    }

    fn scene_space_transform_fingerprint(
        root_transform: &RenderTransform,
        view_transform: &RenderTransform,
        include_view_transform: bool,
    ) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        hash_scene_draw_space_transforms(
            root_transform,
            view_transform,
            include_view_transform,
            &mut hasher,
        );
        hasher.finish()
    }

    #[test]
    fn gpu_static_space_fingerprint_ignores_camera_view_but_not_object_root_transform() {
        let root = RenderTransform {
            position: glam::Vec3::new(1.0, 2.0, 3.0),
            rotation: glam::Quat::IDENTITY,
            scale: glam::Vec3::ONE,
        };
        let first_view = RenderTransform {
            position: glam::Vec3::new(10.0, 20.0, 30.0),
            rotation: glam::Quat::IDENTITY,
            scale: glam::Vec3::ONE,
        };
        let moved_view = RenderTransform {
            position: glam::Vec3::new(40.0, 50.0, 60.0),
            ..first_view
        };
        let moved_root = RenderTransform {
            position: glam::Vec3::new(4.0, 5.0, 6.0),
            ..root
        };

        assert_eq!(
            scene_space_transform_fingerprint(&root, &first_view, false),
            scene_space_transform_fingerprint(&root, &moved_view, false),
            "GPU-static packets refresh current camera cull state instead of embedding the view transform"
        );
        assert_ne!(
            scene_space_transform_fingerprint(&root, &first_view, true),
            scene_space_transform_fingerprint(&root, &moved_view, true),
            "exact packets still depend on the camera view transform"
        );
        assert_ne!(
            scene_space_transform_fingerprint(&root, &first_view, false),
            scene_space_transform_fingerprint(&moved_root, &first_view, false),
            "the root transform remains an object-world-matrix dependency"
        );
    }

    #[test]
    fn cached_draw_hit_refreshes_current_cull_projection_for_every_hit_kind() {
        let (world, _) = cached_draw_plan(1, 9);
        let mut plans = [ViewWorldMeshDrawPlans {
            world,
            shadow_casters: WorldMeshDrawPlan::Empty,
            desktop_overlay: None,
        }];
        let refreshed_projection = WorldMeshCullProjParams {
            world_proj: Mat4::from_scale(glam::Vec3::splat(2.0)),
            overlay_proj: Mat4::from_scale(glam::Vec3::splat(3.0)),
            vr_stereo: None,
        };

        refresh_cached_view_cull_snapshots(
            &mut plans,
            vec![Some(ViewCullSnapshot {
                proj: refreshed_projection,
                hi_z: None,
                hi_z_temporal: None,
            })],
        );

        let refreshed = plans[0]
            .world
            .as_prefetched_view_draws()
            .expect("cached world plan remains prefetched");
        let refreshed_cull = refreshed
            .cull_proj
            .expect("the current cull projection replaces the cached value");
        assert_eq!(refreshed_cull.world_proj, refreshed_projection.world_proj);
        assert_eq!(
            refreshed_cull.overlay_proj,
            refreshed_projection.overlay_proj
        );
        assert!(refreshed_cull.vr_stereo.is_none());
    }

    #[test]
    fn draw_plan_cache_exact_fingerprint_reuses_inner_arcs_and_counts_work() {
        let cache = WorldMeshDrawPlanFrameCache::default();
        let (world, world_arc) = cached_draw_plan(2, 10);
        let (shadow_casters, shadow_arc) = cached_draw_plan(1, 11);
        let (desktop_overlay, overlay_arc) = cached_draw_plan(3, 12);
        let plans = [ViewWorldMeshDrawPlans {
            world,
            shadow_casters,
            desktop_overlay: Some(desktop_overlay),
        }];
        let fingerprint = 0xA5A5_1234_5678_9ABCu64;

        cache.store(
            fingerprint,
            fingerprint,
            false,
            vec![false; plans.len()],
            &plans,
        );
        assert_eq!(
            cache.stats(),
            WorldMeshDrawPlanCacheStats {
                stores: 1,
                ..Default::default()
            }
        );

        let reused = cache
            .try_reuse(fingerprint, fingerprint)
            .expect("the exact dependency fingerprint should hit")
            .0;
        assert_eq!(reused.len(), 1);
        let WorldMeshDrawPlan::Prefetched(reused_world) = &reused[0].world else {
            panic!("world draw plan should remain prefetched");
        };
        let WorldMeshDrawPlan::Prefetched(reused_shadows) = &reused[0].shadow_casters else {
            panic!("shadow draw plan should remain prefetched");
        };
        let WorldMeshDrawPlan::Prefetched(reused_overlay) = reused[0]
            .desktop_overlay
            .as_ref()
            .expect("desktop overlay plan")
        else {
            panic!("desktop overlay draw plan should remain prefetched");
        };
        assert!(Arc::ptr_eq(&world_arc, reused_world));
        assert!(Arc::ptr_eq(&shadow_arc, reused_shadows));
        assert!(Arc::ptr_eq(&overlay_arc, reused_overlay));
        assert_eq!(
            cache.stats(),
            WorldMeshDrawPlanCacheStats {
                hits: 1,
                stores: 1,
                reused_views: 1,
                reused_draws: 6,
                ..Default::default()
            }
        );
    }

    #[test]
    fn draw_plan_cache_mismatched_fingerprint_misses_without_evicting_exact_entry() {
        let cache = WorldMeshDrawPlanFrameCache::default();
        let (world, world_arc) = cached_draw_plan(2, 20);
        let plans = [ViewWorldMeshDrawPlans {
            world,
            shadow_casters: WorldMeshDrawPlan::Empty,
            desktop_overlay: None,
        }];
        let fingerprint = 0x1111_2222_3333_4444u64;

        cache.store(
            fingerprint,
            fingerprint,
            false,
            vec![false; plans.len()],
            &plans,
        );
        assert!(cache.try_reuse(fingerprint ^ 1, fingerprint ^ 1).is_none());
        assert_eq!(
            cache.stats(),
            WorldMeshDrawPlanCacheStats {
                misses: 1,
                stores: 1,
                ..Default::default()
            }
        );

        let reused = cache
            .try_reuse(fingerprint, fingerprint)
            .expect("a mismatched lookup must not evict the exact entry")
            .0;
        let WorldMeshDrawPlan::Prefetched(reused_world) = &reused[0].world else {
            panic!("world draw plan should remain prefetched");
        };
        assert!(Arc::ptr_eq(&world_arc, reused_world));
        assert_eq!(
            cache.stats(),
            WorldMeshDrawPlanCacheStats {
                hits: 1,
                misses: 1,
                stores: 1,
                reused_views: 1,
                reused_draws: 2,
                ..Default::default()
            }
        );
    }

    #[test]
    fn draw_plan_cache_clear_drops_device_relative_probe_selections() {
        let cache = WorldMeshDrawPlanFrameCache::default();
        let (world, _world_arc) = cached_draw_plan(1, 25);
        let plans = [ViewWorldMeshDrawPlans {
            world,
            shadow_casters: WorldMeshDrawPlan::Empty,
            desktop_overlay: None,
        }];

        cache.store(7, 7, false, vec![false; plans.len()], &plans);
        cache.clear();

        assert!(cache.try_reuse(7, 7).is_none());
    }

    #[test]
    fn draw_plan_cache_gpu_static_hit_survives_camera_only_exact_key_change() {
        let cache = WorldMeshDrawPlanFrameCache::default();
        let (world, world_arc) = cached_draw_plan(4, 30);
        let plans = [ViewWorldMeshDrawPlans {
            world,
            shadow_casters: WorldMeshDrawPlan::Empty,
            desktop_overlay: None,
        }];

        cache.store(100, 77, true, vec![false; plans.len()], &plans);
        let (reused, hit_kind) = cache
            .try_reuse(101, 77)
            .expect("GPU-static plans should reuse across camera-only exact-key changes");

        assert_eq!(hit_kind, WorldMeshDrawPlanCacheHit::GpuStatic);
        let WorldMeshDrawPlan::Prefetched(reused_world) = &reused[0].world else {
            panic!("world draw plan should remain prefetched");
        };
        assert!(Arc::ptr_eq(&world_arc, reused_world));
        assert_eq!(
            cache.stats(),
            WorldMeshDrawPlanCacheStats {
                hits: 1,
                gpu_static_hits: 1,
                stores: 1,
                reused_views: 1,
                reused_draws: 4,
                ..Default::default()
            }
        );
    }

    #[test]
    fn shadow_plans_reuse_across_world_plan_miss_restores_source_pointer_identity() {
        let cache = WorldMeshDrawPlanFrameCache::default();
        let (cached_world, _cached_world_arc) = cached_draw_plan(2, 40);
        let (cached_shadow, cached_shadow_arc) = cached_draw_plan(2, 41);
        let cached_plans = [ViewWorldMeshDrawPlans {
            world: cached_world,
            shadow_casters: cached_shadow,
            desktop_overlay: None,
        }];
        // World plan is not GPU-static reusable (particle draws), shadow plan is retained eligible.
        cache.store(1, 500, false, vec![true], &cached_plans);

        // A later frame collects a fresh distinct-Arc shadow plan under the same static fingerprint.
        let (fresh_world, fresh_world_arc) = cached_draw_plan(2, 40);
        let (fresh_shadow, fresh_shadow_arc) = cached_draw_plan(2, 41);
        assert!(!Arc::ptr_eq(&cached_shadow_arc, &fresh_shadow_arc));
        let mut fresh = vec![ViewWorldMeshDrawPlans {
            world: fresh_world,
            shadow_casters: fresh_shadow,
            desktop_overlay: None,
        }];

        assert_eq!(cache.reuse_shadow_plans(500, &mut fresh, &[true]), 1);
        let WorldMeshDrawPlan::Prefetched(reused_shadow) = &fresh[0].shadow_casters else {
            panic!("shadow plan should remain prefetched");
        };
        assert!(
            Arc::ptr_eq(&cached_shadow_arc, reused_shadow),
            "shadow source draw pointer identity must be restored so downstream caches hit"
        );
        let WorldMeshDrawPlan::Prefetched(untouched_world) = &fresh[0].world else {
            panic!("world plan should remain prefetched");
        };
        assert!(
            Arc::ptr_eq(&fresh_world_arc, untouched_world),
            "shadow reuse must not disturb the freshly collected world plan"
        );

        // A frame that culled its casters (fresh not eligible) is never served the no-cull plan.
        let (guard_shadow, guard_shadow_arc) = cached_draw_plan(2, 41);
        let mut guarded = vec![ViewWorldMeshDrawPlans {
            world: cached_draw_plan(2, 40).0,
            shadow_casters: guard_shadow,
            desktop_overlay: None,
        }];
        assert_eq!(cache.reuse_shadow_plans(500, &mut guarded, &[false]), 0);
        let WorldMeshDrawPlan::Prefetched(guard_ref) = &guarded[0].shadow_casters else {
            panic!("prefetched");
        };
        assert!(Arc::ptr_eq(&guard_shadow_arc, guard_ref));

        // A mismatched static fingerprint must not substitute.
        let (other_shadow, other_shadow_arc) = cached_draw_plan(2, 41);
        let mut other = vec![ViewWorldMeshDrawPlans {
            world: cached_draw_plan(2, 40).0,
            shadow_casters: other_shadow,
            desktop_overlay: None,
        }];
        assert_eq!(cache.reuse_shadow_plans(999, &mut other, &[true]), 0);
        let WorldMeshDrawPlan::Prefetched(other_ref) = &other[0].shadow_casters else {
            panic!("prefetched");
        };
        assert!(Arc::ptr_eq(&other_shadow_arc, other_ref));
    }

    #[test]
    fn draw_filter_fingerprint_is_deterministic_and_preserves_filter_semantics() {
        let mut only_first = HashSet::new();
        only_first.insert(42);
        only_first.insert(7);
        only_first.insert(99);
        let mut exclude_first = HashSet::new();
        exclude_first.insert(81);
        exclude_first.insert(3);

        let mut only_second = HashSet::new();
        only_second.insert(99);
        only_second.insert(42);
        only_second.insert(7);
        let mut exclude_second = HashSet::new();
        exclude_second.insert(3);
        exclude_second.insert(81);

        let first = CameraTransformDrawFilter {
            only: Some(only_first),
            exclude: exclude_first,
        };
        let second = CameraTransformDrawFilter {
            only: Some(only_second),
            exclude: exclude_second,
        };
        assert_eq!(
            draw_filter_fingerprint(Some(&first)),
            draw_filter_fingerprint(Some(&second)),
            "hash-set insertion order must not perturb the draw dependency fingerprint"
        );

        let no_filter = draw_filter_fingerprint(None);
        let exclude_nothing = draw_filter_fingerprint(Some(&CameraTransformDrawFilter::default()));
        let select_nothing = draw_filter_fingerprint(Some(&CameraTransformDrawFilter {
            only: Some(HashSet::new()),
            exclude: HashSet::new(),
        }));
        assert_ne!(no_filter, exclude_nothing);
        assert_ne!(exclude_nothing, select_nothing);

        let mut changed = second;
        changed.exclude.insert(12);
        assert_ne!(
            draw_filter_fingerprint(Some(&first)),
            draw_filter_fingerprint(Some(&changed))
        );
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
        let write_target = OffscreenWriteTarget::host_render_texture(77);
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
    fn desktop_overlay_view_inputs_do_not_inherit_main_camera_filters_or_culling() {
        let mut plan = main_swapchain_plan();
        plan.draw_filter = Some(CameraTransformDrawFilter {
            only: Some(HashSet::from_iter([42])),
            exclude: HashSet::from_iter([7]),
        });
        plan.transform_filter_space = Some(RenderSpaceId(99));
        plan.render_space_scope = ViewRenderSpaceScope::single(RenderSpaceId(99));

        let inputs = desktop_overlay_view_inputs(&plan, 1.25);

        assert_eq!(inputs.render_context, RenderingContext::UserView);
        assert_eq!(
            inputs.head_output_transform,
            plan.host_camera.head_output_transform
        );
        assert_eq!(inputs.view_origin_world, plan.view_origin_world());
        assert!(inputs.culling.is_none());
        assert!(inputs.transform_filter.is_none());
        assert!(inputs.transform_filter_space.is_none());
        assert_eq!(inputs.render_space_scope, ViewRenderSpaceScope::AllActive);
        assert!(inputs.reflection_probes.is_none());
        assert_eq!(inputs.mesh_lod_bias, 1.25);
        assert_eq!(inputs.layer_policy, ViewLayerPolicy::DesktopOverlay);
    }

    #[test]
    fn shadow_caster_view_inputs_preserve_view_scope_without_camera_culling() {
        let mut plan = main_swapchain_plan();
        plan.draw_filter = Some(CameraTransformDrawFilter {
            only: Some(HashSet::from_iter([42])),
            exclude: HashSet::from_iter([7]),
        });
        plan.transform_filter_space = Some(RenderSpaceId(99));
        plan.render_space_scope = ViewRenderSpaceScope::single(RenderSpaceId(99));
        plan.layer_policy = ViewLayerPolicy::MainView;

        let cull_input = WorldMeshCullInput {
            proj: WorldMeshCullProjParams {
                world_proj: Mat4::IDENTITY,
                overlay_proj: Mat4::IDENTITY,
                vr_stereo: None,
            },
            host_camera: &plan.host_camera,
            hi_z: None,
            hi_z_temporal: None,
        };
        let inputs = shadow_caster_view_inputs(&plan, Some(&cull_input), 1.25);

        assert_eq!(inputs.render_context, RenderingContext::UserView);
        assert_eq!(
            inputs.head_output_transform,
            plan.host_camera.head_output_transform
        );
        assert_eq!(inputs.view_origin_world, plan.view_origin_world());
        assert!(inputs.culling.is_none());
        assert!(inputs.lod_selection_culling.is_some());
        assert!(inputs.transform_filter.is_some());
        assert_eq!(inputs.transform_filter_space, Some(RenderSpaceId(99)));
        assert_eq!(
            inputs.render_space_scope,
            ViewRenderSpaceScope::single(RenderSpaceId(99))
        );
        assert!(inputs.reflection_probes.is_none());
        assert_eq!(inputs.mesh_lod_bias, 1.25);
        assert_eq!(inputs.layer_policy, ViewLayerPolicy::MainView);
    }

    #[test]
    fn overlay_trace_stats_reports_plan_presence_and_counters() {
        assert_eq!(overlay_trace_stats(None), OverlayTraceStats::default());

        let empty = WorldMeshDrawPlan::Empty;
        assert_eq!(
            overlay_trace_stats(Some(&empty)),
            OverlayTraceStats {
                plan_present: true,
                ..Default::default()
            }
        );

        let draw = dummy_world_mesh_draw_item(DummyDrawItemSpec {
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
        let prefetched =
            WorldMeshDrawPlan::Prefetched(Arc::new(PrefetchedWorldMeshViewDraws::new(
                WorldMeshDrawCollection {
                    items: vec![draw].into(),
                    draws_pre_cull: 5,
                    draws_culled: 2,
                    draws_hi_z_culled: 1,
                    visibility: Default::default(),
                    arrangement: Default::default(),
                },
                None,
            )));

        assert_eq!(
            overlay_trace_stats(Some(&prefetched)),
            OverlayTraceStats {
                plan_present: true,
                draws: 1,
                draws_pre_cull: 5,
                draws_culled: 2,
                draws_hi_z_culled: 1,
            }
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
    fn arrange_parallelism_uses_draw_heavy_threshold_independent_of_collection() {
        let policy = FrameParallelPolicy::new(4);

        assert_eq!(
            select_arrange_parallelism_for_draws_with_policy(
                policy,
                policy.draw_heavy_threshold() - 1,
            ),
            WorldMeshDrawArrangeParallelism::Serial
        );
        assert_eq!(
            select_arrange_parallelism_for_draws_with_policy(policy, policy.draw_heavy_threshold()),
            WorldMeshDrawArrangeParallelism::Full
        );
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

        let mut camera_skinned = skinned.clone();
        camera_skinned.render_context = RenderingContext::Camera;

        let mut overlay = rigid.clone();
        overlay.node_id = 9;
        overlay.renderable_index = 9;
        overlay.instance_id = crate::scene::MeshRendererInstanceId(9);
        overlay.blendshape_deformed = true;

        let plans = [ViewWorldMeshDrawPlans {
            world: WorldMeshDrawPlan::Prefetched(Arc::new(PrefetchedWorldMeshViewDraws::new(
                WorldMeshDrawCollection {
                    items: vec![
                        rigid,
                        blend.clone(),
                        skinned.clone(),
                        camera_skinned.clone(),
                    ]
                    .into(),
                    draws_pre_cull: 4,
                    draws_culled: 0,
                    draws_hi_z_culled: 0,
                    visibility: Default::default(),
                    arrangement: Default::default(),
                },
                None,
            ))),
            shadow_casters: WorldMeshDrawPlan::Empty,
            desktop_overlay: Some(WorldMeshDrawPlan::Prefetched(Arc::new(
                PrefetchedWorldMeshViewDraws::new(
                    WorldMeshDrawCollection {
                        items: vec![overlay.clone()].into(),
                        draws_pre_cull: 1,
                        draws_culled: 0,
                        draws_hi_z_culled: 0,
                        visibility: Default::default(),
                        arrangement: Default::default(),
                    },
                    None,
                ),
            ))),
        }];

        let keys = visible_mesh_deform_keys_from_draw_plans(&plans);

        assert_eq!(keys.len(), 4);
        assert!(keys.contains(&SkinCacheKey::new(
            blend.space_id,
            RenderingContext::UserView,
            SkinCacheRendererKind::Static,
            blend.instance_id,
        )));
        assert!(keys.contains(&SkinCacheKey::new(
            skinned.space_id,
            RenderingContext::UserView,
            SkinCacheRendererKind::Skinned,
            skinned.instance_id,
        )));
        assert!(keys.contains(&SkinCacheKey::new(
            camera_skinned.space_id,
            RenderingContext::Camera,
            SkinCacheRendererKind::Skinned,
            camera_skinned.instance_id,
        )));
        assert!(keys.contains(&SkinCacheKey::new(
            overlay.space_id,
            RenderingContext::UserView,
            SkinCacheRendererKind::Static,
            overlay.instance_id,
        )));
    }
}
