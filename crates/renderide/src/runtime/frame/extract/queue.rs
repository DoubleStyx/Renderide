//! Per-view world-mesh draw queueing for frame extraction.

use rayon::prelude::*;

use crate::backend::ExtractedFrameShared;
use crate::cpu_parallelism::{FrameCpuWorkload, FrameParallelPolicy};
use crate::world_mesh::{
    DrawCollectionFrameCaches, DrawCollectionInputs, DrawCollectionMaterialInputs,
    DrawCollectionSceneAssets, DrawCollectionViewInputs, PrefetchedWorldMeshViewDraws,
    QueuedWorldMeshDraws, ViewLayerPolicy, WorldMeshCullInput, WorldMeshCullProjParams,
    WorldMeshDrawCollectParallelism, WorldMeshDrawPlan, queue_draws_with_parallelism,
    queue_prepared_draws_for_views_with_parallelism,
};

use super::super::view_plan::FrameViewPlan;
use super::ViewWorldMeshDrawPlans;
use super::cull::ViewCullSnapshot;

/// Prepared view plans assigned to one draw-collection worker.
const VIEW_COLLECTION_PARALLEL_CHUNK_VIEWS: usize = 1;

/// Queued draw candidates and cull projection for one view.
pub(super) struct QueuedViewDraws {
    /// Regular camera-world draw candidates before final phase sorting.
    world: QueuedWorldMeshDraws,
    /// Desktop overlay draw candidates before final phase sorting.
    desktop_overlay: Option<QueuedWorldMeshDraws>,
    /// Projection parameters matching the view's camera/viewport.
    cull_proj: Option<WorldMeshCullProjParams>,
}

impl QueuedViewDraws {
    /// Number of queued draw candidates before final sorting and arrangement.
    pub(super) fn queued_draw_count(&self) -> usize {
        self.world.len()
            + self
                .desktop_overlay
                .as_ref()
                .map_or(0, QueuedWorldMeshDraws::len)
    }

    /// Sorts this view's queued draws and packages the final draw plan.
    pub(super) fn sort_and_package(
        self,
        parallelism: crate::world_mesh::WorldMeshDrawArrangeParallelism,
        command_cache: &crate::world_mesh::WorldMeshCommandCache,
    ) -> ViewWorldMeshDrawPlans {
        let world = sort_and_package_one_view_draw_plan(
            self.world,
            self.cull_proj.as_ref(),
            parallelism,
            command_cache,
        );
        let desktop_overlay = self.desktop_overlay.map(|queued| {
            sort_and_package_one_view_draw_plan(queued, None, parallelism, command_cache)
        });
        ViewWorldMeshDrawPlans {
            world,
            desktop_overlay,
        }
    }
}

fn sort_and_package_one_view_draw_plan(
    queued: QueuedWorldMeshDraws,
    cull_proj: Option<&WorldMeshCullProjParams>,
    parallelism: crate::world_mesh::WorldMeshDrawArrangeParallelism,
    command_cache: &crate::world_mesh::WorldMeshCommandCache,
) -> WorldMeshDrawPlan {
    let collection = queued.sort_and_arrange_with_cache(parallelism, Some(command_cache));
    WorldMeshDrawPlan::Prefetched(Box::new(PrefetchedWorldMeshViewDraws::new(
        collection, cull_proj,
    )))
}

/// Queues world-mesh draws for every prepared view in parallel.
///
/// Returns one queued draw packet per prepared view, preserving input order so the compiled graph
/// never has to infer whether draws were intentionally omitted or merely missing after sorting.
///
/// Takes ownership of `cull_snapshots` so each view moves its `hi_z` / `hi_z_temporal` payloads
/// (already `Arc`-shared internally) into the cull input instead of cloning, avoiding a per-view
/// refcount bump on the heaviest-view path.
pub(super) fn queue_view_draws(
    setup: &ExtractedFrameShared<'_>,
    prepared: &[FrameViewPlan<'_>],
    cull_snapshots: Vec<Option<ViewCullSnapshot>>,
    mesh_lod_bias: f32,
) -> Vec<QueuedViewDraws> {
    profiling::scope!("render::queue_view_draws");
    let dict = {
        profiling::scope!("queue::shared_dictionary");
        crate::materials::host_data::MaterialDictionary::new(setup.property_store)
    };
    let max_prepared_draw_count = max_prepared_draw_count_for_views(setup, prepared);
    let inner_parallelism = select_inner_parallelism_for_prepared_work(
        prepared.len(),
        max_prepared_draw_count,
        setup.inner_parallelism,
    );
    let parallelize_views =
        should_parallelize_view_collection(prepared.len(), max_prepared_draw_count);
    let (cull_inputs, cull_projs) = build_view_cull_inputs(prepared, cull_snapshots);
    let contexts = {
        profiling::scope!("render::queue_view_draws::build_contexts");
        prepared
            .iter()
            .zip(cull_inputs.iter())
            .map(|(prep, culling)| {
                let shader_perm = prep.shader_permutation();
                let render_context = prep.render_context();
                let material_cache = {
                    profiling::scope!("render::queue_view_draws::material_cache_lookup");
                    setup.material_cache_for(render_context, shader_perm)
                };
                DrawCollectionInputs {
                    scene_assets: DrawCollectionSceneAssets {
                        scene: setup.scene,
                        mesh_pool: setup.mesh_pool,
                    },
                    materials: DrawCollectionMaterialInputs {
                        dict: &dict,
                        router: setup.router,
                        pipeline_property_ids: &setup.pipeline_property_ids,
                        shader_perm,
                    },
                    view: DrawCollectionViewInputs {
                        render_context,
                        head_output_transform: prep.host_camera.head_output_transform,
                        view_origin_world: prep.view_origin_world(),
                        culling: culling.as_ref(),
                        mesh_lod_bias,
                        transform_filter: prep.draw_filter.as_ref(),
                        render_space_filter: prep.render_space_filter,
                        layer_policy: prep.layer_policy,
                        reflection_probes: Some(setup.reflection_probes),
                    },
                    caches: DrawCollectionFrameCaches {
                        material_cache,
                        prepared: setup.prepared_renderables_for(render_context),
                    },
                }
            })
            .collect::<Vec<_>>()
    };
    let mut view_draws = if let Some(queued) =
        queue_prepared_draws_for_views_with_parallelism(&contexts, inner_parallelism)
    {
        queued
            .into_iter()
            .zip(cull_projs)
            .map(|(world, cull_proj)| QueuedViewDraws {
                world,
                desktop_overlay: None,
                cull_proj,
            })
            .collect()
    } else {
        collect_view_draws_with_strategy(
            &contexts,
            &cull_projs,
            parallelize_views,
            inner_parallelism,
        )
    };
    queue_desktop_overlay_draws(
        setup,
        prepared,
        &dict,
        mesh_lod_bias,
        inner_parallelism,
        &mut view_draws,
    );
    view_draws
}

fn max_prepared_draw_count_for_views(
    setup: &ExtractedFrameShared<'_>,
    prepared: &[FrameViewPlan<'_>],
) -> usize {
    prepared
        .iter()
        .filter_map(|prep| setup.prepared_renderables_for(prep.render_context()))
        .map(crate::world_mesh::FramePreparedRenderables::len)
        .max()
        .unwrap_or(0)
}

fn build_view_cull_inputs<'a>(
    prepared: &'a [FrameViewPlan<'_>],
    cull_snapshots: Vec<Option<ViewCullSnapshot>>,
) -> (
    Vec<Option<WorldMeshCullInput<'a>>>,
    Vec<Option<WorldMeshCullProjParams>>,
) {
    profiling::scope!("render::queue_view_draws::build_cull_inputs");
    let mut cull_inputs = Vec::with_capacity(prepared.len());
    let mut cull_projs = Vec::with_capacity(prepared.len());
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
    (cull_inputs, cull_projs)
}

fn queue_desktop_overlay_draws(
    setup: &ExtractedFrameShared<'_>,
    prepared: &[FrameViewPlan<'_>],
    dict: &crate::materials::host_data::MaterialDictionary<'_>,
    mesh_lod_bias: f32,
    parallelism: WorldMeshDrawCollectParallelism,
    view_draws: &mut [QueuedViewDraws],
) {
    profiling::scope!("render::queue_view_draws::desktop_overlay");
    for (index, prep) in prepared.iter().enumerate() {
        if prep.desktop_overlay_resource_view_id().is_none() {
            continue;
        }
        let shader_perm = prep.shader_permutation();
        let render_context = prep.render_context();
        let material_cache = {
            profiling::scope!("render::queue_view_draws::desktop_overlay::material_cache_lookup");
            setup.material_cache_for(render_context, shader_perm)
        };
        let ctx = DrawCollectionInputs {
            scene_assets: DrawCollectionSceneAssets {
                scene: setup.scene,
                mesh_pool: setup.mesh_pool,
            },
            materials: DrawCollectionMaterialInputs {
                dict,
                router: setup.router,
                pipeline_property_ids: &setup.pipeline_property_ids,
                shader_perm,
            },
            view: desktop_overlay_view_inputs(prep, mesh_lod_bias),
            caches: DrawCollectionFrameCaches {
                material_cache,
                prepared: setup.prepared_renderables_for(render_context),
            },
        };
        if let Some(view_draws) = view_draws.get_mut(index) {
            view_draws.desktop_overlay = Some(queue_draws_with_parallelism(&ctx, parallelism));
        }
    }
}

/// Builds overlay draw-collection inputs for the desktop overlay pass.
pub(super) fn desktop_overlay_view_inputs<'a>(
    prep: &FrameViewPlan<'_>,
    mesh_lod_bias: f32,
) -> DrawCollectionViewInputs<'a> {
    DrawCollectionViewInputs {
        render_context: prep.render_context(),
        head_output_transform: prep.host_camera.head_output_transform,
        view_origin_world: prep.view_origin_world(),
        culling: None,
        mesh_lod_bias,
        transform_filter: None,
        render_space_filter: None,
        layer_policy: ViewLayerPolicy::DesktopOverlay,
        reflection_probes: None,
    }
}

/// Queues one view through the general draw-collection path.
fn collect_one_view_draws(
    ctx: &DrawCollectionInputs<'_>,
    cull_proj: Option<&WorldMeshCullProjParams>,
    parallelism: WorldMeshDrawCollectParallelism,
) -> QueuedViewDraws {
    profiling::scope!("render::queue_view_draws::queue_one");
    let queued = queue_draws_with_parallelism(ctx, parallelism);
    QueuedViewDraws {
        world: queued,
        desktop_overlay: None,
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
    contexts: &[DrawCollectionInputs<'_>],
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
pub(super) fn select_inner_parallelism_for_prepared_work_with_policy(
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

fn should_parallelize_view_collection(view_count: usize, max_prepared_draw_count: usize) -> bool {
    should_parallelize_view_collection_with_policy(
        FrameParallelPolicy::for_current_thread_pool(),
        view_count,
        max_prepared_draw_count,
    )
}

/// Policy-injected implementation for deterministic unit tests.
pub(super) fn should_parallelize_view_collection_with_policy(
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

#[cfg(test)]
mod tests {
    use hashbrown::HashMap;

    use crate::backend::ExtractedFrameShared;
    use crate::camera::{HostCameraFrame, ViewId};
    use crate::gpu_pools::MeshPool;
    use crate::materials::{MaterialRouter, MaterialSystem, RasterPipelineKind, ShaderPermutation};
    use crate::occlusion::OcclusionSystem;
    use crate::reflection_probes::specular::ReflectionProbeFrameSelection;
    use crate::render_graph::{FrameViewClear, RenderPathProfile};
    use crate::scene::{CameraRenderableEntry, RenderSpaceId, SceneCoordinator};
    use crate::shared::{CameraProjection, CameraState, RenderTransform, RenderingContext};
    use crate::world_mesh::{
        FrameMaterialBatchCache, RenderWorld, WorldMeshCommandCache,
        WorldMeshDrawCollectParallelism,
    };

    use super::super::super::view_plan::{FrameViewPlan, FrameViewPlanParams, FrameViewPlanTarget};
    use super::{QueuedViewDraws, queue_view_draws};

    fn dashboard_camera_entry(render_texture_asset_id: i32) -> CameraRenderableEntry {
        CameraRenderableEntry {
            renderable_index: 0,
            transform_id: 0,
            state: CameraState {
                projection: CameraProjection::Orthographic,
                render_texture_asset_id,
                selective_render_count: 1,
                flags: 1,
                ..Default::default()
            },
            selective_transform_ids: vec![0],
            exclude_transform_ids: Vec::new(),
        }
    }

    fn main_desktop_plan() -> FrameViewPlan<'static> {
        FrameViewPlan::new(
            &HostCameraFrame::default(),
            FrameViewPlanParams {
                render_context: RenderingContext::UserView,
                frame_time_seconds: 0.0,
                view_id: ViewId::Main,
                viewport_px: (1280, 720),
                clear: FrameViewClear::default(),
                profile: RenderPathProfile::desktop_main(),
                target: FrameViewPlanTarget::Swapchain,
            },
        )
    }

    #[test]
    fn desktop_overlay_queue_survives_dashboard_render_texture_camera() {
        let mut scene = SceneCoordinator::new();
        let overlay_space = RenderSpaceId(3);
        scene.test_seed_space_identity_worlds(
            overlay_space,
            vec![RenderTransform::default()],
            vec![-1],
        );
        scene.test_set_space_overlay(overlay_space, true);
        scene.test_push_cameras(overlay_space, [dashboard_camera_entry(77)]);

        let mesh_pool = MeshPool::default_pool();
        let materials = MaterialSystem::new();
        let router = MaterialRouter::new(RasterPipelineKind::Null);
        let render_worlds = HashMap::<u8, RenderWorld>::new();
        let material_caches = HashMap::<(u8, ShaderPermutation), FrameMaterialBatchCache>::new();
        let command_cache = WorldMeshCommandCache::default();
        let occlusion = OcclusionSystem::new();
        let reflection_probes = ReflectionProbeFrameSelection::default();
        let pipeline_property_ids = materials.pipeline_property_resolver().resolve();
        let shared = ExtractedFrameShared {
            scene: &scene,
            mesh_pool: &mesh_pool,
            property_store: materials.material_property_store(),
            router: &router,
            pipeline_property_ids,
            render_worlds: &render_worlds,
            material_caches: &material_caches,
            command_cache: &command_cache,
            occlusion: &occlusion,
            reflection_probes: &reflection_probes,
            inner_parallelism: WorldMeshDrawCollectParallelism::Full,
        };
        let prepared = [main_desktop_plan()];

        let draws = queue_view_draws(&shared, &prepared, vec![None], 1.0);

        assert_eq!(draws.len(), 1);
        assert!(overlay_plan_present(&draws));
    }

    fn overlay_plan_present(draws: &[QueuedViewDraws]) -> bool {
        draws
            .first()
            .is_some_and(|draws| draws.desktop_overlay.is_some())
    }
}
