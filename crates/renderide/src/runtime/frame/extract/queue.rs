//! Per-view world-mesh draw queueing for frame extraction.

use crate::backend::ExtractedFrameShared;
use crate::cpu_parallelism::FrameParallelPolicy;
use crate::world_mesh::{
    DrawCollectionFrameCaches, DrawCollectionInputs, DrawCollectionMaterialInputs,
    DrawCollectionSceneAssets, DrawCollectionViewInputs, PrefetchedWorldMeshViewDraws,
    QueuedWorldMeshDraws, ViewLayerPolicy, ViewRenderSpaceScope, WorldMeshCullInput,
    WorldMeshCullProjParams, WorldMeshDrawCollectParallelism, WorldMeshDrawPlan,
    queue_draws_with_parallelism, queue_prepared_draws_for_views_with_parallelism,
};

use super::super::view_plan::FrameViewPlan;
use super::ViewWorldMeshDrawPlans;
use super::cull::ViewCullSnapshot;

/// Queued draw candidates and cull projection for one view.
pub(super) struct QueuedViewDraws {
    /// Regular camera-world draw candidates before final phase sorting.
    world: QueuedWorldMeshDraws,
    /// Shadow-caster draw candidates before final phase sorting.
    shadow_casters: QueuedWorldMeshDraws,
    /// Desktop overlay draw candidates before final phase sorting.
    desktop_overlay: Option<QueuedWorldMeshDraws>,
    /// Projection parameters matching the view's camera/viewport.
    cull_proj: Option<WorldMeshCullProjParams>,
}

impl QueuedViewDraws {
    /// Number of queued draw candidates before final sorting and arrangement.
    pub(super) fn queued_draw_count(&self) -> usize {
        self.world.len()
            + self.shadow_casters.len()
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
        let shadow_casters = sort_and_package_one_view_draw_plan(
            self.shadow_casters,
            None,
            parallelism,
            command_cache,
        );
        ViewWorldMeshDrawPlans {
            world,
            shadow_casters,
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
    let (cull_inputs, cull_projs) = build_view_cull_inputs(prepared, cull_snapshots);
    let contexts =
        build_view_draw_collection_contexts(setup, prepared, &dict, &cull_inputs, mesh_lod_bias);
    let shadow_contexts =
        build_shadow_caster_draw_collection_contexts(setup, prepared, &dict, mesh_lod_bias);
    let world_draws = queue_prepared_draws_for_views_with_parallelism(&contexts, inner_parallelism);
    let shadow_caster_draws =
        queue_prepared_draws_for_views_with_parallelism(&shadow_contexts, inner_parallelism);
    let mut view_draws: Vec<QueuedViewDraws> = world_draws
        .into_iter()
        .zip(shadow_caster_draws)
        .zip(cull_projs)
        .map(|((world, shadow_casters), cull_proj)| QueuedViewDraws {
            world,
            shadow_casters,
            desktop_overlay: None,
            cull_proj,
        })
        .collect();
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

fn build_view_draw_collection_contexts<'a>(
    setup: &'a ExtractedFrameShared<'a>,
    prepared: &'a [FrameViewPlan<'_>],
    dict: &'a crate::materials::host_data::MaterialDictionary<'a>,
    cull_inputs: &'a [Option<WorldMeshCullInput<'a>>],
    mesh_lod_bias: f32,
) -> Vec<DrawCollectionInputs<'a>> {
    profiling::scope!("render::queue_view_draws::build_contexts");
    prepared
        .iter()
        .zip(cull_inputs.iter())
        .map(|(prep, culling)| {
            build_draw_collection_context(
                setup,
                dict,
                prep,
                visible_view_inputs(setup, prep, culling.as_ref(), mesh_lod_bias),
            )
        })
        .collect()
}

fn build_shadow_caster_draw_collection_contexts<'a>(
    setup: &'a ExtractedFrameShared<'a>,
    prepared: &'a [FrameViewPlan<'_>],
    dict: &'a crate::materials::host_data::MaterialDictionary<'a>,
    mesh_lod_bias: f32,
) -> Vec<DrawCollectionInputs<'a>> {
    profiling::scope!("render::queue_view_draws::build_shadow_contexts");
    prepared
        .iter()
        .map(|prep| {
            build_draw_collection_context(
                setup,
                dict,
                prep,
                shadow_caster_view_inputs(prep, mesh_lod_bias),
            )
        })
        .collect()
}

fn build_draw_collection_context<'a>(
    setup: &'a ExtractedFrameShared<'a>,
    dict: &'a crate::materials::host_data::MaterialDictionary<'a>,
    prep: &'a FrameViewPlan<'_>,
    view: DrawCollectionViewInputs<'a>,
) -> DrawCollectionInputs<'a> {
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
            dict,
            router: setup.router,
            pipeline_property_ids: &setup.pipeline_property_ids,
            shader_perm,
        },
        view,
        caches: DrawCollectionFrameCaches {
            material_cache,
            prepared: setup.prepared_renderables_for(render_context),
        },
    }
}

fn visible_view_inputs<'a>(
    setup: &'a ExtractedFrameShared<'a>,
    prep: &'a FrameViewPlan<'_>,
    culling: Option<&'a WorldMeshCullInput<'a>>,
    mesh_lod_bias: f32,
) -> DrawCollectionViewInputs<'a> {
    DrawCollectionViewInputs {
        render_context: prep.render_context(),
        head_output_transform: prep.host_camera.head_output_transform,
        view_origin_world: prep.view_origin_world(),
        culling,
        mesh_lod_bias,
        transform_filter: prep.draw_filter.as_ref(),
        transform_filter_space: prep.transform_filter_space,
        render_space_scope: prep.render_space_scope,
        layer_policy: prep.layer_policy,
        reflection_probes: Some(setup.reflection_probes),
    }
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
        transform_filter_space: None,
        render_space_scope: ViewRenderSpaceScope::AllActive,
        layer_policy: ViewLayerPolicy::DesktopOverlay,
        reflection_probes: None,
    }
}

/// Builds shadow-caster draw-collection inputs for a view without camera visibility culling.
pub(super) fn shadow_caster_view_inputs<'a>(
    prep: &'a FrameViewPlan<'_>,
    mesh_lod_bias: f32,
) -> DrawCollectionViewInputs<'a> {
    DrawCollectionViewInputs {
        render_context: prep.render_context(),
        head_output_transform: prep.host_camera.head_output_transform,
        view_origin_world: prep.view_origin_world(),
        culling: None,
        mesh_lod_bias,
        transform_filter: prep.draw_filter.as_ref(),
        transform_filter_space: prep.transform_filter_space,
        render_space_scope: prep.render_space_scope,
        layer_policy: prep.layer_policy,
        reflection_probes: None,
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
