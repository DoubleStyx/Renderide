//! CPU draw-preparation ownership behind the backend facade.

use hashbrown::HashMap;

use crate::materials::host_data::{MaterialDictionary, MaterialPropertyStore};
use crate::materials::{MaterialPipelinePropertyIds, MaterialRouter, RasterPipelineKind};
use crate::reflection_probes::specular::ReflectionProbeFrameSelection;
use crate::scene::{SceneApplyReport, SceneCacheFlushReport, SceneCoordinator};
use crate::shared::RenderingContext;
use crate::world_mesh::{FrameMaterialBatchCache, RenderWorld};

use crate::backend::AssetTransferQueue;
use crate::materials::{MaterialSystem, ShaderPermutation};
use crate::occlusion::OcclusionSystem;

use super::frame_packet::ExtractedFrameShared;

/// Inputs for one backend draw-preparation extraction.
pub(super) struct DrawPreparationExtractDesc<'a, 'v> {
    /// Scene after cache flush for world-matrix lookups and cull evaluation.
    pub(super) scene: &'a SceneCoordinator,
    /// Material registry, routes, and property data.
    pub(super) materials: &'a MaterialSystem,
    /// Asset upload queues and resident GPU pools.
    pub(super) asset_transfers: &'a AssetTransferQueue,
    /// Shared occlusion state used for Hi-Z snapshots and temporal cull data.
    pub(super) occlusion: &'a OcclusionSystem,
    /// CPU-side specular reflection-probe selector for per-object probe assignment.
    pub(super) reflection_probes: &'a ReflectionProbeFrameSelection,
    /// Rayon parallelism tier for each view's inner walk.
    pub(super) inner_parallelism: crate::world_mesh::WorldMeshDrawCollectParallelism,
    /// Render context and shader permutation used by each prepared view this tick.
    pub(super) view_draw_preparations: &'v [(RenderingContext, ShaderPermutation)],
}

/// Backend-owned CPU draw-preparation caches.
pub(super) struct BackendDrawPreparation {
    /// Fallback router used before any embedded-material registry is available.
    null_material_router: MaterialRouter,
    /// Persistent resolved-material caches keyed by render context and shader permutation.
    material_batch_caches: HashMap<(u8, ShaderPermutation), FrameMaterialBatchCache>,
    /// Backend-owned CPU render-world caches used to amortize draw preparation per context.
    render_worlds: HashMap<u8, RenderWorld>,
}

impl BackendDrawPreparation {
    /// Creates empty draw-preparation caches.
    pub(super) fn new() -> Self {
        Self {
            null_material_router: MaterialRouter::new(RasterPipelineKind::Null),
            material_batch_caches: HashMap::new(),
            render_worlds: HashMap::new(),
        }
    }

    /// Applies scene mutation reports to backend-owned CPU render-world caches.
    pub(super) fn note_scene_apply_report(&mut self, report: &SceneApplyReport) {
        for render_world in self.render_worlds.values_mut() {
            render_world.note_scene_apply_report(report);
        }
    }

    /// Applies world-cache flush reports to backend-owned CPU render-world caches.
    pub(super) fn note_scene_cache_flush_report(&mut self, report: &SceneCacheFlushReport) {
        for render_world in self.render_worlds.values_mut() {
            render_world.note_cache_flush_report(report);
        }
    }

    /// Refreshes backend-owned draw-prep state and returns the immutable frame setup.
    pub(super) fn extract_frame_shared<'a>(
        &'a mut self,
        desc: DrawPreparationExtractDesc<'a, '_>,
    ) -> ExtractedFrameShared<'a> {
        let DrawPreparationExtractDesc {
            scene,
            materials,
            asset_transfers,
            occlusion,
            reflection_probes,
            inner_parallelism,
            view_draw_preparations,
        } = desc;
        let Self {
            null_material_router,
            material_batch_caches,
            render_worlds,
        } = self;
        let (property_store, router, pipeline_property_ids) = {
            profiling::scope!("render::extract_frame_shared::material_inputs");
            let property_store = materials.material_property_store();
            let router = materials
                .material_registry()
                .map_or(&*null_material_router, |registry| registry.router());
            let pipeline_property_ids = materials.pipeline_property_resolver().resolve();
            (property_store, router, pipeline_property_ids)
        };
        {
            profiling::scope!("render::build_frame_prepared_renderables");
            prepare_render_worlds_for_views(
                render_worlds,
                scene,
                asset_transfers.mesh_pool(),
                view_draw_preparations,
            );
        }

        refresh_material_caches(
            material_batch_caches,
            render_worlds,
            property_store,
            router,
            &pipeline_property_ids,
            view_draw_preparations,
        );

        ExtractedFrameShared {
            scene,
            mesh_pool: asset_transfers.mesh_pool(),
            property_store,
            router,
            pipeline_property_ids,
            render_worlds,
            material_caches: material_batch_caches,
            occlusion,
            reflection_probes,
            inner_parallelism,
        }
    }
}

fn render_context_key(render_context: RenderingContext) -> u8 {
    render_context as u8
}

fn prepare_render_worlds_for_views(
    render_worlds: &mut HashMap<u8, RenderWorld>,
    scene: &SceneCoordinator,
    mesh_pool: &crate::gpu_pools::MeshPool,
    view_draw_preparations: &[(RenderingContext, ShaderPermutation)],
) {
    profiling::scope!("render::prepare_render_worlds_for_views");
    for (index, &(render_context, _)) in view_draw_preparations.iter().enumerate() {
        let key = render_context_key(render_context);
        if view_draw_preparations[..index]
            .iter()
            .any(|&(previous_context, _)| render_context_key(previous_context) == key)
        {
            continue;
        }
        {
            profiling::scope!("render::prepare_render_worlds_for_views::context");
            render_worlds
                .entry(key)
                .or_insert_with(|| RenderWorld::new(render_context))
                .prepare_for_frame(scene, mesh_pool, render_context);
        }
    }
}

fn refresh_material_caches(
    material_batch_caches: &mut HashMap<(u8, ShaderPermutation), FrameMaterialBatchCache>,
    render_worlds: &HashMap<u8, RenderWorld>,
    property_store: &MaterialPropertyStore,
    router: &MaterialRouter,
    pipeline_property_ids: &MaterialPipelinePropertyIds,
    view_draw_preparations: &[(RenderingContext, ShaderPermutation)],
) {
    profiling::scope!("render::build_frame_material_cache");
    let dict = {
        profiling::scope!("render::build_frame_material_cache::dictionary");
        MaterialDictionary::new(property_store)
    };
    for (index, &(render_context, view_perm)) in view_draw_preparations.iter().enumerate() {
        let context_key = render_context_key(render_context);
        let Some(render_world) = render_worlds.get(&context_key) else {
            continue;
        };
        if is_first_context_request(view_draw_preparations, index, context_key) {
            {
                profiling::scope!("render::build_frame_material_cache::default_permutation");
                refresh_material_cache(
                    material_batch_caches,
                    render_world,
                    &dict,
                    router,
                    pipeline_property_ids,
                    context_key,
                    ShaderPermutation(0),
                );
            }
        }
        if view_perm != ShaderPermutation(0)
            && is_first_permutation_request(view_draw_preparations, index, context_key, view_perm)
        {
            {
                profiling::scope!("render::build_frame_material_cache::view_permutation");
                refresh_material_cache(
                    material_batch_caches,
                    render_world,
                    &dict,
                    router,
                    pipeline_property_ids,
                    context_key,
                    view_perm,
                );
            }
        }
    }
}

fn refresh_material_cache(
    material_batch_caches: &mut HashMap<(u8, ShaderPermutation), FrameMaterialBatchCache>,
    render_world: &RenderWorld,
    dict: &MaterialDictionary<'_>,
    router: &MaterialRouter,
    pipeline_property_ids: &MaterialPipelinePropertyIds,
    context_key: u8,
    shader_perm: ShaderPermutation,
) {
    profiling::scope!("render::refresh_material_cache");
    material_batch_caches
        .entry((context_key, shader_perm))
        .or_default()
        .refresh_for_prepared(
            render_world.prepared(),
            dict,
            router,
            pipeline_property_ids,
            shader_perm,
        );
}

fn is_first_context_request(
    view_draw_preparations: &[(RenderingContext, ShaderPermutation)],
    index: usize,
    context_key: u8,
) -> bool {
    !view_draw_preparations[..index]
        .iter()
        .any(|&(previous_context, _)| render_context_key(previous_context) == context_key)
}

fn is_first_permutation_request(
    view_draw_preparations: &[(RenderingContext, ShaderPermutation)],
    index: usize,
    context_key: u8,
    shader_perm: ShaderPermutation,
) -> bool {
    !view_draw_preparations[..index]
        .iter()
        .any(|&(previous_context, previous_perm)| {
            render_context_key(previous_context) == context_key && previous_perm == shader_perm
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_pools::MeshPool;

    #[test]
    fn prepare_render_worlds_keeps_one_cache_per_render_context() {
        let mut render_worlds = HashMap::new();
        let scene = SceneCoordinator::new();
        let mesh_pool = MeshPool::default_pool();
        let views = [
            (RenderingContext::ExternalView, ShaderPermutation(1)),
            (RenderingContext::Camera, ShaderPermutation(0)),
            (RenderingContext::Camera, ShaderPermutation(0)),
        ];

        prepare_render_worlds_for_views(&mut render_worlds, &scene, &mesh_pool, &views);

        assert_eq!(render_worlds.len(), 2);
        assert_eq!(
            render_worlds
                .get(&render_context_key(RenderingContext::ExternalView))
                .map(|world| world.prepared().render_context()),
            Some(RenderingContext::ExternalView)
        );
        assert_eq!(
            render_worlds
                .get(&render_context_key(RenderingContext::Camera))
                .map(|world| world.prepared().render_context()),
            Some(RenderingContext::Camera)
        );
    }
}
