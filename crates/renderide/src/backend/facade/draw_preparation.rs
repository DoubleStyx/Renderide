//! CPU draw-preparation ownership behind the backend facade.

use hashbrown::HashMap;
use rayon::prelude::*;

use crate::materials::host_data::{MaterialDictionary, MaterialPropertyStore};
use crate::materials::{MaterialPipelinePropertyIds, MaterialRouter, RasterPipelineKind};
use crate::reflection_probes::specular::ReflectionProbeFrameSelection;
use crate::scene::{SceneApplyReport, SceneCacheFlushReport, SceneCoordinator};
use crate::shared::RenderingContext;
use crate::world_mesh::{FrameMaterialBatchCache, RenderWorld, RenderWorldMaintenanceStats};

use crate::backend::AssetTransferQueue;
use crate::materials::{MaterialSystem, ShaderPermutation};
use crate::occlusion::OcclusionSystem;

use super::frame_packet::ExtractedFrameShared;

/// Unique render contexts assigned to one render-world preparation worker.
const RENDER_WORLD_PREP_PARALLEL_CHUNK_CONTEXTS: usize = 1;
/// Unique render-context count required before render-world preparation fans out.
const RENDER_WORLD_PREP_PARALLEL_MIN_CONTEXTS: usize =
    RENDER_WORLD_PREP_PARALLEL_CHUNK_CONTEXTS * 2;
/// Unique material caches assigned to one cache-refresh worker.
const MATERIAL_CACHE_PREP_PARALLEL_CHUNK_CACHES: usize = 1;
/// Unique material-cache count required before cache refresh fans out.
const MATERIAL_CACHE_PREP_PARALLEL_MIN_CACHES: usize =
    MATERIAL_CACHE_PREP_PARALLEL_CHUNK_CACHES * 2;

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
    pub(super) fn note_scene_cache_flush_report(&self, report: &SceneCacheFlushReport) {
        for render_world in self.render_worlds.values() {
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
                asset_transfers.point_render_buffers(),
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

    /// Aggregated retained render-world maintenance counters for diagnostics.
    pub(super) fn render_world_maintenance_stats(&self) -> RenderWorldMaintenanceStats {
        let mut stats = RenderWorldMaintenanceStats::default();
        for render_world in self.render_worlds.values() {
            stats.accumulate(render_world.maintenance_stats());
        }
        stats
    }
}

/// Converts a render context into a compact cache-map key.
fn render_context_key(render_context: RenderingContext) -> u8 {
    render_context as u8
}

/// Refreshes every unique render-context cache required by this frame's views.
fn prepare_render_worlds_for_views(
    render_worlds: &mut HashMap<u8, RenderWorld>,
    scene: &SceneCoordinator,
    mesh_pool: &crate::gpu_pools::MeshPool,
    point_render_buffers: &HashMap<i32, crate::particles::PointRenderBufferAsset>,
    view_draw_preparations: &[(RenderingContext, ShaderPermutation)],
) {
    profiling::scope!("render::prepare_render_worlds_for_views");
    let mut work = unique_render_context_work(view_draw_preparations, render_worlds);
    if work.len() >= RENDER_WORLD_PREP_PARALLEL_MIN_CONTEXTS {
        profiling::scope!("render::prepare_render_worlds_for_views::parallel_contexts");
        work.par_iter_mut()
            .with_min_len(RENDER_WORLD_PREP_PARALLEL_CHUNK_CONTEXTS)
            .for_each(|(_, render_context, render_world)| {
                profiling::scope!("render::prepare_render_worlds_for_views::context_worker");
                render_world.prepare_for_frame(
                    scene,
                    mesh_pool,
                    point_render_buffers,
                    *render_context,
                );
            });
    } else {
        for (_, render_context, render_world) in &mut work {
            profiling::scope!("render::prepare_render_worlds_for_views::context");
            render_world.prepare_for_frame(scene, mesh_pool, point_render_buffers, *render_context);
        }
    }
    for (key, _, render_world) in work {
        render_worlds.insert(key, render_world);
    }
}

/// Removes unique render-world caches from the map for worker-owned preparation.
fn unique_render_context_work(
    view_draw_preparations: &[(RenderingContext, ShaderPermutation)],
    render_worlds: &mut HashMap<u8, RenderWorld>,
) -> Vec<(u8, RenderingContext, RenderWorld)> {
    let mut work = Vec::new();
    for (index, &(render_context, _)) in view_draw_preparations.iter().enumerate() {
        let key = render_context_key(render_context);
        if !is_first_context_request(view_draw_preparations, index, key) {
            continue;
        }
        let render_world = render_worlds
            .remove(&key)
            .unwrap_or_else(|| RenderWorld::new(render_context));
        work.push((key, render_context, render_world));
    }
    work
}

/// Refreshes material batch caches for every unique context and shader permutation.
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
    let mut work = unique_material_cache_work(view_draw_preparations, material_batch_caches);
    if work.len() >= MATERIAL_CACHE_PREP_PARALLEL_MIN_CACHES {
        profiling::scope!("render::build_frame_material_cache::parallel_caches");
        work.par_iter_mut()
            .with_min_len(MATERIAL_CACHE_PREP_PARALLEL_CHUNK_CACHES)
            .for_each(|(context_key, shader_perm, cache)| {
                profiling::scope!("render::build_frame_material_cache::cache_worker");
                if let Some(render_world) = render_worlds.get(context_key) {
                    cache.refresh_for_prepared(
                        render_world.prepared(),
                        &dict,
                        router,
                        pipeline_property_ids,
                        *shader_perm,
                    );
                }
            });
    } else {
        for (context_key, shader_perm, cache) in &mut work {
            profiling::scope!("render::build_frame_material_cache::cache");
            if let Some(render_world) = render_worlds.get(context_key) {
                cache.refresh_for_prepared(
                    render_world.prepared(),
                    &dict,
                    router,
                    pipeline_property_ids,
                    *shader_perm,
                );
            }
        }
    }
    for (context_key, shader_perm, cache) in work {
        material_batch_caches.insert((context_key, shader_perm), cache);
    }
}

/// Removes unique material caches from the map for worker-owned refresh.
fn unique_material_cache_work(
    view_draw_preparations: &[(RenderingContext, ShaderPermutation)],
    material_batch_caches: &mut HashMap<(u8, ShaderPermutation), FrameMaterialBatchCache>,
) -> Vec<(u8, ShaderPermutation, FrameMaterialBatchCache)> {
    let mut work = Vec::new();
    for (index, &(render_context, view_perm)) in view_draw_preparations.iter().enumerate() {
        let context_key = render_context_key(render_context);
        if is_first_context_request(view_draw_preparations, index, context_key) {
            let shader_perm = ShaderPermutation(0);
            let cache = material_batch_caches
                .remove(&(context_key, shader_perm))
                .unwrap_or_default();
            work.push((context_key, shader_perm, cache));
        }
        if view_perm != ShaderPermutation(0)
            && is_first_permutation_request(view_draw_preparations, index, context_key, view_perm)
        {
            let cache = material_batch_caches
                .remove(&(context_key, view_perm))
                .unwrap_or_default();
            work.push((context_key, view_perm, cache));
        }
    }
    work
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
        let point_render_buffers = HashMap::new();
        let views = [
            (RenderingContext::ExternalView, ShaderPermutation(1)),
            (RenderingContext::Camera, ShaderPermutation(0)),
            (RenderingContext::Camera, ShaderPermutation(0)),
        ];

        prepare_render_worlds_for_views(
            &mut render_worlds,
            &scene,
            &mesh_pool,
            &point_render_buffers,
            &views,
        );

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
