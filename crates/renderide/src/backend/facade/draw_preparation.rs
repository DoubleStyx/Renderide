//! CPU draw-preparation ownership behind the backend facade.

use hashbrown::{HashMap, HashSet};
use rayon::prelude::*;

use crate::cpu_parallelism::{FrameCpuWorkload, FrameParallelPolicy, record_parallel_admission};
use crate::gpu_pools::MeshPool;
use crate::materials::host_data::{MaterialDictionary, MaterialPropertyStore};
use crate::materials::{MaterialPipelinePropertyIds, MaterialRouter, RasterPipelineKind};
use crate::reflection_probes::specular::ReflectionProbeFrameSelection;
use crate::scene::{
    SceneApplyReport, SceneCacheFlushReport, SceneCoordinator, SceneSpaceRead, WorldMeshSceneRead,
};
use crate::shared::RenderingContext;
use crate::world_mesh::{
    FrameMaterialBatchCache, RenderWorld, RenderWorldMaintenanceStats, WorldMeshCommandCache,
    WorldMeshCommandCacheStats,
};

use crate::backend::AssetTransferQueue;
use crate::materials::{MaterialSystem, ShaderPermutation};
use crate::occlusion::OcclusionSystem;

use super::frame_packet::ExtractedFrameShared;

/// Asset tables needed by CPU draw-preparation.
pub(super) trait DrawPrepAssetRead {
    /// Resident mesh pool used for bounds, submeshes, and deform metadata.
    fn mesh_pool(&self) -> &MeshPool;
    /// CPU point render-buffer assets expanded into prepared particle draws.
    fn point_render_buffers(&self) -> &HashMap<i32, crate::particles::PointRenderBufferAsset>;
}

impl DrawPrepAssetRead for AssetTransferQueue {
    fn mesh_pool(&self) -> &MeshPool {
        AssetTransferQueue::mesh_pool(self)
    }

    fn point_render_buffers(&self) -> &HashMap<i32, crate::particles::PointRenderBufferAsset> {
        AssetTransferQueue::point_render_buffers(self)
    }
}

/// Unique render contexts assigned to one render-world preparation worker.
const RENDER_WORLD_PREP_PARALLEL_CHUNK_CONTEXTS: usize = 1;
/// Unique material caches assigned to one cache-refresh worker.
const MATERIAL_CACHE_PREP_PARALLEL_CHUNK_CACHES: usize = 1;
/// Shared cache key for render contexts with no draw-prep overrides.
const CONTEXT_INVARIANT_RENDER_WORLD_KEY: u8 = u8::MAX;

/// Inputs for one backend draw-preparation extraction.
pub(super) struct DrawPreparationExtractDesc<'a, 'v> {
    /// Scene after cache flush for world-matrix lookups and cull evaluation.
    pub(super) scene: &'a SceneCoordinator,
    /// Material registry, routes, and property data.
    pub(super) materials: &'a MaterialSystem,
    /// Asset tables and resident GPU pools needed for draw preparation.
    pub(super) assets: &'a dyn DrawPrepAssetRead,
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
    /// Retained arranged draw command lists keyed by visible draw fingerprints.
    command_cache: WorldMeshCommandCache,
}

impl BackendDrawPreparation {
    /// Creates empty draw-preparation caches.
    pub(super) fn new() -> Self {
        Self {
            null_material_router: MaterialRouter::new(RasterPipelineKind::Null),
            material_batch_caches: HashMap::new(),
            render_worlds: HashMap::new(),
            command_cache: WorldMeshCommandCache::default(),
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
            assets,
            occlusion,
            reflection_probes,
            inner_parallelism,
            view_draw_preparations,
        } = desc;
        let Self {
            null_material_router,
            material_batch_caches,
            render_worlds,
            command_cache,
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
                assets.mesh_pool(),
                assets.point_render_buffers(),
                view_draw_preparations,
            );
        }

        refresh_material_caches(
            material_batch_caches,
            render_worlds,
            scene,
            property_store,
            router,
            &pipeline_property_ids,
            view_draw_preparations,
        );

        ExtractedFrameShared {
            scene,
            mesh_pool: assets.mesh_pool(),
            property_store,
            router,
            pipeline_property_ids,
            render_worlds,
            material_caches: material_batch_caches,
            command_cache,
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

    /// Retained draw command-list cache counters for diagnostics.
    pub(super) fn command_cache_stats(&self) -> WorldMeshCommandCacheStats {
        self.command_cache.stats()
    }
}

/// Converts a render context into a compact cache-map key.
pub(super) fn render_context_cache_key(
    scene: &(impl SceneSpaceRead + ?Sized),
    render_context: RenderingContext,
) -> u8 {
    if scene.render_context_affects_draw_prep(render_context) {
        render_context as u8
    } else {
        CONTEXT_INVARIANT_RENDER_WORLD_KEY
    }
}

/// Refreshes every unique render-context cache required by this frame's views.
fn prepare_render_worlds_for_views<S>(
    render_worlds: &mut HashMap<u8, RenderWorld>,
    scene: &S,
    mesh_pool: &MeshPool,
    point_render_buffers: &HashMap<i32, crate::particles::PointRenderBufferAsset>,
    view_draw_preparations: &[(RenderingContext, ShaderPermutation)],
) where
    S: WorldMeshSceneRead + Sync + ?Sized,
{
    profiling::scope!("render::prepare_render_worlds_for_views");
    let mut work = unique_render_context_work(scene, view_draw_preparations, render_worlds);
    let admission = FrameParallelPolicy::for_current_thread_pool().admit_independent_items(
        FrameCpuWorkload::independent_items(work.len()),
        RENDER_WORLD_PREP_PARALLEL_CHUNK_CONTEXTS,
    );
    record_parallel_admission(
        "render_world_prepare_contexts",
        work.len(),
        work.len(),
        admission,
    );
    if admission.is_parallel() {
        profiling::scope!("render::prepare_render_worlds_for_views::parallel_contexts");
        work.par_iter_mut()
            .with_min_len(admission.chunk_size().unwrap_or(1))
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
    scene: &(impl SceneSpaceRead + ?Sized),
    view_draw_preparations: &[(RenderingContext, ShaderPermutation)],
    render_worlds: &mut HashMap<u8, RenderWorld>,
) -> Vec<(u8, RenderingContext, RenderWorld)> {
    let mut work = Vec::new();
    let mut seen_contexts = HashSet::with_capacity(view_draw_preparations.len());
    for &(render_context, _) in view_draw_preparations {
        let key = render_context_cache_key(scene, render_context);
        if !seen_contexts.insert(key) {
            continue;
        }
        let render_world = render_worlds.remove(&key).unwrap_or_else(|| {
            if key == CONTEXT_INVARIANT_RENDER_WORLD_KEY {
                RenderWorld::new_context_invariant(render_context)
            } else {
                RenderWorld::new(render_context)
            }
        });
        work.push((key, render_context, render_world));
    }
    work
}

/// Refreshes material batch caches for every unique context and shader permutation.
fn refresh_material_caches(
    material_batch_caches: &mut HashMap<(u8, ShaderPermutation), FrameMaterialBatchCache>,
    render_worlds: &HashMap<u8, RenderWorld>,
    scene: &(impl SceneSpaceRead + ?Sized),
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
    let mut work = unique_material_cache_work(scene, view_draw_preparations, material_batch_caches);
    let admission = FrameParallelPolicy::for_current_thread_pool().admit_independent_items(
        FrameCpuWorkload::independent_items(work.len()),
        MATERIAL_CACHE_PREP_PARALLEL_CHUNK_CACHES,
    );
    record_parallel_admission(
        "render_world_material_caches",
        work.len(),
        work.len(),
        admission,
    );
    if admission.is_parallel() {
        profiling::scope!("render::build_frame_material_cache::parallel_caches");
        work.par_iter_mut()
            .with_min_len(admission.chunk_size().unwrap_or(1))
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
    scene: &(impl SceneSpaceRead + ?Sized),
    view_draw_preparations: &[(RenderingContext, ShaderPermutation)],
    material_batch_caches: &mut HashMap<(u8, ShaderPermutation), FrameMaterialBatchCache>,
) -> Vec<(u8, ShaderPermutation, FrameMaterialBatchCache)> {
    let mut work = Vec::new();
    let mut seen_contexts = HashSet::with_capacity(view_draw_preparations.len());
    let mut seen_permutations = HashSet::with_capacity(view_draw_preparations.len() * 2);
    for &(render_context, view_perm) in view_draw_preparations {
        let context_key = render_context_cache_key(scene, render_context);
        if seen_contexts.insert(context_key) {
            let shader_perm = ShaderPermutation(0);
            let cache = material_batch_caches
                .remove(&(context_key, shader_perm))
                .unwrap_or_default();
            work.push((context_key, shader_perm, cache));
        }
        if view_perm != ShaderPermutation(0) && seen_permutations.insert((context_key, view_perm)) {
            let cache = material_batch_caches
                .remove(&(context_key, view_perm))
                .unwrap_or_default();
            work.push((context_key, view_perm, cache));
        }
    }
    work
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_pools::MeshPool;
    use crate::scene::RenderSpaceId;
    use crate::shared::RenderTransform;

    #[test]
    fn prepare_render_worlds_shares_cache_for_contexts_without_overrides() {
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

        assert_eq!(render_worlds.len(), 1);
        assert_eq!(
            render_worlds
                .get(&CONTEXT_INVARIANT_RENDER_WORLD_KEY)
                .map(|world| world.prepared().render_context()),
            Some(RenderingContext::ExternalView)
        );
        assert!(
            render_worlds
                .get(&CONTEXT_INVARIANT_RENDER_WORLD_KEY)
                .is_some_and(|world| world
                    .prepared()
                    .is_compatible_with_render_context(RenderingContext::Camera))
        );
    }

    #[test]
    fn prepare_render_worlds_keeps_exact_cache_for_context_with_overrides() {
        let mut render_worlds = HashMap::new();
        let mut scene = SceneCoordinator::new();
        scene.test_seed_space_identity_worlds(
            RenderSpaceId(1),
            vec![RenderTransform::default()],
            vec![-1],
        );
        scene.test_push_scale_render_transform_override(
            RenderSpaceId(1),
            0,
            RenderingContext::Camera,
            glam::Vec3::splat(2.0),
        );
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
        assert!(render_worlds.contains_key(&CONTEXT_INVARIANT_RENDER_WORLD_KEY));
        assert_eq!(
            render_worlds
                .get(&render_context_cache_key(&scene, RenderingContext::Camera))
                .map(|world| world.prepared().render_context()),
            Some(RenderingContext::Camera)
        );
    }
}
