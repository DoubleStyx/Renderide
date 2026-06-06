//! Backend-owned frame extraction helpers and read-only draw-preparation views.

use hashbrown::{HashMap, HashSet};

use crate::backend::FrameLightViewDesc;
use crate::gpu_pools::MeshPool;
use crate::materials::ShaderPermutation;
use crate::materials::host_data::MaterialPropertyStore;
use crate::materials::{MaterialPipelinePropertyIds, MaterialRouter};
use crate::reflection_probes::specular::ReflectionProbeFrameSelection;
use crate::scene::{RenderSpaceId, SceneApplyReport, SceneCacheFlushReport, SceneCoordinator};
use crate::shared::RenderingContext;
use crate::world_mesh::{
    FrameMaterialBatchCache, FramePreparedRenderables, RenderWorld, WorldMeshCommandCache,
    WorldMeshDrawCollectParallelism, WorldMeshDrawPlan,
};

use super::draw_preparation::{DrawPreparationExtractDesc, render_context_cache_key};
use super::{OcclusionSystem, RenderBackend};

/// Immutable backend-owned extraction snapshot produced by [`RenderBackend::extract_frame_shared`].
///
/// This is the runtime/backend hand-off for CPU-side world-mesh draw collection: the runtime owns
/// view planning while the backend owns material routing, resolved-material caching, prepared
/// renderables, and occlusion state.
pub(crate) struct ExtractedFrameShared<'a> {
    /// Scene after cache flush for world-matrix lookups and cull evaluation.
    pub(crate) scene: &'a SceneCoordinator,
    /// Mesh GPU asset pool queried for bounds and skinning metadata during draw collection.
    pub(crate) mesh_pool: &'a MeshPool,
    /// Property store backing [`crate::materials::host_data::MaterialDictionary::new`].
    pub(crate) property_store: &'a MaterialPropertyStore,
    /// Resolved raster pipeline selection for embedded materials.
    pub(crate) router: &'a MaterialRouter,
    /// Registry of renderer-side property ids used by the pipeline selector.
    pub(crate) pipeline_property_ids: MaterialPipelinePropertyIds,
    /// Prepared render-world caches for every render context used by this tick's views.
    pub(crate) render_worlds: &'a HashMap<u8, RenderWorld>,
    /// Persistent material batch caches keyed by render context and [`ShaderPermutation`].
    pub(crate) material_caches: &'a HashMap<(u8, ShaderPermutation), FrameMaterialBatchCache>,
    /// Persistent arranged draw command-list cache shared by per-view sorting.
    pub(crate) command_cache: &'a WorldMeshCommandCache,
    /// Shared occlusion state used for Hi-Z snapshots and temporal cull data.
    pub(crate) occlusion: &'a OcclusionSystem,
    /// CPU-side specular reflection-probe selector for per-object probe assignment.
    pub(crate) reflection_probes: &'a ReflectionProbeFrameSelection,
    /// Rayon parallelism tier for each view's inner walk.
    pub(crate) inner_parallelism: WorldMeshDrawCollectParallelism,
}

impl ExtractedFrameShared<'_> {
    /// Dense draw-prep snapshot matching `render_context`, if it was prepared for this frame.
    pub(crate) fn prepared_renderables_for(
        &self,
        render_context: RenderingContext,
    ) -> Option<&FramePreparedRenderables> {
        self.render_worlds
            .get(&render_context_cache_key(self.scene, render_context))
            .map(RenderWorld::prepared)
    }

    /// Material batch cache matching one view's render context and shader permutation.
    pub(crate) fn material_cache_for(
        &self,
        render_context: RenderingContext,
        shader_perm: ShaderPermutation,
    ) -> Option<&FrameMaterialBatchCache> {
        self.material_caches.get(&(
            render_context_cache_key(self.scene, render_context),
            shader_perm,
        ))
    }
}

impl RenderBackend {
    /// Applies scene mutation reports to backend-owned CPU render-world caches.
    pub(crate) fn note_scene_apply_report(&mut self, report: &SceneApplyReport) {
        self.draw_preparation.note_scene_apply_report(report);
        self.purge_closed_render_space_resources(&report.removed_spaces);
    }

    /// Applies world-cache flush reports to backend-owned CPU render-world caches.
    pub(crate) fn note_scene_cache_flush_report(&self, report: &SceneCacheFlushReport) {
        self.draw_preparation.note_scene_cache_flush_report(report);
    }

    fn purge_closed_render_space_resources(&mut self, removed_spaces: &[RenderSpaceId]) {
        if removed_spaces.is_empty() {
            return;
        }
        profiling::scope!("backend::purge_closed_render_space_resources");

        self.reflection_probes
            .purge_render_space_resources(removed_spaces);
        let retired_views = self.retire_views_for_render_spaces(removed_spaces);
        let skin_entries = self.frame_services.purge_skin_cache_spaces(removed_spaces);

        logger::info!(
            "world-close resource purge: spaces={} views={} skin_entries={}",
            removed_spaces.len(),
            retired_views,
            skin_entries
        );
    }

    fn retire_views_for_render_spaces(&mut self, spaces: &[RenderSpaceId]) -> usize {
        if spaces.is_empty() {
            return 0;
        }
        let removed_spaces: HashSet<RenderSpaceId> = spaces.iter().copied().collect();
        let retired = self.graph_state.retire_views_where(|view_id| {
            view_id
                .render_space_id()
                .is_some_and(|space_id| removed_spaces.contains(&space_id))
        });
        if retired.is_empty() {
            return 0;
        }
        logger::debug!(
            "retiring {} view-scoped resource sets for closed render spaces",
            retired.len()
        );
        self.world_mesh_frame_planner
            .release_view_resources(&retired);
        for &view_id in &retired {
            self.frame_services.frame_resources.retire_view(view_id);
            self.graph_state.history_registry_mut().retire_view(view_id);
            let _ = self.occlusion.retire_view(view_id);
        }
        retired.len()
    }

    /// Prepares clustered-light frame resources for the planned views in one graph submission.
    pub(crate) fn prepare_lights_for_views<I>(&mut self, scene: &SceneCoordinator, views: I)
    where
        I: IntoIterator<Item = FrameLightViewDesc>,
    {
        self.frame_services
            .frame_resources
            .prepare_lights_for_views(scene, views, Some(&self.asset_transfers));
    }

    /// Prepares realtime shadow assignments and atlas render views for the sorted view draw plans.
    pub(crate) fn prepare_shadow_frame_for_views<'a, I>(&mut self, views: I)
    where
        I: IntoIterator<Item = (crate::camera::ViewId, &'a WorldMeshDrawPlan)>,
    {
        self.frame_services
            .frame_resources
            .prepare_shadow_frame_for_views(self.shadow_quality(), views);
    }

    /// Drains completed Hi-Z readbacks into CPU snapshots at the top of the tick.
    pub(crate) fn hi_z_begin_frame_readback(&self, device: &wgpu::Device) {
        self.occlusion.hi_z_begin_frame_readback(device);
    }

    /// Refreshes backend-owned draw-prep state and returns the immutable frame setup used by the
    /// runtime's per-view draw collection stage.
    ///
    /// `view_draw_preparations` lists each prepared view's render context and shader permutation;
    /// one material batch cache is refreshed per distinct pair so multi-view frames (e.g. VR
    /// stereo + a secondary camera) do not pay an O(materials x pipeline_property_ids) walk per
    /// view.
    pub(crate) fn extract_frame_shared<'a>(
        &'a mut self,
        scene: &'a SceneCoordinator,
        inner_parallelism: WorldMeshDrawCollectParallelism,
        view_draw_preparations: &[(RenderingContext, ShaderPermutation)],
    ) -> ExtractedFrameShared<'a> {
        self.draw_preparation
            .extract_frame_shared(DrawPreparationExtractDesc {
                scene,
                materials: &self.materials,
                asset_transfers: &self.asset_transfers,
                occlusion: &self.occlusion,
                reflection_probes: self.reflection_probes.selection(),
                inner_parallelism,
                view_draw_preparations,
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::{SetRenderTextureFormat, SetTexture2DFormat};

    #[test]
    fn closed_render_space_cleanup_preserves_renderer_session_asset_catalogs() {
        let mut backend = RenderBackend::new();
        backend.asset_transfers.catalogs.texture_formats.insert(
            10,
            SetTexture2DFormat {
                asset_id: 10,
                ..Default::default()
            },
        );
        backend
            .asset_transfers
            .catalogs
            .render_texture_formats
            .insert(
                20,
                SetRenderTextureFormat {
                    asset_id: 20,
                    ..Default::default()
                },
            );
        let report = SceneApplyReport {
            removed_spaces: vec![RenderSpaceId(7)],
            ..Default::default()
        };

        backend.note_scene_apply_report(&report);

        assert!(
            backend
                .asset_transfers
                .catalogs
                .texture_formats
                .contains_key(&10)
        );
        assert!(
            backend
                .asset_transfers
                .catalogs
                .render_texture_formats
                .contains_key(&20)
        );
    }
}
