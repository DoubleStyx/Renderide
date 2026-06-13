//! [`GraphFrameResources`] trait impl for [`FrameResourceManager`].

use std::sync::Arc;

use hashbrown::HashSet;

use crate::backend::frame_gpu::{LIGHT_COOKIE_ATLAS_PASS_NAME, SHADOW_ATLAS_PASS_NAME};
use crate::camera::ViewId;
use crate::gpu::frame_globals::SkyboxSpecularUniformParams;
use crate::graph_inputs::PreRecordViewResourceLayout;
use crate::mesh_deform::{PaddedPerDrawUniforms, SkinCacheKey};
use crate::passes::MaterialBatchBoundary;
use crate::render_graph::execution_backend::{
    GraphAssetResources, GraphClusterBufferRefs, GraphFrameResources, ShadowAtlasEncodeParams,
};
use crate::render_graph::frame_upload_batch::GraphUploadSink;

use super::manager::FrameResourceManager;

impl GraphFrameResources for FrameResourceManager {
    fn has_frame_gpu(&self) -> bool {
        self.frame_gpu().is_some()
    }

    fn frame_light_count_u32(&self, view_id: ViewId) -> u32 {
        self.frame_light_count_for_view_u32(view_id)
    }

    fn lights_buffer(&self, view_id: ViewId) -> Option<wgpu::Buffer> {
        self.per_view_frame(view_id)
            .map(|state| state.lights_buffer.clone())
    }

    fn frame_uniform_buffer(&self) -> Option<wgpu::Buffer> {
        self.frame_gpu().map(|fgpu| fgpu.frame_uniform.clone())
    }

    fn shared_cluster_buffer_refs(&self) -> Option<GraphClusterBufferRefs> {
        self.shared_cluster_buffer_refs()
            .map(|refs| GraphClusterBufferRefs {
                cluster_light_counts: refs.cluster_light_counts.clone(),
                cluster_light_indices: refs.cluster_light_indices.clone(),
            })
    }

    fn shared_cluster_version(&self) -> u64 {
        self.shared_cluster_version()
    }

    fn per_view_cluster_params_buffer(&self, view_id: ViewId) -> Option<wgpu::Buffer> {
        self.per_view_frame(view_id)
            .map(|state| state.cluster_params_buffer.clone())
    }

    fn per_view_frame_bind_group_and_buffer(
        &self,
        view_id: ViewId,
    ) -> Option<(Arc<wgpu::BindGroup>, wgpu::Buffer)> {
        self.per_view_frame(view_id).map(|state| {
            (
                Arc::clone(&state.frame_bind_group),
                state.frame_uniform_buffer.clone(),
            )
        })
    }

    fn per_view_named_scene_color_frame_bind_group(
        &self,
        view_id: ViewId,
    ) -> Option<Arc<wgpu::BindGroup>> {
        self.per_view_frame(view_id)
            .map(|state| Arc::clone(&state.named_scene_color_frame_bind_group))
    }

    fn ensure_per_view_per_draw_capacity(
        &self,
        device: &wgpu::Device,
        view_id: ViewId,
        draw_count: usize,
    ) -> Option<wgpu::Buffer> {
        let per_draw_slot = self.per_view_per_draw(view_id)?;
        let mut per_draw = per_draw_slot.lock();
        let _ = per_draw.ensure_draw_slot_capacity(device, draw_count);
        Some(per_draw.per_draw_storage.clone())
    }

    fn with_per_view_per_draw_scratch(
        &self,
        view_id: ViewId,
        f: &mut dyn FnMut(&mut Vec<PaddedPerDrawUniforms>),
    ) -> bool {
        let Some(scratch_slot) = self.per_view_per_draw_scratch(view_id) else {
            return false;
        };
        let mut scratch_guard = scratch_slot.lock();
        let scratch = &mut *scratch_guard;
        let uniforms = &mut scratch.uniforms;
        f(uniforms);
        drop(scratch_guard);
        true
    }

    fn with_per_view_material_batch_scratch(
        &self,
        view_id: ViewId,
        f: &mut dyn FnMut(&mut Vec<MaterialBatchBoundary>),
    ) -> bool {
        let Some(scratch_slot) = self.per_view_per_draw_scratch(view_id) else {
            return false;
        };
        let mut scratch_guard = scratch_slot.lock();
        f(&mut scratch_guard.material_batch_boundaries);
        drop(scratch_guard);
        true
    }

    fn per_view_per_draw_storage(&self, view_id: ViewId) -> Option<wgpu::Buffer> {
        self.per_view_per_draw(view_id)
            .map(|per_draw| per_draw.lock().per_draw_storage.clone())
    }

    fn per_view_per_draw_bind_group(&self, view_id: ViewId) -> Option<Arc<wgpu::BindGroup>> {
        self.per_view_per_draw(view_id)
            .map(|per_draw| Arc::clone(&per_draw.lock().bind_group))
    }

    fn empty_material_bind_group(&self) -> Option<Arc<wgpu::BindGroup>> {
        self.empty_material()
            .map(|empty| Arc::clone(&empty.bind_group))
    }

    fn copy_scene_depth_snapshot_for_view(
        &self,
        view_id: ViewId,
        encoder: &mut wgpu::CommandEncoder,
        source_depth: &wgpu::Texture,
        viewport: (u32, u32),
        multiview: bool,
    ) -> bool {
        self.copy_scene_depth_snapshot_for_view(view_id, encoder, source_depth, viewport, multiview)
    }

    fn copy_scene_color_snapshot_for_view(
        &self,
        view_id: ViewId,
        encoder: &mut wgpu::CommandEncoder,
        source_color: &wgpu::Texture,
        viewport: (u32, u32),
        multiview: bool,
    ) -> bool {
        self.copy_scene_color_snapshot_for_view(view_id, encoder, source_color, viewport, multiview)
    }

    fn copy_named_scene_color_snapshot_for_view(
        &self,
        view_id: ViewId,
        encoder: &mut wgpu::CommandEncoder,
        source_color: &wgpu::Texture,
        viewport: (u32, u32),
        multiview: bool,
    ) -> bool {
        self.copy_named_scene_color_snapshot_for_view(
            view_id,
            encoder,
            source_color,
            viewport,
            multiview,
        )
    }

    fn skybox_specular_uniform_params(&self) -> SkyboxSpecularUniformParams {
        self.skybox_specular_uniform_params()
    }

    fn mesh_deform_dispatched_this_submission(&self) -> bool {
        self.mesh_deform_dispatched_this_submission()
    }

    fn set_mesh_deform_dispatched_this_submission(&self) {
        self.set_mesh_deform_dispatched_this_submission();
    }

    fn visible_mesh_deform_keys_snapshot(&self) -> Option<HashSet<SkinCacheKey>> {
        self.visible_mesh_deform_keys_snapshot()
    }

    fn frame_global_pass_is_inactive(&self, pass_name: &str) -> bool {
        match pass_name {
            "MeshDeform" => self.visible_mesh_deform_filter_is_empty(),
            LIGHT_COOKIE_ATLAS_PASS_NAME => !self.has_light_cookie_requests(),
            SHADOW_ATLAS_PASS_NAME => !self.has_shadow_atlas_requests(),
            _ => false,
        }
    }

    fn ensure_per_view_frame_resources(
        &mut self,
        view_id: ViewId,
        device: &wgpu::Device,
        layout: PreRecordViewResourceLayout,
    ) -> bool {
        self.per_view_frame_or_create(view_id, device, layout)
            .is_some()
    }

    fn ensure_per_view_per_draw_resources(
        &mut self,
        view_id: ViewId,
        device: &wgpu::Device,
    ) -> bool {
        self.per_view_per_draw_or_create(view_id, device).is_some()
    }

    fn ensure_per_view_per_draw_scratch(&mut self, view_id: ViewId) {
        let _ = self.per_view_per_draw_scratch_or_create(view_id);
    }

    fn pre_record_sync_for_views(
        &mut self,
        device: &wgpu::Device,
        uploads: GraphUploadSink<'_>,
        view_layouts: &[PreRecordViewResourceLayout],
    ) {
        self.pre_record_sync_for_views(device, uploads, view_layouts);
    }

    fn retain_submit_resources(&self, resources: &mut crate::gpu::GpuRetainedResources) {
        if let Some(fgpu) = self.frame_gpu() {
            fgpu.retain_submit_resources(resources);
        }
        if let Some(empty) = self.empty_material() {
            resources.retain_bind_group(empty.bind_group.as_ref().clone());
        }
        for state in self.per_view_frame.values() {
            resources.retain_buffer(state.frame_uniform_buffer.clone());
            resources.retain_buffer(state.lights_buffer.clone());
            resources.retain_bind_group(state.frame_bind_group.as_ref().clone());
            resources.retain_bind_group(state.named_scene_color_frame_bind_group.as_ref().clone());
            resources.retain_buffer(state.cluster_params_buffer.clone());
            state.scene_snapshots.retain_submit_resources(resources);
        }
        for per_draw in self.per_view_draw.values() {
            per_draw.lock().retain_submit_resources(resources);
        }
    }

    fn has_light_cookie_requests(&self) -> bool {
        self.frame_gpu()
            .is_some_and(|fgpu| fgpu.has_light_cookie_requests())
    }

    fn encode_light_cookie_atlas(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        asset_resources: &dyn GraphAssetResources,
        profiler: Option<&crate::profiling::GpuProfilerHandle>,
    ) {
        if let Some(fgpu) = self.frame_gpu() {
            fgpu.encode_light_cookie_atlas(device, encoder, asset_resources, profiler);
        }
    }

    fn has_shadow_atlas_requests(&self) -> bool {
        self.frame_gpu().is_some() && !self.shadow_frame_plan().render_views.is_empty()
    }

    fn encode_shadow_atlas(&self, params: ShadowAtlasEncodeParams<'_, '_, '_>) {
        if let Some(fgpu) = self.frame_gpu() {
            fgpu.encode_shadow_atlas(self.shadow_frame_plan(), params);
        }
    }
}
