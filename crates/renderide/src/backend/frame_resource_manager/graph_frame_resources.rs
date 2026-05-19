//! [`GraphFrameResources`] trait impl for [`FrameResourceManager`].

use std::any::Any;
use std::sync::Arc;

use hashbrown::HashSet;

use crate::camera::ViewId;
use crate::gpu::frame_globals::SkyboxSpecularUniformParams;
use crate::mesh_deform::{PaddedPerDrawUniforms, SkinCacheKey};
use crate::passes::MaterialBatchBoundary;
use crate::render_graph::execution_backend::{
    GraphClusterBufferRefs, GraphFrameResources, GraphShadowMetadataBuffers,
};
use crate::render_graph::frame_params::PreRecordViewResourceLayout;
use crate::render_graph::frame_upload_batch::GraphUploadSink;
use crate::shared::QualityConfig;

use super::super::frame_gpu::FrameGpuResources;
use super::super::light_gpu::GpuLight;
use super::manager::FrameResourceManager;

impl GraphFrameResources for FrameResourceManager {
    fn has_frame_gpu(&self) -> bool {
        self.frame_gpu().is_some()
    }

    fn frame_lights(&self, view_id: ViewId) -> &[GpuLight] {
        self.frame_lights_for_view(view_id)
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

    fn ensure_per_view_per_draw_capacity(
        &self,
        device: &wgpu::Device,
        view_id: ViewId,
        draw_count: usize,
    ) -> Option<wgpu::Buffer> {
        let per_draw_slot = self.per_view_per_draw(view_id)?;
        let mut per_draw = per_draw_slot.lock();
        per_draw.ensure_draw_slot_capacity(device, draw_count);
        Some(per_draw.per_draw_storage.clone())
    }

    fn with_per_view_per_draw_scratch(
        &self,
        view_id: ViewId,
        f: &mut dyn FnMut(&mut Vec<PaddedPerDrawUniforms>, &mut Vec<u8>),
    ) -> bool {
        let Some(scratch_slot) = self.per_view_per_draw_scratch(view_id) else {
            return false;
        };
        let mut scratch_guard = scratch_slot.lock();
        let scratch = &mut *scratch_guard;
        let uniforms = &mut scratch.uniforms;
        let slab_bytes = &mut scratch.slab_bytes;
        f(uniforms, slab_bytes);
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

    fn quality_config(&self) -> QualityConfig {
        self.quality_config()
    }

    fn shadow_resolution(&self) -> u32 {
        self.shadow_resolution()
    }

    fn shadow_layer_view(&self, layer: usize) -> Option<Arc<wgpu::TextureView>> {
        self.shadow_layer_view(layer)
    }

    fn shadow_metadata_buffers(&self, view_id: ViewId) -> Option<GraphShadowMetadataBuffers> {
        self.shadow_metadata_buffers_for_view(view_id)
            .map(|(shadow_lights, shadow_views)| GraphShadowMetadataBuffers {
                shadow_lights,
                shadow_views,
            })
    }

    fn write_shadow_lights(
        &self,
        uploads: GraphUploadSink<'_>,
        shadow_lights_buffer: &wgpu::Buffer,
        lights: &[crate::gpu::GpuShadowLight],
    ) {
        FrameGpuResources::write_shadow_lights_buffer_to(uploads, shadow_lights_buffer, lights);
    }

    fn write_shadow_views(
        &self,
        uploads: GraphUploadSink<'_>,
        shadow_views_buffer: &wgpu::Buffer,
        views: &[crate::gpu::GpuShadowView],
    ) {
        FrameGpuResources::write_shadow_views_buffer_to(uploads, shadow_views_buffer, views);
    }

    fn ensure_per_view_shadow_per_draw_capacity(
        &self,
        device: &wgpu::Device,
        view_id: ViewId,
        draw_count: usize,
    ) -> Option<wgpu::Buffer> {
        let per_draw_slot = self.per_view_shadow_per_draw(view_id)?;
        let mut per_draw = per_draw_slot.lock();
        per_draw.ensure_draw_slot_capacity(device, draw_count);
        Some(per_draw.per_draw_storage.clone())
    }

    fn with_per_view_shadow_per_draw_scratch(
        &self,
        view_id: ViewId,
        f: &mut dyn FnMut(&mut Vec<PaddedPerDrawUniforms>, &mut Vec<u8>),
    ) -> bool {
        let Some(scratch_slot) = self.per_view_shadow_per_draw_scratch(view_id) else {
            return false;
        };
        let mut scratch_guard = scratch_slot.lock();
        let scratch = &mut *scratch_guard;
        let uniforms = &mut scratch.uniforms;
        let slab_bytes = &mut scratch.slab_bytes;
        f(uniforms, slab_bytes);
        drop(scratch_guard);
        true
    }

    fn with_per_view_shadow_phase_cache(
        &self,
        view_id: ViewId,
        f: &mut dyn FnMut(&mut dyn Any),
    ) -> bool {
        let Some(cache_slot) = self.per_view_shadow_phase_cache(view_id) else {
            return false;
        };
        let mut cache_guard = cache_slot.lock();
        f(&mut *cache_guard);
        drop(cache_guard);
        true
    }

    fn per_view_shadow_per_draw_bind_group(&self, view_id: ViewId) -> Option<Arc<wgpu::BindGroup>> {
        self.per_view_shadow_per_draw(view_id)
            .map(|per_draw| Arc::clone(&per_draw.lock().bind_group))
    }

    fn per_view_shadow_writer_frame_bind_group(
        &self,
        view_id: ViewId,
    ) -> Option<Arc<wgpu::BindGroup>> {
        self.per_view_frame(view_id)
            .map(|state| Arc::clone(&state.shadow_writer_frame_bind_group))
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

    fn skybox_specular_uniform_params(&self) -> SkyboxSpecularUniformParams {
        self.skybox_specular_uniform_params()
    }

    fn visible_mesh_deform_filter_is_empty(&self) -> bool {
        self.visible_mesh_deform_filter_is_empty()
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

    fn ensure_per_view_shadow_per_draw_resources(
        &mut self,
        view_id: ViewId,
        device: &wgpu::Device,
    ) -> bool {
        self.per_view_shadow_per_draw_or_create(view_id, device)
            .is_some()
    }

    fn ensure_per_view_per_draw_scratch(&mut self, view_id: ViewId) {
        let _ = self.per_view_per_draw_scratch_or_create(view_id);
    }

    fn ensure_per_view_shadow_per_draw_scratch(&mut self, view_id: ViewId) {
        let _ = self.per_view_shadow_per_draw_scratch_or_create(view_id);
        let _ = self.per_view_shadow_phase_cache_or_create(view_id);
    }

    fn pre_record_sync_for_views(
        &mut self,
        device: &wgpu::Device,
        uploads: GraphUploadSink<'_>,
        view_layouts: &[PreRecordViewResourceLayout],
    ) {
        self.pre_record_sync_for_views(device, uploads, view_layouts);
    }
}
