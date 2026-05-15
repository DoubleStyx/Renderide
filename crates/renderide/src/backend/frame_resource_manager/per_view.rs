//! Per-view frame/per-draw lifecycle, scene snapshot copies, cluster sync, and mesh-deform
//! submission flags for [`FrameResourceManager`].

use std::sync::Arc;
use std::sync::atomic::Ordering;

use hashbrown::HashSet;
use parking_lot::Mutex;

use crate::backend::cluster_gpu::ClusterBufferRefs;
use crate::camera::ViewId;
use crate::gpu::frame_globals::{FrameGpuUniforms, SkyboxSpecularUniformParams};
use crate::mesh_deform::SkinCacheKey;
use crate::render_graph::frame_params::PreRecordViewResourceLayout;
use crate::render_graph::frame_upload_batch::GraphUploadSink;

use super::super::frame_gpu::{
    EmptyMaterialBindGroup, FrameGpuResources, PerViewSceneSnapshots,
    ReflectionProbeSpecularResources,
};
use super::super::per_draw_resources::PerDrawResources;
use super::cluster_layout::{
    cluster_index_capacity_for_layout, make_cluster_params_buffer, per_view_snapshot_sync_params,
    unique_cluster_pre_record_layouts,
};
use super::manager::FrameResourceManager;
use super::per_view_state::{PerViewFrameState, PerViewPerDrawScratch};

impl FrameResourceManager {
    /// Clears per-tick frame-resource flags. Call once per winit frame from
    /// [`crate::runtime::RendererRuntime::tick_frame_wall_clock_begin`].
    ///
    /// The flag store uses [`Ordering::Release`] so a worker that observes the cleared state on
    /// the next tick is guaranteed to see the prior tick's GPU writes that produced the work.
    pub fn reset_light_prep_for_tick(&self) {
        self.mesh_deform_dispatched_this_submission
            .store(false, Ordering::Release);
        *self.visible_mesh_deform_keys.lock() = None;
    }

    /// Starts a graph submission with the visible deformed renderers gathered during draw prep.
    ///
    /// A tick can record multiple graph submissions: camera readbacks, reflection probes, HMD
    /// frames, and the desktop mirror/main view. Each submission has its own draw list, so the
    /// mesh-deform coalescing flag must reset here instead of only at the wall-clock tick
    /// boundary.
    pub fn begin_mesh_deform_submission(&mut self, keys: HashSet<SkinCacheKey>) {
        *self.visible_mesh_deform_keys.get_mut() = Some(keys);
        self.mesh_deform_dispatched_this_submission
            .store(false, Ordering::Release);
    }

    /// Whether [`crate::passes::MeshDeformPass`] already dispatched for this graph submission.
    ///
    /// Acquire-load pairs with the [`Ordering::Release`] store in
    /// [`Self::set_mesh_deform_dispatched_this_submission`] so a multi-view worker that sees
    /// `true` is guaranteed to see the prior dispatch's encoder/queue writes.
    pub fn mesh_deform_dispatched_this_submission(&self) -> bool {
        self.mesh_deform_dispatched_this_submission
            .load(Ordering::Acquire)
    }

    /// Marks mesh deform as dispatched for this graph submission.
    pub fn set_mesh_deform_dispatched_this_submission(&self) {
        self.mesh_deform_dispatched_this_submission
            .store(true, Ordering::Release);
    }

    /// Clones the current visible deform filter for lock-free worker iteration.
    pub fn visible_mesh_deform_keys_snapshot(&self) -> Option<HashSet<SkinCacheKey>> {
        self.visible_mesh_deform_keys.lock().clone()
    }

    /// Returns `true` when draw collection proved there is no visible deform work this frame.
    pub fn visible_mesh_deform_filter_is_empty(&self) -> bool {
        self.visible_mesh_deform_keys
            .lock()
            .as_ref()
            .is_some_and(HashSet::is_empty)
    }

    /// Shared `@group(0)` frame globals (camera + lights), after attach.
    pub fn frame_gpu(&self) -> Option<&FrameGpuResources> {
        self.frame_gpu.as_ref()
    }

    /// Mutable shared frame globals (cluster resize, uniform upload).
    pub fn frame_gpu_mut(&mut self) -> Option<&mut FrameGpuResources> {
        self.frame_gpu.as_mut()
    }

    /// Empty `@group(1)` bind group for shaders without per-material bindings.
    pub fn empty_material(&self) -> Option<&EmptyMaterialBindGroup> {
        self.empty_material.as_ref()
    }

    /// Returns the per-view frame state for `view_id`, creating it lazily if it does not exist.
    ///
    /// Grows the shared cluster buffers (on [`FrameGpuResources`]) to cover this view's layout
    /// when needed and rebuilds the `@group(0)` bind group whenever the shared cluster buffers,
    /// reflection-probe resources, or this view's snapshots change.
    ///
    /// Returns `None` when the manager has not been attached (no GPU resources available) or
    /// when cluster buffers cannot be allocated for the given viewport.
    pub fn per_view_frame_or_create(
        &mut self,
        view_id: ViewId,
        device: &wgpu::Device,
        layout: PreRecordViewResourceLayout,
    ) -> Option<&mut PerViewFrameState> {
        profiling::scope!("render::ensure_per_view_frame");
        let limits = Arc::clone(self.limits.as_ref()?);
        let viewport = (layout.width, layout.height);
        let stereo = layout.stereo;
        let index_capacity_words = cluster_index_capacity_for_layout(
            layout,
            self.frame_light_count_for_view_u32(view_id),
        )?;
        let snapshot_sync = per_view_snapshot_sync_params(layout);

        let per_view_frame = &mut self.per_view_frame;
        let frame_gpu_opt = &mut self.frame_gpu;
        let fgpu = frame_gpu_opt.as_mut()?;
        // Grow the shared cluster buffers to cover this view if needed; `sync_cluster_viewport`
        // is grow-only so repeated calls from different views consolidate to the max envelope.
        fgpu.sync_cluster_viewport(device, viewport, stereo, index_capacity_words)?;
        let cluster_ver = fgpu.cluster_cache.version;
        let skybox_specular_version = fgpu.skybox_specular_version();

        if !per_view_frame.contains_key(view_id) {
            let frame_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("per_view_frame_uniform"),
                size: size_of::<FrameGpuUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            crate::profiling::note_resource_churn!(Buffer, "backend::per_view_frame_uniform");
            let lights_buffer =
                FrameGpuResources::create_lights_storage_buffer(device, "per_view_lights_storage");
            crate::profiling::note_resource_churn!(Buffer, "backend::per_view_lights_storage");
            let cluster_params_buffer = make_cluster_params_buffer(device, stereo);
            let mut scene_snapshots =
                PerViewSceneSnapshots::new(device, layout.depth_format, layout.color_format);
            scene_snapshots.sync(device, limits.as_ref(), snapshot_sync);
            let refs = fgpu.cluster_cache.current_refs()?;
            let frame_bind_group = fgpu.build_per_view_bind_group(
                device,
                &frame_uniform_buffer,
                &lights_buffer,
                refs,
                scene_snapshots.views(),
            );
            let state = PerViewFrameState {
                frame_uniform_buffer,
                lights_buffer,
                frame_bind_group,
                cluster_params_buffer,
                scene_snapshots,
                last_cluster_version: cluster_ver,
                last_skybox_specular_version: skybox_specular_version,
                last_stereo: stereo,
            };
            let _ = per_view_frame.get_or_insert_with(view_id, || state);
        }

        let entry = per_view_frame.get_mut(view_id)?;

        // Resize per-view params buffer on mono->stereo transition (grow-only for consistency).
        if stereo && !entry.last_stereo {
            entry.cluster_params_buffer = make_cluster_params_buffer(device, true);
            entry.last_stereo = true;
        }

        let snapshots_changed = entry
            .scene_snapshots
            .sync(device, limits.as_ref(), snapshot_sync);
        let needs_rebuild = cluster_ver != entry.last_cluster_version
            || skybox_specular_version != entry.last_skybox_specular_version
            || snapshots_changed;

        if needs_rebuild {
            let refs = fgpu.cluster_cache.current_refs()?;
            let new_bg = fgpu.build_per_view_bind_group(
                device,
                &entry.frame_uniform_buffer,
                &entry.lights_buffer,
                refs,
                entry.scene_snapshots.views(),
            );
            entry.frame_bind_group = new_bg;
            entry.last_cluster_version = cluster_ver;
            entry.last_skybox_specular_version = skybox_specular_version;
        }

        per_view_frame.get_mut(view_id)
    }

    /// Uniform parameters for the disabled direct skybox specular slot.
    pub fn skybox_specular_uniform_params(&self) -> SkyboxSpecularUniformParams {
        self.frame_gpu.as_ref().map_or_else(
            SkyboxSpecularUniformParams::disabled,
            FrameGpuResources::skybox_specular_uniform_params,
        )
    }

    /// Synchronizes the frame-global reflection-probe specular resources.
    pub fn sync_reflection_probe_specular_resources(
        &mut self,
        device: &wgpu::Device,
        resources: Option<ReflectionProbeSpecularResources>,
    ) -> bool {
        self.frame_gpu
            .as_mut()
            .is_some_and(|fgpu| fgpu.sync_reflection_probe_specular_resources(device, resources))
    }

    /// Refs to the shared cluster buffers. All views share these.
    pub fn shared_cluster_buffer_refs(&self) -> Option<ClusterBufferRefs<'_>> {
        self.frame_gpu.as_ref()?.cluster_cache.current_refs()
    }

    /// Current cluster cache version on the shared cache. Used for bind-group invalidation
    /// caches that key on cluster-buffer reallocations.
    pub fn shared_cluster_version(&self) -> u64 {
        self.frame_gpu
            .as_ref()
            .map_or(0, |fgpu| fgpu.cluster_cache.version)
    }

    /// Returns the per-view frame state for `view_id`, or `None` if not yet created.
    pub fn per_view_frame(&self, view_id: ViewId) -> Option<&PerViewFrameState> {
        self.per_view_frame.get(view_id)
    }

    /// Frees per-view frame bind resources for a view that is no longer active.
    pub fn retire_per_view_frame(&mut self, view_id: ViewId) {
        self.per_view_frame.retire(view_id);
    }

    /// Returns the per-draw slab for the given view, creating it if it does not yet exist.
    pub fn per_view_per_draw_or_create(
        &mut self,
        view_id: ViewId,
        device: &wgpu::Device,
    ) -> Option<&Mutex<PerDrawResources>> {
        profiling::scope!("render::ensure_per_view_per_draw");
        let layout = self.per_draw_bind_group_layout.clone()?;
        let limits = self.limits.clone()?;
        let _ = self.per_view_per_draw_scratch_or_create(view_id);
        Some(self.per_view_draw.get_or_insert_with(view_id, || {
            Mutex::new(PerDrawResources::new_with_layout(device, layout, limits))
        }))
    }

    /// Returns the per-draw slab for the given view, or `None` if it has not been created yet.
    pub fn per_view_per_draw(&self, view_id: ViewId) -> Option<&Mutex<PerDrawResources>> {
        self.per_view_draw.get(view_id)
    }

    /// Frees the per-draw slab for a view that is no longer active.
    pub fn retire_per_view_per_draw(&mut self, view_id: ViewId) {
        self.per_view_draw.retire(view_id);
    }

    /// Returns the per-view scratch slot used for per-draw uniform packing, creating it on first use.
    pub fn per_view_per_draw_scratch_or_create(
        &mut self,
        view_id: ViewId,
    ) -> &Mutex<PerViewPerDrawScratch> {
        profiling::scope!("render::ensure_per_view_per_draw_scratch");
        self.per_view_per_draw_scratch
            .get_or_insert_with(view_id, || Mutex::new(PerViewPerDrawScratch::default()))
    }

    /// Returns the per-view scratch slot, or `None` if it has not been created yet.
    pub fn per_view_per_draw_scratch(
        &self,
        view_id: ViewId,
    ) -> Option<&Mutex<PerViewPerDrawScratch>> {
        self.per_view_per_draw_scratch.get(view_id)
    }

    /// Frees the per-view scratch buffers for a view that is no longer active.
    pub fn retire_per_view_per_draw_scratch(&mut self, view_id: ViewId) {
        self.per_view_per_draw_scratch.retire(view_id);
    }

    /// Retires all view-scoped frame resources for `view_id`.
    pub fn retire_view(&mut self, view_id: ViewId) {
        self.retire_per_view_frame(view_id);
        self.retire_per_view_per_draw(view_id);
        self.retire_per_view_per_draw_scratch(view_id);
        let _ = self.per_view_lights.retire(view_id);
    }

    /// Pre-synchronizes shared cluster buffers for every unique view layout before per-view
    /// recording starts and uploads each view's packed lights buffer.
    pub fn pre_record_sync_for_views(
        &mut self,
        device: &wgpu::Device,
        uploads: GraphUploadSink<'_>,
        view_layouts: &[PreRecordViewResourceLayout],
    ) {
        profiling::scope!("render::pre_record_sync_for_views");
        let cluster_layouts = unique_cluster_pre_record_layouts(view_layouts, |view_id| {
            self.frame_light_count_for_view_u32(view_id)
        });
        for layout in cluster_layouts {
            profiling::scope!("render::pre_record_sync_for_views::cluster_viewport");
            let Some(fgpu) = self.frame_gpu_mut() else {
                return;
            };
            if fgpu
                .sync_cluster_viewport(
                    device,
                    (layout.width, layout.height),
                    layout.stereo,
                    layout.index_capacity_words,
                )
                .is_none()
            {
                logger::warn!(
                    "pre-record cluster sync failed for viewport {}x{} stereo={} index_capacity={}",
                    layout.width,
                    layout.height,
                    layout.stereo,
                    layout.index_capacity_words
                );
            }
        }
        {
            profiling::scope!("render::pre_record_sync_for_views::write_lights");
            for layout in view_layouts {
                let Some(state) = self.per_view_frame(layout.view_id) else {
                    continue;
                };
                FrameGpuResources::write_lights_buffer_to(
                    uploads,
                    &state.lights_buffer,
                    self.frame_lights_for_view(layout.view_id),
                );
            }
        }
    }

    /// Copies the main depth attachment into this view's scene-depth snapshot.
    pub fn copy_scene_depth_snapshot_for_view(
        &self,
        view_id: ViewId,
        encoder: &mut wgpu::CommandEncoder,
        source_depth: &wgpu::Texture,
        viewport: (u32, u32),
        multiview: bool,
    ) -> bool {
        let Some(state) = self.per_view_frame.get(view_id) else {
            logger::warn!("scene depth snapshot copy: missing per-view frame for {view_id:?}");
            return false;
        };
        state
            .scene_snapshots
            .encode_depth_copy(encoder, source_depth, viewport, multiview)
    }

    /// Copies the main color attachment into this view's scene-color snapshot.
    pub fn copy_scene_color_snapshot_for_view(
        &self,
        view_id: ViewId,
        encoder: &mut wgpu::CommandEncoder,
        source_color: &wgpu::Texture,
        viewport: (u32, u32),
        multiview: bool,
    ) -> bool {
        let Some(state) = self.per_view_frame.get(view_id) else {
            logger::warn!("scene color snapshot copy: missing per-view frame for {view_id:?}");
            return false;
        };
        state
            .scene_snapshots
            .encode_color_copy(encoder, source_color, viewport, multiview)
    }
}
