//! Per-frame GPU bind groups, light staging, per-view cluster buffers, and per-view per-draw
//! instance resources.
//!
//! [`FrameResourceManager`] owns the shared `@group(0)` frame uniform/light bind group
//! ([`FrameGpuResources`]), the empty `@group(1)` fallback ([`EmptyMaterialBindGroup`]),
//! per-view cluster buffer caches and `@group(0)` bind groups ([`PerViewFrameState`]), a
//! `@group(2)` per-draw instance storage slab per render view ([`PerDrawResources`]), and the
//! CPU-side packed light buffer used by [`crate::render_graph::passes::ClusteredLightPass`] and
//! the forward pass.
//!
//! Per-view cluster buffers are each view's own independent storage so that views cannot stomp
//! one another's clustered light lists under single-submit semantics. Per-view state is keyed by
//! [`OcclusionViewId`] and created lazily on first use; retired explicitly when a secondary RT
//! camera is destroyed.
//!
//! Per-draw resources follow the same ownership model: one grow-on-demand slab per
//! [`OcclusionViewId`], created lazily so no view can exhaust another view's per-draw capacity.

use std::cell::Cell;
use std::sync::Arc;

use hashbrown::HashMap;

use crate::backend::cluster_gpu::{ClusterBufferCache, ClusterBufferRefs, CLUSTER_COUNT_Z};
use crate::gpu::frame_globals::FrameGpuUniforms;
use crate::gpu::GpuLimits;
use crate::render_graph::OcclusionViewId;

use super::frame_gpu::{EmptyMaterialBindGroup, FrameGpuResources};
use super::frame_gpu_bindings::{FrameGpuBindings, FrameGpuBindingsError};
use super::light_gpu::{order_lights_for_clustered_shading_in_place, GpuLight, MAX_LIGHTS};
use super::mesh_deform::PaddedPerDrawUniforms;
use super::per_draw_resources::PerDrawResources;
use crate::scene::{light_contributes, ResolvedLight, SceneCoordinator};

/// Per-view `@group(0)` cluster buffers, frame uniform buffer, and bind group.
///
/// Each [`OcclusionViewId`] owns one of these to ensure its clustered light lists are never
/// overwritten by another view's compute dispatch under single-submit semantics. The bind group
/// is rebuilt whenever the view's cluster buffers change (viewport or stereo change) or when the
/// shared scene snapshot textures are recreated.
pub struct PerViewFrameState {
    /// Per-view cluster buffer cache (light counts, light indices, compute params).
    pub cluster_cache: ClusterBufferCache,
    /// Per-view `@group(0)` frame uniform buffer written by the prepare pass each frame.
    pub frame_uniform_buffer: wgpu::Buffer,
    /// Per-view `@group(0)` bind group referencing [`Self::frame_uniform_buffer`] and the
    /// per-view cluster buffers alongside shared lights and scene snapshots.
    pub frame_bind_group: Arc<wgpu::BindGroup>,
    /// [`ClusterBufferCache::version`] at which [`Self::frame_bind_group`] was last built.
    last_cluster_version: u64,
    /// Viewport for which [`Self::cluster_cache`] was last provisioned.
    last_viewport: (u32, u32),
    /// Stereo flag for which [`Self::cluster_cache`] was last provisioned.
    last_stereo: bool,
    /// [`FrameGpuResources::snapshot_version`] at which [`Self::frame_bind_group`] was last built.
    last_snapshot_version: u64,
}

impl PerViewFrameState {
    /// Returns the cluster buffer refs using the key that was active when the cache was last
    /// provisioned, without needing to repeat the viewport/stereo parameters.
    pub fn cluster_buffer_refs(&self) -> Option<ClusterBufferRefs<'_>> {
        self.cluster_cache
            .get_buffers(self.last_viewport, CLUSTER_COUNT_Z, self.last_stereo)
    }
}

/// Immutable snapshot of `@group(0)` / empty `@group(1)` resources for one frame.
///
/// Obtained via [`FrameResourceManager::gpu_bind_context`]; intended to narrow pass APIs that
/// should not take the full [`super::RenderBackend`].
pub struct FrameGpuBindContext<'a> {
    /// Camera + lights (`@group(0)`).
    pub frame_gpu: Option<&'a FrameGpuResources>,
    /// Fallback material (`@group(1)`).
    pub empty_material: Option<&'a EmptyMaterialBindGroup>,
}

/// Per-frame GPU state: shared frame/light resources, per-view cluster buffers and bind groups,
/// per-view per-draw storage slabs, and the CPU-side packed light buffer.
pub struct FrameResourceManager {
    /// Shared `@group(0)` frame globals (lights buffer, snapshot textures, bind group layout).
    pub(crate) frame_gpu: Option<FrameGpuResources>,
    /// Placeholder `@group(1)` for materials without per-material bindings.
    pub(crate) empty_material: Option<EmptyMaterialBindGroup>,
    /// Per-view cluster buffers, frame uniform buffer, and `@group(0)` bind group.
    ///
    /// Created lazily on first use per [`OcclusionViewId`]; retired when a secondary RT camera
    /// is destroyed via [`Self::retire_per_view_frame`].
    per_view_frame: HashMap<OcclusionViewId, PerViewFrameState>,
    /// One grow-on-demand per-draw slab per stable render-view identity.
    ///
    /// Created lazily; keyed by [`OcclusionViewId`] so secondary RT cameras never compete
    /// with the main view (or each other) for buffer space.
    per_view_draw: HashMap<OcclusionViewId, PerDrawResources>,
    /// Shared `@group(2)` bind group layout, reflected once at attach time.
    per_draw_bind_group_layout: Option<Arc<wgpu::BindGroupLayout>>,
    /// GPU limits stored at attach time for lazy per-view slab/cluster creation.
    limits: Option<Arc<GpuLimits>>,
    /// Last packed lights for the frame (after [`Self::prepare_lights_from_scene`]).
    light_scratch: Vec<GpuLight>,
    /// Reused each frame to flatten all spaces' [`crate::scene::ResolvedLight`] before ordering and GPU pack.
    resolved_flatten_scratch: Vec<ResolvedLight>,
    /// When true, [`Self::prepare_lights_from_scene`] is a no-op until [`Self::reset_light_prep_for_tick`] runs.
    ///
    /// Cleared at the start of each winit tick so multiple graph entry points in one tick (e.g. secondary
    /// RT passes then main swapchain) share one CPU light pack.
    light_prep_done_this_tick: bool,
    /// When true, the packed light buffer was already uploaded to the GPU this tick (multi-view path).
    ///
    /// Reset with [`Self::reset_light_prep_for_tick`]. [`crate::render_graph::passes::ClusteredLightPass`]
    /// skips redundant `write_lights_buffer` while still dispatching per view.
    lights_gpu_uploaded_this_tick: Cell<bool>,
    /// When true, [`crate::render_graph::passes::MeshDeformPass`] already dispatched this tick.
    ///
    /// In VR, the HMD graph runs mesh deform first; secondary cameras skip it via this flag.
    /// Reset with [`Self::reset_light_prep_for_tick`].
    mesh_deform_dispatched_this_tick: Cell<bool>,
    /// Reused per-view scratch for per-draw VP/pack before [`crate::backend::mesh_deform::write_per_draw_uniform_slab`].
    ///
    /// Keyed by [`OcclusionViewId`] so parallel per-view recording never aliases the same scratch
    /// across rayon workers. Mirrors the `per_view_frame` / `per_view_draw` pattern.
    pub(crate) per_draw_uniforms_scratch_by_view:
        HashMap<OcclusionViewId, Vec<PaddedPerDrawUniforms>>,
    /// Reused byte slab for [`crate::backend::mesh_deform::write_per_draw_uniform_slab`], keyed per view.
    ///
    /// Keyed by [`OcclusionViewId`] so parallel per-view recording never aliases the same scratch
    /// across rayon workers. Mirrors the `per_view_frame` / `per_view_draw` pattern.
    pub(crate) per_draw_slab_byte_scratch_by_view: HashMap<OcclusionViewId, Vec<u8>>,
}

impl Default for FrameResourceManager {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameResourceManager {
    /// Creates an empty manager with no GPU resources.
    pub fn new() -> Self {
        Self {
            frame_gpu: None,
            empty_material: None,
            per_view_frame: HashMap::new(),
            per_view_draw: HashMap::new(),
            per_draw_bind_group_layout: None,
            limits: None,
            light_scratch: Vec::new(),
            resolved_flatten_scratch: Vec::new(),
            light_prep_done_this_tick: false,
            lights_gpu_uploaded_this_tick: Cell::new(false),
            mesh_deform_dispatched_this_tick: Cell::new(false),
            per_draw_uniforms_scratch_by_view: HashMap::new(),
            per_draw_slab_byte_scratch_by_view: HashMap::new(),
        }
    }

    /// Allocates GPU resources for this manager. Called from [`super::RenderBackend::attach`].
    ///
    /// On success, `@group(0)` / `@group(1)` / `@group(2)` layout are present.
    /// Per-view per-draw slabs and per-view cluster buffers are created lazily on first use.
    /// On error, frame bind fields remain unset (no partial attach).
    pub fn attach(
        &mut self,
        device: &wgpu::Device,
        limits: Arc<GpuLimits>,
    ) -> Result<(), FrameGpuBindingsError> {
        let binds = FrameGpuBindings::try_new(device, Arc::clone(&limits))?;
        self.frame_gpu = Some(binds.frame_gpu);
        self.empty_material = Some(binds.empty_material);
        self.per_draw_bind_group_layout = Some(binds.per_draw_bind_group_layout);
        self.limits = Some(limits);
        Ok(())
    }

    /// Clears the per-tick light prep coalescing flag. Call once per winit frame from
    /// [`crate::runtime::RendererRuntime::tick_frame_wall_clock_begin`].
    pub fn reset_light_prep_for_tick(&mut self) {
        self.light_prep_done_this_tick = false;
        self.lights_gpu_uploaded_this_tick.set(false);
        self.mesh_deform_dispatched_this_tick.set(false);
    }

    /// Whether [`crate::render_graph::passes::ClusteredLightPass`] already uploaded lights this tick.
    pub fn lights_gpu_uploaded_this_tick(&self) -> bool {
        self.lights_gpu_uploaded_this_tick.get()
    }

    /// Whether [`crate::render_graph::passes::MeshDeformPass`] already dispatched this tick.
    pub fn mesh_deform_dispatched_this_tick(&self) -> bool {
        self.mesh_deform_dispatched_this_tick.get()
    }

    /// Marks mesh deform as dispatched for this tick.
    pub fn set_mesh_deform_dispatched_this_tick(&self) {
        self.mesh_deform_dispatched_this_tick.set(true);
    }

    /// Packed GPU lights from the last [`Self::prepare_lights_from_scene`] call.
    pub fn frame_lights(&self) -> &[GpuLight] {
        &self.light_scratch
    }

    /// Light count for frame uniforms and shaders (`min(len, [`MAX_LIGHTS`])`).
    pub fn frame_light_count_u32(&self) -> u32 {
        self.light_scratch.len().min(MAX_LIGHTS) as u32
    }

    /// Writes camera frame uniform and, if lights were not yet uploaded this tick, the lights storage buffer.
    ///
    /// Skips [`FrameGpuResources::write_lights_buffer`] when [`Self::lights_gpu_uploaded_this_tick`] is already
    /// true (e.g. [`crate::render_graph::passes::ClusteredLightPass`] ran first), avoiding duplicate uploads
    /// on multi-view paths while still refreshing frame uniforms every view.
    pub fn write_frame_uniform_and_lights_from_scratch(
        &mut self,
        queue: &wgpu::Queue,
        uniforms: &FrameGpuUniforms,
    ) {
        let Some(fgpu) = self.frame_gpu.as_ref() else {
            return;
        };
        fgpu.write_frame_uniform(queue, uniforms);
        if !self.lights_gpu_uploaded_this_tick.get() {
            fgpu.write_lights_buffer(queue, &self.light_scratch);
            self.lights_gpu_uploaded_this_tick.set(true);
        }
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
    /// Internally provisions per-view cluster buffers for `viewport`/`stereo` via
    /// [`ClusterBufferCache::ensure_buffers`] and rebuilds the `@group(0)` bind group whenever
    /// the cluster buffer version or shared snapshot version changes.
    ///
    /// Returns `None` when the manager has not been attached (no GPU resources available) or
    /// when cluster buffers cannot be allocated for the given viewport.
    pub fn per_view_frame_or_create(
        &mut self,
        view_id: OcclusionViewId,
        device: &wgpu::Device,
        viewport: (u32, u32),
        stereo: bool,
    ) -> Option<&mut PerViewFrameState> {
        let limits = self.limits.clone()?;

        // Use field-level split borrows: per_view_frame and frame_gpu are disjoint fields.
        let per_view_frame = &mut self.per_view_frame;
        let frame_gpu_opt = &mut self.frame_gpu;
        let fgpu = frame_gpu_opt.as_mut()?;
        let snapshot_ver = fgpu.snapshot_version;
        let placeholder_bg = fgpu.bind_group.clone();

        // Ensure entry exists with initial cluster buffers.
        if !per_view_frame.contains_key(&view_id) {
            let frame_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("per_view_frame_uniform"),
                size: std::mem::size_of::<FrameGpuUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut cluster_cache = ClusterBufferCache::new();
            let _ =
                cluster_cache.ensure_buffers(device, &limits, viewport, CLUSTER_COUNT_Z, stereo);
            let cluster_ver = cluster_cache.version;
            let frame_bind_group = cluster_cache
                .get_buffers(viewport, CLUSTER_COUNT_Z, stereo)
                .map(|refs| fgpu.build_per_view_bind_group(device, &frame_uniform_buffer, refs))
                .unwrap_or_else(|| placeholder_bg);
            logger::debug!("per-view frame state: allocating for view {view_id:?}");
            per_view_frame.insert(
                view_id,
                PerViewFrameState {
                    cluster_cache,
                    frame_uniform_buffer,
                    frame_bind_group,
                    last_cluster_version: cluster_ver,
                    last_viewport: viewport,
                    last_stereo: stereo,
                    last_snapshot_version: snapshot_ver,
                },
            );
        }

        let entry = per_view_frame.get_mut(&view_id)?;

        // Ensure cluster buffers are provisioned for the current viewport/stereo.
        let _ =
            entry
                .cluster_cache
                .ensure_buffers(device, &limits, viewport, CLUSTER_COUNT_Z, stereo);
        let cluster_ver = entry.cluster_cache.version;

        let needs_rebuild = cluster_ver != entry.last_cluster_version
            || viewport != entry.last_viewport
            || stereo != entry.last_stereo
            || snapshot_ver != entry.last_snapshot_version;

        if needs_rebuild {
            if let Some(refs) = entry
                .cluster_cache
                .get_buffers(viewport, CLUSTER_COUNT_Z, stereo)
            {
                let new_bg =
                    fgpu.build_per_view_bind_group(device, &entry.frame_uniform_buffer, refs);
                entry.frame_bind_group = new_bg;
            }
            entry.last_cluster_version = cluster_ver;
            entry.last_viewport = viewport;
            entry.last_stereo = stereo;
            entry.last_snapshot_version = snapshot_ver;
        }

        per_view_frame.get_mut(&view_id)
    }

    /// Returns the per-view frame state for `view_id`, or `None` if not yet created.
    pub fn per_view_frame(&self, view_id: OcclusionViewId) -> Option<&PerViewFrameState> {
        self.per_view_frame.get(&view_id)
    }

    /// Frees per-view cluster buffers and bind group for a view that is no longer active.
    ///
    /// Call alongside [`Self::retire_per_view_per_draw`] when a secondary RT camera is destroyed.
    /// Has no effect if the view was never allocated.
    pub fn retire_per_view_frame(&mut self, view_id: OcclusionViewId) {
        if self.per_view_frame.remove(&view_id).is_some() {
            logger::debug!("per-view frame state: retired for view {view_id:?}");
        }
    }

    /// Returns the per-draw slab for the given view, creating it if it does not yet exist.
    ///
    /// Returns `None` when the manager has not been attached (no device limits / layout available).
    pub fn per_view_per_draw_or_create(
        &mut self,
        view_id: OcclusionViewId,
        device: &wgpu::Device,
    ) -> Option<&mut PerDrawResources> {
        let layout = self.per_draw_bind_group_layout.clone()?;
        let limits = self.limits.clone()?;
        Some(self.per_view_draw.entry(view_id).or_insert_with(|| {
            logger::debug!("per-draw slab: allocating new slab for view {view_id:?}");
            PerDrawResources::new_with_layout(device, layout, limits)
        }))
    }

    /// Returns the per-draw slab for the given view, or `None` if it has not been created yet.
    pub fn per_view_per_draw(&self, view_id: OcclusionViewId) -> Option<&PerDrawResources> {
        self.per_view_draw.get(&view_id)
    }

    /// Frees the per-draw slab for a view that is no longer active (e.g. render-texture camera destroyed).
    ///
    /// Has no effect if the view was never allocated.
    pub fn retire_per_view_per_draw(&mut self, view_id: OcclusionViewId) {
        if self.per_view_draw.remove(&view_id).is_some() {
            logger::debug!("per-draw slab: retired slab for view {view_id:?}");
        }
    }

    /// Returns the per-view byte scratch slab used for per-draw uniform packing, creating it on first use.
    ///
    /// Keyed per [`OcclusionViewId`] so parallel per-view recording cannot alias the same scratch
    /// across rayon workers.
    pub fn per_draw_scratch_mut(&mut self, view_id: OcclusionViewId) -> &mut Vec<u8> {
        self.per_draw_slab_byte_scratch_by_view
            .entry(view_id)
            .or_default()
    }

    /// Returns the per-view byte scratch slab, or `None` if no packing has happened for this view yet.
    pub fn per_draw_scratch(&self, view_id: OcclusionViewId) -> Option<&Vec<u8>> {
        self.per_draw_slab_byte_scratch_by_view.get(&view_id)
    }

    /// Frees the per-view scratch buffers (uniforms + byte slab) for a view that is no longer active.
    ///
    /// Call alongside [`Self::retire_per_view_per_draw`] and [`Self::retire_per_view_frame`] when a
    /// secondary RT camera is destroyed. Has no effect if the view was never allocated.
    pub fn retire_per_view_per_draw_scratch(&mut self, view_id: OcclusionViewId) {
        let removed_uniforms = self
            .per_draw_uniforms_scratch_by_view
            .remove(&view_id)
            .is_some();
        let removed_slab = self
            .per_draw_slab_byte_scratch_by_view
            .remove(&view_id)
            .is_some();
        if removed_uniforms || removed_slab {
            logger::debug!("per-draw slab scratch: retired for view {view_id:?}");
        }
    }

    /// Fills the light scratch buffer from [`SceneCoordinator`] (all spaces, clustered ordering,
    /// capped at [`super::MAX_LIGHTS`]).
    ///
    /// After the first successful run in a winit tick, subsequent calls are skipped until
    /// [`Self::reset_light_prep_for_tick`] runs, so secondary RT and main passes share one pack.
    /// Non-contributing lights are filtered via [`light_contributes`] before clustered ordering.
    ///
    /// Per-space [`SceneCoordinator::resolve_lights_world_into`] is read-only on the scene and is
    /// fanned out across rayon workers when more than one render space exists. Single-space
    /// scenes (the common case) take the serial fast path to avoid rayon overhead.
    pub fn prepare_lights_from_scene(&mut self, scene: &SceneCoordinator) {
        if self.light_prep_done_this_tick {
            return;
        }
        profiling::scope!("render::prepare_lights");
        self.light_scratch.clear();
        self.resolved_flatten_scratch.clear();

        let space_ids: Vec<_> = scene.render_space_ids().collect();
        match space_ids.len() {
            0 => {}
            1 => {
                scene.resolve_lights_world_into(space_ids[0], &mut self.resolved_flatten_scratch);
            }
            _ => {
                use rayon::prelude::*;
                let per_space: Vec<Vec<ResolvedLight>> = space_ids
                    .par_iter()
                    .map(|&id| {
                        let mut local = Vec::new();
                        scene.resolve_lights_world_into(id, &mut local);
                        local
                    })
                    .collect();
                let total: usize = per_space.iter().map(Vec::len).sum();
                self.resolved_flatten_scratch.reserve(total);
                for chunk in per_space {
                    self.resolved_flatten_scratch.extend(chunk);
                }
            }
        }

        self.resolved_flatten_scratch.retain(light_contributes);
        order_lights_for_clustered_shading_in_place(&mut self.resolved_flatten_scratch);
        self.light_scratch
            .reserve(self.resolved_flatten_scratch.len().min(MAX_LIGHTS));
        self.light_scratch.extend(
            self.resolved_flatten_scratch
                .iter()
                .map(GpuLight::from_resolved),
        );
        self.light_prep_done_this_tick = true;
        self.lights_gpu_uploaded_this_tick.set(false);
    }

    /// Bundles frame/empty-material bind resources for render passes.
    pub fn gpu_bind_context(&self) -> FrameGpuBindContext<'_> {
        FrameGpuBindContext {
            frame_gpu: self.frame_gpu.as_ref(),
            empty_material: self.empty_material.as_ref(),
        }
    }

    /// Syncs the global cluster viewport and uploads the packed light buffer once per tick.
    ///
    /// The global cluster viewport sync keeps the shared bind group consistent with the current
    /// viewport/stereo. Per-view cluster buffers (in [`PerViewFrameState`]) are synced separately
    /// via [`Self::per_view_frame_or_create`]. Lights upload is coalesced: after the first
    /// successful upload this tick, subsequent calls skip
    /// [`super::frame_gpu::FrameGpuResources::write_lights_buffer`].
    pub fn sync_cluster_viewport_ensure_lights_upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        viewport: (u32, u32),
        stereo: bool,
    ) -> Option<&mut FrameGpuResources> {
        let skip = self.lights_gpu_uploaded_this_tick.get();
        {
            let fgpu = self.frame_gpu_mut()?;
            fgpu.sync_cluster_viewport(device, viewport, stereo);
        }
        if !skip {
            let fgpu = self.frame_gpu.as_ref()?;
            fgpu.write_lights_buffer(queue, &self.light_scratch);
            self.lights_gpu_uploaded_this_tick.set(true);
        }
        self.frame_gpu_mut()
    }

    /// Copies the main depth attachment into the scene-depth snapshot and rebuilds all per-view
    /// `@group(0)` bind groups whose snapshot version is now stale.
    ///
    /// This must be called instead of [`FrameGpuResources::copy_scene_depth_snapshot`] when
    /// per-view bind groups are in use, so that intersection draw passes see the updated texture
    /// view in their `@group(0)` binding.
    pub fn copy_scene_depth_snapshot_for_view(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        source_depth: &wgpu::Texture,
        viewport: (u32, u32),
        multiview: bool,
        stereo_cluster: bool,
    ) {
        let per_view_frame = &mut self.per_view_frame;
        let Some(fgpu) = self.frame_gpu.as_mut() else {
            return;
        };
        fgpu.copy_scene_depth_snapshot(
            device,
            encoder,
            source_depth,
            viewport,
            multiview,
            stereo_cluster,
        );
        let snapshot_ver = fgpu.snapshot_version;
        for entry in per_view_frame.values_mut() {
            if entry.last_snapshot_version == snapshot_ver {
                continue;
            }
            if let Some(refs) = entry.cluster_cache.get_buffers(
                entry.last_viewport,
                CLUSTER_COUNT_Z,
                entry.last_stereo,
            ) {
                let new_bg =
                    fgpu.build_per_view_bind_group(device, &entry.frame_uniform_buffer, refs);
                entry.frame_bind_group = new_bg;
            }
            entry.last_snapshot_version = snapshot_ver;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_manager_has_no_per_view_draw() {
        let mgr = FrameResourceManager::new();
        assert!(mgr.per_view_per_draw(OcclusionViewId::Main).is_none());
        assert!(mgr
            .per_view_per_draw(OcclusionViewId::OffscreenRenderTexture(42))
            .is_none());
    }

    #[test]
    fn new_manager_has_no_per_view_frame() {
        let mgr = FrameResourceManager::new();
        assert!(mgr.per_view_frame(OcclusionViewId::Main).is_none());
        assert!(mgr
            .per_view_frame(OcclusionViewId::OffscreenRenderTexture(42))
            .is_none());
    }

    #[test]
    fn retire_nonexistent_is_noop() {
        let mut mgr = FrameResourceManager::new();
        mgr.retire_per_view_per_draw(OcclusionViewId::Main);
        mgr.retire_per_view_per_draw(OcclusionViewId::OffscreenRenderTexture(99));
        mgr.retire_per_view_frame(OcclusionViewId::Main);
        mgr.retire_per_view_frame(OcclusionViewId::OffscreenRenderTexture(99));
    }
}
