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

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use hashbrown::{HashMap, HashSet};
use parking_lot::Mutex;

use crate::backend::cluster_gpu::{ClusterBufferRefs, CLUSTER_PARAMS_UNIFORM_SIZE};
use crate::gpu::frame_globals::FrameGpuUniforms;
use crate::gpu::GpuLimits;
use crate::render_graph::OcclusionViewId;

use super::frame_gpu::{EmptyMaterialBindGroup, FrameGpuResources};
use super::frame_gpu_bindings::{FrameGpuBindings, FrameGpuBindingsError};
use super::light_gpu::{order_lights_for_clustered_shading_in_place, GpuLight, MAX_LIGHTS};
use super::mesh_deform::PaddedPerDrawUniforms;
use super::per_draw_resources::PerDrawResources;
use crate::scene::{light_contributes, ResolvedLight, SceneCoordinator};

/// Per-view `@group(0)` frame uniform buffer + bind group.
///
/// The large cluster storage buffers (`cluster_light_counts`, `cluster_light_indices`) are
/// shared across all views via [`FrameGpuResources::cluster_cache`] and are safe to share
/// because GPU in-order execution within a single submit ensures each view's compute→raster
/// pair retires before the next view's compute overwrites.
///
/// [`Self::cluster_params_buffer`] is intentionally **per-view**: it is written by
/// `ClusteredLightPass::record` via `FrameUploadBatch`, which accumulates writes from rayon
/// workers. Since insertion order into the batch is non-deterministic, a shared params buffer
/// would mean the last view to push wins — corrupting every other view's cluster culling and
/// causing strobe flicker. Keeping params per-view eliminates the race at the cost of ~512 B
/// per view (completely negligible).
pub struct PerViewFrameState {
    /// Per-view `@group(0)` frame uniform buffer written by the prepare pass each frame.
    pub frame_uniform_buffer: wgpu::Buffer,
    /// Per-view `@group(0)` bind group referencing [`Self::frame_uniform_buffer`] and the
    /// shared cluster buffers alongside shared lights and scene snapshots.
    pub frame_bind_group: Arc<wgpu::BindGroup>,
    /// Per-view uniform buffer for `ClusterParams` (camera matrix, projection, viewport, etc.).
    ///
    /// Sized `CLUSTER_PARAMS_UNIFORM_SIZE × eye_multiplier`. Must be per-view — see struct doc.
    pub cluster_params_buffer: wgpu::Buffer,
    /// Shared [`ClusterBufferCache::version`] at which [`Self::frame_bind_group`] was last built.
    last_cluster_version: u64,
    /// [`FrameGpuResources::snapshot_version`] at which [`Self::frame_bind_group`] was last built.
    last_snapshot_version: u64,
    /// Stereo flag at which [`Self::cluster_params_buffer`] was last allocated.
    last_stereo: bool,
}

/// Per-view CPU scratch used to pack `@group(2)` per-draw uniforms before upload.
#[derive(Default)]
pub struct PerViewPerDrawScratch {
    /// Packed per-draw uniforms before serializing into the byte slab.
    pub uniforms: Vec<PaddedPerDrawUniforms>,
    /// Serialized byte slab uploaded into [`PerDrawResources::per_draw_storage`].
    pub slab_bytes: Vec<u8>,
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
    per_view_draw: HashMap<OcclusionViewId, Mutex<PerDrawResources>>,
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
    lights_gpu_uploaded_this_tick: AtomicBool,
    /// When true, [`crate::render_graph::passes::MeshDeformPass`] already dispatched this tick.
    ///
    /// In VR, the HMD graph runs mesh deform first; secondary cameras skip it via this flag.
    /// Reset with [`Self::reset_light_prep_for_tick`].
    mesh_deform_dispatched_this_tick: AtomicBool,
    /// Reused per-view scratch for per-draw VP/pack before [`crate::backend::mesh_deform::write_per_draw_uniform_slab`].
    ///
    /// Each view owns its own mutex-wrapped slot so rayon workers never alias the same scratch.
    per_view_per_draw_scratch: HashMap<OcclusionViewId, Mutex<PerViewPerDrawScratch>>,
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
            lights_gpu_uploaded_this_tick: AtomicBool::new(false),
            mesh_deform_dispatched_this_tick: AtomicBool::new(false),
            per_view_per_draw_scratch: HashMap::new(),
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
        self.lights_gpu_uploaded_this_tick
            .store(false, Ordering::Relaxed);
        self.mesh_deform_dispatched_this_tick
            .store(false, Ordering::Relaxed);
    }

    /// Whether [`crate::render_graph::passes::ClusteredLightPass`] already uploaded lights this tick.
    pub fn lights_gpu_uploaded_this_tick(&self) -> bool {
        self.lights_gpu_uploaded_this_tick.load(Ordering::Relaxed)
    }

    /// Whether [`crate::render_graph::passes::MeshDeformPass`] already dispatched this tick.
    pub fn mesh_deform_dispatched_this_tick(&self) -> bool {
        self.mesh_deform_dispatched_this_tick
            .load(Ordering::Relaxed)
    }

    /// Marks mesh deform as dispatched for this tick.
    pub fn set_mesh_deform_dispatched_this_tick(&self) {
        self.mesh_deform_dispatched_this_tick
            .store(true, Ordering::Relaxed);
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
        &self,
        queue: &wgpu::Queue,
        uniforms: &FrameGpuUniforms,
    ) {
        profiling::scope!("render::write_frame_uniforms");
        let Some(fgpu) = self.frame_gpu.as_ref() else {
            return;
        };
        fgpu.write_frame_uniform(queue, uniforms);
        if !self.lights_gpu_uploaded_this_tick.load(Ordering::Relaxed) {
            fgpu.write_lights_buffer(queue, &self.light_scratch);
            self.lights_gpu_uploaded_this_tick
                .store(true, Ordering::Relaxed);
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
    /// Grows the shared cluster buffers (on [`FrameGpuResources`]) to cover this view's
    /// `viewport` / `stereo` when needed and rebuilds the `@group(0)` bind group whenever the
    /// shared cluster version or snapshot version changes.
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
        profiling::scope!("render::ensure_per_view_frame");
        let _ = self.limits.as_ref()?; // confirm attached

        let per_view_frame = &mut self.per_view_frame;
        let frame_gpu_opt = &mut self.frame_gpu;
        let fgpu = frame_gpu_opt.as_mut()?;
        // Grow the shared cluster buffers to cover this view if needed; `sync_cluster_viewport`
        // is grow-only so repeated calls from different views consolidate to the max envelope.
        fgpu.sync_cluster_viewport(device, viewport, stereo);
        let snapshot_ver = fgpu.snapshot_version;
        let cluster_ver = fgpu.cluster_cache.version;
        let placeholder_bg = fgpu.bind_group.clone();

        if !per_view_frame.contains_key(&view_id) {
            let frame_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("per_view_frame_uniform"),
                size: std::mem::size_of::<FrameGpuUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let cluster_params_buffer = make_cluster_params_buffer(device, stereo);
            let frame_bind_group = fgpu
                .cluster_cache
                .current_refs()
                .map(|refs| fgpu.build_per_view_bind_group(device, &frame_uniform_buffer, refs))
                .unwrap_or_else(|| placeholder_bg);
            logger::debug!("per-view frame state: allocating for view {view_id:?}");
            per_view_frame.insert(
                view_id,
                PerViewFrameState {
                    frame_uniform_buffer,
                    frame_bind_group,
                    cluster_params_buffer,
                    last_cluster_version: cluster_ver,
                    last_snapshot_version: snapshot_ver,
                    last_stereo: stereo,
                },
            );
        }

        let entry = per_view_frame.get_mut(&view_id)?;

        // Resize per-view params buffer on mono→stereo transition (grow-only for consistency).
        if stereo && !entry.last_stereo {
            entry.cluster_params_buffer = make_cluster_params_buffer(device, true);
            entry.last_stereo = true;
        }

        let needs_rebuild = cluster_ver != entry.last_cluster_version
            || snapshot_ver != entry.last_snapshot_version;

        if needs_rebuild {
            if let Some(refs) = fgpu.cluster_cache.current_refs() {
                let new_bg =
                    fgpu.build_per_view_bind_group(device, &entry.frame_uniform_buffer, refs);
                entry.frame_bind_group = new_bg;
            }
            entry.last_cluster_version = cluster_ver;
            entry.last_snapshot_version = snapshot_ver;
        }

        per_view_frame.get_mut(&view_id)
    }

    /// Refs to the shared cluster buffers (see [`ClusterBufferCache`]). All views share these.
    pub fn shared_cluster_buffer_refs(&self) -> Option<ClusterBufferRefs<'_>> {
        self.frame_gpu.as_ref()?.cluster_cache.current_refs()
    }

    /// Current [`ClusterBufferCache::version`] on the shared cache. Used for bind-group
    /// invalidation caches that key on cluster-buffer reallocations.
    pub fn shared_cluster_version(&self) -> u64 {
        self.frame_gpu
            .as_ref()
            .map(|fgpu| fgpu.cluster_cache.version)
            .unwrap_or(0)
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
    ) -> Option<&Mutex<PerDrawResources>> {
        profiling::scope!("render::ensure_per_view_per_draw");
        let layout = self.per_draw_bind_group_layout.clone()?;
        let limits = self.limits.clone()?;
        let _ = self.per_view_per_draw_scratch_or_create(view_id);
        Some(self.per_view_draw.entry(view_id).or_insert_with(|| {
            logger::debug!("per-draw slab: allocating new slab for view {view_id:?}");
            Mutex::new(PerDrawResources::new_with_layout(device, layout, limits))
        }))
    }

    /// Returns the per-draw slab for the given view, or `None` if it has not been created yet.
    pub fn per_view_per_draw(&self, view_id: OcclusionViewId) -> Option<&Mutex<PerDrawResources>> {
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

    /// Returns the per-view scratch slot used for per-draw uniform packing, creating it on first use.
    ///
    /// Keyed per [`OcclusionViewId`] so parallel per-view recording cannot alias the same scratch
    /// across rayon workers.
    pub fn per_view_per_draw_scratch_or_create(
        &mut self,
        view_id: OcclusionViewId,
    ) -> &Mutex<PerViewPerDrawScratch> {
        profiling::scope!("render::ensure_per_view_per_draw_scratch");
        self.per_view_per_draw_scratch
            .entry(view_id)
            .or_insert_with(|| {
                logger::debug!("per-draw scratch: allocating for view {view_id:?}");
                Mutex::new(PerViewPerDrawScratch::default())
            })
    }

    /// Returns the per-view scratch slot, or `None` if it has not been created yet.
    pub fn per_view_per_draw_scratch(
        &self,
        view_id: OcclusionViewId,
    ) -> Option<&Mutex<PerViewPerDrawScratch>> {
        self.per_view_per_draw_scratch.get(&view_id)
    }

    /// Frees the per-view scratch buffers for a view that is no longer active.
    ///
    /// Call alongside [`Self::retire_per_view_per_draw`] and [`Self::retire_per_view_frame`] when a
    /// secondary RT camera is destroyed. Has no effect if the view was never allocated.
    pub fn retire_per_view_per_draw_scratch(&mut self, view_id: OcclusionViewId) {
        if self.per_view_per_draw_scratch.remove(&view_id).is_some() {
            logger::debug!("per-draw slab scratch: retired for view {view_id:?}");
        }
    }

    /// Fills the light scratch buffer from [`SceneCoordinator`] (active render spaces only,
    /// clustered ordering, capped at [`super::MAX_LIGHTS`]).
    ///
    /// Inactive spaces are skipped so lights from a previously focused world do not persist into
    /// the next frame's shading. This matches how renderables, mesh deform, secondary cameras,
    /// and the material-batch cache already filter by [`crate::scene::RenderSpaceState::is_active`].
    ///
    /// After the first successful run in a winit tick, subsequent calls are skipped until
    /// [`Self::reset_light_prep_for_tick`] runs, so secondary RT and main passes share one pack.
    /// Non-contributing lights are filtered via [`light_contributes`] before clustered ordering.
    ///
    /// Per-space [`SceneCoordinator::resolve_lights_world_into`] is read-only on the scene and is
    /// fanned out across rayon workers when more than one active render space exists.
    /// Single-space scenes (the common case) take the serial fast path to avoid rayon overhead.
    pub fn prepare_lights_from_scene(&mut self, scene: &SceneCoordinator) {
        if self.light_prep_done_this_tick {
            return;
        }
        profiling::scope!("render::prepare_lights");
        self.light_scratch.clear();
        self.resolved_flatten_scratch.clear();

        let space_ids: Vec<_> = scene
            .render_space_ids()
            .filter(|id| scene.space(*id).map(|s| s.is_active).unwrap_or(false))
            .collect();
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
        self.lights_gpu_uploaded_this_tick
            .store(false, Ordering::Relaxed);
    }

    /// Bundles frame/empty-material bind resources for render passes.
    pub fn gpu_bind_context(&self) -> FrameGpuBindContext<'_> {
        FrameGpuBindContext {
            frame_gpu: self.frame_gpu.as_ref(),
            empty_material: self.empty_material.as_ref(),
        }
    }

    /// Pre-synchronizes the shared cluster viewport for every unique view layout before per-view
    /// recording starts and uploads the packed lights buffer at most once for the tick.
    pub fn pre_record_sync_for_views(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        viewports_and_stereo: &[(u32, u32, bool)],
    ) {
        let mut seen = HashSet::new();
        for &(width, height, stereo) in viewports_and_stereo {
            if !seen.insert((width, height, stereo)) {
                continue;
            }
            let Some(fgpu) = self.frame_gpu_mut() else {
                return;
            };
            fgpu.sync_cluster_viewport(device, (width, height), stereo);
        }
        if self
            .lights_gpu_uploaded_this_tick
            .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
        {
            let Some(fgpu) = self.frame_gpu.as_ref() else {
                return;
            };
            fgpu.write_lights_buffer(queue, &self.light_scratch);
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
        let skip = self.lights_gpu_uploaded_this_tick.load(Ordering::Relaxed);
        {
            let fgpu = self.frame_gpu_mut()?;
            fgpu.sync_cluster_viewport(device, viewport, stereo);
        }
        if !skip {
            let fgpu = self.frame_gpu.as_ref()?;
            fgpu.write_lights_buffer(queue, &self.light_scratch);
            self.lights_gpu_uploaded_this_tick
                .store(true, Ordering::Relaxed);
        }
        self.frame_gpu_mut()
    }

    /// Copies the main depth attachment into the scene-depth snapshot that was already
    /// provisioned by [`Self::pre_record_sync_for_views`].
    pub fn copy_scene_depth_snapshot_for_view(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source_depth: &wgpu::Texture,
        viewport: (u32, u32),
        multiview: bool,
    ) {
        let Some(fgpu) = self.frame_gpu.as_ref() else {
            return;
        };
        fgpu.encode_scene_depth_snapshot_copy(encoder, source_depth, viewport, multiview);
    }
}

/// Allocates the per-view `ClusterParams` uniform buffer. Sized for one slot (mono) or two
/// slots (stereo). Used by `ClusteredLightPass` to write camera matrices per-view without
/// racing against other views' writes in the shared `FrameUploadBatch`.
fn make_cluster_params_buffer(device: &wgpu::Device, stereo: bool) -> wgpu::Buffer {
    let eye_multiplier = if stereo { 2 } else { 1 };
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("per_view_cluster_params_uniform"),
        size: CLUSTER_PARAMS_UNIFORM_SIZE * eye_multiplier,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use glam::{Quat, Vec3};

    use crate::scene::RenderSpaceId;
    use crate::shared::{
        LightData, LightType, LightsBufferRendererState, RenderTransform, ShadowType,
    };

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

    fn make_light_data(color_x: f32) -> LightData {
        LightData {
            point: Vec3::ZERO,
            orientation: Quat::IDENTITY,
            color: Vec3::new(color_x, 0.0, 0.0),
            intensity: 1.0,
            range: 10.0,
            angle: 45.0,
        }
    }

    fn make_state(global_unique_id: i32) -> LightsBufferRendererState {
        LightsBufferRendererState {
            renderable_index: 0,
            global_unique_id,
            shadow_strength: 0.0,
            shadow_near_plane: 0.0,
            shadow_map_resolution: 0,
            shadow_bias: 0.0,
            shadow_normal_bias: 0.0,
            cookie_texture_asset_id: -1,
            light_type: LightType::Point,
            shadow_type: ShadowType::None,
            _padding: [0; 2],
        }
    }

    fn seed_space_with_light(
        scene: &mut SceneCoordinator,
        space_id: RenderSpaceId,
        global_unique_id: i32,
        color_x: f32,
    ) {
        scene.test_seed_space_identity_worlds(space_id, vec![RenderTransform::default()], vec![-1]);
        let cache = scene.light_cache_mut();
        cache.store_full(global_unique_id, vec![make_light_data(color_x)]);
        cache.apply_update(space_id.0, &[], &[0], &[make_state(global_unique_id)]);
    }

    /// Lights from inactive render spaces must not leak into the frame's GPU light buffer.
    ///
    /// Regression: `prepare_lights_from_scene` used to iterate every tracked render space, so
    /// after a world switch (host marks the old space `is_active = false` but keeps it resident)
    /// its lights persisted into the new world's shading. Every other per-space pipeline
    /// (renderables, deform, secondary cameras, material-batch cache) filters by `is_active`;
    /// lights must follow the same rule.
    #[test]
    fn prepare_lights_from_scene_skips_inactive_spaces() {
        let mut scene = SceneCoordinator::new();
        let space_a = RenderSpaceId(1);
        let space_b = RenderSpaceId(2);
        seed_space_with_light(&mut scene, space_a, 100, 1.0);
        seed_space_with_light(&mut scene, space_b, 200, 0.5);

        // Both spaces active: both lights contribute.
        let mut mgr = FrameResourceManager::new();
        mgr.prepare_lights_from_scene(&scene);
        assert_eq!(mgr.frame_lights().len(), 2);

        // Focus space A only.
        scene.test_set_space_active(space_b, false);
        mgr.reset_light_prep_for_tick();
        mgr.prepare_lights_from_scene(&scene);
        let packed = mgr.frame_lights();
        assert_eq!(packed.len(), 1);
        assert!((packed[0].color[0] - 1.0).abs() < 1e-5);

        // Switch focus to space B; A's light must not carry over.
        scene.test_set_space_active(space_a, false);
        scene.test_set_space_active(space_b, true);
        mgr.reset_light_prep_for_tick();
        mgr.prepare_lights_from_scene(&scene);
        let packed = mgr.frame_lights();
        assert_eq!(packed.len(), 1);
        assert!((packed[0].color[0] - 0.5).abs() < 1e-5);
    }
}
