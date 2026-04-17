//! Per-frame GPU bind groups, light staging, and per-draw instance resources.
//!
//! [`FrameResourceManager`] owns the CPU-side packed light buffer and tick coalescing flags.
//! After GPU attach, [`super::FrameGpuBindings`] (when present) holds `@group(0)` / `@group(1)` /
//! `@group(2)` resources used by [`crate::render_graph::passes::ClusteredLightPass`] and the forward pass.

use std::cell::Cell;
use std::sync::Arc;

use super::frame_gpu::{EmptyMaterialBindGroup, FrameGpuResources};
use super::frame_gpu_bindings::FrameGpuBindings;
use super::light_gpu::{order_lights_for_clustered_shading_in_place, GpuLight, MAX_LIGHTS};
use super::per_draw_resources::PerDrawResources;
use crate::gpu::frame_globals::FrameGpuUniforms;
use crate::scene::{ResolvedLight, SceneCoordinator};

/// Immutable snapshot of `@group(0)` / empty `@group(1)` / per-draw `@group(2)` resources for one frame.
///
/// Obtained via [`FrameResourceManager::gpu_bind_context`]; intended to narrow pass APIs that
/// should not take the full [`super::RenderBackend`].
pub struct FrameGpuBindContext<'a> {
    /// Frame / empty-material / per-draw binds when GPU attach succeeded.
    pub binds: Option<&'a FrameGpuBindings>,
}

impl<'a> FrameGpuBindContext<'a> {
    /// Camera + lights (`@group(0)`), if attached.
    pub fn frame_gpu(&self) -> Option<&'a FrameGpuResources> {
        self.binds.map(|b| b.frame_gpu())
    }

    /// Fallback material (`@group(1)`), if attached.
    pub fn empty_material(&self) -> Option<&'a EmptyMaterialBindGroup> {
        self.binds.map(|b| b.empty_material())
    }

    /// Per-draw instance storage (`@group(2)`), if attached.
    pub fn per_draw(&self) -> Option<&'a PerDrawResources> {
        self.binds.map(|b| b.per_draw())
    }
}

/// Per-frame GPU state: camera/light bind group, empty material fallback, per-draw storage, and
/// the CPU-side packed light buffer.
pub struct FrameResourceManager {
    /// Frame / empty / per-draw binds after a successful GPU attach bundle.
    pub(crate) gpu_binds: Option<FrameGpuBindings>,
    /// Last packed lights for the frame (after [`Self::prepare_lights_from_scene`]).
    light_scratch: Vec<GpuLight>,
    /// Reused each frame to flatten all spaces’ [`crate::scene::ResolvedLight`] before ordering and GPU pack.
    resolved_flatten_scratch: Vec<ResolvedLight>,
    /// When true, [`Self::prepare_lights_from_scene`] is a no-op until [`Self::reset_light_prep_for_tick`].
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
            gpu_binds: None,
            light_scratch: Vec::new(),
            resolved_flatten_scratch: Vec::new(),
            light_prep_done_this_tick: false,
            lights_gpu_uploaded_this_tick: Cell::new(false),
            mesh_deform_dispatched_this_tick: Cell::new(false),
        }
    }

    /// Installs a pre-built `@group(0/1/2)` bundle after a successful transactional attach.
    pub(crate) fn set_gpu_binds(&mut self, binds: FrameGpuBindings) {
        self.gpu_binds = Some(binds);
    }

    /// Clears the per-tick light prep coalescing flag. Call once per winit frame from
    /// [`super::RenderBackend::reset_light_prep_for_tick`] (which also advances the GPU skin cache frame counter).
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
        let Some(binds) = self.gpu_binds.as_ref() else {
            return;
        };
        let fgpu = binds.frame_gpu();
        fgpu.write_frame_uniform(queue, uniforms);
        if !self.lights_gpu_uploaded_this_tick.get() {
            fgpu.write_lights_buffer(queue, &self.light_scratch);
            self.lights_gpu_uploaded_this_tick.set(true);
        }
    }

    /// Per-frame `@group(0)` bind group (camera + lights), after attach.
    pub fn frame_gpu(&self) -> Option<&FrameGpuResources> {
        self.gpu_binds.as_ref().map(|b| b.frame_gpu())
    }

    /// Mutable frame globals (cluster resize, uniform upload).
    pub fn frame_gpu_mut(&mut self) -> Option<&mut FrameGpuResources> {
        self.gpu_binds.as_mut().map(|b| b.frame_gpu_mut())
    }

    /// Empty `@group(1)` bind group for shaders without per-material bindings.
    pub fn empty_material(&self) -> Option<&EmptyMaterialBindGroup> {
        self.gpu_binds.as_ref().map(|b| b.empty_material())
    }

    /// Cloned [`Arc`] bind groups for mesh forward (`@group(0)` frame + `@group(1)` empty material).
    ///
    /// Used when the pass also needs `&mut` access to other fields (avoids borrow conflicts).
    pub fn mesh_forward_frame_bind_groups(
        &self,
    ) -> Option<(Arc<wgpu::BindGroup>, Arc<wgpu::BindGroup>)> {
        self.gpu_binds
            .as_ref()
            .map(FrameGpuBindings::mesh_forward_frame_bind_groups)
    }

    /// Fills the light scratch buffer from [`SceneCoordinator`] (all spaces, clustered ordering,
    /// capped at [`super::MAX_LIGHTS`]).
    ///
    /// After the first successful run in a winit tick, subsequent calls are skipped until
    /// [`Self::reset_light_prep_for_tick`], so secondary RT and main passes share one pack.
    pub fn prepare_lights_from_scene(&mut self, scene: &SceneCoordinator) {
        if self.light_prep_done_this_tick {
            return;
        }
        self.light_scratch.clear();
        self.resolved_flatten_scratch.clear();
        for id in scene.render_space_ids() {
            scene.resolve_lights_world_into(id, &mut self.resolved_flatten_scratch);
        }
        order_lights_for_clustered_shading_in_place(&mut self.resolved_flatten_scratch);
        self.light_scratch
            .reserve(self.resolved_flatten_scratch.len().min(MAX_LIGHTS));
        self.light_scratch.extend(
            self.resolved_flatten_scratch
                .iter()
                .map(GpuLight::from_resolved),
        );
        self.light_prep_done_this_tick = true;
    }

    /// Per-draw mesh forward storage: 256-byte slots, indexed by instance or dynamic offset.
    pub fn per_draw(&self) -> Option<&PerDrawResources> {
        self.gpu_binds.as_ref().map(|b| b.per_draw())
    }

    /// Mutable per-draw slab for uploads and capacity growth.
    pub fn per_draw_mut(&mut self) -> Option<&mut PerDrawResources> {
        self.gpu_binds.as_mut().map(|b| b.per_draw_mut())
    }

    /// Bundled frame / empty-material / per-draw bind resources for render passes.
    pub fn gpu_bind_context(&self) -> FrameGpuBindContext<'_> {
        FrameGpuBindContext {
            binds: self.gpu_binds.as_ref(),
        }
    }

    /// Syncs cluster viewport and uploads the packed light buffer once per tick (multi-view path).
    ///
    /// Reads lights from [`Self::light_scratch`] (no clone). After the first successful GPU upload in a tick,
    /// [`Self::lights_gpu_uploaded_this_tick`] is set and subsequent calls skip
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
            let fgpu = self.frame_gpu()?;
            fgpu.write_lights_buffer(queue, &self.light_scratch);
            self.lights_gpu_uploaded_this_tick.set(true);
        }
        self.frame_gpu_mut()
    }
}
