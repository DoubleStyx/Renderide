//! Per-frame GPU bind groups, light staging, and debug draw resources.
//!
//! [`FrameResourceManager`] owns the `@group(0)` frame uniform/light bind group
//! ([`FrameGpuResources`]), the empty `@group(1)` fallback ([`EmptyMaterialBindGroup`]),
//! the `@group(2)` per-draw debug slab ([`DebugDrawResources`]), and the CPU-side packed light
//! buffer used by [`crate::render_graph::passes::ClusteredLightPass`] and the forward pass.

use std::sync::Arc;

use super::debug_draw::DebugDrawResources;
use super::frame_gpu::{EmptyMaterialBindGroup, FrameGpuResources};
use super::light_gpu::{order_lights_for_clustered_shading, GpuLight};
use crate::scene::SceneCoordinator;

/// Per-frame GPU state: camera/light bind group, empty material fallback, debug draw slab, and
/// the CPU-side packed light buffer.
pub struct FrameResourceManager {
    /// Per-frame `@group(0)` camera + lights (after GPU attach).
    pub(crate) frame_gpu: Option<FrameGpuResources>,
    /// Placeholder `@group(1)` for materials without per-material bindings.
    pub(crate) empty_material: Option<EmptyMaterialBindGroup>,
    /// Uniforms + bind group for debug mesh draws (`@group(2)` dynamic slab).
    pub(crate) debug_draw: Option<DebugDrawResources>,
    /// Last packed lights for the frame (after [`Self::prepare_lights_from_scene`]).
    light_scratch: Vec<GpuLight>,
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
            debug_draw: None,
            light_scratch: Vec::new(),
        }
    }

    /// Allocates GPU resources for this manager. Called from [`super::RenderBackend::attach`].
    pub fn attach(&mut self, device: &wgpu::Device) {
        self.frame_gpu = Some(FrameGpuResources::new(device));
        self.empty_material = Some(EmptyMaterialBindGroup::new(device));
        self.debug_draw = Some(DebugDrawResources::new(device));
    }

    /// Packed GPU lights from the last [`Self::prepare_lights_from_scene`] call.
    pub fn frame_lights(&self) -> &[GpuLight] {
        &self.light_scratch
    }

    /// Per-frame `@group(0)` bind group (camera + lights), after attach.
    pub fn frame_gpu(&self) -> Option<&FrameGpuResources> {
        self.frame_gpu.as_ref()
    }

    /// Mutable frame globals (cluster resize, uniform upload).
    pub fn frame_gpu_mut(&mut self) -> Option<&mut FrameGpuResources> {
        self.frame_gpu.as_mut()
    }

    /// Empty `@group(1)` bind group for shaders without per-material bindings.
    pub fn empty_material(&self) -> Option<&EmptyMaterialBindGroup> {
        self.empty_material.as_ref()
    }

    /// Cloned [`Arc`] bind groups for mesh forward (`@group(0)` frame + `@group(1)` empty material).
    ///
    /// Used when the pass also needs `&mut` access to other fields (avoids borrow conflicts).
    pub fn mesh_forward_frame_bind_groups(
        &self,
    ) -> Option<(Arc<wgpu::BindGroup>, Arc<wgpu::BindGroup>)> {
        let f = self.frame_gpu.as_ref()?;
        let e = self.empty_material.as_ref()?;
        Some((f.bind_group.clone(), e.bind_group.clone()))
    }

    /// Fills the light scratch buffer from [`SceneCoordinator`] (all spaces, clustered ordering,
    /// capped at [`super::MAX_LIGHTS`]).
    pub fn prepare_lights_from_scene(&mut self, scene: &SceneCoordinator) {
        self.light_scratch.clear();
        let mut all = Vec::new();
        for id in scene.render_space_ids() {
            all.extend(scene.resolve_lights_world(id));
        }
        let ordered = order_lights_for_clustered_shading(&all);
        self.light_scratch
            .extend(ordered.iter().map(GpuLight::from_resolved));
    }

    /// Per-draw debug mesh uniforms: 256-byte dynamic uniform slab.
    pub fn debug_draw(&self) -> Option<&DebugDrawResources> {
        self.debug_draw.as_ref()
    }
}
