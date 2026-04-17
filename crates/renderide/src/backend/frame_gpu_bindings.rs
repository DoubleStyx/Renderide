//! Frame-global and per-draw GPU bind bundles created atomically at GPU attach.
//!
//! [`FrameGpuBindings::try_new`] groups `@group(0)`, `@group(1)`, and `@group(2)` allocation so
//! downstream code does not handle three independent optional attach failures.

use std::sync::Arc;

use crate::gpu::GpuLimits;
use crate::materials::PipelineBuildError;

use super::frame_gpu::{EmptyMaterialBindGroup, FrameGpuResources};
use super::frame_gpu_error::FrameGpuInitError;
use super::per_draw_resources::PerDrawResources;

/// Allocation or wiring failure for one of the frame / per-draw GPU bind bundles.
#[derive(Debug, thiserror::Error)]
pub enum FrameGpuBindingsError {
    /// `@group(0)` frame uniforms / lights / cluster buffers could not be created.
    #[error("frame GPU resources: {0}")]
    Frame(#[from] FrameGpuInitError),
    /// `@group(2)` per-draw slab could not be created.
    #[error("per-draw resources: {0}")]
    PerDraw(#[from] PipelineBuildError),
}

/// `@group(0)` frame resources, empty `@group(1)` fallback, and `@group(2)` per-draw slab.
///
/// Built only via [`Self::try_new`] so attach does not leave a partially wired bind set.
pub struct FrameGpuBindings {
    /// Camera + lights + cluster buffers (`@group(0)`).
    pub(crate) frame_gpu: FrameGpuResources,
    /// Placeholder `@group(1)` for materials without per-material bindings.
    pub(crate) empty_material: EmptyMaterialBindGroup,
    /// Per-draw instance storage (`@group(2)`).
    pub(crate) per_draw: PerDrawResources,
}

impl FrameGpuBindings {
    /// Allocates all three bind domains; returns an error if any step fails (no partial bundle).
    pub fn try_new(
        device: &wgpu::Device,
        limits: Arc<GpuLimits>,
    ) -> Result<Self, FrameGpuBindingsError> {
        let frame_gpu = FrameGpuResources::new(device, Arc::clone(&limits))?;
        let empty_material = EmptyMaterialBindGroup::new(device);
        let per_draw = PerDrawResources::new(device, limits)?;
        Ok(Self {
            frame_gpu,
            empty_material,
            per_draw,
        })
    }

    /// Frame camera + lights bind group owner.
    pub fn frame_gpu(&self) -> &FrameGpuResources {
        &self.frame_gpu
    }

    /// Mutable frame globals (cluster resize, uniform upload).
    pub fn frame_gpu_mut(&mut self) -> &mut FrameGpuResources {
        &mut self.frame_gpu
    }

    /// Empty `@group(1)` bind group.
    pub fn empty_material(&self) -> &EmptyMaterialBindGroup {
        &self.empty_material
    }

    /// Per-draw slab (`@group(2)`).
    pub fn per_draw(&self) -> &PerDrawResources {
        &self.per_draw
    }

    /// Mutable per-draw slab.
    pub fn per_draw_mut(&mut self) -> &mut PerDrawResources {
        &mut self.per_draw
    }

    /// Cloned [`Arc`] bind groups for mesh forward (`@group(0)` frame + `@group(1)` empty material).
    pub fn mesh_forward_frame_bind_groups(&self) -> (Arc<wgpu::BindGroup>, Arc<wgpu::BindGroup>) {
        (
            self.frame_gpu.bind_group.clone(),
            self.empty_material.bind_group.clone(),
        )
    }
}
