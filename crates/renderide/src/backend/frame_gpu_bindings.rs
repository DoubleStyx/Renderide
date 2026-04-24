//! Transactional allocation of `@group(0)` frame resources, empty `@group(1)`, and the shared
//! `@group(2)` per-draw bind group layout.
//!
//! [`FrameGpuBindings::try_new`] succeeds only when all required resources are created, avoiding
//! a partially wired frame bind set. Per-draw [`crate::backend::PerDrawResources`] instances are
//! allocated lazily per view by [`crate::backend::FrameResourceManager`].

use std::sync::Arc;

use crate::gpu::GpuLimits;
use crate::materials::PipelineBuildError;
use thiserror::Error;

use super::frame_gpu::{EmptyMaterialBindGroup, FrameGpuResources};
use super::frame_gpu_error::FrameGpuInitError;
use crate::pipelines::raster::DebugWorldNormalsFamily;

/// Either frame globals failed to allocate, or the per-draw bind group layout could not be built.
#[derive(Debug, Error)]
pub enum FrameGpuBindingsError {
    /// `@group(0)` frame buffers / cluster bootstrap failed.
    #[error(transparent)]
    FrameGpuInit(#[from] FrameGpuInitError),
    /// `@group(2)` per-draw layout reflection failed.
    #[error(transparent)]
    PipelineBuild(#[from] PipelineBuildError),
}

/// All mesh-forward frame bind resources allocated together ([`FrameGpuBindings::try_new`]).
pub struct FrameGpuBindings {
    /// Camera + lights (`@group(0)`).
    pub frame_gpu: FrameGpuResources,
    /// Fallback material (`@group(1)`).
    pub empty_material: EmptyMaterialBindGroup,
    /// Shared `@group(2)` bind group layout for per-view [`crate::backend::PerDrawResources`].
    ///
    /// Derived once from naga reflection and shared across all per-view slab instances so the
    /// shader is not re-reflected every time a new view slab is created.
    pub per_draw_bind_group_layout: Arc<wgpu::BindGroupLayout>,
}

impl FrameGpuBindings {
    /// Allocates frame globals, empty material bind group, and the per-draw bind group layout
    /// in one step.
    ///
    /// On error, nothing is returned; callers must not treat any partial state as attached.
    pub fn try_new(
        device: &wgpu::Device,
        limits: Arc<GpuLimits>,
    ) -> Result<Self, FrameGpuBindingsError> {
        let frame_gpu = FrameGpuResources::new(device, Arc::clone(&limits))?;
        let empty_material = EmptyMaterialBindGroup::new(device);
        let per_draw_bind_group_layout =
            Arc::new(DebugWorldNormalsFamily::per_draw_bind_group_layout(device)?);
        Ok(Self {
            frame_gpu,
            empty_material,
            per_draw_bind_group_layout,
        })
    }
}
