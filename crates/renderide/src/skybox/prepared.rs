//! Prepared skybox draws produced during world-mesh forward preparation.

use std::sync::Arc;

/// Prepared draw that fills the forward color target before world meshes.
pub enum PreparedSkybox {
    /// Host material-driven skybox draw.
    Material(PreparedMaterialSkybox),
    /// Solid color background for host cameras using `CameraClearMode::Color`.
    ClearColor(PreparedClearColorSkybox),
}

/// Prepared material-driven skybox resources.
pub struct PreparedMaterialSkybox {
    /// Cached render pipeline for the skybox family and view target layout.
    pub pipeline: Arc<wgpu::RenderPipeline>,
    /// `@group(1)` material bind group resolved from the host material store.
    pub material_bind_group: Arc<wgpu::BindGroup>,
    /// `@group(2)` draw-local skybox view uniform bind group.
    pub view_bind_group: Arc<wgpu::BindGroup>,
}

/// Prepared solid-color background resources.
pub struct PreparedClearColorSkybox {
    /// Cached render pipeline for the color background draw.
    pub pipeline: Arc<wgpu::RenderPipeline>,
    /// `@group(0)` bind group carrying the background color uniform.
    pub view_bind_group: Arc<wgpu::BindGroup>,
}
