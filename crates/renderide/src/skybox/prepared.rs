//! Prepared skybox draws produced during world-mesh forward preparation.

use std::sync::Arc;

/// Prepared draw that fills the forward color target after opaque world meshes.
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
    /// Dynamic offset for the material uniform arena, when the skybox material has one.
    pub material_uniform_dynamic_offset: Option<u32>,
    /// `@group(2)` draw-local skybox view uniform bind group.
    pub view_bind_group: Arc<wgpu::BindGroup>,
    /// Geometry used to draw the skybox.
    pub geometry: PreparedMaterialSkyboxGeometry,
}

/// Prepared material skybox geometry.
pub enum PreparedMaterialSkyboxGeometry {
    /// Fullscreen triangle geometry generated from `@builtin(vertex_index)`.
    FullscreenTriangle {
        /// Vertex count for the fullscreen draw.
        vertex_count: u32,
    },
    /// Fixed mesh geometry sourced from a build-time asset.
    Mesh {
        /// Cached fixed mesh used by the draw.
        mesh: Arc<PreparedMaterialSkyboxMesh>,
    },
}

/// Prepared fixed-mesh skybox geometry.
pub struct PreparedMaterialSkyboxMesh {
    /// Vertex buffer containing tightly packed `vec3<f32>` positions.
    pub vertex_buffer: wgpu::Buffer,
    /// Vertex count for the mesh draw.
    pub vertex_count: u32,
}

/// Prepared solid-color background resources.
pub struct PreparedClearColorSkybox {
    /// Cached render pipeline for the color background draw.
    pub pipeline: Arc<wgpu::RenderPipeline>,
    /// `@group(0)` bind group carrying the background color uniform.
    pub view_bind_group: Arc<wgpu::BindGroup>,
}
