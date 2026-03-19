//! Uniform struct layouts for pipeline bindings.

/// MVP + model matrix for non-skinned pipelines.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Uniforms {
    pub mvp: [[f32; 4]; 4],
    pub model: [[f32; 4]; 4],
}

/// Overlay stencil uniforms: MVP, model, and clip rect (x, y, width, height).
/// Pad to 256 bytes for dynamic offset alignment.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct OverlayStencilUniforms {
    pub mvp: [[f32; 4]; 4],
    pub model: [[f32; 4]; 4],
    pub clip_rect: [f32; 4],
    pub _pad: [f32; 16],
}

/// Scene uniforms for PBR pipeline: view position, cluster counts, light count.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SceneUniforms {
    pub view_position: [f32; 3],
    pub _pad0: f32,
    pub cluster_count_x: u32,
    pub cluster_count_y: u32,
    pub light_count: u32,
    pub _pad1: u32,
}

/// MVP + 256 bone matrices + blendshape weights for skinned pipeline.
///
/// Blendshape weights are applied in the vertex shader before bone skinning.
/// Weights stored as 32× vec4 ([`super::core::MAX_BLENDSHAPE_WEIGHTS`] floats) for WGSL uniform 16-byte alignment.
/// Meshes with more than [`super::core::MAX_BLENDSHAPE_WEIGHTS`] blendshapes are truncated; consider a storage
/// buffer for unbounded weight counts if needed.
/// Padding before blendshape_weights matches WGSL layout (vec4 requires 16-byte alignment).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct SkinnedUniforms {
    pub mvp: [[f32; 4]; 4],
    pub bone_matrices: [[[f32; 4]; 4]; 256],
    pub num_blendshapes: u32,
    pub num_vertices: u32,
    /// Padding so blendshape_weights is 16-byte aligned (WGSL vec4 alignment).
    pub _pad: [u32; 2],
    /// Blendshape weights packed as 32 vec4s ([`super::core::MAX_BLENDSHAPE_WEIGHTS`] floats). Weights beyond
    /// index 127 are truncated.
    pub blendshape_weights: [[f32; 4]; 32],
}
