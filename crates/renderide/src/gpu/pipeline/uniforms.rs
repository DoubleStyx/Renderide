//! Uniform struct layouts for pipeline bindings.

#[cfg(test)]
mod scene_uniforms_tests {
    use super::{SceneUniforms, Uniforms};
    use std::mem::size_of;

    /// WGSL `SceneUniforms` must stay byte-compatible with the PBR family shaders.
    #[test]
    fn scene_uniforms_size_matches_wgsl_layout() {
        assert_eq!(
            size_of::<SceneUniforms>(),
            64,
            "SceneUniforms must be 64 bytes for WGSL uniform alignment"
        );
        assert_eq!(size_of::<SceneUniforms>() % 16, 0);
    }

    /// Non-skinned ring slot must match WGSL `UniformsSlot` / `uniform_ring.wgsl` (256 bytes).
    #[test]
    fn uniforms_slot_stride_is_256_bytes() {
        assert_eq!(size_of::<Uniforms>(), 256);
        assert_eq!(
            size_of::<Uniforms>() as u64,
            super::super::core::UNIFORM_ALIGNMENT
        );
    }
}

/// MVP + model matrix for non-skinned pipelines, with optional host-driven PBR factors in the uniform ring.
///
/// Byte layout matches [`uniform_ring::UniformsSlot`](../../../../wgsl_modules/uniform_ring.wgsl) and all WGSL
/// `UniformsSlot` copies in debug MRT and PBR sources (256 bytes per dynamic slot).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Uniforms {
    pub mvp: [[f32; 4]; 4],
    pub model: [[f32; 4]; 4],
    /// RGB when `.w >= 0.5`; else fragment shaders use built-in `(0.8, 0.8, 0.8)`.
    pub host_base_color: [f32; 4],
    /// Metallic (`.x`) and roughness (`.y`) when `.z >= 0.5`.
    pub host_metallic_roughness: [f32; 4],
    pub _pad: [f32; 24],
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

/// Scene uniforms for PBR pipeline: view position, cluster depth row, cluster counts, clip planes, light count, viewport.
///
/// Layout matches WGSL `SceneUniforms` in the PBR shader sources (64 bytes, 16-byte aligned).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SceneUniforms {
    pub view_position: [f32; 3],
    pub _pad0: f32,
    /// Coefficients for view-space Z: `dot(xyz, world_position) + w`, matching clustered-light culling eye space.
    pub view_space_z_coeffs: [f32; 4],
    pub cluster_count_x: u32,
    pub cluster_count_y: u32,
    pub cluster_count_z: u32,
    pub near_clip: f32,
    pub far_clip: f32,
    pub light_count: u32,
    /// Viewport width in pixels; fragment cluster XY uses the same 16px tiles as clustered light compute.
    pub viewport_width: u32,
    /// Viewport height in pixels.
    pub viewport_height: u32,
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
