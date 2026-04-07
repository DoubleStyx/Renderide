//! Uniform layout for ray-traced shadow tuning and atlas sampling in PBR ray-query shaders.

/// WGSL `shadow_mode`: trace shadow rays in the fragment shader.
pub const RT_SHADOW_MODE_TRACE: u32 = 0;
/// WGSL `shadow_mode`: sample visibility from the compute-filled shadow atlas ([`crate::render::pass::RtShadowComputePass`]).
pub const RT_SHADOW_MODE_ATLAS: u32 = 1;

/// GPU uniform for soft shadow sample count, cone width, frame jitter, and atlas metadata.
///
/// Bound at `@group(1) @binding(5)` in PBR ray-query WGSL. Must match the `RtShadowUniforms`
/// struct in [`super::shaders::pbr_ray_query`] sources (48 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RtShadowUniforms {
    /// Soft shadow ray count, clamped to 1–16 in WGSL. Hard shadows always use one ray.
    pub soft_shadow_sample_count: u32,
    /// Multiplier for the soft cone offset (1.0 matches the historical 0.025 tangent offset).
    pub soft_cone_scale: f32,
    /// Per-frame value for hashing soft shadow directions (reduces static speckle).
    pub frame_counter: u32,
    /// [`RT_SHADOW_MODE_TRACE`] or [`RT_SHADOW_MODE_ATLAS`].
    pub shadow_mode: u32,
    pub full_viewport_width: u32,
    pub full_viewport_height: u32,
    pub shadow_atlas_width: u32,
    pub shadow_atlas_height: u32,
    /// Same origin added back when decoding MRT positions (camera-relative G-buffer).
    pub gbuffer_origin: [f32; 3],
    pub _pad0: f32,
}

/// Optional scene bind group inputs for PBR ray-query pipelines (group 1, bindings 5–7).
pub struct RtShadowSceneBind<'a> {
    /// [`RtShadowUniforms`] uploaded each frame.
    pub uniform_buffer: &'a wgpu::Buffer,
    /// Layer `i` is visibility for cluster slot `i` (see [`crate::render::pass::RtShadowComputePass`]).
    pub atlas_view: &'a wgpu::TextureView,
    pub sampler: &'a wgpu::Sampler,
}

#[cfg(test)]
mod rt_shadow_uniforms_tests {
    use super::*;
    use std::mem::size_of;

    #[test]
    fn rt_shadow_uniforms_size_matches_wgsl() {
        assert_eq!(size_of::<RtShadowUniforms>(), 48);
        assert_eq!(size_of::<RtShadowUniforms>() % 16, 0);
    }
}
