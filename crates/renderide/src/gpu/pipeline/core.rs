//! Core pipeline abstractions: RenderPipeline trait, UniformData, and shared constants.

use super::super::mesh::GpuMeshBuffers;
use nalgebra::Matrix4;

/// Converts a nalgebra Matrix4 to WGSL column-major layout.
///
/// WGSL `mat4x4f` expects matrices in column-major order: column 0 (4 floats), then column 1,
/// then column 2, then column 3. Nalgebra is also column-major, so this explicitly ensures
/// the byte layout matches WGSL expectations when uploading to uniform buffers.
pub fn matrix4_to_wgsl_column_major(mat: &Matrix4<f32>) -> [[f32; 4]; 4] {
    let mut out = [[0.0f32; 4]; 4];
    for c in 0..4 {
        for r in 0..4 {
            out[c][r] = mat[(r, c)];
        }
    }
    out
}

/// Per-draw uniform data; pipelines extract what they need.
#[derive(Clone, Copy)]
pub enum UniformData<'a> {
    /// MVP + model for non-skinned.
    Simple {
        mvp: Matrix4<f32>,
        model: Matrix4<f32>,
    },
    /// MVP + bone matrices for skinned.
    Skinned {
        mvp: Matrix4<f32>,
        bone_matrices: &'a [[[f32; 4]; 4]],
    },
}

/// Abstraction for a render pipeline (shader, bind groups, draw logic).
pub trait RenderPipeline {
    /// Binds this pipeline to the render pass. Call once per pipeline group.
    fn bind_pipeline(&self, _pass: &mut wgpu::RenderPass) {
        // Default: no-op for placeholder pipelines.
    }

    /// Binds per-draw bind group and dynamic offset. Call once per draw.
    /// `batch_index` selects the uniform slot; `draw_bind_group` is used by skinned pipelines.
    fn bind_draw(
        &self,
        _pass: &mut wgpu::RenderPass,
        _batch_index: Option<u32>,
        _frame_index: u64,
        _draw_bind_group: Option<&wgpu::BindGroup>,
    ) {
        // Default: no-op for placeholder pipelines.
    }

    /// Binds this pipeline and its bind groups to the render pass.
    /// `frame_index` is used by batched pipelines for ring buffer offset; ignored by others.
    /// `draw_bind_group` is used by skinned pipeline for per-draw bind group (uniform + blendshape buffer).
    fn bind(
        &self,
        pass: &mut wgpu::RenderPass,
        batch_index: Option<u32>,
        frame_index: u64,
        draw_bind_group: Option<&wgpu::BindGroup>,
    ) {
        self.bind_pipeline(pass);
        self.bind_draw(pass, batch_index, frame_index, draw_bind_group);
    }

    /// Binds per-frame scene bind group (e.g. view position, lights, cluster buffers).
    /// Called once per pipeline group before bind_draw. Default no-op.
    fn bind_scene(
        &self,
        _pass: &mut wgpu::RenderPass,
        _scene_bind_group: Option<&wgpu::BindGroup>,
    ) {
        // Default: no-op for pipelines that don't use scene bind group.
    }

    /// Draws a non-skinned mesh. No-op for pipelines that only support skinned.
    fn draw_mesh(
        &self,
        pass: &mut wgpu::RenderPass,
        buffers: &GpuMeshBuffers,
        _uniforms: &UniformData<'_>,
    ) {
        let _ = (pass, buffers, _uniforms);
    }

    /// Sets vertex and index buffers for a non-skinned mesh.
    ///
    /// Used by the recording loop to avoid redundant buffer binding when consecutive draws
    /// share the same mesh. No-op for pipelines that only support skinned.
    fn set_mesh_buffers(&self, _pass: &mut wgpu::RenderPass, _buffers: &GpuMeshBuffers) {
        // Default: no-op for skinned-only pipelines.
    }

    /// Issues draw_indexed calls for a non-skinned mesh. Buffers must already be set.
    ///
    /// Used by the recording loop after optionally calling [`set_mesh_buffers`](Self::set_mesh_buffers).
    /// No-op for pipelines that only support skinned.
    fn draw_mesh_indexed(&self, _pass: &mut wgpu::RenderPass, _buffers: &GpuMeshBuffers) {
        // Default: no-op for skinned-only pipelines.
    }

    /// Whether this pipeline supports instanced drawing for same-mesh runs.
    /// When true, the recording loop may batch consecutive same-mesh draws into one instanced call.
    fn supports_instancing(&self) -> bool {
        false
    }

    /// Issues an instanced draw_indexed for a run of same-mesh draws.
    ///
    /// Called only when [`supports_instancing`](Self::supports_instancing) is true and
    /// `instance_count` > 1. Caller must have already bound with [`bind_draw`](Self::bind_draw)
    /// using `run_start` as batch_index. Default falls back to per-draw loop.
    fn draw_mesh_indexed_instanced(
        &self,
        _pass: &mut wgpu::RenderPass,
        _buffers: &GpuMeshBuffers,
        _instance_count: u32,
    ) {
        // Default: no-op; recording loop uses per-draw path when not supported.
    }

    /// Draws a skinned mesh. No-op for pipelines that only support non-skinned.
    fn draw_skinned(
        &self,
        pass: &mut wgpu::RenderPass,
        buffers: &GpuMeshBuffers,
        _uniforms: &UniformData<'_>,
    ) {
        let _ = (pass, buffers, _uniforms);
    }

    /// Sets vertex and index buffers for a skinned mesh.
    ///
    /// Used by the recording loop to avoid redundant buffer binding when consecutive draws
    /// share the same mesh. No-op for pipelines that only support non-skinned.
    fn set_skinned_buffers(&self, _pass: &mut wgpu::RenderPass, _buffers: &GpuMeshBuffers) {
        // Default: no-op for non-skinned pipelines.
    }

    /// Issues draw_indexed calls for a skinned mesh. Buffers must already be set.
    ///
    /// Used by the recording loop after optionally calling [`set_skinned_buffers`](Self::set_skinned_buffers).
    /// No-op for pipelines that only support non-skinned.
    fn draw_skinned_indexed(&self, _pass: &mut wgpu::RenderPass, _buffers: &GpuMeshBuffers) {
        // Default: no-op for non-skinned pipelines.
    }

    /// Uploads batched uniforms for non-skinned draws. No-op for pipelines that don't batch.
    /// `frame_index` advances each frame for ring buffer region selection.
    fn upload_batch(
        &self,
        _queue: &wgpu::Queue,
        _mvp_models: &[(Matrix4<f32>, Matrix4<f32>)],
        _frame_index: u64,
    ) {
    }

    /// Uploads batched uniforms for overlay stencil draws (includes clip_rect).
    /// Default calls `upload_batch` with mvp_models only.
    #[allow(clippy::type_complexity)]
    fn upload_batch_overlay(
        &self,
        queue: &wgpu::Queue,
        items: &[(Matrix4<f32>, Matrix4<f32>, Option<[f32; 4]>)],
        frame_index: u64,
    ) {
        let mvp_models: Vec<_> = items.iter().map(|(m, p, _)| (*m, *p)).collect();
        self.upload_batch(queue, &mvp_models, frame_index);
    }

    /// Uploads skinned uniforms for a single draw. No-op for non-skinned pipelines.
    fn upload_skinned(
        &self,
        _queue: &wgpu::Queue,
        _mvp: Matrix4<f32>,
        _bone_matrices: &[[[f32; 4]; 4]],
    ) {
    }

    /// Uploads batched skinned uniforms to the ring buffer. No-op for pipelines that don't batch.
    #[allow(clippy::type_complexity)]
    fn upload_skinned_batch(
        &self,
        _queue: &wgpu::Queue,
        _items: &[(Matrix4<f32>, &[[[f32; 4]; 4]], Option<&[f32]>, u32)],
        _frame_index: u64,
    ) {
    }

    /// Creates a per-draw bind group for skinned pipeline (uniform + blendshape buffer).
    /// Returns None for non-skinned pipelines.
    fn create_skinned_draw_bind_group(
        &self,
        _device: &wgpu::Device,
        _buffers: &GpuMeshBuffers,
    ) -> Option<wgpu::BindGroup> {
        None
    }

    /// Writes scene uniforms to the pipeline's scene uniform buffer.
    /// Call before using a cached scene bind group. No-op for pipelines without scene bind group.
    fn write_scene_uniform(&self, _queue: &wgpu::Queue, _scene: &[u8]) {}

    /// Creates the per-frame scene bind group for pipelines that use it (e.g. PBR).
    /// Returns None for pipelines that don't use a scene bind group.
    #[allow(clippy::too_many_arguments)]
    fn create_scene_bind_group(
        &self,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _view_position: [f32; 3],
        _view_space_z_coeffs: [f32; 4],
        _cluster_count_x: u32,
        _cluster_count_y: u32,
        _cluster_count_z: u32,
        _near_clip: f32,
        _far_clip: f32,
        _light_count: u32,
        _viewport_width: u32,
        _viewport_height: u32,
        _light_buffer: &wgpu::Buffer,
        _cluster_light_counts: &wgpu::Buffer,
        _cluster_light_indices: &wgpu::Buffer,
    ) -> Option<wgpu::BindGroup> {
        None
    }
}

/// Maximum instances per instanced draw when batching same-mesh draws.
pub const MAX_INSTANCE_RUN: u32 = 64;

/// Alignment for dynamic uniform buffer offsets (wgpu/Vulkan minimum).
pub(crate) const UNIFORM_ALIGNMENT: u64 = 256;
/// Number of frame regions in the ring buffer (avoids overwriting in-flight data).
///
/// 4 frames allows the CPU to stay one extra GPU-frame ahead before the ring-buffer fence stalls
/// it waiting for the oldest submission.  With RTAO GPU time ~12 ms and CPU pre-render ~17 ms,
/// 3 frames was tight enough to cause a stall every frame; 4 removes it.
///
/// Memory cost: 4 × 16 384 × 256 B ≈ 16 MB (rigid ring) + 4 × 512 × 17 152 B ≈ 35 MB (skinned ring).
/// [`crate::gpu::GpuFrameScheduler`] caps concurrent uniform-ring submissions using this value.
pub const NUM_FRAMES_IN_FLIGHT: usize = 4;
/// Slots per frame region. Draws exceeding this are split into multiple chunks.
pub(crate) const SLOTS_PER_FRAME: usize = 16_384;

/// Slot stride for skinned uniforms (mvp + 256 bones + num_blendshapes + 128 weights).
/// Aligned to 256 for dynamic offset. Struct size 16964, round up to 17152.
pub(crate) const SKINNED_SLOT_STRIDE: u64 = 17_152;
/// Slots per frame for skinned draws. Smaller than `SLOTS_PER_FRAME` to limit memory (~25 MB).
pub(crate) const SKINNED_SLOTS_PER_FRAME: usize = 512;

/// Maximum blendshape weights per draw. Meshes with more blendshapes are truncated; weights
/// beyond this index are ignored. The host (SkinnedMeshRendererManager) has no limit.
pub const MAX_BLENDSHAPE_WEIGHTS: usize = 128;

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies that matrix4_to_wgsl_column_major produces byte layout matching WGSL mat4x4f:
    /// column 0 at offset 0, column 1 at 16, column 2 at 32, column 3 at 48.
    #[test]
    fn test_matrix4_to_wgsl_column_major_layout() {
        let mat = Matrix4::from_fn(|r, c| (r * 4 + c) as f32);
        let arr = matrix4_to_wgsl_column_major(&mat);
        let bytes: &[u8] = bytemuck::bytes_of(&arr);
        assert_eq!(bytes.len(), 64);
        for c in 0..4 {
            let offset = c * 16;
            for r in 0..4 {
                let val = f32::from_le_bytes(
                    bytes[offset + r * 4..offset + r * 4 + 4]
                        .try_into()
                        .expect("4-byte slice for f32"),
                );
                assert_eq!(val, (r * 4 + c) as f32, "col {} row {}", c, r);
            }
        }
    }

    /// Verifies translation in column 3 (typical model matrix layout).
    #[test]
    fn test_matrix4_wgsl_translation_column3() {
        let mat = Matrix4::new_translation(&nalgebra::Vector3::new(1.0, 2.0, 3.0));
        let arr = matrix4_to_wgsl_column_major(&mat);
        let bytes: &[u8] = bytemuck::bytes_of(&arr);
        let col3_offset = 48;
        let tx = f32::from_le_bytes(
            bytes[col3_offset..col3_offset + 4]
                .try_into()
                .expect("4-byte slice for f32"),
        );
        let ty = f32::from_le_bytes(
            bytes[col3_offset + 4..col3_offset + 8]
                .try_into()
                .expect("4-byte slice for f32"),
        );
        let tz = f32::from_le_bytes(
            bytes[col3_offset + 8..col3_offset + 12]
                .try_into()
                .expect("4-byte slice for f32"),
        );
        assert!((tx - 1.0).abs() < 1e-6);
        assert!((ty - 2.0).abs() < 1e-6);
        assert!((tz - 3.0).abs() < 1e-6);
    }
}
