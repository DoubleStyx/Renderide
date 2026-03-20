//! Skinned overlay stencil pipelines for GraphicsChunk masking.
//!
//! Three pipeline phases (Content, MaskWrite, MaskClear) each wrap a [`SkinnedPipeline`]
//! configured with the appropriate stencil state. All three delegate every
//! [`RenderPipeline`] method identically to their inner pipeline; the
//! `impl_overlay_stencil_skinned!` macro generates these impls to avoid the 3×8-method
//! repetition that previously lived here.

use nalgebra::Matrix4;

use super::super::mesh::GpuMeshBuffers;
use super::core::{RenderPipeline, UniformData};
use super::overlay_stencil::OverlayStencilPhase;
use super::skinned::SkinnedPipeline;

// ─── Macro ────────────────────────────────────────────────────────────────────

/// Generates a `RenderPipeline` impl that forwards every method to `self.inner`.
///
/// The stencil phase difference is baked into the inner [`SkinnedPipeline`] at construction;
/// at runtime all three types delegate identically.
macro_rules! impl_overlay_stencil_skinned {
    ($ty:ty) => {
        impl RenderPipeline for $ty {
            fn bind_pipeline(&self, pass: &mut wgpu::RenderPass) {
                self.inner.bind_pipeline(pass);
            }

            fn bind_draw(
                &self,
                pass: &mut wgpu::RenderPass,
                batch_index: Option<u32>,
                frame_index: u64,
                draw_bind_group: Option<&wgpu::BindGroup>,
            ) {
                self.inner
                    .bind_draw(pass, batch_index, frame_index, draw_bind_group);
            }

            fn create_skinned_draw_bind_group(
                &self,
                device: &wgpu::Device,
                buffers: &GpuMeshBuffers,
            ) -> Option<wgpu::BindGroup> {
                self.inner.create_skinned_draw_bind_group(device, buffers)
            }

            fn draw_skinned(
                &self,
                pass: &mut wgpu::RenderPass,
                buffers: &GpuMeshBuffers,
                uniforms: &UniformData<'_>,
            ) {
                self.inner.draw_skinned(pass, buffers, uniforms);
            }

            fn set_skinned_buffers(&self, pass: &mut wgpu::RenderPass, buffers: &GpuMeshBuffers) {
                self.inner.set_skinned_buffers(pass, buffers);
            }

            fn draw_skinned_indexed(&self, pass: &mut wgpu::RenderPass, buffers: &GpuMeshBuffers) {
                self.inner.draw_skinned_indexed(pass, buffers);
            }

            #[allow(clippy::type_complexity)]
            fn upload_skinned_batch(
                &self,
                queue: &wgpu::Queue,
                items: &[(Matrix4<f32>, &[[[f32; 4]; 4]], Option<&[f32]>, u32)],
                frame_index: u64,
            ) {
                self.inner.upload_skinned_batch(queue, items, frame_index);
            }
        }
    };
}

// ─── Pipeline types ────────────────────────────────────────────────────────────

/// Skinned overlay stencil Content pipeline: compare=Equal, pass_op=Keep, write_mask=0.
///
/// Same semantics as [`super::overlay_stencil::OverlayStencilPipeline`] but for
/// bone-weighted meshes. Call `set_stencil_reference` before each draw.
pub struct OverlayStencilSkinnedPipeline {
    inner: SkinnedPipeline,
}

impl OverlayStencilSkinnedPipeline {
    /// Creates a skinned overlay stencil Content pipeline.
    pub fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> Self {
        Self {
            inner: SkinnedPipeline::new(device, config, Some(OverlayStencilPhase::Content), false),
        }
    }
}

/// Skinned overlay stencil MaskWrite pipeline: compare=Always, pass_op=Replace, write_mask=0xFF.
///
/// Writes the stencil mask shape for GraphicsChunk masking.
pub struct OverlayStencilMaskWriteSkinnedPipeline {
    inner: SkinnedPipeline,
}

impl OverlayStencilMaskWriteSkinnedPipeline {
    /// Creates a skinned overlay stencil MaskWrite pipeline.
    pub fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> Self {
        Self {
            inner: SkinnedPipeline::new(
                device,
                config,
                Some(OverlayStencilPhase::MaskWrite),
                false,
            ),
        }
    }
}

/// Skinned overlay stencil MaskClear pipeline: compare=Always, pass_op=Zero, write_mask=0xFF.
///
/// Clears the stencil mask after a GraphicsChunk group finishes.
pub struct OverlayStencilMaskClearSkinnedPipeline {
    inner: SkinnedPipeline,
}

impl OverlayStencilMaskClearSkinnedPipeline {
    /// Creates a skinned overlay stencil MaskClear pipeline.
    pub fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> Self {
        Self {
            inner: SkinnedPipeline::new(
                device,
                config,
                Some(OverlayStencilPhase::MaskClear),
                false,
            ),
        }
    }
}

// ─── RenderPipeline impls (macro-generated) ────────────────────────────────────

impl_overlay_stencil_skinned!(OverlayStencilSkinnedPipeline);
impl_overlay_stencil_skinned!(OverlayStencilMaskWriteSkinnedPipeline);
impl_overlay_stencil_skinned!(OverlayStencilMaskClearSkinnedPipeline);
