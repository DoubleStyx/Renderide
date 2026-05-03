//! Small GPU cache primitives for render-graph effect passes.
//!
//! Internal organisation:
//!
//! - [`once`] -- one-shot lazy slot ([`OnceGpu`]).
//! - [`cache`] -- generic locked map with double-check insertion (private).
//! - [`pipeline`] -- typed wrapper around the generic cache for [`wgpu::RenderPipeline`].
//! - [`bindgroup`] -- typed wrapper for [`wgpu::BindGroup`].
//! - [`shader`] -- WGSL shader-module construction helper.
//! - [`samplers`] -- sampler / view / uniform-buffer helpers.
//! - [`fullscreen`] -- fullscreen-triangle pipeline builders + stereo multiview mask helpers.

mod bindgroup;
mod cache;
mod fullscreen;
mod once;
mod pipeline;
mod samplers;
mod shader;

pub(crate) use bindgroup::BindGroupMap;
pub(crate) use fullscreen::{
    FullscreenPipelineVariantDesc, FullscreenRenderPipelineDesc, FullscreenShaderVariants,
    create_fullscreen_render_pipeline, fullscreen_pipeline_variant, raster_stereo_mask_override,
    stereo_mask_or_template,
};
pub(crate) use once::OnceGpu;
pub(crate) use pipeline::RenderPipelineMap;
pub(crate) use samplers::{
    create_d2_array_view, create_linear_clamp_sampler, create_uniform_buffer,
};
pub(crate) use shader::create_wgsl_shader_module;

// Bind-group layout entry helpers moved to `crate::gpu::bind_layout`. Re-exported here so
// existing render_graph internal callers keep using `super::gpu_cache::*` paths.
#[expect(
    unused_imports,
    reason = "back-compat: render_graph internals reach these via super::gpu_cache::*"
)]
pub(crate) use crate::gpu::bind_layout::{
    fragment_filterable_d2_array_entry, fragment_filtering_sampler_entry, sampler_layout_entry,
    storage_texture_layout_entry, texture_layout_entry, uniform_buffer_layout_entry,
};

#[cfg(test)]
mod tests {
    use crate::gpu::bind_layout::texture_layout_entry;

    #[test]
    fn texture_layout_entry_uses_requested_binding_shape() {
        let entry = texture_layout_entry(
            7,
            wgpu::ShaderStages::COMPUTE,
            wgpu::TextureSampleType::Depth,
            wgpu::TextureViewDimension::D2Array,
            true,
        );

        assert_eq!(entry.binding, 7);
        assert_eq!(entry.visibility, wgpu::ShaderStages::COMPUTE);
        assert!(matches!(
            entry.ty,
            wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Depth,
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled: true,
            }
        ));
    }
}
