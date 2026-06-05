//! Fullscreen-triangle pipeline builders and stereo multiview helpers.

use std::num::NonZeroU32;
use std::sync::Arc;

use super::shader::create_wgsl_shader_module;
use crate::gpu_resource::RenderPipelineMap;
use crate::render_graph::context::RasterPassCtx;
use crate::render_graph::pass::RenderPassTemplate;

/// Descriptor for a fullscreen triangle render pipeline.
pub(crate) struct FullscreenRenderPipelineDesc<'a> {
    /// Debug label applied to the pipeline layout and render pipeline.
    pub(crate) label: &'a str,
    /// Bind group layouts used by the pipeline layout.
    pub(crate) bind_group_layouts: &'a [Option<&'a wgpu::BindGroupLayout>],
    /// Shader module containing `vs_main` and the selected fragment entry point.
    pub(crate) shader: &'a wgpu::ShaderModule,
    /// Fragment entry point for the pass.
    pub(crate) fragment_entry: &'a str,
    /// Single color attachment format.
    pub(crate) output_format: wgpu::TextureFormat,
    /// Optional color blend state.
    pub(crate) blend: Option<wgpu::BlendState>,
    /// Whether the pipeline records as a two-eye multiview pass.
    pub(crate) multiview_stereo: bool,
}

/// WGSL sources and labels for a mono/multiview fullscreen shader pair.
pub(crate) struct FullscreenShaderVariants<'a> {
    /// Debug label for the mono shader and pipeline.
    pub(crate) mono_label: &'a str,
    /// WGSL source for the mono shader.
    pub(crate) mono_source: &'a str,
    /// Debug label for the multiview shader and pipeline.
    pub(crate) multiview_label: &'a str,
    /// WGSL source for the multiview shader.
    pub(crate) multiview_source: &'a str,
}

impl FullscreenShaderVariants<'_> {
    /// Selects the label and WGSL source for the requested view mode.
    fn select(&self, multiview_stereo: bool) -> (&str, &str) {
        if multiview_stereo {
            (self.multiview_label, self.multiview_source)
        } else {
            (self.mono_label, self.mono_source)
        }
    }
}

/// Descriptor for selecting and building a cached fullscreen pipeline variant.
pub(crate) struct FullscreenPipelineVariantDesc<'a> {
    /// Color target format used as the per-cache key.
    pub(crate) output_format: wgpu::TextureFormat,
    /// Whether the multiview shader and pipeline cache should be used.
    pub(crate) multiview_stereo: bool,
    /// Cache for mono render pipelines keyed by target format.
    pub(crate) mono: &'a RenderPipelineMap<wgpu::TextureFormat>,
    /// Cache for multiview render pipelines keyed by target format.
    pub(crate) multiview: &'a RenderPipelineMap<wgpu::TextureFormat>,
    /// Paired WGSL shader labels and sources.
    pub(crate) shader: FullscreenShaderVariants<'a>,
    /// Bind group layouts used by the generated pipeline layout.
    pub(crate) bind_group_layouts: &'a [Option<&'a wgpu::BindGroupLayout>],
    /// Short pass name included in pipeline creation logs.
    pub(crate) log_name: &'a str,
}

/// Returns the `0b11` multiview mask used for stereo eye layers.
pub(crate) fn stereo_multiview_mask() -> Option<NonZeroU32> {
    NonZeroU32::new(3)
}

/// Returns the stereo multiview mask when `multiview_stereo` is active.
pub(crate) fn multiview_mask(multiview_stereo: bool) -> Option<NonZeroU32> {
    multiview_stereo.then(stereo_multiview_mask).flatten()
}

/// Returns the stereo mask for active multiview, otherwise preserves a template mask.
pub(crate) fn stereo_mask_or_template(
    multiview_stereo: bool,
    template_mask: Option<NonZeroU32>,
) -> Option<NonZeroU32> {
    if multiview_stereo {
        stereo_multiview_mask()
    } else {
        template_mask
    }
}

/// Returns the stereo mask override for the current raster frame, otherwise preserves a template mask.
pub(crate) fn raster_stereo_mask_override(
    ctx: &RasterPassCtx<'_, '_>,
    template: &RenderPassTemplate,
) -> Option<NonZeroU32> {
    let stereo = ctx.frame.view.multiview_stereo;
    stereo_mask_or_template(stereo, template.multiview_mask)
}

/// Builds a standard fullscreen triangle render pipeline.
pub(crate) fn create_fullscreen_render_pipeline(
    device: &wgpu::Device,
    desc: FullscreenRenderPipelineDesc<'_>,
) -> wgpu::RenderPipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(desc.label),
        bind_group_layouts: desc.bind_group_layouts,
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(desc.label),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: desc.shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: desc.shader,
            entry_point: Some(desc.fragment_entry),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: desc.output_format,
                blend: desc.blend,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: multiview_mask(desc.multiview_stereo),
        cache: None,
    });
    crate::profiling::note_resource_churn!(RenderPipeline, "render_graph::fullscreen_pipeline");
    pipeline
}

/// Returns or builds a fullscreen pipeline from paired mono/multiview caches.
pub(crate) fn fullscreen_pipeline_variant(
    device: &wgpu::Device,
    desc: FullscreenPipelineVariantDesc<'_>,
) -> Arc<wgpu::RenderPipeline> {
    let map = if desc.multiview_stereo {
        desc.multiview
    } else {
        desc.mono
    };
    map.get_or_create(desc.output_format, |output_format| {
        logger::debug!(
            "{}: building pipeline (dst format = {:?}, multiview = {})",
            desc.log_name,
            output_format,
            desc.multiview_stereo
        );
        let (label, source) = desc.shader.select(desc.multiview_stereo);
        let shader = create_wgsl_shader_module(device, label, source);
        create_fullscreen_render_pipeline(
            device,
            FullscreenRenderPipelineDesc {
                label,
                bind_group_layouts: desc.bind_group_layouts,
                shader: &shader,
                fragment_entry: "fs_main",
                output_format: *output_format,
                blend: None,
                multiview_stereo: desc.multiview_stereo,
            },
        )
    })
}
