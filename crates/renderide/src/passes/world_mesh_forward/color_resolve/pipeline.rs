//! Cached pipelines, bind layouts, and per-frame UBO for [`super::WorldMeshForwardColorResolvePass`].
//!
//! The pass replaces wgpu's automatic linear MSAA color resolve with a Karis HDR-aware bracket
//! (compress / linear-average / uncompress) so contrast edges between very bright and very dark
//! samples don't alias under tonemapping. Bind layouts:
//!
//! - **Mono**: `params: ResolveParams` (UBO, sample count) + `src_msaa: texture_multisampled_2d<f32>`
//! - **Stereo / multiview**: `params` UBO + two `texture_multisampled_2d<f32>` bindings, one per
//!   eye layer of the multisampled HDR scene-color source. naga 29 does not yet expose
//!   `texture_multisampled_2d_array`, so the shader picks between the two bindings using
//!   `@builtin(view_index)` (uniform within a multiview draw).

use std::sync::Arc;

use crate::embedded_shaders::{MSAA_RESOLVE_HDR_DEFAULT_WGSL, MSAA_RESOLVE_HDR_MULTIVIEW_WGSL};
use crate::gpu::bind_layout::{texture_layout_entry, uniform_buffer_layout_entry};
use crate::render_graph::gpu_cache::{
    FullscreenPipelineVariantDesc, FullscreenShaderVariants, OnceGpu, RenderPipelineMap,
    create_uniform_buffer, fullscreen_pipeline_variant,
};

/// Debug label for the mono pipeline.
const PIPELINE_LABEL_MONO: &str = "msaa_resolve_hdr_default";
/// Debug label for the multiview pipeline.
const PIPELINE_LABEL_MULTIVIEW: &str = "msaa_resolve_hdr_multiview";

/// CPU-side `ResolveParams` mirror for the WGSL UBO.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct ResolveParamsUbo {
    /// Runtime MSAA sample count for the source attachment (1, 2, 4, or 8).
    pub sample_count: u32,
    /// Padding so the buffer matches WGSL's 16-byte UBO alignment.
    pub _pad: [u32; 3],
}

impl ResolveParamsUbo {
    /// Size in bytes of the WGSL `ResolveParams` struct (one `u32` plus 12 bytes of padding).
    pub const SIZE: u64 = size_of::<Self>() as u64;
}

/// GPU state shared across all MSAA color resolve invocations: bind layouts, pipelines, and the
/// per-frame `ResolveParams` UBO.
#[derive(Default)]
pub(super) struct MsaaResolveHdrPipelineCache {
    /// Bind group layout for the mono resolve variant.
    bind_group_layout_mono: OnceGpu<wgpu::BindGroupLayout>,
    /// Bind group layout for the multiview resolve variant.
    bind_group_layout_multiview: OnceGpu<wgpu::BindGroupLayout>,
    /// One pipeline per output color format (matches scene_color_hdr's runtime format).
    mono: RenderPipelineMap<wgpu::TextureFormat>,
    /// Same, but with `multiview_mask = 3` so the shader runs once per eye layer.
    multiview: RenderPipelineMap<wgpu::TextureFormat>,
    /// Lazily-allocated UBO holding the live sample count. Re-uploaded each frame via
    /// [`wgpu::Queue::write_buffer`] before the pass records its draw.
    params_ubo: OnceGpu<wgpu::Buffer>,
}

impl MsaaResolveHdrPipelineCache {
    /// Returns the per-frame `ResolveParams` UBO, lazily creating it on first call.
    pub(super) fn params_ubo(&self, device: &wgpu::Device) -> &wgpu::Buffer {
        self.params_ubo.get_or_create(|| {
            create_uniform_buffer(device, "msaa_resolve_hdr_params", ResolveParamsUbo::SIZE)
        })
    }

    /// Bind group layout for the mono variant: `params` + one `texture_multisampled_2d<f32>`.
    pub(super) fn bind_group_layout_mono(&self, device: &wgpu::Device) -> &wgpu::BindGroupLayout {
        self.bind_group_layout_mono.get_or_create(|| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("msaa_resolve_hdr_mono_bgl"),
                entries: &[
                    uniform_buffer_layout_entry(
                        0,
                        wgpu::ShaderStages::FRAGMENT,
                        wgpu::BufferSize::new(ResolveParamsUbo::SIZE),
                    ),
                    texture_layout_entry(
                        1,
                        wgpu::ShaderStages::FRAGMENT,
                        wgpu::TextureSampleType::Float { filterable: false },
                        wgpu::TextureViewDimension::D2,
                        true,
                    ),
                ],
            })
        })
    }

    /// Bind group layout for the multiview variant: `params` + two `texture_multisampled_2d<f32>`
    /// bindings (one per eye layer of the source MSAA scene color).
    pub(super) fn bind_group_layout_multiview(
        &self,
        device: &wgpu::Device,
    ) -> &wgpu::BindGroupLayout {
        self.bind_group_layout_multiview.get_or_create(|| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("msaa_resolve_hdr_multiview_bgl"),
                entries: &[
                    uniform_buffer_layout_entry(
                        0,
                        wgpu::ShaderStages::FRAGMENT,
                        wgpu::BufferSize::new(ResolveParamsUbo::SIZE),
                    ),
                    texture_layout_entry(
                        1,
                        wgpu::ShaderStages::FRAGMENT,
                        wgpu::TextureSampleType::Float { filterable: false },
                        wgpu::TextureViewDimension::D2,
                        true,
                    ),
                    texture_layout_entry(
                        2,
                        wgpu::ShaderStages::FRAGMENT,
                        wgpu::TextureSampleType::Float { filterable: false },
                        wgpu::TextureViewDimension::D2,
                        true,
                    ),
                ],
            })
        })
    }

    /// Returns or builds a pipeline for `output_format` and the requested view configuration.
    pub(super) fn pipeline(
        &self,
        device: &wgpu::Device,
        output_format: wgpu::TextureFormat,
        multiview_stereo: bool,
    ) -> Arc<wgpu::RenderPipeline> {
        let layout_bgl = if multiview_stereo {
            self.bind_group_layout_multiview(device)
        } else {
            self.bind_group_layout_mono(device)
        };
        fullscreen_pipeline_variant(
            device,
            FullscreenPipelineVariantDesc {
                output_format,
                multiview_stereo,
                mono: &self.mono,
                multiview: &self.multiview,
                shader: FullscreenShaderVariants {
                    mono_label: PIPELINE_LABEL_MONO,
                    mono_source: MSAA_RESOLVE_HDR_DEFAULT_WGSL,
                    multiview_label: PIPELINE_LABEL_MULTIVIEW,
                    multiview_source: MSAA_RESOLVE_HDR_MULTIVIEW_WGSL,
                },
                bind_group_layouts: &[Some(layout_bgl)],
                log_name: "msaa_resolve_hdr",
            },
        )
    }
}
