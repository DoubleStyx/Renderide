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

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use parking_lot::Mutex;

use crate::embedded_shaders::{MSAA_RESOLVE_HDR_DEFAULT_WGSL, MSAA_RESOLVE_HDR_MULTIVIEW_WGSL};

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
    pub const SIZE: u64 = std::mem::size_of::<Self>() as u64;
}

/// GPU state shared across all MSAA color resolve invocations: bind layouts, pipelines, and the
/// per-frame `ResolveParams` UBO.
pub(super) struct MsaaResolveHdrPipelineCache {
    bind_group_layout_mono: OnceLock<wgpu::BindGroupLayout>,
    bind_group_layout_multiview: OnceLock<wgpu::BindGroupLayout>,
    /// One pipeline per output color format (matches scene_color_hdr's runtime format).
    mono: Mutex<HashMap<wgpu::TextureFormat, Arc<wgpu::RenderPipeline>>>,
    /// Same, but with `multiview_mask = 3` so the shader runs once per eye layer.
    multiview: Mutex<HashMap<wgpu::TextureFormat, Arc<wgpu::RenderPipeline>>>,
    /// Lazily-allocated UBO holding the live sample count. Re-uploaded each frame via
    /// [`wgpu::Queue::write_buffer`] before the pass records its draw.
    params_ubo: OnceLock<wgpu::Buffer>,
}

impl Default for MsaaResolveHdrPipelineCache {
    fn default() -> Self {
        Self {
            bind_group_layout_mono: OnceLock::new(),
            bind_group_layout_multiview: OnceLock::new(),
            mono: Mutex::new(HashMap::new()),
            multiview: Mutex::new(HashMap::new()),
            params_ubo: OnceLock::new(),
        }
    }
}

impl MsaaResolveHdrPipelineCache {
    /// Returns the per-frame `ResolveParams` UBO, lazily creating it on first call.
    pub(super) fn params_ubo(&self, device: &wgpu::Device) -> &wgpu::Buffer {
        self.params_ubo.get_or_init(|| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("msaa_resolve_hdr_params"),
                size: ResolveParamsUbo::SIZE,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        })
    }

    /// Bind group layout for the mono variant: `params` + one `texture_multisampled_2d<f32>`.
    pub(super) fn bind_group_layout_mono(&self, device: &wgpu::Device) -> &wgpu::BindGroupLayout {
        self.bind_group_layout_mono.get_or_init(|| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("msaa_resolve_hdr_mono_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(ResolveParamsUbo::SIZE),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: true,
                        },
                        count: None,
                    },
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
        self.bind_group_layout_multiview.get_or_init(|| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("msaa_resolve_hdr_multiview_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(ResolveParamsUbo::SIZE),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: true,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: true,
                        },
                        count: None,
                    },
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
        let map = if multiview_stereo {
            &self.multiview
        } else {
            &self.mono
        };
        {
            let guard = map.lock();
            if let Some(p) = guard.get(&output_format) {
                return Arc::clone(p);
            }
        }
        let (label, source, layout_bgl) = if multiview_stereo {
            (
                PIPELINE_LABEL_MULTIVIEW,
                MSAA_RESOLVE_HDR_MULTIVIEW_WGSL,
                self.bind_group_layout_multiview(device),
            )
        } else {
            (
                PIPELINE_LABEL_MONO,
                MSAA_RESOLVE_HDR_DEFAULT_WGSL,
                self.bind_group_layout_mono(device),
            )
        };
        logger::debug!(
            "msaa_resolve_hdr: building pipeline (output format = {output_format:?}, multiview = {multiview_stereo})"
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[Some(layout_bgl)],
            immediate_size: 0,
        });
        let pipeline = Arc::new(
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: output_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                // Output is single-sample (`scene_color_hdr`), so the pipeline itself is non-MSAA.
                multisample: Default::default(),
                multiview_mask: multiview_stereo
                    .then(|| std::num::NonZeroU32::new(3))
                    .flatten(),
                cache: None,
            }),
        );
        let mut guard = map.lock();
        if let Some(existing) = guard.get(&output_format) {
            return Arc::clone(existing);
        }
        guard.insert(output_format, Arc::clone(&pipeline));
        pipeline
    }
}
