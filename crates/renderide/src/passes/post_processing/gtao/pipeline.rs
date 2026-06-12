//! Cached pipelines, bind layouts, sampler, and per-pass uniform buffer for the GTAO chain
//! (`gtao_prefilter_*` -> `gtao_main` -> optional `gtao_denoise` -> `gtao_apply`).
//!
//! Four independent caches are exposed:
//!
//! - [`GtaoDepthPrefilterPipelineCache`] -- compute depth prefilter pipeline for raw depth ->
//!   all view-space depth levels.
//! - [`GtaoMainPipelineCache`] -- main AO production pass with two `R8Unorm` color targets
//!   (visibility scaled by `1 / OCCLUSION_TERM_SCALE` + packed edges). Built manually
//!   because the shared fullscreen helper is single-color-target only.
//! - [`GtaoDenoisePipelineCache`] -- bilateral denoise iteration with one `R8Unorm` color
//!   target (denoised AO).
//! - [`GtaoApplyPipelineCache`] -- final-apply pass that folds the denoise kernel into in-place
//!   opaque HDR modulation through multiplicative destination-color blending.
//!
//! Each cache holds mono + multiview variants. One process-wide `GtaoParams` uniform buffer
//! is shared across all three caches and rewritten per-record from the live
//! [`crate::config::GtaoSettings`] with stage-appropriate `denoise_blur_beta` / `final_apply`
//! values (see the per-stage `record` paths in `main_pass.rs`, `denoise_pass.rs`,
//! `apply_pass.rs`).
//!
//! WGSL is sourced from the runtime shader package; the build script auto-
//! discovers `shaders/passes/post/*.wgsl` and emits one `<name>_default` / `<name>_multiview`
//! pair per source.

mod params;

use std::sync::Arc;

use crate::embedded_shaders::embedded_wgsl;
use crate::gpu::bind_layout::{
    fragment_filterable_d2_array_entry, storage_texture_layout_entry, texture_layout_entry,
    uniform_buffer_layout_entry,
};
use crate::gpu_resource::{BindGroupMap, OnceGpu, RenderPipelineMap};
use crate::render_graph::gpu_cache::{
    FullscreenPipelineVariantDesc, FullscreenShaderVariants, create_d2_array_view,
    create_wgsl_shader_module, fullscreen_pipeline_variant, stereo_mask_or_template,
};
pub(super) use params::{
    AO_TERM_FORMAT, EDGES_FORMAT, GtaoParamsBuffer, GtaoParamsGpu, VIEW_DEPTH_FORMAT,
    VIEW_DEPTH_MIP_COUNT,
};

/// Upper bound for cached bind groups per cache before the cache is flushed.
///
/// Expected occupancy is one entry per active view (desktop / HMD / each secondary RT camera).
/// The cap protects against unbounded growth when views cycle during resize / MSAA / camera
/// churn.
const MAX_CACHED_BIND_GROUPS: usize = 16;

// ---- main (AO production) pipeline cache ----------------------------------

/// Cache key for [`GtaoMainPipelineCache::bind_groups`].
#[derive(Clone, Eq, Hash, PartialEq)]
struct GtaoMainBindGroupKey {
    view_depth_mips: [wgpu::Texture; VIEW_DEPTH_MIP_COUNT as usize],
    view_normals_texture: wgpu::Texture,
    frame_uniforms: wgpu::Buffer,
    view_depth_mip_count: u32,
    multiview_stereo: bool,
}

/// Runtime resources used to build or fetch a `gtao_main` bind group.
pub(super) struct GtaoMainBindGroupResources<'a> {
    /// Whether the current view uses a two-layer multiview texture binding.
    pub(super) multiview_stereo: bool,
    /// View-space depth textures containing the valid runtime depth levels.
    pub(super) view_depth_mips: [&'a wgpu::Texture; VIEW_DEPTH_MIP_COUNT as usize],
    /// Number of valid mips in `view_depth_texture`.
    pub(super) view_depth_mip_count: u32,
    /// Smooth view-space normal texture produced by the normal prepass.
    pub(super) view_normals_texture: &'a wgpu::Texture,
    /// Per-view frame uniform buffer.
    pub(super) frame_uniforms: &'a wgpu::Buffer,
}

/// Cache and bind-group layout for `gtao_main` (AO production pass).
pub(super) struct GtaoMainPipelineCache {
    bind_group_layout_mono: OnceGpu<wgpu::BindGroupLayout>,
    bind_group_layout_stereo: OnceGpu<wgpu::BindGroupLayout>,
    pipeline_mono: OnceGpu<Arc<wgpu::RenderPipeline>>,
    pipeline_stereo: OnceGpu<Arc<wgpu::RenderPipeline>>,
    bind_groups: BindGroupMap<GtaoMainBindGroupKey>,
}

impl Default for GtaoMainPipelineCache {
    fn default() -> Self {
        Self {
            bind_group_layout_mono: OnceGpu::default(),
            bind_group_layout_stereo: OnceGpu::default(),
            pipeline_mono: OnceGpu::default(),
            pipeline_stereo: OnceGpu::default(),
            bind_groups: BindGroupMap::with_max_entries(MAX_CACHED_BIND_GROUPS),
        }
    }
}

impl GtaoMainPipelineCache {
    pub(super) fn bind_group_layout(
        &self,
        device: &wgpu::Device,
        multiview_stereo: bool,
    ) -> &wgpu::BindGroupLayout {
        let slot = if multiview_stereo {
            &self.bind_group_layout_stereo
        } else {
            &self.bind_group_layout_mono
        };
        slot.get_or_create(|| {
            let depth_view_dim = if multiview_stereo {
                wgpu::TextureViewDimension::D2Array
            } else {
                wgpu::TextureViewDimension::D2
            };
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(if multiview_stereo {
                    "gtao-main-multiview"
                } else {
                    "gtao-main-mono"
                }),
                entries: &[
                    texture_layout_entry(
                        0,
                        wgpu::ShaderStages::FRAGMENT,
                        wgpu::TextureSampleType::Float { filterable: false },
                        depth_view_dim,
                        false,
                    ),
                    texture_layout_entry(
                        1,
                        wgpu::ShaderStages::FRAGMENT,
                        wgpu::TextureSampleType::Float { filterable: false },
                        depth_view_dim,
                        false,
                    ),
                    texture_layout_entry(
                        2,
                        wgpu::ShaderStages::FRAGMENT,
                        wgpu::TextureSampleType::Float { filterable: false },
                        depth_view_dim,
                        false,
                    ),
                    texture_layout_entry(
                        3,
                        wgpu::ShaderStages::FRAGMENT,
                        wgpu::TextureSampleType::Float { filterable: false },
                        depth_view_dim,
                        false,
                    ),
                    texture_layout_entry(
                        4,
                        wgpu::ShaderStages::FRAGMENT,
                        wgpu::TextureSampleType::Float { filterable: false },
                        depth_view_dim,
                        false,
                    ),
                    texture_layout_entry(
                        5,
                        wgpu::ShaderStages::FRAGMENT,
                        wgpu::TextureSampleType::Float { filterable: false },
                        depth_view_dim,
                        false,
                    ),
                    uniform_buffer_layout_entry(6, wgpu::ShaderStages::FRAGMENT, None),
                    uniform_buffer_layout_entry(7, wgpu::ShaderStages::FRAGMENT, None),
                ],
            })
        })
    }

    pub(super) fn pipeline(
        &self,
        device: &wgpu::Device,
        multiview_stereo: bool,
    ) -> Arc<wgpu::RenderPipeline> {
        let slot = if multiview_stereo {
            &self.pipeline_stereo
        } else {
            &self.pipeline_mono
        };
        slot.get_or_create(|| {
            let (label, source) = if multiview_stereo {
                ("gtao_main_multiview", embedded_wgsl!("gtao_main_multiview"))
            } else {
                ("gtao_main_default", embedded_wgsl!("gtao_main_default"))
            };
            logger::debug!("gtao_main: building pipeline (multiview = {multiview_stereo})");
            let shader = create_wgsl_shader_module(device, label, source);
            let layout = self.bind_group_layout(device, multiview_stereo);
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(layout)],
                immediate_size: 0,
            });
            let ao_target = wgpu::ColorTargetState {
                format: AO_TERM_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            };
            let edges_target = wgpu::ColorTargetState {
                format: EDGES_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            };
            let pipeline = Arc::new(device.create_render_pipeline(
                &wgpu::RenderPipelineDescriptor {
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
                        targets: &[Some(ao_target), Some(edges_target)],
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        ..Default::default()
                    },
                    depth_stencil: None,
                    multisample: Default::default(),
                    multiview_mask: stereo_mask_or_template(multiview_stereo, None),
                    cache: None,
                },
            ));
            crate::profiling::note_resource_churn!(RenderPipeline, "passes::gtao_main_pipeline");
            pipeline
        })
        .clone()
    }

    pub(super) fn bind_group(
        &self,
        device: &wgpu::Device,
        resources: GtaoMainBindGroupResources<'_>,
        params_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let key = GtaoMainBindGroupKey {
            view_depth_mips: resources.view_depth_mips.map(wgpu::Texture::clone),
            view_normals_texture: resources.view_normals_texture.clone(),
            frame_uniforms: resources.frame_uniforms.clone(),
            view_depth_mip_count: resources
                .view_depth_mip_count
                .clamp(1, VIEW_DEPTH_MIP_COUNT),
            multiview_stereo: resources.multiview_stereo,
        };
        self.bind_groups.get_or_create(key, |key| {
            let (depth_dim, depth_layer_count) = if key.multiview_stereo {
                (wgpu::TextureViewDimension::D2Array, Some(2))
            } else {
                (wgpu::TextureViewDimension::D2, Some(1))
            };
            let depth_views = key.view_depth_mips.each_ref().map(|texture| {
                let view = texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("gtao_main_view_depth"),
                    aspect: wgpu::TextureAspect::All,
                    dimension: Some(depth_dim),
                    mip_level_count: Some(1),
                    array_layer_count: depth_layer_count,
                    ..Default::default()
                });
                crate::profiling::note_resource_churn!(TextureView, "passes::gtao_main_depth_view");
                view
            });
            let normals_view = key
                .view_normals_texture
                .create_view(&wgpu::TextureViewDescriptor {
                    label: Some("gtao_main_view_normals"),
                    aspect: wgpu::TextureAspect::All,
                    dimension: Some(depth_dim),
                    mip_level_count: Some(1),
                    array_layer_count: depth_layer_count,
                    ..Default::default()
                });
            crate::profiling::note_resource_churn!(TextureView, "passes::gtao_main_normals_view");
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gtao_main"),
                layout: self.bind_group_layout(device, key.multiview_stereo),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&depth_views[0]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&depth_views[1]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&depth_views[2]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&depth_views[3]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&depth_views[4]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(&normals_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: key.frame_uniforms.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });
            crate::profiling::note_resource_churn!(BindGroup, "passes::gtao_main_bind_group");
            bind_group
        })
    }
}

// ---- depth prefilter pipeline cache ---------------------------------------

/// Compute pipelines and layouts for the combined GTAO view-space depth prefilter.
#[derive(Default)]
pub(super) struct GtaoDepthPrefilterPipelineCache {
    bind_group_layout_mono: OnceGpu<wgpu::BindGroupLayout>,
    bind_group_layout_stereo: OnceGpu<wgpu::BindGroupLayout>,
    pipeline_mono: OnceGpu<Arc<wgpu::ComputePipeline>>,
    pipeline_stereo: OnceGpu<Arc<wgpu::ComputePipeline>>,
}

impl GtaoDepthPrefilterPipelineCache {
    /// Bind group layout for raw depth -> all view-space depth levels.
    pub(super) fn bind_group_layout(
        &self,
        device: &wgpu::Device,
        multiview_stereo: bool,
    ) -> &wgpu::BindGroupLayout {
        let slot = if multiview_stereo {
            &self.bind_group_layout_stereo
        } else {
            &self.bind_group_layout_mono
        };
        slot.get_or_create(|| {
            let view_dimension = prefilter_view_dimension(multiview_stereo);
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(if multiview_stereo {
                    "gtao-prefilter-combined-stereo"
                } else {
                    "gtao-prefilter-combined-mono"
                }),
                entries: &[
                    texture_layout_entry(
                        0,
                        wgpu::ShaderStages::COMPUTE,
                        wgpu::TextureSampleType::Depth,
                        view_dimension,
                        false,
                    ),
                    uniform_buffer_layout_entry(1, wgpu::ShaderStages::COMPUTE, None),
                    uniform_buffer_layout_entry(2, wgpu::ShaderStages::COMPUTE, None),
                    storage_texture_layout_entry(
                        3,
                        wgpu::ShaderStages::COMPUTE,
                        wgpu::StorageTextureAccess::WriteOnly,
                        VIEW_DEPTH_FORMAT,
                        view_dimension,
                    ),
                    storage_texture_layout_entry(
                        4,
                        wgpu::ShaderStages::COMPUTE,
                        wgpu::StorageTextureAccess::WriteOnly,
                        VIEW_DEPTH_FORMAT,
                        view_dimension,
                    ),
                    storage_texture_layout_entry(
                        5,
                        wgpu::ShaderStages::COMPUTE,
                        wgpu::StorageTextureAccess::WriteOnly,
                        VIEW_DEPTH_FORMAT,
                        view_dimension,
                    ),
                    storage_texture_layout_entry(
                        6,
                        wgpu::ShaderStages::COMPUTE,
                        wgpu::StorageTextureAccess::WriteOnly,
                        VIEW_DEPTH_FORMAT,
                        view_dimension,
                    ),
                    storage_texture_layout_entry(
                        7,
                        wgpu::ShaderStages::COMPUTE,
                        wgpu::StorageTextureAccess::WriteOnly,
                        VIEW_DEPTH_FORMAT,
                        view_dimension,
                    ),
                ],
            })
        })
    }

    /// Compute pipeline for raw depth -> all view-space depth levels.
    pub(super) fn pipeline(
        &self,
        device: &wgpu::Device,
        multiview_stereo: bool,
    ) -> Arc<wgpu::ComputePipeline> {
        let slot = if multiview_stereo {
            &self.pipeline_stereo
        } else {
            &self.pipeline_mono
        };
        slot.get_or_create(|| {
            let (label, source) = if multiview_stereo {
                (
                    "gtao_prefilter_mip0_multiview",
                    embedded_wgsl!("gtao_prefilter_mip0_multiview"),
                )
            } else {
                (
                    "gtao_prefilter_mip0_default",
                    embedded_wgsl!("gtao_prefilter_mip0_default"),
                )
            };
            let shader = create_wgsl_shader_module(device, label, source);
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(self.bind_group_layout(device, multiview_stereo))],
                immediate_size: 0,
            });
            let pipeline = Arc::new(device.create_compute_pipeline(
                &wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(&layout),
                    module: &shader,
                    entry_point: Some("cs_main"),
                    compilation_options: Default::default(),
                    cache: None,
                },
            ));
            crate::profiling::note_resource_churn!(
                ComputePipeline,
                "passes::gtao_prefilter_combined_pipeline"
            );
            pipeline
        })
        .clone()
    }
}

fn prefilter_view_dimension(multiview_stereo: bool) -> wgpu::TextureViewDimension {
    if multiview_stereo {
        wgpu::TextureViewDimension::D2Array
    } else {
        wgpu::TextureViewDimension::D2
    }
}

// ---- denoise (intermediate) pipeline cache --------------------------------

/// Cache key for [`GtaoDenoisePipelineCache::bind_groups`].
#[derive(Clone, Eq, Hash, PartialEq)]
struct GtaoDenoiseBindGroupKey {
    ao_term: wgpu::Texture,
    ao_edges: wgpu::Texture,
    multiview_stereo: bool,
}

/// Cache and bind-group layout for `gtao_denoise` (intermediate denoise pass).
pub(super) struct GtaoDenoisePipelineCache {
    bind_group_layout: OnceGpu<wgpu::BindGroupLayout>,
    pipeline_mono: RenderPipelineMap<wgpu::TextureFormat>,
    pipeline_multiview: RenderPipelineMap<wgpu::TextureFormat>,
    bind_groups: BindGroupMap<GtaoDenoiseBindGroupKey>,
}

impl Default for GtaoDenoisePipelineCache {
    fn default() -> Self {
        Self {
            bind_group_layout: OnceGpu::default(),
            pipeline_mono: RenderPipelineMap::default(),
            pipeline_multiview: RenderPipelineMap::default(),
            bind_groups: BindGroupMap::with_max_entries(MAX_CACHED_BIND_GROUPS),
        }
    }
}

impl GtaoDenoisePipelineCache {
    pub(super) fn bind_group_layout(&self, device: &wgpu::Device) -> &wgpu::BindGroupLayout {
        self.bind_group_layout.get_or_create(|| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gtao-denoise"),
                entries: &[
                    fragment_filterable_d2_array_entry(0),
                    fragment_filterable_d2_array_entry(1),
                    uniform_buffer_layout_entry(2, wgpu::ShaderStages::FRAGMENT, None),
                ],
            })
        })
    }

    pub(super) fn pipeline(
        &self,
        device: &wgpu::Device,
        multiview_stereo: bool,
    ) -> Arc<wgpu::RenderPipeline> {
        let bind_group_layout = self.bind_group_layout(device);
        fullscreen_pipeline_variant(
            device,
            FullscreenPipelineVariantDesc {
                output_format: AO_TERM_FORMAT,
                multiview_stereo,
                mono: &self.pipeline_mono,
                multiview: &self.pipeline_multiview,
                shader: FullscreenShaderVariants {
                    mono_label: "gtao_denoise_default",
                    mono_source: embedded_wgsl!("gtao_denoise_default"),
                    multiview_label: "gtao_denoise_multiview",
                    multiview_source: embedded_wgsl!("gtao_denoise_multiview"),
                },
                bind_group_layouts: &[Some(bind_group_layout)],
                log_name: "gtao_denoise",
            },
        )
    }

    pub(super) fn bind_group(
        &self,
        device: &wgpu::Device,
        multiview_stereo: bool,
        ao_term: &wgpu::Texture,
        ao_edges: &wgpu::Texture,
        params_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let key = GtaoDenoiseBindGroupKey {
            ao_term: ao_term.clone(),
            ao_edges: ao_edges.clone(),
            multiview_stereo,
        };
        self.bind_groups.get_or_create(key, |key| {
            let ao_view =
                create_d2_array_view(&key.ao_term, "gtao_denoise_ao", key.multiview_stereo);
            let edges_view =
                create_d2_array_view(&key.ao_edges, "gtao_denoise_edges", key.multiview_stereo);
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gtao_denoise"),
                layout: self.bind_group_layout(device),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&ao_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&edges_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });
            crate::profiling::note_resource_churn!(BindGroup, "passes::gtao_denoise_bind_group");
            bind_group
        })
    }
}

// ---- apply (final denoise + modulation) pipeline cache --------------------

/// Cache key for [`GtaoApplyPipelineCache::pipeline_mono`] and
/// [`GtaoApplyPipelineCache::pipeline_multiview`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct GtaoApplyPipelineKey {
    /// Color target format of the opaque HDR attachment.
    output_format: wgpu::TextureFormat,
    /// Render target sample count.
    sample_count: u32,
}

/// Cache key for [`GtaoApplyPipelineCache::bind_groups`].
#[derive(Clone, Eq, Hash, PartialEq)]
struct GtaoApplyBindGroupKey {
    /// AO term texture sampled by the final bilateral kernel.
    ao_term: wgpu::Texture,
    /// Packed edge texture sampled by the final bilateral kernel.
    ao_edges: wgpu::Texture,
    /// Whether the texture views bind both stereo array layers.
    multiview_stereo: bool,
}

/// Cache and bind-group layout for `gtao_apply` (opaque final-apply pass).
pub(super) struct GtaoApplyPipelineCache {
    bind_group_layout: OnceGpu<wgpu::BindGroupLayout>,
    pipeline_mono: RenderPipelineMap<GtaoApplyPipelineKey>,
    pipeline_multiview: RenderPipelineMap<GtaoApplyPipelineKey>,
    bind_groups: BindGroupMap<GtaoApplyBindGroupKey>,
}

impl Default for GtaoApplyPipelineCache {
    fn default() -> Self {
        Self {
            bind_group_layout: OnceGpu::default(),
            pipeline_mono: RenderPipelineMap::default(),
            pipeline_multiview: RenderPipelineMap::default(),
            bind_groups: BindGroupMap::with_max_entries(MAX_CACHED_BIND_GROUPS),
        }
    }
}

impl GtaoApplyPipelineCache {
    pub(super) fn bind_group_layout(&self, device: &wgpu::Device) -> &wgpu::BindGroupLayout {
        self.bind_group_layout.get_or_create(|| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gtao-apply"),
                entries: &[
                    fragment_filterable_d2_array_entry(0),
                    fragment_filterable_d2_array_entry(1),
                    uniform_buffer_layout_entry(2, wgpu::ShaderStages::FRAGMENT, None),
                ],
            })
        })
    }

    pub(super) fn pipeline(
        &self,
        device: &wgpu::Device,
        output_format: wgpu::TextureFormat,
        sample_count: u32,
        multiview_stereo: bool,
    ) -> Arc<wgpu::RenderPipeline> {
        let map = if multiview_stereo {
            &self.pipeline_multiview
        } else {
            &self.pipeline_mono
        };
        map.get_or_create(
            GtaoApplyPipelineKey {
                output_format,
                sample_count: sample_count.max(1),
            },
            |key| {
                logger::debug!(
                    "gtao_apply: building pipeline (dst format = {:?}, samples = {}, multiview = {})",
                    key.output_format,
                    key.sample_count,
                    multiview_stereo
                );
                let (label, source) = if multiview_stereo {
                    ("gtao_apply_multiview", embedded_wgsl!("gtao_apply_multiview"))
                } else {
                    ("gtao_apply_default", embedded_wgsl!("gtao_apply_default"))
                };
                let shader = create_wgsl_shader_module(device, label, source);
                let layout = self.bind_group_layout(device);
                create_gtao_apply_pipeline(
                    device,
                    label,
                    &shader,
                    layout,
                    key.output_format,
                    key.sample_count,
                    multiview_stereo,
                )
            },
        )
    }

    pub(super) fn bind_group(
        &self,
        device: &wgpu::Device,
        multiview_stereo: bool,
        ao_term: &wgpu::Texture,
        ao_edges: &wgpu::Texture,
        params_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let key = GtaoApplyBindGroupKey {
            ao_term: ao_term.clone(),
            ao_edges: ao_edges.clone(),
            multiview_stereo,
        };
        self.bind_groups.get_or_create(key, |key| {
            let ao_view = create_d2_array_view(&key.ao_term, "gtao_apply_ao", key.multiview_stereo);
            let edges_view =
                create_d2_array_view(&key.ao_edges, "gtao_apply_edges", key.multiview_stereo);
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gtao_apply"),
                layout: self.bind_group_layout(device),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&ao_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&edges_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });
            crate::profiling::note_resource_churn!(BindGroup, "passes::gtao_apply_bind_group");
            bind_group
        })
    }
}

/// Builds the `gtao_apply` fullscreen pipeline for the active opaque color target.
fn create_gtao_apply_pipeline(
    device: &wgpu::Device,
    label: &str,
    shader: &wgpu::ShaderModule,
    bind_group_layout: &wgpu::BindGroupLayout,
    output_format: wgpu::TextureFormat,
    sample_count: u32,
    multiview_stereo: bool,
) -> wgpu::RenderPipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(bind_group_layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: output_format,
                blend: Some(gtao_multiply_blend_state()),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: sample_count.max(1),
            ..Default::default()
        },
        multiview_mask: stereo_mask_or_template(multiview_stereo, None),
        cache: None,
    });
    crate::profiling::note_resource_churn!(RenderPipeline, "passes::gtao_apply_pipeline");
    pipeline
}

/// Returns the blend state that computes `dst.rgb = src.rgb * dst.rgb` and preserves `dst.a`.
fn gtao_multiply_blend_state() -> wgpu::BlendState {
    wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::Dst,
            dst_factor: wgpu::BlendFactor::Zero,
            operation: wgpu::BlendOperation::Add,
        },
        alpha: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::Zero,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        },
    }
}

/// Bundles the pipeline caches plus the shared GTAO params UBO so callers can grab
/// them from a single process-wide singleton (see `gtao_pipelines()` in the parent module).
#[derive(Default)]
pub(super) struct GtaoPipelines {
    pub(super) depth_prefilter: GtaoDepthPrefilterPipelineCache,
    pub(super) main: GtaoMainPipelineCache,
    pub(super) denoise: GtaoDenoisePipelineCache,
    pub(super) apply: GtaoApplyPipelineCache,
    pub(super) params: GtaoParamsBuffer,
}
