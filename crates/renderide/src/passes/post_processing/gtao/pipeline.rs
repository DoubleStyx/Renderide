//! Cached pipelines, bind layouts, sampler, and per-pass uniform buffer for the GTAO
//! three-pass chain (`gtao_main` → optional `gtao_denoise` → `gtao_apply`).
//!
//! Three independent caches are exposed:
//!
//! - [`GtaoMainPipelineCache`] — main AO production pass with two `R8Unorm` color targets
//!   (visibility scaled by `1 / OCCLUSION_TERM_SCALE` + packed edges). Built manually
//!   because the shared fullscreen helper is single-color-target only.
//! - [`GtaoDenoisePipelineCache`] — bilateral denoise iteration with one `R8Unorm` color
//!   target (denoised AO).
//! - [`GtaoApplyPipelineCache`] — final-apply pass that folds the denoise kernel into HDR
//!   modulation; one color target whose format follows the post-processing chain.
//!
//! Each cache holds mono + multiview variants. One process-wide `GtaoParams` uniform buffer
//! is shared across all three caches and rewritten per-record from the live
//! [`crate::config::GtaoSettings`] with stage-appropriate `denoise_blur_beta` / `final_apply`
//! values (see the per-stage `record` paths in `main_pass.rs`, `denoise_pass.rs`,
//! `apply_pass.rs`).
//!
//! WGSL is sourced from the build-time embedded shader registry; the build script auto-
//! discovers `shaders/passes/post/*.wgsl` and emits one `<name>_default` / `<name>_multiview`
//! pair per source.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use crate::embedded_shaders::{
    GTAO_APPLY_DEFAULT_WGSL, GTAO_APPLY_MULTIVIEW_WGSL, GTAO_DENOISE_DEFAULT_WGSL,
    GTAO_DENOISE_MULTIVIEW_WGSL, GTAO_MAIN_DEFAULT_WGSL, GTAO_MAIN_MULTIVIEW_WGSL,
};
use crate::render_graph::gpu_cache::{
    BindGroupMap, FullscreenPipelineVariantDesc, FullscreenShaderVariants, OnceGpu,
    RenderPipelineMap, create_d2_array_view, create_linear_clamp_sampler, create_uniform_buffer,
    create_wgsl_shader_module, fragment_filterable_d2_array_entry,
    fragment_filtering_sampler_entry, fullscreen_pipeline_variant, stereo_mask_or_template,
    texture_layout_entry, uniform_buffer_layout_entry,
};

/// AO term and packed-edges target format. R8 unorm matches XeGTAO's reference shape (the AO
/// term is 8-bit `Texture2D<uint>` and the edges are `Texture2D<unorm float>` in the reference;
/// we collapse both to `R8Unorm` here so wgpu can render-attach them and we can sample with
/// floating-point math throughout).
pub(super) const AO_TERM_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;
/// Packed-edges target format (mirrors the AO term).
pub(super) const EDGES_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;

/// Upper bound for cached bind groups per cache before the cache is flushed.
///
/// Expected occupancy is one entry per active view (desktop / HMD / each secondary RT camera).
/// The cap protects against unbounded growth when views cycle during resize / MSAA / camera
/// churn.
const MAX_CACHED_BIND_GROUPS: usize = 16;

/// CPU mirror of the WGSL `GtaoParams` uniform (32 bytes, 16-byte aligned).
///
/// Rewritten every record from the live [`crate::config::GtaoSettings`] (with `final_apply`
/// and `denoise_blur_beta` adjusted per-stage). Kept separate from `FrameGpuUniforms` so
/// GTAO's per-effect knobs don't bloat the shared frame-globals block.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(super) struct GtaoParamsGpu {
    /// World-space search radius (meters).
    pub radius_world: f32,
    /// Cap on the horizon search in pixels.
    pub max_pixel_radius: f32,
    /// AO strength exponent applied to the raw visibility factor.
    pub intensity: f32,
    /// Horizon steps per side.
    pub step_count: u32,
    /// Distance-falloff range as a fraction of `radius_world`.
    pub falloff_range: f32,
    /// Gray-albedo proxy for the multi-bounce fit (paper Eq. 10).
    pub albedo_multibounce: f32,
    /// Bilateral blur strength for the active denoise stage. Production binds `0.0` (kernel
    /// inert at that stage); the intermediate denoise pass binds `denoise_blur_beta / 5.0`
    /// (XeGTAO's intermediate ratio); the apply pass binds the full `denoise_blur_beta`, or
    /// `0.0` when the user disabled the denoise filter (which short-circuits the kernel).
    pub denoise_blur_beta: f32,
    /// Set to `1` on the apply stage, `0` on production and intermediate denoise. Forwarded as
    /// a `u32` to align with WGSL's lack of `bool` in uniform structs.
    pub final_apply: u32,
}

/// Process-wide `GtaoParams` uniform buffer, shared across the three pipeline caches.
#[derive(Default)]
pub(super) struct GtaoParamsBuffer {
    buffer: OnceGpu<wgpu::Buffer>,
}

impl GtaoParamsBuffer {
    pub(super) fn get(&self, device: &wgpu::Device) -> &wgpu::Buffer {
        self.buffer.get_or_create(|| {
            create_uniform_buffer(device, "gtao-params", size_of::<GtaoParamsGpu>() as u64)
        })
    }
}

// ---- main (AO production) pipeline cache ----------------------------------

/// Cache key for [`GtaoMainPipelineCache::bind_groups`].
#[derive(Clone, Eq, Hash, PartialEq)]
struct GtaoMainBindGroupKey {
    scene_depth_texture: wgpu::Texture,
    frame_uniforms: wgpu::Buffer,
    multiview_stereo: bool,
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
                        wgpu::TextureSampleType::Depth,
                        depth_view_dim,
                        false,
                    ),
                    uniform_buffer_layout_entry(1, wgpu::ShaderStages::FRAGMENT, None),
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
        let slot = if multiview_stereo {
            &self.pipeline_stereo
        } else {
            &self.pipeline_mono
        };
        slot.get_or_create(|| {
            let (label, source) = if multiview_stereo {
                ("gtao_main_multiview", GTAO_MAIN_MULTIVIEW_WGSL)
            } else {
                ("gtao_main_default", GTAO_MAIN_DEFAULT_WGSL)
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
            Arc::new(
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
                }),
            )
        })
        .clone()
    }

    pub(super) fn bind_group(
        &self,
        device: &wgpu::Device,
        multiview_stereo: bool,
        scene_depth_texture: &wgpu::Texture,
        frame_uniforms: &wgpu::Buffer,
        params_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let key = GtaoMainBindGroupKey {
            scene_depth_texture: scene_depth_texture.clone(),
            frame_uniforms: frame_uniforms.clone(),
            multiview_stereo,
        };
        self.bind_groups.get_or_create(key, |key| {
            let (depth_dim, depth_layer_count) = if key.multiview_stereo {
                (wgpu::TextureViewDimension::D2Array, Some(2))
            } else {
                (wgpu::TextureViewDimension::D2, Some(1))
            };
            let depth_view = key
                .scene_depth_texture
                .create_view(&wgpu::TextureViewDescriptor {
                    label: Some("gtao_main_scene_depth"),
                    aspect: wgpu::TextureAspect::DepthOnly,
                    dimension: Some(depth_dim),
                    array_layer_count: depth_layer_count,
                    ..Default::default()
                });
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gtao_main"),
                layout: self.bind_group_layout(device, key.multiview_stereo),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: key.frame_uniforms.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            })
        })
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
                    mono_source: GTAO_DENOISE_DEFAULT_WGSL,
                    multiview_label: "gtao_denoise_multiview",
                    multiview_source: GTAO_DENOISE_MULTIVIEW_WGSL,
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
            device.create_bind_group(&wgpu::BindGroupDescriptor {
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
            })
        })
    }
}

// ---- apply (final denoise + modulation) pipeline cache --------------------

/// Cache key for [`GtaoApplyPipelineCache::bind_groups`].
#[derive(Clone, Eq, Hash, PartialEq)]
struct GtaoApplyBindGroupKey {
    scene_color: wgpu::Texture,
    ao_term: wgpu::Texture,
    ao_edges: wgpu::Texture,
    multiview_stereo: bool,
}

/// Cache, sampler, and bind-group layout for `gtao_apply` (final-apply pass).
pub(super) struct GtaoApplyPipelineCache {
    bind_group_layout: OnceGpu<wgpu::BindGroupLayout>,
    sampler: OnceGpu<wgpu::Sampler>,
    pipeline_mono: RenderPipelineMap<wgpu::TextureFormat>,
    pipeline_multiview: RenderPipelineMap<wgpu::TextureFormat>,
    bind_groups: BindGroupMap<GtaoApplyBindGroupKey>,
}

impl Default for GtaoApplyPipelineCache {
    fn default() -> Self {
        Self {
            bind_group_layout: OnceGpu::default(),
            sampler: OnceGpu::default(),
            pipeline_mono: RenderPipelineMap::default(),
            pipeline_multiview: RenderPipelineMap::default(),
            bind_groups: BindGroupMap::with_max_entries(MAX_CACHED_BIND_GROUPS),
        }
    }
}

impl GtaoApplyPipelineCache {
    fn sampler(&self, device: &wgpu::Device) -> &wgpu::Sampler {
        self.sampler
            .get_or_create(|| create_linear_clamp_sampler(device, "gtao_apply"))
    }

    pub(super) fn bind_group_layout(&self, device: &wgpu::Device) -> &wgpu::BindGroupLayout {
        self.bind_group_layout.get_or_create(|| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gtao-apply"),
                entries: &[
                    fragment_filterable_d2_array_entry(0),
                    fragment_filtering_sampler_entry(1),
                    fragment_filterable_d2_array_entry(2),
                    fragment_filterable_d2_array_entry(3),
                    uniform_buffer_layout_entry(4, wgpu::ShaderStages::FRAGMENT, None),
                ],
            })
        })
    }

    pub(super) fn pipeline(
        &self,
        device: &wgpu::Device,
        output_format: wgpu::TextureFormat,
        multiview_stereo: bool,
    ) -> Arc<wgpu::RenderPipeline> {
        let bind_group_layout = self.bind_group_layout(device);
        fullscreen_pipeline_variant(
            device,
            FullscreenPipelineVariantDesc {
                output_format,
                multiview_stereo,
                mono: &self.pipeline_mono,
                multiview: &self.pipeline_multiview,
                shader: FullscreenShaderVariants {
                    mono_label: "gtao_apply_default",
                    mono_source: GTAO_APPLY_DEFAULT_WGSL,
                    multiview_label: "gtao_apply_multiview",
                    multiview_source: GTAO_APPLY_MULTIVIEW_WGSL,
                },
                bind_group_layouts: &[Some(bind_group_layout)],
                log_name: "gtao_apply",
            },
        )
    }

    pub(super) fn bind_group(
        &self,
        device: &wgpu::Device,
        multiview_stereo: bool,
        scene_color: &wgpu::Texture,
        ao_term: &wgpu::Texture,
        ao_edges: &wgpu::Texture,
        params_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let key = GtaoApplyBindGroupKey {
            scene_color: scene_color.clone(),
            ao_term: ao_term.clone(),
            ao_edges: ao_edges.clone(),
            multiview_stereo,
        };
        let sampler = self.sampler(device).clone();
        self.bind_groups.get_or_create(key, |key| {
            let scene_color_view = create_d2_array_view(
                &key.scene_color,
                "gtao_apply_scene_color",
                key.multiview_stereo,
            );
            let ao_view = create_d2_array_view(&key.ao_term, "gtao_apply_ao", key.multiview_stereo);
            let edges_view =
                create_d2_array_view(&key.ao_edges, "gtao_apply_edges", key.multiview_stereo);
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gtao_apply"),
                layout: self.bind_group_layout(device),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&scene_color_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&ao_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&edges_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            })
        })
    }
}

/// Bundles the three pipeline caches plus the shared GTAO params UBO so callers can grab
/// them from a single process-wide singleton (see `gtao_pipelines()` in the parent module).
#[derive(Default)]
pub(super) struct GtaoPipelines {
    pub(super) main: GtaoMainPipelineCache,
    pub(super) denoise: GtaoDenoisePipelineCache,
    pub(super) apply: GtaoApplyPipelineCache,
    pub(super) params: GtaoParamsBuffer,
}
