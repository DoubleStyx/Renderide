//! Cached pipelines, bind layouts, and per-frame UBO for HDR-aware color resolves.
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

use crate::embedded_shaders::embedded_wgsl;
use crate::gpu::bind_layout::{texture_layout_entry, uniform_buffer_layout_entry};
use crate::gpu_resource::{BindGroupMap, OnceGpu, RenderPipelineMap};
use crate::render_graph::gpu_cache::{
    FullscreenPipelineVariantDesc, FullscreenShaderVariants, create_uniform_buffer,
    fullscreen_pipeline_variant,
};

/// Debug label for the mono pipeline.
const PIPELINE_LABEL_MONO: &str = "msaa_resolve_hdr_default";
/// Debug label for the multiview pipeline.
const PIPELINE_LABEL_MULTIVIEW: &str = "msaa_resolve_hdr_multiview";

/// Upper bound on cached resolve bind groups. The working set is bounded by the small number of
/// `(scene-color MSAA texture, params UBO)` pairs alive at any given moment (per stage, per
/// runtime view); the cap protects against unbounded growth when the transient pool recycles
/// allocations rapidly (e.g. viewport resize storms, MSAA tier flips).
const MAX_CACHED_BIND_GROUPS: usize = 16;

/// Cache key for resolve bind groups. `source_texture` is the multisampled HDR scene-color source
/// from the transient pool; `params_ubo` is the lazily-allocated per-pass UBO. Both are Arc-backed
/// in wgpu, so equality is identity-based and reallocated resources produce new keys.
#[derive(Clone, Eq, Hash, PartialEq)]
struct ResolveBindGroupKey {
    source_texture: wgpu::Texture,
    params_ubo: wgpu::Buffer,
}

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
pub(super) struct MsaaResolveHdrPipelineCache {
    /// Bind group layout for the mono resolve variant.
    bind_group_layout_mono: OnceGpu<wgpu::BindGroupLayout>,
    /// Bind group layout for the multiview resolve variant.
    bind_group_layout_multiview: OnceGpu<wgpu::BindGroupLayout>,
    /// One pipeline per output color format (matches scene_color_hdr's runtime format).
    mono: RenderPipelineMap<wgpu::TextureFormat>,
    /// Same, but with `multiview_mask = 3` so the shader runs once per eye layer.
    multiview: RenderPipelineMap<wgpu::TextureFormat>,
    /// Lazily-allocated UBO holding the live sample count. Re-uploaded each frame through the
    /// graph upload sink before the pass records its draw.
    params_ubo: OnceGpu<wgpu::Buffer>,
    /// Mono bind groups keyed by `(source MSAA texture, params UBO)`. Stale entries are orphaned
    /// when the transient pool recycles the scene-color allocation.
    bind_groups_mono: BindGroupMap<ResolveBindGroupKey>,
    /// Multiview bind groups keyed the same way as `bind_groups_mono`; the cached value holds the
    /// pair of single-layer texture views alive alongside the bind group itself.
    bind_groups_multiview: BindGroupMap<ResolveBindGroupKey>,
}

impl Default for MsaaResolveHdrPipelineCache {
    fn default() -> Self {
        Self {
            bind_group_layout_mono: OnceGpu::default(),
            bind_group_layout_multiview: OnceGpu::default(),
            mono: RenderPipelineMap::default(),
            multiview: RenderPipelineMap::default(),
            params_ubo: OnceGpu::default(),
            bind_groups_mono: BindGroupMap::with_max_entries(MAX_CACHED_BIND_GROUPS),
            bind_groups_multiview: BindGroupMap::with_max_entries(MAX_CACHED_BIND_GROUPS),
        }
    }
}

impl MsaaResolveHdrPipelineCache {
    /// Returns the per-frame `ResolveParams` UBO, lazily creating it on first call.
    pub(super) fn params_ubo(&self, device: &wgpu::Device) -> &wgpu::Buffer {
        self.params_ubo.get_or_create(|| {
            create_uniform_buffer(device, "msaa_resolve_hdr_params", ResolveParamsUbo::SIZE)
        })
    }

    /// Bind group layout for the mono variant: `params` + one `texture_multisampled_2d<f32>`.
    fn bind_group_layout_mono(&self, device: &wgpu::Device) -> &wgpu::BindGroupLayout {
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
    fn bind_group_layout_multiview(&self, device: &wgpu::Device) -> &wgpu::BindGroupLayout {
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

    /// Returns a cached resolve bind group for the given source texture, building it on miss.
    ///
    /// The cached bind group owns the per-eye single-layer texture views (multiview path) or the
    /// single-layer view (mono path), so view churn is amortized alongside the bind group itself.
    pub(super) fn bind_group(
        &self,
        device: &wgpu::Device,
        source_texture: &wgpu::Texture,
        params_ubo: &wgpu::Buffer,
        multiview_stereo: bool,
    ) -> wgpu::BindGroup {
        let key = ResolveBindGroupKey {
            source_texture: source_texture.clone(),
            params_ubo: params_ubo.clone(),
        };
        if multiview_stereo {
            self.bind_groups_multiview.get_or_create(key, |k| {
                let layer0 = k.source_texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("msaa_resolve_hdr_src_msaa_left"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: 0,
                    array_layer_count: Some(1),
                    ..Default::default()
                });
                crate::profiling::note_resource_churn!(
                    TextureView,
                    "passes::world_mesh_color_resolve_left_view"
                );
                let layer1 = k.source_texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("msaa_resolve_hdr_src_msaa_right"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: 1,
                    array_layer_count: Some(1),
                    ..Default::default()
                });
                crate::profiling::note_resource_churn!(
                    TextureView,
                    "passes::world_mesh_color_resolve_right_view"
                );
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("msaa_resolve_hdr_bg_multiview"),
                    layout: self.bind_group_layout_multiview(device),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: k.params_ubo.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&layer0),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&layer1),
                        },
                    ],
                });
                crate::profiling::note_resource_churn!(
                    BindGroup,
                    "passes::world_mesh_color_resolve_multiview_bg"
                );
                bind_group
            })
        } else {
            self.bind_groups_mono.get_or_create(key, |k| {
                let view = k.source_texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("msaa_resolve_hdr_src_msaa"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: 0,
                    array_layer_count: Some(1),
                    ..Default::default()
                });
                crate::profiling::note_resource_churn!(
                    TextureView,
                    "passes::world_mesh_color_resolve_mono_view"
                );
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("msaa_resolve_hdr_bg_mono"),
                    layout: self.bind_group_layout_mono(device),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: k.params_ubo.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&view),
                        },
                    ],
                });
                crate::profiling::note_resource_churn!(
                    BindGroup,
                    "passes::world_mesh_color_resolve_mono_bg"
                );
                bind_group
            })
        }
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
                    mono_source: embedded_wgsl!("msaa_resolve_hdr_default"),
                    multiview_label: PIPELINE_LABEL_MULTIVIEW,
                    multiview_source: embedded_wgsl!("msaa_resolve_hdr_multiview"),
                },
                bind_group_layouts: &[Some(layout_bgl)],
                log_name: "msaa_resolve_hdr",
            },
        )
    }
}
