//! Cached pipelines and bind layout for [`super::AcesTonemapPass`].
//!
//! Mirrors the structure of [`crate::passes::scene_color_compose`]'s pipeline
//! cache: per-output-format `wgpu::RenderPipeline` map for mono and multiview, with a single
//! linear-clamp sampler shared across all instances.
//!
//! WGSL is sourced from the build-time embedded shader registry
//! ([`crate::embedded_shaders::embedded_target_wgsl`]) so the same
//! `shaders/passes/post/aces_tonemap.wgsl` source is composed once into mono and multiview
//! variants by the build script's `#ifdef MULTIVIEW` path (no runtime composition needed).

use std::sync::Arc;

use crate::embedded_shaders::{ACES_TONEMAP_DEFAULT_WGSL, ACES_TONEMAP_MULTIVIEW_WGSL};
use crate::render_graph::gpu_cache::{
    BindGroupMap, FullscreenPipelineVariantDesc, FullscreenShaderVariants, OnceGpu,
    RenderPipelineMap, create_d2_array_view, create_linear_clamp_sampler,
    fragment_filterable_d2_array_entry, fragment_filtering_sampler_entry,
    fullscreen_pipeline_variant,
};

/// Debug label for the mono variant pipeline.
const PIPELINE_LABEL_MONO: &str = "aces_tonemap_default";
/// Debug label for the multiview variant pipeline.
const PIPELINE_LABEL_MULTIVIEW: &str = "aces_tonemap_multiview";

/// GPU state shared by all ACES tonemap passes (bind layout + sampler + per-format pipelines).
pub(super) struct AcesTonemapPipelineCache {
    /// Bind group layout shared by mono and multiview variants.
    bind_group_layout: OnceGpu<wgpu::BindGroupLayout>,
    /// Linear sampler used to read HDR scene color.
    sampler: OnceGpu<wgpu::Sampler>,
    /// Mono pipelines keyed by output color format.
    mono: RenderPipelineMap<wgpu::TextureFormat>,
    /// Multiview pipelines keyed by output color format.
    multiview: RenderPipelineMap<wgpu::TextureFormat>,
    /// Bind groups keyed by scene-color texture identity + multiview flag. `wgpu::Texture`
    /// implements `Eq + Hash` over its internal handle, so entries automatically follow the
    /// transient pool's allocation lifecycle -- when the pool drops and recreates a texture,
    /// the stale entry is orphaned and cleaned up by bounded cache clearing.
    bind_groups: BindGroupMap<(wgpu::Texture, bool)>,
}

impl Default for AcesTonemapPipelineCache {
    fn default() -> Self {
        Self {
            bind_group_layout: OnceGpu::default(),
            sampler: OnceGpu::default(),
            mono: RenderPipelineMap::default(),
            multiview: RenderPipelineMap::default(),
            bind_groups: BindGroupMap::with_max_entries(MAX_CACHED_BIND_GROUPS),
        }
    }
}

impl AcesTonemapPipelineCache {
    /// Linear clamp sampler used to read the HDR scene color.
    pub(super) fn sampler(&self, device: &wgpu::Device) -> &wgpu::Sampler {
        self.sampler
            .get_or_create(|| create_linear_clamp_sampler(device, "aces_tonemap"))
    }

    /// Bind group layout for the HDR scene color texture array + sampler.
    fn bind_group_layout(&self, device: &wgpu::Device) -> &wgpu::BindGroupLayout {
        self.bind_group_layout.get_or_create(|| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("aces_tonemap"),
                entries: &[
                    fragment_filterable_d2_array_entry(0),
                    fragment_filtering_sampler_entry(1),
                ],
            })
        })
    }

    /// Returns or builds a render pipeline for `output_format` and multiview stereo.
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
                mono: &self.mono,
                multiview: &self.multiview,
                shader: FullscreenShaderVariants {
                    mono_label: PIPELINE_LABEL_MONO,
                    mono_source: ACES_TONEMAP_DEFAULT_WGSL,
                    multiview_label: PIPELINE_LABEL_MULTIVIEW,
                    multiview_source: ACES_TONEMAP_MULTIVIEW_WGSL,
                },
                bind_group_layouts: &[Some(bind_group_layout)],
                log_name: "aces_tonemap",
            },
        )
    }

    /// Bind group for one frame's scene-color texture, cached by `(Texture, multiview_stereo)`.
    ///
    /// Builds a fresh `D2Array` view on cache miss so the cached bind group outlives any single
    /// per-frame view clone. Hit is a `HashMap` lookup + `wgpu::BindGroup::clone` (Arc bump).
    pub(super) fn bind_group(
        &self,
        device: &wgpu::Device,
        scene_color_texture: &wgpu::Texture,
        multiview_stereo: bool,
    ) -> wgpu::BindGroup {
        let key = (scene_color_texture.clone(), multiview_stereo);
        self.bind_groups.get_or_create(key, |key| {
            let (scene_color_texture, multiview_stereo) = key;
            let view = create_d2_array_view(
                scene_color_texture,
                "aces_tonemap_sampled",
                *multiview_stereo,
            );
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("aces_tonemap"),
                layout: self.bind_group_layout(device),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(self.sampler(device)),
                    },
                ],
            })
        })
    }
}

/// Upper bound for cached ACES bind groups before the cache is flushed.
///
/// The scene-color transient texture is stable across most frames -- the cache normally holds
/// one or two entries (mono + multiview). This cap protects against unbounded growth when the
/// swapchain / MSAA setting flips repeatedly and the transient pool cycles allocations.
const MAX_CACHED_BIND_GROUPS: usize = 8;
