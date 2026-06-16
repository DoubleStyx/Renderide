//! Shared cache for fullscreen blits that sample a single D2-array HDR texture.
//!
//! ACES tonemap, AgX tonemap, and scene-color compose all read one filterable D2-array texture
//! at `@binding(0)`, sample it with a linear-clamp `@binding(1)` sampler, and write a fullscreen
//! triangle to a color attachment. This cache owns the per-effect bind layout, sampler,
//! mono/multiview pipeline maps keyed by output format, and the (texture, multiview) bind-group
//! cache so each effect only has to supply its shader variants and a debug label.

use std::sync::Arc;

use crate::gpu::bind_layout::{
    fragment_filterable_d2_array_entry, fragment_filtering_sampler_entry,
};
use crate::gpu_resource::{BindGroupMap, OnceGpu, RenderPipelineMap};
use crate::render_graph::gpu_cache::{
    FullscreenPipelineVariantDesc, FullscreenShaderVariants, create_d2_array_view,
    create_linear_clamp_sampler, fullscreen_pipeline_variant,
};

/// Debug/log-only identity for the cached pipelines, samplers, and bind groups.
#[derive(Clone, Copy)]
pub(in crate::passes) struct FullscreenD2ArrayPipelineLabels {
    /// Base label shared by the bind layout, sampler, and bind groups.
    pub(in crate::passes) base: &'static str,
    /// Label suffix for the cached sampled texture view.
    pub(in crate::passes) sampled_view: &'static str,
}

/// Mono / multiview WGSL pair for a fullscreen-D2Array effect.
///
/// All four strings are `'static` because the runtime shader package exposes shader sources as
/// `&'static str` and the cache that owns this struct is itself a process-wide `LazyLock`.
#[derive(Clone, Copy)]
pub(in crate::passes) struct FullscreenD2ArrayShaders {
    /// Debug label for the mono shader and pipeline.
    pub(in crate::passes) mono_label: &'static str,
    /// WGSL source for the mono shader.
    pub(in crate::passes) mono_source: &'static str,
    /// Debug label for the multiview shader and pipeline.
    pub(in crate::passes) multiview_label: &'static str,
    /// WGSL source for the multiview shader.
    pub(in crate::passes) multiview_source: &'static str,
}

impl FullscreenD2ArrayShaders {
    fn as_variants(&self) -> FullscreenShaderVariants<'_> {
        FullscreenShaderVariants {
            mono_label: self.mono_label,
            mono_source: self.mono_source,
            multiview_label: self.multiview_label,
            multiview_source: self.multiview_source,
        }
    }
}

/// GPU state for a fullscreen blit that samples one filterable D2-array texture.
///
/// Holds a single bind layout (`@binding(0) texture_2d_array<f32>` + `@binding(1) sampler`), a
/// linear-clamp sampler, separate `mono` / `multiview` pipeline maps keyed by output format, and a
/// `(texture, multiview)` bind-group cache.
pub(in crate::passes) struct FullscreenD2ArraySampledPipelineCache {
    labels: FullscreenD2ArrayPipelineLabels,
    shaders: FullscreenD2ArrayShaders,
    bind_group_layout: OnceGpu<wgpu::BindGroupLayout>,
    sampler: OnceGpu<wgpu::Sampler>,
    mono: RenderPipelineMap<wgpu::TextureFormat>,
    multiview: RenderPipelineMap<wgpu::TextureFormat>,
    bind_groups: BindGroupMap<(wgpu::Texture, bool)>,
}

impl FullscreenD2ArraySampledPipelineCache {
    /// Builds an empty cache wired to `shaders` and labeled with `labels`.
    ///
    /// `max_cached_bind_groups` bounds the `(texture, multiview)` map so repeated transient-pool
    /// allocation cycles do not grow the cache without limit.
    pub(in crate::passes) fn new(
        labels: FullscreenD2ArrayPipelineLabels,
        shaders: FullscreenD2ArrayShaders,
        max_cached_bind_groups: usize,
    ) -> Self {
        Self {
            labels,
            shaders,
            bind_group_layout: OnceGpu::default(),
            sampler: OnceGpu::default(),
            mono: RenderPipelineMap::default(),
            multiview: RenderPipelineMap::default(),
            bind_groups: BindGroupMap::with_max_entries(max_cached_bind_groups),
        }
    }

    /// Linear clamp sampler used to read the HDR scene color.
    pub(in crate::passes) fn sampler(&self, device: &wgpu::Device) -> &wgpu::Sampler {
        self.sampler
            .get_or_create(|| create_linear_clamp_sampler(device, self.labels.base))
    }

    /// Bind group layout for the sampled texture array + sampler.
    fn bind_group_layout(&self, device: &wgpu::Device) -> &wgpu::BindGroupLayout {
        self.bind_group_layout.get_or_create(|| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(self.labels.base),
                entries: &[
                    fragment_filterable_d2_array_entry(0),
                    fragment_filtering_sampler_entry(1),
                ],
            })
        })
    }

    /// Returns or builds a render pipeline for `output_format` and multiview stereo.
    pub(in crate::passes) fn pipeline(
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
                shader: self.shaders.as_variants(),
                bind_group_layouts: &[Some(bind_group_layout)],
                log_name: self.labels.base,
            },
        )
    }

    /// Bind group for one frame's sampled texture, cached by `(Texture, multiview_stereo)`.
    ///
    /// `note_churn` fires once per cache miss so callers can record a per-site
    /// [`crate::profiling::note_resource_churn!`] counter without leaking the macro's call-site
    /// requirement into this generic cache.
    pub(in crate::passes) fn bind_group(
        &self,
        device: &wgpu::Device,
        sampled_texture: &wgpu::Texture,
        multiview_stereo: bool,
        note_churn: impl FnOnce(),
    ) -> wgpu::BindGroup {
        let key = (sampled_texture.clone(), multiview_stereo);
        self.bind_groups.get_or_create(key, |key| {
            let (sampled_texture, multiview_stereo) = key;
            let view =
                create_d2_array_view(sampled_texture, self.labels.sampled_view, *multiview_stereo);
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(self.labels.base),
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
            });
            note_churn();
            bind_group
        })
    }
}

/// Defines a typed wrapper around [`FullscreenD2ArraySampledPipelineCache`].
macro_rules! define_fullscreen_d2_array_pipeline_cache {
    (
        $(#[$attr:meta])*
        $vis:vis $name:ident {
            base: $base:literal,
            sampled_view: $sampled_view:literal,
            mono: $mono:literal,
            multiview: $multiview:literal,
            max_bind_groups: $max_bind_groups:expr,
            churn_site: $churn_site:literal $(,)?
        }
    ) => {
        $(#[$attr])*
        $vis struct $name($crate::passes::helpers::FullscreenD2ArraySampledPipelineCache);

        impl Default for $name {
            fn default() -> Self {
                Self($crate::passes::helpers::FullscreenD2ArraySampledPipelineCache::new(
                    $crate::passes::helpers::FullscreenD2ArrayPipelineLabels {
                        base: $base,
                        sampled_view: $sampled_view,
                    },
                    $crate::passes::helpers::FullscreenD2ArrayShaders {
                        mono_label: $mono,
                        mono_source: $crate::embedded_shaders::embedded_wgsl!($mono),
                        multiview_label: $multiview,
                        multiview_source: $crate::embedded_shaders::embedded_wgsl!($multiview),
                    },
                    $max_bind_groups,
                ))
            }
        }

        impl $name {
            /// Returns or builds a render pipeline for `output_format` and multiview stereo.
            $vis fn pipeline(
                &self,
                device: &wgpu::Device,
                output_format: wgpu::TextureFormat,
                multiview_stereo: bool,
            ) -> std::sync::Arc<wgpu::RenderPipeline> {
                self.0.pipeline(device, output_format, multiview_stereo)
            }

            /// Bind group for one frame's scene-color texture, cached by `(Texture, multiview_stereo)`.
            $vis fn bind_group(
                &self,
                device: &wgpu::Device,
                scene_color_texture: &wgpu::Texture,
                multiview_stereo: bool,
            ) -> wgpu::BindGroup {
                self.0
                    .bind_group(device, scene_color_texture, multiview_stereo, || {
                        $crate::profiling::note_resource_churn!(BindGroup, $churn_site);
                    })
            }
        }
    };
}

pub(in crate::passes) use define_fullscreen_d2_array_pipeline_cache;
