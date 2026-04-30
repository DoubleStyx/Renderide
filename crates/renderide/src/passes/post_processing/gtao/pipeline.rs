//! Cached pipelines, bind layouts, sampler, and per-pass uniform buffer for [`super::GtaoPass`].
//!
//! Two bind-group layouts are cached (mono vs multiview) because the depth sampling type differs
//! (`texture_depth_2d` vs `texture_depth_2d_array`). The normal prepass target is sampled as a
//! `texture_2d_array<f32>` for both paths so mono and stereo share one shader-side shape. One
//! GPU-side uniform buffer (`GtaoParams`) is shared across every GTAO pass instance and rewritten
//! from the CPU each record — GTAO is a singleton effect in the chain, so a process-wide buffer
//! avoids per-frame allocation churn.
//!
//! WGSL is sourced from the build-time embedded shader registry ([`embedded_target_wgsl`]) so
//! the same `shaders/passes/post/gtao.wgsl` source is composed once into mono and multiview
//! variants by the build script's `#ifdef MULTIVIEW` path.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use crate::embedded_shaders::{GTAO_DEFAULT_WGSL, GTAO_MULTIVIEW_WGSL};
use crate::render_graph::gpu_cache::{
    BindGroupMap, FullscreenPipelineVariantDesc, FullscreenShaderVariants, OnceGpu,
    RenderPipelineMap, create_d2_array_view, create_linear_clamp_sampler, create_uniform_buffer,
    fragment_filterable_d2_array_entry, fragment_filtering_sampler_entry,
    fullscreen_pipeline_variant, texture_layout_entry, uniform_buffer_layout_entry,
};

/// Debug label for the mono variant pipeline.
const PIPELINE_LABEL_MONO: &str = "gtao_default";
/// Debug label for the multiview variant pipeline.
const PIPELINE_LABEL_MULTIVIEW: &str = "gtao_multiview";

/// CPU mirror of the WGSL `GtaoParams` uniform (32 bytes, 16-byte aligned).
///
/// Rewritten every record from the live [`crate::config::GtaoSettings`]. Kept separate from
/// `FrameGpuUniforms` so GTAO's per-effect knobs don't bloat the shared frame-globals block.
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
    /// Distance-falloff range as a fraction of `radius_world`; candidate samples are linearly
    /// faded toward the tangent-plane horizon over the last `falloff_range · radius_world` of
    /// the search radius (matches XeGTAO's `FalloffRange`).
    pub falloff_range: f32,
    /// Gray-albedo proxy for the multi-bounce fit (paper Eq. 10).
    pub albedo_multibounce: f32,
    /// Padding to 16-byte alignment (two f32 slots, matching WGSL `align_pad_tail: vec2<f32>`).
    pub align_pad_tail: [f32; 2],
}

/// Cache key for [`GtaoPipelineCache::bind_groups`].
///
/// `wgpu::Texture` and `wgpu::Buffer` both implement `Eq + Hash` via their internal handles, so
/// entries automatically follow the transient pool's / frame-resource manager's allocation
/// lifecycle: when any of the three backing resources is dropped and recreated, the stale
/// cache entry falls out on overflow eviction.
#[derive(Clone, Eq, Hash, PartialEq)]
struct GtaoBindGroupKey {
    /// Scene-color HDR source texture.
    scene_color_texture: wgpu::Texture,
    /// Scene-depth source texture (aspect view derived internally).
    scene_depth_texture: wgpu::Texture,
    /// GTAO normal prepass texture.
    normal_texture: wgpu::Texture,
    /// Per-view frame-uniforms buffer.
    frame_uniforms: wgpu::Buffer,
    /// Mono vs multiview-stereo view shape.
    multiview_stereo: bool,
}

/// GPU state shared by all GTAO pass instances (layouts + sampler + per-format pipelines + UBO).
pub(super) struct GtaoPipelineCache {
    /// Bind-group layout for the mono pipeline (depth as `texture_depth_2d`).
    bind_group_layout_mono: OnceGpu<wgpu::BindGroupLayout>,
    /// Bind-group layout for the multiview pipeline (depth as `texture_depth_2d_array`).
    bind_group_layout_stereo: OnceGpu<wgpu::BindGroupLayout>,
    /// Linear-clamp sampler used to read the HDR scene color.
    sampler: OnceGpu<wgpu::Sampler>,
    /// Process-wide `GtaoParams` uniform buffer (rewritten every record).
    params_buffer: OnceGpu<wgpu::Buffer>,
    /// Cached pipelines keyed by output format (mono variant).
    mono: RenderPipelineMap<wgpu::TextureFormat>,
    /// Cached pipelines keyed by output format (multiview variant).
    multiview: RenderPipelineMap<wgpu::TextureFormat>,
    /// Bind groups keyed by `(scene_color, scene_depth, normals, frame_uniforms, multiview_stereo)`.
    /// Normally one entry per active view (desktop / HMD / each secondary RT camera).
    bind_groups: BindGroupMap<GtaoBindGroupKey>,
}

impl Default for GtaoPipelineCache {
    fn default() -> Self {
        Self {
            bind_group_layout_mono: OnceGpu::default(),
            bind_group_layout_stereo: OnceGpu::default(),
            sampler: OnceGpu::default(),
            params_buffer: OnceGpu::default(),
            mono: RenderPipelineMap::default(),
            multiview: RenderPipelineMap::default(),
            bind_groups: BindGroupMap::with_max_entries(MAX_CACHED_BIND_GROUPS),
        }
    }
}

impl GtaoPipelineCache {
    /// Linear-clamp sampler used to read the HDR scene color.
    fn sampler(&self, device: &wgpu::Device) -> &wgpu::Sampler {
        self.sampler
            .get_or_create(|| create_linear_clamp_sampler(device, "gtao"))
    }

    /// Process-wide `GtaoParams` uniform buffer. Created on first access.
    pub(super) fn params_buffer(&self, device: &wgpu::Device) -> &wgpu::Buffer {
        self.params_buffer.get_or_create(|| {
            create_uniform_buffer(device, "gtao-params", size_of::<GtaoParamsGpu>() as u64)
        })
    }

    /// Bind-group layout for the selected variant. Entries 0..=5 are, in order:
    /// HDR scene-color array, linear sampler, scene depth, frame globals UBO, GTAO params UBO,
    /// GTAO normal prepass texture.
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
                    "gtao-multiview"
                } else {
                    "gtao-mono"
                }),
                entries: &[
                    fragment_filterable_d2_array_entry(0),
                    fragment_filtering_sampler_entry(1),
                    texture_layout_entry(
                        2,
                        wgpu::ShaderStages::FRAGMENT,
                        wgpu::TextureSampleType::Depth,
                        depth_view_dim,
                        false,
                    ),
                    uniform_buffer_layout_entry(3, wgpu::ShaderStages::FRAGMENT, None),
                    uniform_buffer_layout_entry(4, wgpu::ShaderStages::FRAGMENT, None),
                    texture_layout_entry(
                        5,
                        wgpu::ShaderStages::FRAGMENT,
                        wgpu::TextureSampleType::Float { filterable: false },
                        wgpu::TextureViewDimension::D2Array,
                        false,
                    ),
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
        let bind_group_layout = self.bind_group_layout(device, multiview_stereo);
        fullscreen_pipeline_variant(
            device,
            FullscreenPipelineVariantDesc {
                output_format,
                multiview_stereo,
                mono: &self.mono,
                multiview: &self.multiview,
                shader: FullscreenShaderVariants {
                    mono_label: PIPELINE_LABEL_MONO,
                    mono_source: GTAO_DEFAULT_WGSL,
                    multiview_label: PIPELINE_LABEL_MULTIVIEW,
                    multiview_source: GTAO_MULTIVIEW_WGSL,
                },
                bind_group_layouts: &[Some(bind_group_layout)],
                log_name: "gtao",
            },
        )
    }

    /// Bind group for one frame's set of textures + UBOs, cached by
    /// `(scene_color_texture, scene_depth_texture, normal_texture, frame_uniforms, multiview_stereo)`.
    ///
    /// Builds the per-dispatch `D2Array` color view and depth-aspect view on miss so the cached
    /// bind group outlives any single per-frame view clone. Hit is a `HashMap` lookup +
    /// `wgpu::BindGroup::clone` (Arc bump).
    pub(super) fn bind_group(
        &self,
        device: &wgpu::Device,
        multiview_stereo: bool,
        scene_color_texture: &wgpu::Texture,
        scene_depth_texture: &wgpu::Texture,
        normal_texture: &wgpu::Texture,
        frame_uniforms: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let key = GtaoBindGroupKey {
            scene_color_texture: scene_color_texture.clone(),
            scene_depth_texture: scene_depth_texture.clone(),
            normal_texture: normal_texture.clone(),
            frame_uniforms: frame_uniforms.clone(),
            multiview_stereo,
        };
        self.bind_groups.get_or_create(key, |key| {
            let scene_color_view = create_d2_array_view(
                &key.scene_color_texture,
                "gtao_scene_color_sampled",
                key.multiview_stereo,
            );
            let (depth_dim, depth_layer_count) = if key.multiview_stereo {
                (wgpu::TextureViewDimension::D2Array, Some(2))
            } else {
                (wgpu::TextureViewDimension::D2, Some(1))
            };
            let scene_depth_view =
                key.scene_depth_texture
                    .create_view(&wgpu::TextureViewDescriptor {
                        label: Some("gtao_scene_depth_sampled"),
                        aspect: wgpu::TextureAspect::DepthOnly,
                        dimension: Some(depth_dim),
                        array_layer_count: depth_layer_count,
                        ..Default::default()
                    });
            let normal_view = create_d2_array_view(
                &key.normal_texture,
                "gtao_normals_sampled",
                key.multiview_stereo,
            );
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gtao"),
                layout: self.bind_group_layout(device, key.multiview_stereo),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&scene_color_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(self.sampler(device)),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&scene_depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: key.frame_uniforms.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.params_buffer(device).as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(&normal_view),
                    },
                ],
            })
        })
    }
}

/// Upper bound for cached GTAO bind groups before the cache is flushed.
///
/// Expected occupancy is one entry per active view (desktop / HMD / each secondary RT camera).
/// The cap protects against unbounded growth when views cycle during resize / MSAA / camera
/// churn.
const MAX_CACHED_BIND_GROUPS: usize = 16;
