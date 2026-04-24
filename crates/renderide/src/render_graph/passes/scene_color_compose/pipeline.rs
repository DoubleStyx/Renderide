//! Cached pipelines and bind layout for [`super::SceneColorComposePass`].
//!
//! WGSL is sourced from the build-time embedded shader registry
//! ([`crate::embedded_shaders::embedded_target_wgsl`]): the single source
//! `shaders/source/post/scene_color_compose.wgsl` is composed twice with `#ifdef MULTIVIEW` into
//! `scene_color_compose_default` and `scene_color_compose_multiview` by the build script.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use parking_lot::Mutex;

use crate::embedded_shaders::embedded_target_wgsl;

/// Embedded shader stem for the mono variant.
const WGSL_STEM_MONO: &str = "scene_color_compose_default";
/// Embedded shader stem for the multiview variant.
const WGSL_STEM_MULTIVIEW: &str = "scene_color_compose_multiview";

/// GPU state shared by all compose passes (bind layout + sampler).
pub(super) struct SceneColorComposePipelineCache {
    bind_group_layout: OnceLock<wgpu::BindGroupLayout>,
    sampler: OnceLock<wgpu::Sampler>,
    mono: Mutex<HashMap<wgpu::TextureFormat, Arc<wgpu::RenderPipeline>>>,
    multiview: Mutex<HashMap<wgpu::TextureFormat, Arc<wgpu::RenderPipeline>>>,
    /// Bind groups keyed by scene-color texture identity + multiview flag; avoids rebuilding
    /// on every frame when the transient pool reuses the same allocation.
    bind_groups: Mutex<HashMap<(wgpu::Texture, bool), wgpu::BindGroup>>,
}

impl Default for SceneColorComposePipelineCache {
    fn default() -> Self {
        Self {
            bind_group_layout: OnceLock::new(),
            sampler: OnceLock::new(),
            mono: Mutex::new(HashMap::new()),
            multiview: Mutex::new(HashMap::new()),
            bind_groups: Mutex::new(HashMap::new()),
        }
    }
}

impl SceneColorComposePipelineCache {
    /// Linear clamp sampler for HDR scene color.
    pub(super) fn sampler(&self, device: &wgpu::Device) -> &wgpu::Sampler {
        self.sampler.get_or_init(|| {
            device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("scene_color_compose"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                ..Default::default()
            })
        })
    }

    fn bind_group_layout(&self, device: &wgpu::Device) -> &wgpu::BindGroupLayout {
        self.bind_group_layout.get_or_init(|| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("scene_color_compose"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
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
        let map = if multiview_stereo {
            &self.multiview
        } else {
            &self.mono
        };
        let mut guard = map.lock();
        if let Some(p) = guard.get(&output_format) {
            return Arc::clone(p);
        }
        logger::debug!(
            "scene_color_compose: building pipeline (dst format = {:?}, multiview = {})",
            output_format,
            multiview_stereo
        );
        let stem = if multiview_stereo {
            WGSL_STEM_MULTIVIEW
        } else {
            WGSL_STEM_MONO
        };
        #[expect(
            clippy::expect_used,
            reason = "embedded shader is required; absence is a build script regression"
        )]
        let source = embedded_target_wgsl(stem)
            .expect("scene_color_compose: embedded shader missing (build script regression)");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(stem),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(stem),
            bind_group_layouts: &[Some(self.bind_group_layout(device))],
            immediate_size: 0,
        });
        let pipeline = Arc::new(
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(stem),
                layout: Some(&layout),
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
                multisample: Default::default(),
                multiview_mask: multiview_stereo
                    .then(|| std::num::NonZeroU32::new(3))
                    .flatten(),
                cache: None,
            }),
        );
        guard.insert(output_format, Arc::clone(&pipeline));
        pipeline
    }

    /// Bind group for one frame's scene-color texture, cached by `(Texture, multiview_stereo)`.
    pub(super) fn bind_group(
        &self,
        device: &wgpu::Device,
        scene_color_texture: &wgpu::Texture,
        multiview_stereo: bool,
    ) -> wgpu::BindGroup {
        let key = (scene_color_texture.clone(), multiview_stereo);
        {
            let guard = self.bind_groups.lock();
            if let Some(bg) = guard.get(&key) {
                return bg.clone();
            }
        }
        let layers_in_texture = scene_color_texture.size().depth_or_array_layers.max(1);
        let array_layer_count = if multiview_stereo {
            2.min(layers_in_texture)
        } else {
            1
        };
        let view = scene_color_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("scene_color_compose_sampled"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            array_layer_count: Some(array_layer_count),
            ..Default::default()
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene_color_compose"),
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
        let mut guard = self.bind_groups.lock();
        if let Some(existing) = guard.get(&key) {
            return existing.clone();
        }
        if guard.len() >= MAX_CACHED_BIND_GROUPS {
            guard.clear();
        }
        guard.insert(key, bg.clone());
        bg
    }
}

/// Upper bound for cached scene-color-compose bind groups before the cache is flushed.
///
/// Normally one or two entries (mono + multiview). The cap protects against unbounded growth
/// when the transient pool cycles allocations (resize / MSAA toggle).
const MAX_CACHED_BIND_GROUPS: usize = 8;
