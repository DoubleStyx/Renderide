//! Cached pipelines, bind-group layouts, sampler, and shared params UBO for the bloom passes.
//!
//! Bloom is a multi-pass effect (first downsample, N-1 subsequent downsamples, N-1 upsamples, one
//! composite). Every pipeline shares the same WGSL source (`shaders/source/post/bloom.wgsl`) but
//! differs by entry point, blend state, and bind-group layout (downsample/upsample use 1 group;
//! composite uses 2). The cache keys pipelines by [`BloomPipelineKind`] + output format +
//! multiview stereo, mirroring [`super::super::aces_tonemap::pipeline::AcesTonemapPipelineCache`]
//! so all bloom pass instances pay one-time pipeline compilation cost and subsequently hit a
//! `HashMap` lookup.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use bytemuck::{Pod, Zeroable};
use parking_lot::Mutex;

use crate::embedded_shaders::embedded_target_wgsl;

/// Embedded shader stem for the mono variant (no `MULTIVIEW` define).
const WGSL_STEM_MONO: &str = "bloom_default";
/// Embedded shader stem for the multiview variant (with `MULTIVIEW = Bool(true)`).
const WGSL_STEM_MULTIVIEW: &str = "bloom_multiview";

/// `std140`-compatible bloom uniform matching `BloomUniforms` in `bloom.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub(super) struct BloomParamsGpu {
    /// `[threshold, threshold - knee, 2*knee, 0.25/(knee + 1e-4)]`. See `soft_threshold` in WGSL.
    pub threshold_precomputations: [f32; 4],
    /// Composite intensity (scatter factor in linear HDR).
    pub intensity: f32,
    /// `1.0` → energy-conserving composite; `0.0` → additive composite.
    pub energy_conserving: f32,
    /// Alignment pad to 32 bytes (std140 vec2 tail).
    pub _pad: [f32; 2],
}

impl BloomParamsGpu {
    /// Builds the GPU-side params UBO from the current bloom settings. Called each frame by
    /// [`super::BloomDownsampleFirstPass::record`] so slider edits reach the shader without a
    /// graph rebuild.
    pub(super) fn from_settings(settings: &crate::config::BloomSettings) -> Self {
        Self {
            threshold_precomputations: threshold_precomputations(
                settings.prefilter_threshold,
                settings.prefilter_threshold_softness,
            ),
            intensity: settings.intensity.max(0.0),
            energy_conserving: match settings.composite_mode {
                crate::config::BloomCompositeMode::EnergyConserving => 1.0,
                crate::config::BloomCompositeMode::Additive => 0.0,
            },
            _pad: [0.0, 0.0],
        }
    }
}

/// Pipeline variant keyed into the cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum BloomPipelineKind {
    /// First downsample with Karis firefly reduction + optional soft-knee prefilter.
    DownsampleFirst,
    /// Plain 13-tap downsample between intermediate bloom mips.
    Downsample,
    /// 3×3 tent upsample with energy-conserving blend (`src*C + dst*(1-C)`).
    UpsampleEnergyConserving,
    /// 3×3 tent upsample with additive blend (`src*C + dst`).
    UpsampleAdditive,
    /// Composite: samples scene + bloom mip 0, does blend math in shader (Replace blend state).
    Composite,
}

impl BloomPipelineKind {
    fn entry_point(self) -> &'static str {
        match self {
            Self::DownsampleFirst => "fs_downsample_first",
            Self::Downsample => "fs_downsample",
            Self::UpsampleEnergyConserving | Self::UpsampleAdditive => "fs_upsample",
            Self::Composite => "fs_composite",
        }
    }

    fn needs_group_1(self) -> bool {
        matches!(self, Self::Composite)
    }

    fn color_blend(self) -> Option<wgpu::BlendState> {
        match self {
            Self::UpsampleEnergyConserving => Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::Constant,
                    dst_factor: wgpu::BlendFactor::OneMinusConstant,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent::REPLACE,
            }),
            Self::UpsampleAdditive => Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::Constant,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent::REPLACE,
            }),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct PipelineKey {
    kind: BloomPipelineKind,
    output_format: wgpu::TextureFormat,
    multiview_stereo: bool,
}

/// GPU state shared by every bloom pass instance.
pub(super) struct BloomPipelineCache {
    sampler: OnceLock<wgpu::Sampler>,
    bgl_group0: OnceLock<wgpu::BindGroupLayout>,
    bgl_group1: OnceLock<wgpu::BindGroupLayout>,
    params_buffer: OnceLock<wgpu::Buffer>,
    shader_mono: OnceLock<wgpu::ShaderModule>,
    shader_multiview: OnceLock<wgpu::ShaderModule>,
    pipelines: Mutex<HashMap<PipelineKey, Arc<wgpu::RenderPipeline>>>,
    /// Group 0 bind groups keyed by `(source texture, multiview)`. Source is either the chain
    /// HDR input (first downsample / composite) or a bloom mip texture (downsample chain, upsample
    /// chain). Stale entries are orphaned by transient-pool reuse.
    group0_bind_groups: Mutex<HashMap<(wgpu::Texture, bool), wgpu::BindGroup>>,
    /// Group 1 bind groups keyed by `(bloom mip 0 texture, multiview)`. Composite-only.
    group1_bind_groups: Mutex<HashMap<(wgpu::Texture, bool), wgpu::BindGroup>>,
}

impl Default for BloomPipelineCache {
    fn default() -> Self {
        Self {
            sampler: OnceLock::new(),
            bgl_group0: OnceLock::new(),
            bgl_group1: OnceLock::new(),
            params_buffer: OnceLock::new(),
            shader_mono: OnceLock::new(),
            shader_multiview: OnceLock::new(),
            pipelines: Mutex::new(HashMap::new()),
            group0_bind_groups: Mutex::new(HashMap::new()),
            group1_bind_groups: Mutex::new(HashMap::new()),
        }
    }
}

impl BloomPipelineCache {
    /// Linear clamp sampler shared across every bloom stage.
    pub(super) fn sampler(&self, device: &wgpu::Device) -> &wgpu::Sampler {
        self.sampler.get_or_init(|| {
            device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("bloom"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                ..Default::default()
            })
        })
    }

    /// Process-wide bloom params UBO. Overwritten once per frame by the first downsample pass.
    pub(super) fn params_buffer(&self, device: &wgpu::Device) -> &wgpu::Buffer {
        self.params_buffer.get_or_init(|| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("bloom-params"),
                size: std::mem::size_of::<BloomParamsGpu>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        })
    }

    /// Group 0 layout: `src_texture (2D array, filterable)`, `sampler (filtering)`, `uniforms`.
    pub(super) fn bind_group_layout_0(&self, device: &wgpu::Device) -> &wgpu::BindGroupLayout {
        self.bgl_group0.get_or_init(|| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bloom-group0"),
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<
                                BloomParamsGpu,
                            >()
                                as u64),
                        },
                        count: None,
                    },
                ],
            })
        })
    }

    /// Group 1 layout: `bloom_texture (2D array, filterable)`. Composite-only.
    pub(super) fn bind_group_layout_1(&self, device: &wgpu::Device) -> &wgpu::BindGroupLayout {
        self.bgl_group1.get_or_init(|| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bloom-group1"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                }],
            })
        })
    }

    fn shader_module(&self, device: &wgpu::Device, multiview_stereo: bool) -> &wgpu::ShaderModule {
        let slot = if multiview_stereo {
            &self.shader_multiview
        } else {
            &self.shader_mono
        };
        slot.get_or_init(|| {
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
                .expect("bloom: embedded shader missing (build script regression)");
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(stem),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            })
        })
    }

    /// Fetches or builds a pipeline for the given variant. Pipelines are stored in an `Arc` so
    /// concurrent callers share one GPU object; the cache guards the map behind a [`Mutex`].
    pub(super) fn pipeline(
        &self,
        device: &wgpu::Device,
        kind: BloomPipelineKind,
        output_format: wgpu::TextureFormat,
        multiview_stereo: bool,
    ) -> Arc<wgpu::RenderPipeline> {
        let key = PipelineKey {
            kind,
            output_format,
            multiview_stereo,
        };
        {
            let guard = self.pipelines.lock();
            if let Some(p) = guard.get(&key) {
                return Arc::clone(p);
            }
        }
        let shader = self.shader_module(device, multiview_stereo).clone();
        let bgl0 = self.bind_group_layout_0(device);
        let bgl1 = self.bind_group_layout_1(device);
        let layouts: &[Option<&wgpu::BindGroupLayout>] = if kind.needs_group_1() {
            &[Some(bgl0), Some(bgl1)]
        } else {
            &[Some(bgl0)]
        };
        let label = format!("bloom-{:?}", kind);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&label),
            bind_group_layouts: layouts,
            immediate_size: 0,
        });
        let pipeline = Arc::new(
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(&label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(kind.entry_point()),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: output_format,
                        blend: kind.color_blend(),
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
        let mut guard = self.pipelines.lock();
        if let Some(existing) = guard.get(&key) {
            return Arc::clone(existing);
        }
        guard.insert(key, Arc::clone(&pipeline));
        pipeline
    }

    /// Builds or fetches a group-0 bind group for sampling `texture` as the current stage input,
    /// plus the shared sampler and params UBO. Caches per `(texture, multiview_stereo)`.
    pub(super) fn group0_bind_group(
        &self,
        device: &wgpu::Device,
        texture: &wgpu::Texture,
        multiview_stereo: bool,
    ) -> wgpu::BindGroup {
        let key = (texture.clone(), multiview_stereo);
        {
            let guard = self.group0_bind_groups.lock();
            if let Some(bg) = guard.get(&key) {
                return bg.clone();
            }
        }
        let layers_in_texture = texture.size().depth_or_array_layers.max(1);
        let array_layer_count = if multiview_stereo {
            2.min(layers_in_texture)
        } else {
            1
        };
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("bloom-group0-src"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            array_layer_count: Some(array_layer_count),
            ..Default::default()
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom-group0"),
            layout: self.bind_group_layout_0(device),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(self.sampler(device)),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.params_buffer(device).as_entire_binding(),
                },
            ],
        });
        let mut guard = self.group0_bind_groups.lock();
        if let Some(existing) = guard.get(&key) {
            return existing.clone();
        }
        if guard.len() >= MAX_CACHED_BIND_GROUPS {
            guard.clear();
        }
        guard.insert(key, bg.clone());
        bg
    }

    /// Builds or fetches a group-1 bind group for sampling bloom mip 0 during the composite.
    pub(super) fn group1_bind_group(
        &self,
        device: &wgpu::Device,
        bloom_mip0: &wgpu::Texture,
        multiview_stereo: bool,
    ) -> wgpu::BindGroup {
        let key = (bloom_mip0.clone(), multiview_stereo);
        {
            let guard = self.group1_bind_groups.lock();
            if let Some(bg) = guard.get(&key) {
                return bg.clone();
            }
        }
        let layers_in_texture = bloom_mip0.size().depth_or_array_layers.max(1);
        let array_layer_count = if multiview_stereo {
            2.min(layers_in_texture)
        } else {
            1
        };
        let view = bloom_mip0.create_view(&wgpu::TextureViewDescriptor {
            label: Some("bloom-group1-mip0"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            array_layer_count: Some(array_layer_count),
            ..Default::default()
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom-group1"),
            layout: self.bind_group_layout_1(device),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            }],
        });
        let mut guard = self.group1_bind_groups.lock();
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

/// Upper bound on cached bind groups. Bloom texture identities are stable across most frames; the
/// cap protects against unbounded growth when the transient pool recycles allocations rapidly
/// (e.g. viewport resize storms, MSAA flips).
const MAX_CACHED_BIND_GROUPS: usize = 32;

/// Precomputes the `[threshold, threshold-knee, 2*knee, 0.25/(knee + 1e-4)]` vector the shader's
/// `soft_threshold` reads. `threshold` and `softness` come from [`crate::config::BloomSettings`];
/// this matches Bevy's `BloomUniforms` precompute exactly.
pub(super) fn threshold_precomputations(threshold: f32, softness: f32) -> [f32; 4] {
    let threshold = threshold.max(0.0);
    let softness = softness.clamp(0.0, 1.0);
    let knee = threshold * softness;
    let soft_inv_denom = 0.25 / (knee + 1.0e-4);
    [threshold, threshold - knee, 2.0 * knee, soft_inv_denom]
}

#[cfg(test)]
mod tests {
    use super::threshold_precomputations;

    #[test]
    fn threshold_zero_yields_zero_curve() {
        let v = threshold_precomputations(0.0, 0.0);
        assert_eq!(v[0], 0.0);
        assert_eq!(v[1], 0.0);
        assert_eq!(v[2], 0.0);
        // Denominator uses 1e-4 floor so the curve doesn't explode; result is a large but finite
        // value. The shader gates the soft-threshold call on `threshold > 0`, so this constant is
        // only consulted when threshold > 0.
        assert!(v[3].is_finite() && v[3] > 0.0);
    }

    #[test]
    fn threshold_matches_bevy_formula() {
        // threshold=1.0, softness=0.5 → knee=0.5, components: [1.0, 0.5, 1.0, 0.25/0.5001]
        let v = threshold_precomputations(1.0, 0.5);
        assert!((v[0] - 1.0).abs() < 1e-6);
        assert!((v[1] - 0.5).abs() < 1e-6);
        assert!((v[2] - 1.0).abs() < 1e-6);
        let expected_last = 0.25 / (0.5 + 1.0e-4);
        assert!((v[3] - expected_last).abs() < 1e-6);
    }

    #[test]
    fn softness_clamped_to_unit_interval() {
        let over = threshold_precomputations(1.0, 2.0);
        let at_one = threshold_precomputations(1.0, 1.0);
        assert_eq!(over, at_one, "softness > 1 must clamp to 1");
    }
}
