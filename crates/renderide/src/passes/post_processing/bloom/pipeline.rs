//! Cached pipelines, bind-group layouts, sampler, and shared params UBO for the bloom passes.
//!
//! Bloom is a multi-pass effect (first downsample, N-1 subsequent downsamples, N-1 upsamples, one
//! composite). Every pipeline shares the same WGSL source (`shaders/passes/post/bloom.wgsl`) but
//! differs by entry point, blend state, and bind-group layout (downsample/upsample use 1 group;
//! composite uses 2). The cache keys pipelines by [`BloomPipelineKind`] + output format +
//! multiview stereo, mirroring [`super::super::aces_tonemap::pipeline::AcesTonemapPipelineCache`]
//! so all bloom pass instances pay one-time pipeline compilation cost and subsequently hit a
//! `HashMap` lookup.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use crate::embedded_shaders::{BLOOM_DEFAULT_WGSL, BLOOM_MULTIVIEW_WGSL};
use crate::gpu::bind_layout::{
    fragment_filterable_d2_array_entry, fragment_filtering_sampler_entry,
    uniform_buffer_layout_entry,
};
use crate::render_graph::gpu_cache::{
    BindGroupMap, FullscreenRenderPipelineDesc, OnceGpu, RenderPipelineMap, create_d2_array_view,
    create_fullscreen_render_pipeline, create_linear_clamp_sampler, create_uniform_buffer,
    create_wgsl_shader_module,
};

/// Debug label for the mono shader module (no `MULTIVIEW` define).
const SHADER_LABEL_MONO: &str = "bloom_default";
/// Debug label for the multiview shader module (with `MULTIVIEW = Bool(true)`).
const SHADER_LABEL_MULTIVIEW: &str = "bloom_multiview";

/// `std140`-compatible bloom uniform matching `BloomUniforms` in `bloom.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub(super) struct BloomParamsGpu {
    /// `[threshold, threshold - knee, 2*knee, 0.25/(knee + 1e-4)]`. See `soft_threshold` in WGSL.
    pub threshold_precomputations: [f32; 4],
    /// Composite intensity (scatter factor in linear HDR).
    pub intensity: f32,
    /// `1.0` -> energy-conserving composite; `0.0` -> additive composite.
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
    /// 3x3 tent upsample with energy-conserving blend (`src*C + dst*(1-C)`).
    UpsampleEnergyConserving,
    /// 3x3 tent upsample with additive blend (`src*C + dst`).
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
    /// Linear sampler shared by every bloom stage.
    sampler: OnceGpu<wgpu::Sampler>,
    /// Group 0 bind group layout.
    bgl_group0: OnceGpu<wgpu::BindGroupLayout>,
    /// Composite-only group 1 bind group layout.
    bgl_group1: OnceGpu<wgpu::BindGroupLayout>,
    /// Shared bloom parameter uniform buffer.
    params_buffer: OnceGpu<wgpu::Buffer>,
    /// Mono shader module.
    shader_mono: OnceGpu<wgpu::ShaderModule>,
    /// Multiview shader module.
    shader_multiview: OnceGpu<wgpu::ShaderModule>,
    /// Cached pipelines keyed by bloom variant, output format, and view shape.
    pipelines: RenderPipelineMap<PipelineKey>,
    /// Group 0 bind groups keyed by `(source texture, multiview)`. Source is either the chain
    /// HDR input (first downsample / composite) or a bloom mip texture (downsample chain, upsample
    /// chain). Stale entries are orphaned by transient-pool reuse.
    group0_bind_groups: BindGroupMap<(wgpu::Texture, bool)>,
    /// Group 1 bind groups keyed by `(bloom mip 0 texture, multiview)`. Composite-only.
    group1_bind_groups: BindGroupMap<(wgpu::Texture, bool)>,
}

impl Default for BloomPipelineCache {
    fn default() -> Self {
        Self {
            sampler: OnceGpu::default(),
            bgl_group0: OnceGpu::default(),
            bgl_group1: OnceGpu::default(),
            params_buffer: OnceGpu::default(),
            shader_mono: OnceGpu::default(),
            shader_multiview: OnceGpu::default(),
            pipelines: RenderPipelineMap::default(),
            group0_bind_groups: BindGroupMap::with_max_entries(MAX_CACHED_BIND_GROUPS),
            group1_bind_groups: BindGroupMap::with_max_entries(MAX_CACHED_BIND_GROUPS),
        }
    }
}

impl BloomPipelineCache {
    /// Linear clamp sampler shared across every bloom stage.
    pub(super) fn sampler(&self, device: &wgpu::Device) -> &wgpu::Sampler {
        self.sampler
            .get_or_create(|| create_linear_clamp_sampler(device, "bloom"))
    }

    /// Process-wide bloom params UBO. Overwritten once per frame by the first downsample pass.
    pub(super) fn params_buffer(&self, device: &wgpu::Device) -> &wgpu::Buffer {
        self.params_buffer.get_or_create(|| {
            create_uniform_buffer(device, "bloom-params", size_of::<BloomParamsGpu>() as u64)
        })
    }

    /// Group 0 layout: `src_texture (2D array, filterable)`, `sampler (filtering)`, `uniforms`.
    pub(super) fn bind_group_layout_0(&self, device: &wgpu::Device) -> &wgpu::BindGroupLayout {
        self.bgl_group0.get_or_create(|| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bloom-group0"),
                entries: &[
                    fragment_filterable_d2_array_entry(0),
                    fragment_filtering_sampler_entry(1),
                    uniform_buffer_layout_entry(
                        2,
                        wgpu::ShaderStages::FRAGMENT,
                        wgpu::BufferSize::new(size_of::<BloomParamsGpu>() as u64),
                    ),
                ],
            })
        })
    }

    /// Group 1 layout: `bloom_texture (2D array, filterable)`. Composite-only.
    pub(super) fn bind_group_layout_1(&self, device: &wgpu::Device) -> &wgpu::BindGroupLayout {
        self.bgl_group1.get_or_create(|| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bloom-group1"),
                entries: &[fragment_filterable_d2_array_entry(0)],
            })
        })
    }

    fn shader_module(&self, device: &wgpu::Device, multiview_stereo: bool) -> &wgpu::ShaderModule {
        let slot = if multiview_stereo {
            &self.shader_multiview
        } else {
            &self.shader_mono
        };
        slot.get_or_create(|| {
            let (label, source) = if multiview_stereo {
                (SHADER_LABEL_MULTIVIEW, BLOOM_MULTIVIEW_WGSL)
            } else {
                (SHADER_LABEL_MONO, BLOOM_DEFAULT_WGSL)
            };
            create_wgsl_shader_module(device, label, source)
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
        self.pipelines.get_or_create(key, |key| {
            let shader = self.shader_module(device, key.multiview_stereo).clone();
            let bgl0 = self.bind_group_layout_0(device);
            let bgl1 = self.bind_group_layout_1(device);
            let layouts: &[Option<&wgpu::BindGroupLayout>] = if key.kind.needs_group_1() {
                &[Some(bgl0), Some(bgl1)]
            } else {
                &[Some(bgl0)]
            };
            let label = format!("bloom-{:?}", key.kind);
            create_fullscreen_render_pipeline(
                device,
                FullscreenRenderPipelineDesc {
                    label: &label,
                    bind_group_layouts: layouts,
                    shader: &shader,
                    fragment_entry: key.kind.entry_point(),
                    output_format: key.output_format,
                    blend: key.kind.color_blend(),
                    multiview_stereo: key.multiview_stereo,
                },
            )
        })
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
        self.group0_bind_groups.get_or_create(key, |key| {
            let (texture, multiview_stereo) = key;
            let view = create_d2_array_view(texture, "bloom-group0-src", *multiview_stereo);
            device.create_bind_group(&wgpu::BindGroupDescriptor {
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
            })
        })
    }

    /// Builds or fetches a group-1 bind group for sampling bloom mip 0 during the composite.
    pub(super) fn group1_bind_group(
        &self,
        device: &wgpu::Device,
        bloom_mip0: &wgpu::Texture,
        multiview_stereo: bool,
    ) -> wgpu::BindGroup {
        let key = (bloom_mip0.clone(), multiview_stereo);
        self.group1_bind_groups.get_or_create(key, |key| {
            let (bloom_mip0, multiview_stereo) = key;
            let view = create_d2_array_view(bloom_mip0, "bloom-group1-mip0", *multiview_stereo);
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bloom-group1"),
                layout: self.bind_group_layout_1(device),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                }],
            })
        })
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
        // threshold=1.0, softness=0.5 -> knee=0.5, components: [1.0, 0.5, 1.0, 0.25/0.5001]
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
