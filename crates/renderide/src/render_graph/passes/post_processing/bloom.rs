//! Dual-filter physically-based bloom (COD: Advanced Warfare / Bevy port).
//!
//! Registers a subgraph on the post-processing chain: one first downsample (Karis + optional
//! soft-knee threshold) writes bloom mip 0, subsequent downsamples populate the remaining mips,
//! an upsample ladder blends each mip into the next-finer level with a per-mip constant factor,
//! and a final composite pass combines the chain input with bloom mip 0 using the configured
//! composite math (energy-conserving by default). Runs pre-tonemap so it scatters HDR-linear
//! light; the kernel weights and blend math match Bevy's `crates/bevy_post_process/src/bloom/`.

mod composite;
mod downsample;
mod helpers;
mod pipeline;
mod upsample;

use std::sync::OnceLock;

use composite::BloomCompositePass;
use downsample::{BloomDownsampleFirstPass, BloomDownsamplePass};
use pipeline::BloomPipelineCache;
use upsample::BloomUpsamplePass;

use crate::config::{BloomCompositeMode, BloomSettings, PostProcessingSettings};
use crate::render_graph::builder::GraphBuilder;
use crate::render_graph::post_processing::{EffectPasses, PostProcessEffect, PostProcessEffectId};
use crate::render_graph::resources::{
    TextureHandle, TransientArrayLayers, TransientExtent, TransientSampleCount,
    TransientTextureDesc, TransientTextureFormat,
};

/// Storage format for the bloom mip pyramid. 11/11/10-bit float matches Bevy — lower bandwidth
/// than `Rgba16Float` while still covering the HDR range bloom needs to scatter.
const BLOOM_TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg11b10Ufloat;

/// Static labels for each bloom mip transient texture. Sized to cover every reasonable
/// `max_mip_dimension` (log2(65536) = 16).
const BLOOM_MIP_LABELS: [&str; 16] = [
    "bloom_mip_0",
    "bloom_mip_1",
    "bloom_mip_2",
    "bloom_mip_3",
    "bloom_mip_4",
    "bloom_mip_5",
    "bloom_mip_6",
    "bloom_mip_7",
    "bloom_mip_8",
    "bloom_mip_9",
    "bloom_mip_10",
    "bloom_mip_11",
    "bloom_mip_12",
    "bloom_mip_13",
    "bloom_mip_14",
    "bloom_mip_15",
];

/// Effect descriptor plugged into [`crate::render_graph::post_processing::PostProcessChain`].
///
/// Captures a snapshot of [`BloomSettings`] at chain-build time; when the signature-producing
/// fields change (enabled, intensity = 0, effective max mip dimension), the chain is rebuilt.
/// Other parameters (composite mode, low-frequency boost, etc.) take effect on the next frame
/// without a graph rebuild because they're routed through per-pass blend constants and the shared
/// params UBO.
pub struct BloomEffect {
    /// Live bloom tunables captured at chain-build time. Uploaded to the shared params UBO
    /// during the first downsample pass and consumed by the composite for intensity / mode.
    pub settings: BloomSettings,
}

impl PostProcessEffect for BloomEffect {
    fn id(&self) -> PostProcessEffectId {
        PostProcessEffectId::Bloom
    }

    fn is_enabled(&self, settings: &PostProcessingSettings) -> bool {
        settings.enabled && settings.bloom.enabled && settings.bloom.intensity > 0.0
    }

    fn register(
        &self,
        builder: &mut GraphBuilder,
        input: TextureHandle,
        output: TextureHandle,
    ) -> EffectPasses {
        let settings = self.settings;
        let pipelines = bloom_pipelines();
        let max_mip_dimension = settings.effective_max_mip_dimension();
        let mip_count = bloom_mip_count(max_mip_dimension);

        // One transient texture per mip level — avoids needing mip-level render-target views on
        // a single multi-mipped texture, which the graph builder's attachment API doesn't model.
        // The transient pool still aliases across frames and benefits from the per-mip size keys.
        let mip_handles: Vec<TextureHandle> = (0..mip_count)
            .map(|mip| {
                let label = BLOOM_MIP_LABELS
                    .get(mip as usize)
                    .copied()
                    .unwrap_or("bloom_mip_overflow");
                builder.create_texture(TransientTextureDesc {
                    label,
                    format: TransientTextureFormat::Fixed(BLOOM_TEXTURE_FORMAT),
                    extent: TransientExtent::BackbufferScaledMip {
                        max_dim: max_mip_dimension,
                        mip,
                    },
                    mip_levels: 1,
                    sample_count: TransientSampleCount::Fixed(1),
                    dimension: wgpu::TextureDimension::D2,
                    array_layers: TransientArrayLayers::Frame,
                    base_usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    alias: true,
                })
            })
            .collect();

        let first_downsample = builder.add_raster_pass(Box::new(BloomDownsampleFirstPass::new(
            input,
            mip_handles[0],
            settings,
            pipelines,
        )));
        let mut prev = first_downsample;

        for i in 1..(mip_count as usize) {
            let pass = builder.add_raster_pass(Box::new(BloomDownsamplePass::new(
                mip_handles[i - 1],
                mip_handles[i],
                i as u32,
                pipelines,
            )));
            builder.add_edge(prev, pass);
            prev = pass;
        }

        let max_mip = (mip_count.saturating_sub(1)) as f32;
        for i in (1..(mip_count as usize)).rev() {
            let pass = builder.add_raster_pass(Box::new(BloomUpsamplePass::new(
                mip_handles[i],
                mip_handles[i - 1],
                i as u32,
                max_mip,
                settings,
                pipelines,
            )));
            builder.add_edge(prev, pass);
            prev = pass;
        }

        let composite = builder.add_raster_pass(Box::new(BloomCompositePass::new(
            input,
            mip_handles[0],
            output,
            pipelines,
        )));
        builder.add_edge(prev, composite);

        EffectPasses {
            first: first_downsample,
            last: composite,
        }
    }
}

/// Process-wide bloom pipeline cache shared across every chain rebuild.
fn bloom_pipelines() -> &'static BloomPipelineCache {
    static CACHE: OnceLock<BloomPipelineCache> = OnceLock::new();
    CACHE.get_or_init(BloomPipelineCache::default)
}

/// Bloom pyramid mip count (matches Bevy `prepare_bloom_textures`): `max(2, log2(max_mip_dim)) - 1`.
/// Returns at least 1 so the first downsample always has somewhere to write.
fn bloom_mip_count(max_mip_dim: u32) -> u32 {
    let log2 = u32::BITS - max_mip_dim.max(1).leading_zeros() - 1;
    log2.max(2) - 1
}

/// Per-mip upsample blend factor (direct port of Bevy `compute_blend_factor`). `mip` is the
/// source mip being read (higher = lower frequency); `max_mip` is `mip_count - 1`. The factor
/// is uploaded via [`wgpu::RenderPass::set_blend_constant`] and consumed by the GPU blend unit
/// as `src * C + dst * (1-C)` (energy-conserving) or `src * C + dst` (additive).
pub(super) fn compute_blend_factor(settings: &BloomSettings, mip: f32, max_mip: f32) -> f32 {
    let epsilon = 1.0e-6_f32;
    let max_mip = max_mip.max(epsilon);
    let curvature = settings
        .low_frequency_boost_curvature
        .clamp(0.0, 1.0 - epsilon);
    let hpf = settings.high_pass_frequency.max(epsilon);

    let mut lf_boost = (1.0 - (1.0 - (mip / max_mip)).powf(1.0 / (1.0 - curvature)))
        * settings.low_frequency_boost.max(0.0);
    let high_pass_lq =
        1.0 - (((mip / max_mip) - settings.high_pass_frequency) / hpf).clamp(0.0, 1.0);

    lf_boost *= match settings.composite_mode {
        BloomCompositeMode::EnergyConserving => (1.0 - settings.intensity).max(0.0),
        BloomCompositeMode::Additive => 1.0,
    };

    ((settings.intensity.max(0.0) + lf_boost) * high_pass_lq).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    use crate::render_graph::pass::RasterPass;
    use crate::render_graph::resources::TextureHandle;

    #[test]
    fn mip_count_matches_bevy_formula() {
        assert_eq!(bloom_mip_count(512), 8);
        assert_eq!(bloom_mip_count(256), 7);
        assert_eq!(bloom_mip_count(1024), 9);
        // Clamp: log2 < 2 still yields at least 1 mip.
        assert_eq!(bloom_mip_count(1), 1);
        assert_eq!(bloom_mip_count(2), 1);
        assert_eq!(bloom_mip_count(4), 1);
        assert_eq!(bloom_mip_count(8), 2);
    }

    #[test]
    fn gpu_profile_labels_identify_each_bloom_mip_pass() {
        let pipelines = bloom_pipelines();
        let settings = BloomSettings::default();
        let first = downsample::BloomDownsampleFirstPass::new(
            TextureHandle(0),
            TextureHandle(1),
            settings,
            pipelines,
        );
        let downsample =
            downsample::BloomDownsamplePass::new(TextureHandle(1), TextureHandle(2), 1, pipelines);
        let upsample = upsample::BloomUpsamplePass::new(
            TextureHandle(2),
            TextureHandle(1),
            2,
            2.0,
            settings,
            pipelines,
        );

        let labels = [
            first.profiling_label().into_owned(),
            downsample.profiling_label().into_owned(),
            upsample.profiling_label().into_owned(),
        ];

        assert_eq!(labels[0], "BloomDownsampleFirst.mip0");
        assert_eq!(labels[1], "BloomDownsample.mip1");
        assert_eq!(labels[2], "BloomUpsample.mip2_to_mip1");
        assert_eq!(labels.iter().collect::<HashSet<_>>().len(), labels.len());
    }

    #[test]
    fn blend_factor_at_mip_zero_equals_intensity() {
        let s = BloomSettings {
            intensity: 0.5,
            low_frequency_boost: 0.7,
            low_frequency_boost_curvature: 0.95,
            high_pass_frequency: 1.0,
            composite_mode: BloomCompositeMode::EnergyConserving,
            ..BloomSettings::default()
        };
        let f = compute_blend_factor(&s, 0.0, 7.0);
        assert!((f - s.intensity).abs() < 1e-5, "got {f}");
    }

    #[test]
    fn blend_factor_clamped_to_unit_interval() {
        let s = BloomSettings {
            intensity: 0.9,
            low_frequency_boost: 5.0, // intentionally too large
            low_frequency_boost_curvature: 0.95,
            high_pass_frequency: 1.0,
            composite_mode: BloomCompositeMode::Additive,
            ..BloomSettings::default()
        };
        for i in 0..=7 {
            let f = compute_blend_factor(&s, i as f32, 7.0);
            assert!(
                (0.0..=1.0).contains(&f),
                "factor {f} out of [0, 1] at mip {i}"
            );
        }
    }

    #[test]
    fn energy_conserving_factor_never_exceeds_one() {
        let s = BloomSettings {
            intensity: 0.15,
            low_frequency_boost: 0.7,
            low_frequency_boost_curvature: 0.95,
            high_pass_frequency: 1.0,
            composite_mode: BloomCompositeMode::EnergyConserving,
            ..BloomSettings::default()
        };
        for i in 0..=7 {
            let f = compute_blend_factor(&s, i as f32, 7.0);
            assert!(
                (0.0..=1.0).contains(&f),
                "factor {f} out of range at mip {i}"
            );
        }
    }
}
