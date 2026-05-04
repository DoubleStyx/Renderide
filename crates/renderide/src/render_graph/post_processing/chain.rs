//! Post-processing chain: ordered effects + graph wiring helpers.

use crate::config::{PostProcessingSettings, TonemapMode};
use crate::render_graph::builder::GraphBuilder;
use crate::render_graph::ids::PassId;
use crate::render_graph::resources::TextureHandle;

use super::effect::{PostProcessEffect, PostProcessEffectId};
use super::ping_pong::{PingPongCursor, PingPongHdrSlots};

/// Topology fingerprint for the post-processing chain at graph compile time.
///
/// Changes to any field force a render-graph rebuild. Non-topology parameters (intensity,
/// threshold, composite mode, etc.) flow to the passes via per-view blackboard slots
/// ([`crate::passes::post_processing::settings_slot::BloomSettingsSlot`],
/// [`crate::passes::post_processing::settings_slot::GtaoSettingsSlot`],
/// [`crate::passes::post_processing::settings_slot::AutoExposureSettingsSlot`]) and therefore do
/// **not** need to be tracked here.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct PostProcessChainSignature {
    /// Ground-Truth Ambient Occlusion pass active.
    pub gtao: bool,
    /// Number of GTAO depth-aware denoise iterations baked into the graph (`0..=3`).
    /// Topology field: `>= 2` adds an intermediate denoise pass and a second AO ping-pong
    /// transient; `3` adds another ping-pong iteration, so a change must rebuild. `0` when
    /// GTAO is inactive.
    pub gtao_denoise_passes: u32,
    /// Dual-filter bloom pass active.
    pub bloom: bool,
    /// Histogram-based auto-exposure pass active.
    pub auto_exposure: bool,
    /// Stephen Hill ACES Fitted tonemap pass active.
    pub aces_tonemap: bool,
    /// Effective bloom mip 0 target height (px). Baked into the mip-chain transient texture
    /// extents at graph-build time via
    /// [`crate::render_graph::resources::TransientExtent::BackbufferScaledMip`], so a change here
    /// must rebuild. `0` when bloom is inactive.
    pub bloom_max_mip_dimension: u32,
}

impl PostProcessChainSignature {
    /// Derives the signature from live [`PostProcessingSettings`].
    pub fn from_settings(settings: &PostProcessingSettings) -> Self {
        let master = settings.enabled;
        let gtao = master && settings.gtao.enabled;
        let bloom = master && settings.bloom.enabled && settings.bloom.intensity > 0.0;
        let auto_exposure = master && settings.auto_exposure.enabled;
        Self {
            gtao,
            gtao_denoise_passes: if gtao {
                settings.gtao.denoise_passes.min(3)
            } else {
                0
            },
            bloom,
            auto_exposure,
            aces_tonemap: master && matches!(settings.tonemap.mode, TonemapMode::AcesFitted),
            bloom_max_mip_dimension: if bloom {
                settings.bloom.effective_max_mip_dimension()
            } else {
                0
            },
        }
    }

    /// Returns `true` when no effects are active and the chain should be skipped entirely.
    pub fn is_empty(self) -> bool {
        !self.gtao && !self.bloom && !self.auto_exposure && !self.aces_tonemap
    }

    /// Number of active effects.
    pub fn active_count(self) -> usize {
        usize::from(self.gtao)
            + usize::from(self.bloom)
            + usize::from(self.auto_exposure)
            + usize::from(self.aces_tonemap)
    }
}

/// Result of [`PostProcessChain::build_into_graph`].
#[derive(Clone, Copy, Debug)]
#[expect(
    variant_size_differences,
    reason = "Copy enum; `Chained` carries pass-id range inline to avoid heap for a one-shot result"
)]
pub enum ChainOutput {
    /// No effects ran; the chain forwards the original input handle.
    PassThrough(TextureHandle),
    /// One or more effects ran; the chain output and pass-id range are returned so the caller
    /// can wire explicit edges.
    Chained {
        /// Final HDR output of the chain.
        final_handle: TextureHandle,
        /// First pass added by the chain.
        first_pass: PassId,
        /// Last pass added by the chain.
        last_pass: PassId,
    },
}

impl ChainOutput {
    /// Returns the final HDR handle the next consumer should read.
    pub fn final_handle(self) -> TextureHandle {
        match self {
            Self::PassThrough(h) => h,
            Self::Chained { final_handle, .. } => final_handle,
        }
    }

    /// Returns the first/last pass ids when the chain produced any pass.
    pub fn pass_range(self) -> Option<(PassId, PassId)> {
        match self {
            Self::PassThrough(_) => None,
            Self::Chained {
                first_pass,
                last_pass,
                ..
            } => Some((first_pass, last_pass)),
        }
    }
}

/// Ordered, configurable list of [`PostProcessEffect`] trait objects.
pub struct PostProcessChain {
    effects: Vec<Box<dyn PostProcessEffect>>,
}

impl PostProcessChain {
    /// Empty chain (no effects).
    pub fn new() -> Self {
        Self {
            effects: Vec::new(),
        }
    }

    /// Pushes an effect onto the chain.
    pub fn push(&mut self, effect: Box<dyn PostProcessEffect>) {
        self.effects.push(effect);
    }

    /// Iterates over effect identities in execution order.
    pub fn effect_ids(&self) -> impl Iterator<Item = PostProcessEffectId> + '_ {
        self.effects.iter().map(|e| e.id())
    }

    /// Number of registered effects (regardless of enable state).
    pub fn len(&self) -> usize {
        self.effects.len()
    }

    /// Whether the chain has zero registered effects.
    pub fn is_empty(&self) -> bool {
        self.effects.is_empty()
    }

    /// Inserts the chain's enabled passes into `builder`, returning the wiring info.
    pub fn build_into_graph(
        &self,
        builder: &mut GraphBuilder,
        input: TextureHandle,
        settings: &PostProcessingSettings,
    ) -> ChainOutput {
        if !settings.enabled || !self.effects.iter().any(|e| e.is_enabled(settings)) {
            return ChainOutput::PassThrough(input);
        }

        let active: Vec<&'static str> = self
            .effects
            .iter()
            .filter(|e| e.is_enabled(settings))
            .map(|e| e.id().label())
            .collect();
        logger::info!(
            "post-processing chain: {} effect(s) active: {}",
            active.len(),
            active.join(", ")
        );

        let mut cursor = PingPongCursor::start(PingPongHdrSlots::new(builder), input);
        let mut first_pass: Option<PassId> = None;
        let mut last_pass: Option<PassId> = None;

        for effect in self.effects.iter().filter(|e| e.is_enabled(settings)) {
            let registered = effect.register(builder, cursor.input(), cursor.output());
            if let Some(prev_tail) = last_pass {
                builder.add_edge(prev_tail, registered.first);
            }
            first_pass.get_or_insert(registered.first);
            last_pass = Some(registered.last);
            cursor.advance();
        }

        let Some((first_pass, last_pass)) = first_pass.zip(last_pass) else {
            return ChainOutput::PassThrough(input);
        };
        ChainOutput::Chained {
            final_handle: cursor.last_output(),
            first_pass,
            last_pass,
        }
    }
}

impl Default for PostProcessChain {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TonemapSettings;
    use crate::render_graph::context::RasterPassCtx;
    use crate::render_graph::error::{RenderPassError, SetupError};
    use crate::render_graph::pass::{PassBuilder, RasterPass};
    use crate::render_graph::post_processing::effect::EffectPasses;
    use crate::render_graph::resources::{
        TransientArrayLayers, TransientExtent, TransientSampleCount, TransientTextureDesc,
        TransientTextureFormat,
    };

    fn post_process_color_transient_desc(label: &'static str) -> TransientTextureDesc {
        TransientTextureDesc {
            label,
            format: TransientTextureFormat::SceneColorHdr,
            extent: TransientExtent::Backbuffer,
            mip_levels: 1,
            sample_count: TransientSampleCount::Fixed(1),
            dimension: wgpu::TextureDimension::D2,
            array_layers: TransientArrayLayers::Frame,
            base_usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING,
            alias: true,
        }
    }

    struct MockEffect {
        id: PostProcessEffectId,
        enabled: bool,
    }

    impl PostProcessEffect for MockEffect {
        fn id(&self) -> PostProcessEffectId {
            self.id
        }

        fn is_enabled(&self, _settings: &PostProcessingSettings) -> bool {
            self.enabled
        }

        fn register(
            &self,
            builder: &mut GraphBuilder,
            input: TextureHandle,
            output: TextureHandle,
        ) -> EffectPasses {
            let pass_id = builder.add_raster_pass(Box::new(MockPass {
                name: self.id.label(),
                input,
                output,
            }));
            EffectPasses::single(pass_id)
        }
    }

    struct MockPass {
        name: &'static str,
        input: TextureHandle,
        output: TextureHandle,
    }

    impl RasterPass for MockPass {
        fn name(&self) -> &str {
            self.name
        }

        fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
            use crate::render_graph::resources::TextureAccess;
            b.read_texture_resource(
                self.input,
                TextureAccess::Sampled {
                    stages: wgpu::ShaderStages::FRAGMENT,
                },
            );
            let mut r = b.raster();
            r.color(
                self.output,
                wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                Option::<TextureHandle>::None,
            );
            Ok(())
        }

        fn record(
            &self,
            _ctx: &mut RasterPassCtx<'_, '_>,
            _rpass: &mut wgpu::RenderPass<'_>,
        ) -> Result<(), RenderPassError> {
            Ok(())
        }
    }

    fn fake_input(builder: &mut GraphBuilder) -> TextureHandle {
        builder.create_texture(post_process_color_transient_desc("scene_color_hdr"))
    }

    #[test]
    fn empty_chain_returns_pass_through() {
        let mut builder = GraphBuilder::new();
        let input = fake_input(&mut builder);
        let chain = PostProcessChain::new();
        let settings = PostProcessingSettings {
            enabled: true,
            ..Default::default()
        };
        let out = chain.build_into_graph(&mut builder, input, &settings);
        assert!(matches!(out, ChainOutput::PassThrough(h) if h == input));
    }

    #[test]
    fn disabled_master_returns_pass_through_even_with_effects() {
        let mut builder = GraphBuilder::new();
        let input = fake_input(&mut builder);
        let mut chain = PostProcessChain::new();
        chain.push(Box::new(MockEffect {
            id: PostProcessEffectId::AcesTonemap,
            enabled: true,
        }));
        let settings = PostProcessingSettings {
            enabled: false,
            ..Default::default()
        };
        let out = chain.build_into_graph(&mut builder, input, &settings);
        assert!(matches!(out, ChainOutput::PassThrough(h) if h == input));
    }

    #[test]
    fn single_enabled_effect_creates_one_pass_and_chains_handles() {
        let mut builder = GraphBuilder::new();
        let input = fake_input(&mut builder);
        let mut chain = PostProcessChain::new();
        chain.push(Box::new(MockEffect {
            id: PostProcessEffectId::AcesTonemap,
            enabled: true,
        }));
        let settings = PostProcessingSettings {
            enabled: true,
            tonemap: TonemapSettings {
                mode: TonemapMode::AcesFitted,
            },
            ..Default::default()
        };
        let out = chain.build_into_graph(&mut builder, input, &settings);
        match out {
            ChainOutput::Chained {
                final_handle,
                first_pass,
                last_pass,
            } => {
                assert_ne!(
                    final_handle, input,
                    "final handle must be a chain transient"
                );
                assert_eq!(
                    first_pass, last_pass,
                    "single effect produces a single pass"
                );
            }
            other @ ChainOutput::PassThrough(_) => {
                panic!("expected Chained variant, got {other:?}")
            }
        }
    }

    #[test]
    fn multiple_effects_ping_pong_to_pong_slot() {
        let mut builder = GraphBuilder::new();
        let input = fake_input(&mut builder);
        let mut chain = PostProcessChain::new();
        chain.push(Box::new(MockEffect {
            id: PostProcessEffectId::AcesTonemap,
            enabled: true,
        }));
        chain.push(Box::new(MockEffect {
            id: PostProcessEffectId::AcesTonemap,
            enabled: true,
        }));
        let settings = PostProcessingSettings {
            enabled: true,
            ..Default::default()
        };
        let out = chain.build_into_graph(&mut builder, input, &settings);
        match out {
            ChainOutput::Chained {
                final_handle,
                first_pass,
                last_pass,
            } => {
                assert_ne!(final_handle, input);
                assert_ne!(first_pass, last_pass);
            }
            other @ ChainOutput::PassThrough(_) => {
                panic!("expected Chained variant, got {other:?}")
            }
        }
    }

    #[test]
    fn signature_from_settings_matches_master_toggle() {
        let mut s = PostProcessingSettings {
            enabled: false,
            tonemap: TonemapSettings {
                mode: TonemapMode::AcesFitted,
            },
            ..Default::default()
        };
        assert!(PostProcessChainSignature::from_settings(&s).is_empty());

        s.enabled = true;
        let sig = PostProcessChainSignature::from_settings(&s);
        assert!(sig.aces_tonemap);
        assert!(sig.gtao);
        assert!(sig.bloom);
        assert!(sig.auto_exposure);
        assert_eq!(sig.active_count(), 4);

        s.tonemap.mode = TonemapMode::None;
        assert!(PostProcessChainSignature::from_settings(&s).gtao);
        assert!(PostProcessChainSignature::from_settings(&s).bloom);
        assert!(PostProcessChainSignature::from_settings(&s).auto_exposure);
        s.gtao.enabled = false;
        s.bloom.enabled = false;
        s.auto_exposure.enabled = false;
        assert!(PostProcessChainSignature::from_settings(&s).is_empty());
    }

    #[test]
    fn signature_tracks_gtao_toggle_independently_of_tonemap() {
        let mut s = PostProcessingSettings {
            enabled: true,
            tonemap: TonemapSettings {
                mode: TonemapMode::None,
            },
            ..Default::default()
        };
        s.bloom.enabled = false;
        s.auto_exposure.enabled = false;
        let sig = PostProcessChainSignature::from_settings(&s);
        assert!(sig.gtao);
        assert!(!sig.aces_tonemap);
        assert!(!sig.bloom);
        assert_eq!(sig.active_count(), 1);

        s.gtao.enabled = false;
        assert!(PostProcessChainSignature::from_settings(&s).is_empty());

        s.enabled = false;
        assert!(PostProcessChainSignature::from_settings(&s).is_empty());
    }

    #[test]
    fn signature_clamps_gtao_denoise_topology_to_soft_preset() {
        let mut s = PostProcessingSettings {
            enabled: true,
            tonemap: TonemapSettings {
                mode: TonemapMode::None,
            },
            ..Default::default()
        };
        s.bloom.enabled = false;
        s.gtao.denoise_passes = 99;

        let sig = PostProcessChainSignature::from_settings(&s);

        assert!(sig.gtao);
        assert_eq!(sig.gtao_denoise_passes, 3);
    }

    #[test]
    fn signature_tracks_bloom_toggle_and_intensity_gate() {
        let mut s = PostProcessingSettings {
            enabled: true,
            tonemap: TonemapSettings {
                mode: TonemapMode::None,
            },
            ..Default::default()
        };
        s.gtao.enabled = false;
        s.auto_exposure.enabled = false;
        s.bloom.enabled = false;
        assert!(PostProcessChainSignature::from_settings(&s).is_empty());

        s.bloom.enabled = true;
        s.bloom.intensity = 0.15;
        let sig = PostProcessChainSignature::from_settings(&s);
        assert!(sig.bloom);
        assert_eq!(sig.active_count(), 1);

        s.bloom.intensity = 0.0;
        assert!(
            !PostProcessChainSignature::from_settings(&s).bloom,
            "intensity=0 must gate bloom off even when enabled"
        );
    }

    #[test]
    fn signature_tracks_effective_bloom_max_mip_dimension() {
        let mut s = PostProcessingSettings {
            enabled: true,
            tonemap: TonemapSettings {
                mode: TonemapMode::None,
            },
            ..Default::default()
        };
        s.gtao.enabled = false;
        s.auto_exposure.enabled = false;
        s.bloom.max_mip_dimension = 511;

        let sig = PostProcessChainSignature::from_settings(&s);

        assert!(sig.bloom);
        assert_eq!(sig.bloom_max_mip_dimension, 256);
    }

    #[test]
    fn signature_tracks_auto_exposure_toggle_independently_of_tonemap() {
        let mut s = PostProcessingSettings {
            enabled: true,
            tonemap: TonemapSettings {
                mode: TonemapMode::None,
            },
            ..Default::default()
        };
        s.gtao.enabled = false;
        s.bloom.enabled = false;

        let sig = PostProcessChainSignature::from_settings(&s);

        assert!(sig.auto_exposure);
        assert_eq!(sig.active_count(), 1);

        s.auto_exposure.enabled = false;
        assert!(PostProcessChainSignature::from_settings(&s).is_empty());
    }

    #[test]
    fn chain_output_helpers() {
        let h = TextureHandle(7);
        let pt = ChainOutput::PassThrough(h);
        assert_eq!(pt.final_handle(), h);
        assert!(pt.pass_range().is_none());

        let chained = ChainOutput::Chained {
            final_handle: h,
            first_pass: PassId(1),
            last_pass: PassId(2),
        };
        assert_eq!(chained.final_handle(), h);
        assert_eq!(chained.pass_range(), Some((PassId(1), PassId(2))));
    }
}
