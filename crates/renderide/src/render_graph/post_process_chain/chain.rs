//! Ordered list of [`PostProcessEffect`]s and the graph-wiring driver that inserts them between
//! the world-mesh forward HDR producer and the displayable target blit.

use crate::config::PostProcessingSettings;
use crate::render_graph::builder::GraphBuilder;
use crate::render_graph::ids::PassId;
use crate::render_graph::resources::TextureHandle;

use super::effect::{EffectPasses, PostProcessEffect};
use super::output::ChainOutput;
use super::ping_pong::{PingPongCursor, PingPongHdrSlots};

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

    /// Inserts the chain's enabled passes into `builder`, returning the wiring info.
    pub fn build_into_graph(
        &self,
        builder: &mut GraphBuilder,
        input: TextureHandle,
        settings: &PostProcessingSettings,
    ) -> ChainOutput {
        if !settings.enabled {
            for _ in &self.effects {
                builder.note_skipped_pass();
            }
            return ChainOutput::PassThrough(input);
        }
        let enabled_effect_count = self
            .effects
            .iter()
            .filter(|effect| effect.is_enabled(settings))
            .count();
        if enabled_effect_count == 0 {
            for _ in &self.effects {
                builder.note_skipped_pass();
            }
            return ChainOutput::PassThrough(input);
        }

        let mut cursor =
            PingPongCursor::start(PingPongHdrSlots::new(builder, enabled_effect_count), input);
        let mut first_pass: Option<PassId> = None;
        let mut last_pass: Option<PassId> = None;
        let mut registered_effects = Vec::new();

        for effect in &self.effects {
            if !effect.is_enabled(settings) {
                builder.note_skipped_pass();
                continue;
            }
            let registered = effect.register(builder, settings, cursor.input(), cursor.output());
            let EffectPasses::Registered { first, last } = registered else {
                builder.note_skipped_pass();
                continue;
            };
            if let Some(prev_tail) = last_pass {
                builder.add_edge(prev_tail, first);
            }
            first_pass.get_or_insert(first);
            last_pass = Some(last);
            registered_effects.push(effect.id().label());
            cursor.advance();
        }

        let Some((first_pass, last_pass)) = first_pass.zip(last_pass) else {
            return ChainOutput::PassThrough(input);
        };
        logger::info!(
            "post-processing chain: {} effect(s) active: {}",
            registered_effects.len(),
            registered_effects.join(", ")
        );
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
    use crate::config::{TonemapMode, TonemapSettings};
    use crate::render_graph::context::RasterPassCtx;
    use crate::render_graph::error::{RenderPassError, SetupError};
    use crate::render_graph::pass::{PassBuilder, RasterPass};
    use crate::render_graph::post_process_chain::effect::{EffectPasses, PostProcessEffectId};
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
        pass_through: bool,
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
            _settings: &PostProcessingSettings,
            input: TextureHandle,
            output: TextureHandle,
        ) -> EffectPasses {
            if self.pass_through {
                return EffectPasses::pass_through();
            }
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
            pass_through: false,
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
            pass_through: false,
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
            pass_through: false,
        }));
        chain.push(Box::new(MockEffect {
            id: PostProcessEffectId::AcesTonemap,
            enabled: true,
            pass_through: false,
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
    fn pass_through_effect_does_not_advance_chain_output() {
        let mut builder = GraphBuilder::new();
        let input = fake_input(&mut builder);
        let mut chain = PostProcessChain::new();
        chain.push(Box::new(MockEffect {
            id: PostProcessEffectId::MotionBlur,
            enabled: true,
            pass_through: true,
        }));
        let settings = PostProcessingSettings {
            enabled: true,
            ..Default::default()
        };

        let out = chain.build_into_graph(&mut builder, input, &settings);

        assert!(matches!(out, ChainOutput::PassThrough(h) if h == input));
    }
}
