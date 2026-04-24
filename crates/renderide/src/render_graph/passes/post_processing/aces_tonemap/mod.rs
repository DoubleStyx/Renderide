//! Stephen Hill ACES Fitted tonemap render pass.
//!
//! Reads an HDR scene-color array texture, applies the ACES Fitted curve (sRGB → AP1 → RRT+ODT
//! polynomial → AP1 → sRGB → saturate), and writes a chain HDR transient that the next post pass
//! (or [`crate::render_graph::passes::SceneColorComposePass`]) consumes. Output is in `[0, 1]`
//! linear sRGB so the existing sRGB swapchain encodes gamma correctly without a separate gamma
//! pass.

mod pipeline;

use std::num::NonZeroU32;
use std::sync::OnceLock;

use pipeline::AcesTonemapPipelineCache;

use crate::config::PostProcessingSettings;
use crate::render_graph::compiled::RenderPassTemplate;
use crate::render_graph::context::RasterPassCtx;
use crate::render_graph::error::{RenderPassError, SetupError};
use crate::render_graph::pass::{PassBuilder, RasterPass};
use crate::render_graph::post_processing::{PostProcessEffect, PostProcessEffectId};
use crate::render_graph::resources::{TextureAccess, TextureHandle};

/// Graph handles for [`AcesTonemapPass`].
#[derive(Clone, Copy, Debug)]
pub struct AcesTonemapGraphResources {
    /// HDR scene-color input (the previous chain stage's output, or `scene_color_hdr` for the
    /// first effect in the chain).
    pub input: TextureHandle,
    /// HDR chain output written by this pass.
    pub output: TextureHandle,
}

/// Fullscreen render pass applying Stephen Hill ACES Fitted to `input`, writing `output`.
pub struct AcesTonemapPass {
    resources: AcesTonemapGraphResources,
    pipelines: &'static AcesTonemapPipelineCache,
}

impl AcesTonemapPass {
    /// Creates a new ACES tonemap pass instance.
    pub fn new(resources: AcesTonemapGraphResources) -> Self {
        Self {
            resources,
            pipelines: aces_tonemap_pipelines(),
        }
    }
}

/// Process-wide pipeline cache shared by every ACES pass instance.
fn aces_tonemap_pipelines() -> &'static AcesTonemapPipelineCache {
    static CACHE: OnceLock<AcesTonemapPipelineCache> = OnceLock::new();
    CACHE.get_or_init(AcesTonemapPipelineCache::default)
}

impl RasterPass for AcesTonemapPass {
    fn name(&self) -> &str {
        "AcesTonemap"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        b.read_texture_resource(
            self.resources.input,
            TextureAccess::Sampled {
                stages: wgpu::ShaderStages::FRAGMENT,
            },
        );
        let mut r = b.raster();
        r.color(
            self.resources.output,
            wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
            Option::<TextureHandle>::None,
        );
        Ok(())
    }

    fn multiview_mask_override(
        &self,
        ctx: &RasterPassCtx<'_, '_>,
        template: &RenderPassTemplate,
    ) -> Option<NonZeroU32> {
        let stereo = ctx
            .frame
            .as_ref()
            .is_some_and(|frame| frame.view.multiview_stereo);
        if stereo {
            NonZeroU32::new(3)
        } else {
            template.multiview_mask
        }
    }

    fn record(
        &self,
        ctx: &mut RasterPassCtx<'_, '_>,
        rpass: &mut wgpu::RenderPass<'_>,
    ) -> Result<(), RenderPassError> {
        profiling::scope!("post_processing::aces_tonemap");
        let Some(frame) = ctx.frame.as_ref() else {
            return Err(RenderPassError::MissingFrameParams {
                pass: self.name().to_string(),
            });
        };
        let Some(graph_resources) = ctx.graph_resources else {
            return Err(RenderPassError::MissingFrameParams {
                pass: self.name().to_string(),
            });
        };
        let Some(tex) = graph_resources.transient_texture(self.resources.input) else {
            return Err(RenderPassError::MissingFrameParams {
                pass: format!(
                    "{} (missing transient input {:?})",
                    self.name(),
                    self.resources.input
                ),
            });
        };
        let target_format = output_attachment_format(self.resources.output, graph_resources);
        let pipeline =
            self.pipelines
                .pipeline(ctx.device, target_format, frame.view.multiview_stereo);
        let bind_group =
            self.pipelines
                .bind_group(ctx.device, &tex.texture, frame.view.multiview_stereo);
        rpass.set_pipeline(pipeline.as_ref());
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.draw(0..3, 0..1);
        Ok(())
    }
}

/// Resolves the wgpu format the ACES color attachment is bound to this frame.
fn output_attachment_format(
    output: TextureHandle,
    graph_resources: &crate::render_graph::context::GraphResolvedResources,
) -> wgpu::TextureFormat {
    graph_resources
        .transient_texture(output)
        .map(|t| t.texture.format())
        .unwrap_or(wgpu::TextureFormat::Rgba16Float)
}

/// Effect descriptor that contributes an [`AcesTonemapPass`] to the post-processing chain.
pub struct AcesTonemapEffect;

impl PostProcessEffect for AcesTonemapEffect {
    fn id(&self) -> PostProcessEffectId {
        PostProcessEffectId::AcesTonemap
    }

    fn is_enabled(&self, settings: &PostProcessingSettings) -> bool {
        settings.enabled
            && matches!(
                settings.tonemap.mode,
                crate::config::TonemapMode::AcesFitted
            )
    }

    fn build_pass(&self, input: TextureHandle, output: TextureHandle) -> Box<dyn RasterPass> {
        Box::new(AcesTonemapPass::new(AcesTonemapGraphResources {
            input,
            output,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render_graph::pass::node::PassKind;
    use crate::render_graph::pass::PassBuilder;
    use crate::render_graph::resources::{
        AccessKind, TextureAccess, TransientArrayLayers, TransientExtent, TransientSampleCount,
        TransientTextureDesc, TransientTextureFormat,
    };
    use crate::render_graph::GraphBuilder;

    fn fake_textures(builder: &mut GraphBuilder) -> (TextureHandle, TextureHandle) {
        let desc = || TransientTextureDesc {
            label: "pp_hdr",
            format: TransientTextureFormat::SceneColorHdr,
            extent: TransientExtent::Backbuffer,
            mip_levels: 1,
            sample_count: TransientSampleCount::Fixed(1),
            dimension: wgpu::TextureDimension::D2,
            array_layers: TransientArrayLayers::Frame,
            base_usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING,
            alias: true,
        };
        (
            builder.create_texture(desc()),
            builder.create_texture(desc()),
        )
    }

    #[test]
    fn setup_declares_sampled_input_and_raster_output() {
        let mut builder = GraphBuilder::new();
        let (input, output) = fake_textures(&mut builder);
        let mut pass = AcesTonemapPass::new(AcesTonemapGraphResources { input, output });
        let mut b = PassBuilder::new("AcesTonemap");
        pass.setup(&mut b).expect("setup");
        let setup = b.finish().expect("finish");
        assert_eq!(setup.kind, PassKind::Raster);
        assert!(
            setup.accesses.iter().any(|a| matches!(
                &a.access,
                AccessKind::Texture(TextureAccess::Sampled {
                    stages: wgpu::ShaderStages::FRAGMENT,
                    ..
                })
            )),
            "expected sampled HDR input read"
        );
        assert_eq!(setup.color_attachments.len(), 1);
    }

    #[test]
    fn aces_tonemap_effect_id_label() {
        let e = AcesTonemapEffect;
        assert_eq!(e.id(), PostProcessEffectId::AcesTonemap);
    }
}
