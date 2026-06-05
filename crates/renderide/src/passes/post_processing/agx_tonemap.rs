//! Analytic AgX tonemap render pass.
//!
//! Reads an HDR scene-color array texture, applies the analytic AgX transform, and writes a chain
//! HDR transient that the next post pass (or [`crate::passes::SceneColorComposePass`]) consumes.
//! Output is in `[0, 1]` linear sRGB so the existing sRGB swapchain encodes gamma correctly
//! without a separate gamma pass.

mod pipeline;

use std::num::NonZeroU32;
use std::sync::LazyLock;

use pipeline::AgxTonemapPipelineCache;

use crate::config::PostProcessingSettings;
use crate::render_graph::builder::GraphBuilder;
use crate::render_graph::context::RasterPassCtx;
use crate::render_graph::error::{RenderPassError, SetupError};
use crate::render_graph::gpu_cache::raster_stereo_mask_override;
use crate::render_graph::pass::RenderPassTemplate;
use crate::render_graph::pass::{PassBuilder, RasterPass};
use crate::render_graph::post_process_chain::{
    EffectPasses, PostProcessEffect, PostProcessEffectId,
};
use crate::render_graph::resources::TextureHandle;

use super::fullscreen_tonemap::{record_fullscreen_d2_array_blit, setup_fullscreen_d2_array_pass};

/// Graph handles for [`AgxTonemapPass`].
#[derive(Clone, Copy, Debug)]
pub struct AgxTonemapGraphResources {
    /// HDR scene-color input (the previous chain stage's output, or `scene_color_hdr` for the
    /// first effect in the chain).
    pub input: TextureHandle,
    /// HDR chain output written by this pass.
    pub output: TextureHandle,
}

/// Fullscreen render pass applying analytic AgX to `input`, writing `output`.
pub struct AgxTonemapPass {
    resources: AgxTonemapGraphResources,
    pipelines: &'static AgxTonemapPipelineCache,
}

impl AgxTonemapPass {
    /// Creates a new AgX tonemap pass instance.
    pub fn new(resources: AgxTonemapGraphResources) -> Self {
        Self {
            resources,
            pipelines: agx_tonemap_pipelines(),
        }
    }
}

/// Process-wide pipeline cache shared by every AgX pass instance.
fn agx_tonemap_pipelines() -> &'static AgxTonemapPipelineCache {
    static CACHE: LazyLock<AgxTonemapPipelineCache> =
        LazyLock::new(AgxTonemapPipelineCache::default);
    &CACHE
}

impl RasterPass for AgxTonemapPass {
    fn name(&self) -> &str {
        "AgxTonemap"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        setup_fullscreen_d2_array_pass(b, self.resources.input, self.resources.output)
    }

    fn multiview_mask_override(
        &self,
        ctx: &RasterPassCtx<'_, '_>,
        template: &RenderPassTemplate,
    ) -> Option<NonZeroU32> {
        raster_stereo_mask_override(ctx, template)
    }

    fn should_record(&self, ctx: &RasterPassCtx<'_, '_>) -> Result<bool, RenderPassError> {
        Ok(super::view_post_processing_enabled(ctx.frame.view))
    }

    fn record(
        &self,
        ctx: &mut RasterPassCtx<'_, '_>,
        rpass: &mut wgpu::RenderPass<'_>,
    ) -> Result<(), RenderPassError> {
        profiling::scope!("post_processing::agx_tonemap");
        record_fullscreen_d2_array_blit(
            self.name(),
            ctx,
            rpass,
            self.resources.input,
            self.resources.output,
            |device, fmt, mv| self.pipelines.pipeline(device, fmt, mv),
            |device, tex, mv| self.pipelines.bind_group(device, tex, mv),
        )
    }
}

/// Effect descriptor that contributes an [`AgxTonemapPass`] to the post-processing chain.
pub struct AgxTonemapEffect;

impl PostProcessEffect for AgxTonemapEffect {
    fn id(&self) -> PostProcessEffectId {
        PostProcessEffectId::AgxTonemap
    }

    fn is_enabled(&self, settings: &PostProcessingSettings) -> bool {
        settings.enabled && matches!(settings.tonemap.mode, crate::config::TonemapMode::AgX)
    }

    fn register(
        &self,
        builder: &mut GraphBuilder,
        _settings: &PostProcessingSettings,
        input: TextureHandle,
        output: TextureHandle,
    ) -> EffectPasses {
        let pass_id =
            builder.add_raster_pass(Box::new(AgxTonemapPass::new(AgxTonemapGraphResources {
                input,
                output,
            })));
        EffectPasses::single(pass_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render_graph::builder::GraphBuilder;
    use crate::render_graph::pass::PassBuilder;
    use crate::render_graph::pass::node::PassKind;
    use crate::render_graph::resources::{
        AccessKind, TextureAccess, TransientArrayLayers, TransientExtent, TransientSampleCount,
        TransientTextureDesc, TransientTextureFormat,
    };

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
        let mut pass = AgxTonemapPass::new(AgxTonemapGraphResources { input, output });
        let mut b = PassBuilder::new("AgxTonemap");
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
    fn agx_tonemap_effect_id_label() {
        let e = AgxTonemapEffect;
        assert_eq!(e.id(), PostProcessEffectId::AgxTonemap);
    }
}
