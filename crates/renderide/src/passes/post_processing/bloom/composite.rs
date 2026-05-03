//! Bloom composite pass: samples the chain input (scene HDR) + bloom mip 0 and writes the
//! chain's output texture with the configured composite math in-shader.

use std::num::NonZeroU32;

use super::helpers::{attachment_format, stereo_mask_override};
use super::pipeline::{BloomPipelineCache, BloomPipelineKind};
use crate::passes::helpers::{
    color_attachment, missing_pass_resource, read_fragment_sampled_texture,
};
use crate::render_graph::compiled::RenderPassTemplate;
use crate::render_graph::context::RasterPassCtx;
use crate::render_graph::error::{RenderPassError, SetupError};
use crate::render_graph::pass::{PassBuilder, RasterPass};
use crate::render_graph::resources::TextureHandle;

/// Reads the chain HDR input at full resolution, samples bloom mip 0 (the terminating rung of the
/// upsample ladder), combines them with the configured composite math in-shader (no blend state
/// needed), and writes the chain output. This is the tail pass of the bloom subgraph -- the one
/// whose `PassId` gets reported as `EffectPasses::last`.
pub(super) struct BloomCompositePass {
    scene_input: TextureHandle,
    bloom_mip0: TextureHandle,
    output: TextureHandle,
    pipelines: &'static BloomPipelineCache,
}

impl BloomCompositePass {
    pub(super) fn new(
        scene_input: TextureHandle,
        bloom_mip0: TextureHandle,
        output: TextureHandle,
        pipelines: &'static BloomPipelineCache,
    ) -> Self {
        Self {
            scene_input,
            bloom_mip0,
            output,
            pipelines,
        }
    }
}

impl RasterPass for BloomCompositePass {
    fn name(&self) -> &str {
        "BloomComposite"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        read_fragment_sampled_texture(b, self.scene_input);
        read_fragment_sampled_texture(b, self.bloom_mip0);
        color_attachment(
            b,
            self.output,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );
        Ok(())
    }

    fn multiview_mask_override(
        &self,
        ctx: &RasterPassCtx<'_, '_>,
        template: &RenderPassTemplate,
    ) -> Option<NonZeroU32> {
        stereo_mask_override(ctx, template)
    }

    fn record(
        &self,
        ctx: &mut RasterPassCtx<'_, '_>,
        rpass: &mut wgpu::RenderPass<'_>,
    ) -> Result<(), RenderPassError> {
        profiling::scope!("post_processing::bloom::composite");
        let frame = &*ctx.pass_frame;
        let graph_resources = ctx.graph_resources;
        let Some(scene_tex) = graph_resources.transient_texture(self.scene_input) else {
            return Err(missing_pass_resource(self.name(), "missing scene input"));
        };
        let Some(bloom_tex) = graph_resources.transient_texture(self.bloom_mip0) else {
            return Err(missing_pass_resource(self.name(), "missing bloom mip 0"));
        };
        let multiview_stereo = frame.view.multiview_stereo;
        let output_format = attachment_format(graph_resources, self.output);

        let pipeline = self.pipelines.pipeline(
            ctx.device,
            BloomPipelineKind::Composite,
            output_format,
            multiview_stereo,
        );
        let bg_0 =
            self.pipelines
                .group0_bind_group(ctx.device, &scene_tex.texture, multiview_stereo);
        let bg_1 =
            self.pipelines
                .group1_bind_group(ctx.device, &bloom_tex.texture, multiview_stereo);
        rpass.set_pipeline(pipeline.as_ref());
        rpass.set_bind_group(0, &bg_0, &[]);
        rpass.set_bind_group(1, &bg_1, &[]);
        rpass.draw(0..3, 0..1);
        Ok(())
    }
}
