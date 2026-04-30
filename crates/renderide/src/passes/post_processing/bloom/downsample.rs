//! First and subsequent bloom downsample passes.

use std::borrow::Cow;
use std::num::NonZeroU32;

use super::helpers::{attachment_format, stereo_mask_override};
use super::pipeline::{BloomParamsGpu, BloomPipelineCache, BloomPipelineKind};
use crate::config::BloomSettings;
use crate::passes::helpers::{
    color_attachment, missing_frame_params, missing_pass_resource, read_fragment_sampled_texture,
};
use crate::passes::post_processing::settings_slot::BloomSettingsSlot;
use crate::render_graph::compiled::RenderPassTemplate;
use crate::render_graph::context::RasterPassCtx;
use crate::render_graph::error::{RenderPassError, SetupError};
use crate::render_graph::pass::{PassBuilder, RasterPass};
use crate::render_graph::resources::TextureHandle;

/// First downsample: reads the chain's HDR input, applies Karis firefly reduction (and the
/// optional soft-knee prefilter), writes bloom mip 0. Owns the per-frame params UBO upload so
/// every other bloom pass can just bind the already-written buffer.
///
/// Reads [`BloomSettingsSlot`] from the per-view blackboard at record time, so slider edits on
/// non-topology knobs (intensity, threshold, composite mode, etc.) reach the shader without a
/// graph rebuild. `fallback_settings` is used when the blackboard isn't populated (tests / pre-
/// lifecycle paths).
pub(super) struct BloomDownsampleFirstPass {
    /// HDR source texture for the bloom chain.
    input: TextureHandle,
    /// First bloom pyramid mip written by the pass.
    output: TextureHandle,
    /// Settings snapshot used when the live blackboard slot is absent.
    fallback_settings: BloomSettings,
    /// Shared pipeline and bind-group cache for all bloom passes.
    pipelines: &'static BloomPipelineCache,
}

impl BloomDownsampleFirstPass {
    pub(super) fn new(
        input: TextureHandle,
        output: TextureHandle,
        fallback_settings: BloomSettings,
        pipelines: &'static BloomPipelineCache,
    ) -> Self {
        Self {
            input,
            output,
            fallback_settings,
            pipelines,
        }
    }
}

impl RasterPass for BloomDownsampleFirstPass {
    fn name(&self) -> &str {
        "BloomDownsampleFirst"
    }

    fn profiling_label(&self) -> Cow<'_, str> {
        Cow::Borrowed("BloomDownsampleFirst.mip0")
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        read_fragment_sampled_texture(b, self.input);
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
        profiling::scope!("post_processing::bloom::downsample_first");
        let Some(frame) = ctx.frame.as_ref() else {
            return Err(missing_frame_params(self.name()));
        };
        let Some(graph_resources) = ctx.graph_resources else {
            return Err(missing_frame_params(self.name()));
        };
        let Some(input_tex) = graph_resources.transient_texture(self.input) else {
            return Err(missing_pass_resource(
                self.name(),
                "missing transient input",
            ));
        };
        let multiview_stereo = frame.view.multiview_stereo;
        let output_format = attachment_format(graph_resources, self.output);

        // Upload the shared bloom params UBO once per frame via the deferred upload batch
        // (single-producer queue invariant — see `crate::passes::post_processing::gtao`
        // for the equivalent pattern). Params are built from the live blackboard slot so slider
        // edits propagate without rebuilding the graph.
        let settings = ctx
            .blackboard
            .get::<BloomSettingsSlot>()
            .map_or(self.fallback_settings, |slot| slot.0);
        let params = BloomParamsGpu::from_settings(&settings);
        let params_buffer = self.pipelines.params_buffer(ctx.device);
        ctx.upload_batch
            .write_buffer(params_buffer, 0, bytemuck::bytes_of(&params));

        let pipeline = self.pipelines.pipeline(
            ctx.device,
            BloomPipelineKind::DownsampleFirst,
            output_format,
            multiview_stereo,
        );
        let bind_group =
            self.pipelines
                .group0_bind_group(ctx.device, &input_tex.texture, multiview_stereo);
        rpass.set_pipeline(pipeline.as_ref());
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.draw(0..3, 0..1);
        Ok(())
    }
}

/// Plain 13-tap downsample between bloom mips (N-1 → N). No firefly reduction, no threshold —
/// the first pass already absorbed those costs. Shares pipelines and bind groups with the first
/// downsample via [`BloomPipelineCache`].
pub(super) struct BloomDownsamplePass {
    /// Source bloom pyramid mip.
    input: TextureHandle,
    /// Destination bloom pyramid mip.
    output: TextureHandle,
    /// Per-instance profiler label including the destination mip index.
    profile_label: String,
    /// Shared pipeline and bind-group cache for all bloom passes.
    pipelines: &'static BloomPipelineCache,
}

impl BloomDownsamplePass {
    pub(super) fn new(
        input: TextureHandle,
        output: TextureHandle,
        mip: u32,
        pipelines: &'static BloomPipelineCache,
    ) -> Self {
        Self {
            input,
            output,
            profile_label: format!("BloomDownsample.mip{mip}"),
            pipelines,
        }
    }
}

impl RasterPass for BloomDownsamplePass {
    fn name(&self) -> &str {
        "BloomDownsample"
    }

    fn profiling_label(&self) -> Cow<'_, str> {
        Cow::Borrowed(self.profile_label.as_str())
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        read_fragment_sampled_texture(b, self.input);
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
        profiling::scope!("post_processing::bloom::downsample");
        let Some(frame) = ctx.frame.as_ref() else {
            return Err(missing_frame_params(self.name()));
        };
        let Some(graph_resources) = ctx.graph_resources else {
            return Err(missing_frame_params(self.name()));
        };
        let Some(input_tex) = graph_resources.transient_texture(self.input) else {
            return Err(missing_pass_resource(
                self.name(),
                "missing transient input",
            ));
        };
        let multiview_stereo = frame.view.multiview_stereo;
        let output_format = attachment_format(graph_resources, self.output);

        let pipeline = self.pipelines.pipeline(
            ctx.device,
            BloomPipelineKind::Downsample,
            output_format,
            multiview_stereo,
        );
        let bind_group =
            self.pipelines
                .group0_bind_group(ctx.device, &input_tex.texture, multiview_stereo);
        rpass.set_pipeline(pipeline.as_ref());
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.draw(0..3, 0..1);
        Ok(())
    }
}
