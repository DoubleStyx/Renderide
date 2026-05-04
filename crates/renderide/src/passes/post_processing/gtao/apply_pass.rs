//! `gtao_apply` raster pass -- XeGTAO final iteration that folds AO modulation into the
//! kernel.
//!
//! Reads the post-processing chain's HDR scene-color input plus the AO term and packed
//! edges (from `gtao_main` directly when `denoise_passes in {0, 1}`, or from the last
//! intermediate ping-pong target when `denoise_passes >= 2`), runs the bilateral kernel at the
//! full `denoise_blur_beta`, multiplies the resulting AO term by `OCCLUSION_TERM_SCALE` to
//! recover the true visibility (the production pass stored `visibility / 1.5` for kernel
//! headroom), then modulates HDR scene color and writes the chain's HDR output. The shader
//! short-circuits the kernel when `denoise_blur_beta <= 0`, so `denoise_passes == 0`
//! collapses to a "modulate by raw production AO" path without re-binding a different
//! pipeline.

use std::num::NonZeroU32;

use super::pipeline::{GtaoParamsGpu, GtaoPipelines};
use crate::passes::helpers::{
    color_attachment, missing_pass_resource, read_fragment_sampled_texture,
};
use crate::passes::post_processing::settings_slot::GtaoSettingsSlot;
use crate::render_graph::compiled::RenderPassTemplate;
use crate::render_graph::context::RasterPassCtx;
use crate::render_graph::error::{RenderPassError, SetupError};
use crate::render_graph::gpu_cache::raster_stereo_mask_override;
use crate::render_graph::pass::{PassBuilder, RasterPass};
use crate::render_graph::resources::TextureHandle;

/// Handles for one [`GtaoApplyPass`] invocation.
#[derive(Clone, Copy, Debug)]
pub(super) struct GtaoApplyResources {
    /// HDR scene-color input (post-processing chain's previous output, or the forward HDR
    /// target when GTAO is the chain's first effect).
    pub hdr_input: TextureHandle,
    /// AO term sampled by the bilateral kernel.
    pub ao_in: TextureHandle,
    /// Packed-edges texture used for kernel weighting.
    pub edges: TextureHandle,
    /// HDR output written by this pass.
    pub hdr_output: TextureHandle,
}

/// Records the final denoise + apply fragment shader.
pub(super) struct GtaoApplyPass {
    resources: GtaoApplyResources,
    settings: crate::config::GtaoSettings,
    pipelines: &'static GtaoPipelines,
}

impl GtaoApplyPass {
    pub(super) fn new(
        resources: GtaoApplyResources,
        settings: crate::config::GtaoSettings,
        pipelines: &'static GtaoPipelines,
    ) -> Self {
        Self {
            resources,
            settings,
            pipelines,
        }
    }
}

impl RasterPass for GtaoApplyPass {
    fn name(&self) -> &str {
        "GtaoApply"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        read_fragment_sampled_texture(b, self.resources.hdr_input);
        read_fragment_sampled_texture(b, self.resources.ao_in);
        read_fragment_sampled_texture(b, self.resources.edges);
        color_attachment(
            b,
            self.resources.hdr_output,
            wgpu::LoadOp::Clear(wgpu::Color::BLACK),
        );
        Ok(())
    }

    fn multiview_mask_override(
        &self,
        ctx: &RasterPassCtx<'_, '_>,
        template: &RenderPassTemplate,
    ) -> Option<NonZeroU32> {
        raster_stereo_mask_override(ctx, template)
    }

    fn record(
        &self,
        ctx: &mut RasterPassCtx<'_, '_>,
        rpass: &mut wgpu::RenderPass<'_>,
    ) -> Result<(), RenderPassError> {
        profiling::scope!("post_processing::gtao_apply");
        let frame = &*ctx.pass_frame;
        let graph_resources = ctx.graph_resources;
        let Some(hdr_in_tex) = graph_resources.transient_texture(self.resources.hdr_input) else {
            return Err(missing_pass_resource(
                self.name(),
                format_args!("missing hdr_input {:?}", self.resources.hdr_input),
            ));
        };
        let Some(ao_tex) = graph_resources.transient_texture(self.resources.ao_in) else {
            return Err(missing_pass_resource(
                self.name(),
                format_args!("missing ao_in {:?}", self.resources.ao_in),
            ));
        };
        let Some(edges_tex) = graph_resources.transient_texture(self.resources.edges) else {
            return Err(missing_pass_resource(
                self.name(),
                format_args!("missing edges {:?}", self.resources.edges),
            ));
        };
        let Some(out_tex) = graph_resources.transient_texture(self.resources.hdr_output) else {
            return Err(missing_pass_resource(
                self.name(),
                format_args!("missing hdr_output {:?}", self.resources.hdr_output),
            ));
        };

        let multiview_stereo = frame.view.multiview_stereo;
        let output_format = out_tex.texture.format();

        let live = ctx
            .blackboard
            .get::<GtaoSettingsSlot>()
            .map_or(self.settings, |slot| slot.0);
        // `denoise_passes == 0` zeroes the kernel and collapses the apply shader to a raw AO
        // multiply (still scaled-up via OCCLUSION_TERM_SCALE in-shader); otherwise the apply
        // pass uses the full configured blur strength.
        let beta = if live.denoise_passes == 0 {
            0.0
        } else {
            live.denoise_blur_beta.max(0.0)
        };
        let params = GtaoParamsGpu::from_settings(live, beta, true);
        let params_buffer = self.pipelines.params.get(ctx.device);
        ctx.upload_batch
            .write_buffer(params_buffer, 0, bytemuck::bytes_of(&params));

        let pipeline = self
            .pipelines
            .apply
            .pipeline(ctx.device, output_format, multiview_stereo);
        let bind_group = self.pipelines.apply.bind_group(
            ctx.device,
            multiview_stereo,
            &hdr_in_tex.texture,
            &ao_tex.texture,
            &edges_tex.texture,
            params_buffer,
        );
        rpass.set_pipeline(pipeline.as_ref());
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.draw(0..3, 0..1);
        Ok(())
    }
}
