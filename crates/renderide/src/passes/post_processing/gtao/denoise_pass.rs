//! `gtao_denoise` raster pass — XeGTAO bilateral filter, intermediate iteration.
//!
//! Reads the AO term and packed edges produced by [`super::main_pass::GtaoMainPass`], runs
//! the 3×3 edge-preserving kernel (`XeGTAO_Denoise` with `finalApply = false`), and writes
//! a denoised AO term to a ping-pong target. Only registered when
//! [`crate::config::GtaoSettings::denoise_passes`] is `>= 2`. Intermediate iterations use
//! `denoise_blur_beta / 5.0` so two iterations approximate the quality of a single soft
//! pass without over-smoothing silhouettes.

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

/// Handles for one [`GtaoDenoisePass`] invocation.
#[derive(Clone, Copy, Debug)]
pub(super) struct GtaoDenoiseResources {
    /// Source AO term (output of `gtao_main`).
    pub ao_in: TextureHandle,
    /// Packed-edges texture (always written by `gtao_main`).
    pub edges: TextureHandle,
    /// Destination AO term (ping-pong target).
    pub ao_out: TextureHandle,
}

/// Records the bilateral denoise fragment shader.
pub(super) struct GtaoDenoisePass {
    resources: GtaoDenoiseResources,
    settings: crate::config::GtaoSettings,
    pipelines: &'static GtaoPipelines,
}

impl GtaoDenoisePass {
    pub(super) fn new(
        resources: GtaoDenoiseResources,
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

impl RasterPass for GtaoDenoisePass {
    fn name(&self) -> &str {
        "GtaoDenoise"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        read_fragment_sampled_texture(b, self.resources.ao_in);
        read_fragment_sampled_texture(b, self.resources.edges);
        color_attachment(
            b,
            self.resources.ao_out,
            wgpu::LoadOp::Clear(wgpu::Color::WHITE),
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
        profiling::scope!("post_processing::gtao_denoise");
        let frame = &*ctx.pass_frame;
        let graph_resources = ctx.graph_resources;
        let Some(ao_in_tex) = graph_resources.transient_texture(self.resources.ao_in) else {
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

        let multiview_stereo = frame.view.multiview_stereo;
        let live = ctx
            .blackboard
            .get::<GtaoSettingsSlot>()
            .map_or(self.settings, |slot| slot.0);
        // XeGTAO splits the requested blur energy: intermediate uses `beta / 5`, final uses
        // `beta`. Two iterations at full beta would over-smooth.
        let params = GtaoParamsGpu {
            radius_world: live.radius_meters.max(0.0),
            max_pixel_radius: live.max_pixel_radius.max(1.0),
            intensity: live.intensity.max(0.0),
            step_count: live.step_count.max(1),
            falloff_range: live.falloff_range.clamp(0.05, 1.0),
            albedo_multibounce: live.albedo_multibounce.clamp(0.0, 0.99),
            denoise_blur_beta: live.denoise_blur_beta.max(0.0) / 5.0,
            final_apply: 0,
        };
        let params_buffer = self.pipelines.params.get(ctx.device);
        ctx.upload_batch
            .write_buffer(params_buffer, 0, bytemuck::bytes_of(&params));

        let pipeline = self
            .pipelines
            .denoise
            .pipeline(ctx.device, multiview_stereo);
        let bind_group = self.pipelines.denoise.bind_group(
            ctx.device,
            multiview_stereo,
            &ao_in_tex.texture,
            &edges_tex.texture,
            params_buffer,
        );
        rpass.set_pipeline(pipeline.as_ref());
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.draw(0..3, 0..1);
        Ok(())
    }
}
