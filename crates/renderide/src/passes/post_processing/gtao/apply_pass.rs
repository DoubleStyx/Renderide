//! `gtao_apply` raster pass -- GTAO final iteration that composites AO into opaque color.
//!
//! Reads the AO term and packed edges (from `gtao_main` directly when `denoise_passes in {0, 1}`,
//! or from the last intermediate ping-pong target when `denoise_passes >= 2`), runs the bilateral
//! kernel at the full `denoise_blur_beta`, and writes a visibility factor. The pipeline uses
//! multiplicative destination-color blending, so the existing opaque HDR target is modulated in
//! place before transparent draws run. The shader short-circuits the kernel when
//! `denoise_blur_beta <= 0`, so `denoise_passes == 0` collapses to a "modulate by raw production
//! AO" path without re-binding a different pipeline.

use std::num::NonZeroU32;

use super::pipeline::{GtaoParamsGpu, GtaoPipelines};
use crate::passes::helpers::{
    color_attachment, missing_pass_resource, read_fragment_sampled_texture,
};
use crate::passes::post_processing::settings_slots::GtaoSettingsSlot;
use crate::render_graph::context::RasterPassCtx;
use crate::render_graph::error::{RenderPassError, SetupError};
use crate::render_graph::gpu_cache::raster_stereo_mask_override;
use crate::render_graph::pass::RenderPassTemplate;
use crate::render_graph::pass::{PassBuilder, RasterPass};
use crate::render_graph::resources::TextureHandle;

/// Handles for one [`GtaoOpaqueCompositePass`] invocation.
#[derive(Clone, Copy, Debug)]
pub(super) struct GtaoOpaqueCompositeResources {
    /// HDR scene-color target to modulate in place.
    pub target_color: TextureHandle,
    /// Whether [`Self::target_color`] is the multisampled forward target.
    pub target_is_msaa: bool,
    /// AO term sampled by the bilateral kernel.
    pub ao_in: TextureHandle,
    /// Packed-edges texture used for kernel weighting.
    pub edges: TextureHandle,
}

/// Records the final denoise + opaque composite fragment shader.
pub(super) struct GtaoOpaqueCompositePass {
    resources: GtaoOpaqueCompositeResources,
    settings: crate::config::GtaoSettings,
    pipelines: &'static GtaoPipelines,
}

impl GtaoOpaqueCompositePass {
    pub(super) fn new(
        resources: GtaoOpaqueCompositeResources,
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

impl RasterPass for GtaoOpaqueCompositePass {
    fn name(&self) -> &str {
        "GtaoOpaqueComposite"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        b.read_optional_blackboard::<crate::passes::WorldMeshForwardPlanSlot>();
        b.read_blackboard::<GtaoSettingsSlot>();
        read_fragment_sampled_texture(b, self.resources.ao_in);
        read_fragment_sampled_texture(b, self.resources.edges);
        color_attachment(b, self.resources.target_color, wgpu::LoadOp::Load);
        Ok(())
    }

    fn multiview_mask_override(
        &self,
        ctx: &RasterPassCtx<'_, '_>,
        template: &RenderPassTemplate,
    ) -> Option<NonZeroU32> {
        raster_stereo_mask_override(ctx, template)
    }

    fn should_record(&self, ctx: &RasterPassCtx<'_, '_>) -> Result<bool, RenderPassError> {
        Ok(super::gtao_view_recording_needed(
            ctx.blackboard,
            ctx.frame.view,
        ))
    }

    fn record(
        &self,
        ctx: &mut RasterPassCtx<'_, '_>,
        rpass: &mut wgpu::RenderPass<'_>,
    ) -> Result<(), RenderPassError> {
        profiling::scope!("post_processing::gtao_opaque_composite");
        let frame = &ctx.frame;
        let graph_resources = ctx.graph_resources;
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
        let Some(out_tex) = graph_resources.transient_texture(self.resources.target_color) else {
            return Err(missing_pass_resource(
                self.name(),
                format_args!("missing target_color {:?}", self.resources.target_color),
            ));
        };

        let multiview_stereo = frame.view.multiview_stereo;
        let output_format = out_tex.texture.format();
        let sample_count = if self.resources.target_is_msaa {
            frame.view.sample_count.max(1)
        } else {
            1
        };

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
        ctx.write_buffer(params_buffer, 0, bytemuck::bytes_of(&params));

        let pipeline = self.pipelines.apply.pipeline(
            ctx.device,
            output_format,
            sample_count,
            multiview_stereo,
        );
        let bind_group = self.pipelines.apply.bind_group(
            ctx.device,
            multiview_stereo,
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
