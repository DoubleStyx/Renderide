//! `gtao_main` raster pass -- XeGTAO production stage.
//!
//! Reads the XeGTAO view-space depth mip chain plus the smooth view-space normal prepass,
//! evaluates the horizon search, and writes:
//!
//! - `@location(0)` -- `saturate(visibility / OCCLUSION_TERM_SCALE)` to an `R8Unorm` ping-pong
//!   target. The `1 / 1.5` scale is XeGTAO's `XeGTAO_OutputWorkingTerm` headroom convention;
//!   the apply stage multiplies by `OCCLUSION_TERM_SCALE` to recover the true visibility.
//! - `@location(1)` -- packed `LRTB` depth-edge weights (`gtao_pack_edges`) to an `R8Unorm`
//!   ping-pong target sampled by the depth-aware denoise / apply stages.

use std::num::NonZeroU32;

use super::pipeline::{GtaoParamsGpu, GtaoPipelines};
use crate::passes::helpers::{color_attachment, missing_pass_resource};
use crate::passes::post_processing::settings_slot::GtaoSettingsSlot;
use crate::render_graph::compiled::RenderPassTemplate;
use crate::render_graph::context::RasterPassCtx;
use crate::render_graph::error::{RenderPassError, SetupError};
use crate::render_graph::frame_params::PerViewFramePlanSlot;
use crate::render_graph::gpu_cache::raster_stereo_mask_override;
use crate::render_graph::pass::{PassBuilder, RasterPass};
use crate::render_graph::resources::{
    BufferAccess, ImportedBufferHandle, TextureAccess, TextureHandle,
};

/// Graph handles bound to one [`GtaoMainPass`] instance.
#[derive(Clone, Copy, Debug)]
pub(super) struct GtaoMainResources {
    /// View-space depth mip chain sampled by the AO horizon search.
    pub view_depth: TextureHandle,
    /// Smooth view-space normal texture produced by the forward normal prepass.
    pub view_normals: TextureHandle,
    /// Frame-uniforms buffer used as a fallback when the per-view buffer slot is absent.
    pub frame_uniforms: ImportedBufferHandle,
    /// Transient AO-term color attachment written by this pass (`@location(0)`).
    pub ao_term: TextureHandle,
    /// Transient packed-edges color attachment written by this pass (`@location(1)`).
    pub edges: TextureHandle,
}

/// Records the AO production fragment shader.
pub(super) struct GtaoMainPass {
    resources: GtaoMainResources,
    /// Live GTAO tunables captured at chain-build time and rewritten into the GPU UBO each
    /// record (settings_slot may also override per-frame).
    settings: crate::config::GtaoSettings,
    pipelines: &'static GtaoPipelines,
}

impl GtaoMainPass {
    pub(super) fn new(
        resources: GtaoMainResources,
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

impl RasterPass for GtaoMainPass {
    fn name(&self) -> &str {
        "GtaoMain"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        b.read_texture_resource(
            self.resources.view_depth,
            TextureAccess::Sampled {
                stages: wgpu::ShaderStages::FRAGMENT,
            },
        );
        b.read_texture_resource(
            self.resources.view_normals,
            TextureAccess::Sampled {
                stages: wgpu::ShaderStages::FRAGMENT,
            },
        );
        b.import_buffer(
            self.resources.frame_uniforms,
            BufferAccess::Uniform {
                stages: wgpu::ShaderStages::FRAGMENT,
                dynamic_offset: false,
            },
        );
        // Clear AO to white (full visibility) so any fragment that early-outs of `compute_gtao`
        // still leaves the scene unmodulated. Edges clear to zero (= no edges) since the apply
        // shader's no-denoise branch reads the raw AO without sampling edges.
        color_attachment(
            b,
            self.resources.ao_term,
            wgpu::LoadOp::Clear(wgpu::Color::WHITE),
        );
        color_attachment(
            b,
            self.resources.edges,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
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
        profiling::scope!("post_processing::gtao_main");
        let frame = &*ctx.pass_frame;
        let graph_resources = ctx.graph_resources;

        // Bind the per-view frame-uniforms buffer when the per-view plan is populated. The
        // imported `frame_uniforms` handle resolves to the shared FrameResourceManager buffer,
        // which is only written by the shared-frame path -- binding it under per-view rendering
        // would leave the shader reading zeros and producing NaN through `linearize_depth` /
        // `view_pos_from_uv`.
        let per_view_buffer = ctx
            .blackboard
            .get::<PerViewFramePlanSlot>()
            .map(|plan| plan.frame_uniform_buffer.clone());
        let frame_uniform_buffer = match per_view_buffer {
            Some(buf) => buf,
            None => match graph_resources.imported_buffer(self.resources.frame_uniforms) {
                Some(resolved) => resolved.buffer.clone(),
                None => {
                    return Err(missing_pass_resource(
                        self.name(),
                        "frame_uniforms not resolved",
                    ));
                }
            },
        };

        let multiview_stereo = frame.view.multiview_stereo;

        let live = ctx
            .blackboard
            .get::<GtaoSettingsSlot>()
            .map_or(self.settings, |slot| slot.0);
        // Production stage doesn't run the bilateral kernel; `denoise_blur_beta = 0` and
        // `final_apply = 0` keep the shared UBO unambiguous (the production shader doesn't
        // read either field but the apply / denoise shaders share the layout).
        let params = GtaoParamsGpu::from_settings(live, 0.0, false);
        let params_buffer = self.pipelines.params.get(ctx.device);
        ctx.upload_batch
            .write_buffer(params_buffer, 0, bytemuck::bytes_of(&params));

        let Some(view_depth_tex) = graph_resources.transient_texture(self.resources.view_depth)
        else {
            return Err(missing_pass_resource(
                self.name(),
                format_args!("missing view_depth {:?}", self.resources.view_depth),
            ));
        };
        let Some(view_normals_tex) = graph_resources.transient_texture(self.resources.view_normals)
        else {
            return Err(missing_pass_resource(
                self.name(),
                format_args!("missing view_normals {:?}", self.resources.view_normals),
            ));
        };

        let pipeline = self.pipelines.main.pipeline(ctx.device, multiview_stereo);
        let bind_group = self.pipelines.main.bind_group(
            ctx.device,
            multiview_stereo,
            &view_depth_tex.texture,
            &view_normals_tex.texture,
            &frame_uniform_buffer,
            params_buffer,
        );
        rpass.set_pipeline(pipeline.as_ref());
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.draw(0..3, 0..1);
        Ok(())
    }
}
