use std::borrow::Cow;

use crate::graph_inputs::FrameGlobalResourcePass;

/// Encoder pass label for diagnostics.
pub(crate) const LIGHT_COOKIE_ATLAS_PASS_NAME: &str = "light_cookie_atlas";

/// Main-graph frame-global pass that updates light-cookie atlas layers.
pub(crate) struct LightCookieAtlasPass;

impl LightCookieAtlasPass {
    /// Creates the light-cookie atlas update pass.
    pub(crate) const fn new() -> Self {
        Self
    }
}

impl crate::render_graph::pass::EncoderPass for LightCookieAtlasPass {
    fn name(&self) -> &str {
        LIGHT_COOKIE_ATLAS_PASS_NAME
    }

    fn profiling_label(&self) -> Cow<'_, str> {
        Cow::Borrowed("light_cookies::atlas")
    }

    fn setup(
        &mut self,
        builder: &mut crate::render_graph::pass::PassBuilder<'_>,
    ) -> Result<(), crate::render_graph::error::SetupError> {
        builder.encoder();
        builder.cull_exempt();
        builder.never_parallel();
        Ok(())
    }

    fn should_record(
        &self,
        ctx: &crate::render_graph::context::EncoderPassCtx<'_, '_, '_>,
    ) -> Result<bool, crate::render_graph::error::RenderPassError> {
        Ok(ctx
            .frame
            .systems
            .frame_resources
            .has_light_cookie_requests())
    }

    fn record(
        &self,
        ctx: &mut crate::render_graph::context::EncoderPassCtx<'_, '_, '_>,
    ) -> Result<(), crate::render_graph::error::RenderPassError> {
        ctx.frame.systems.frame_resources.encode_light_cookie_atlas(
            ctx.device,
            ctx.encoder,
            ctx.frame.systems.asset_resources,
            ctx.profiler,
        );
        Ok(())
    }

    fn phase(&self) -> crate::render_graph::pass::PassPhase {
        crate::render_graph::pass::PassPhase::FrameGlobal
    }

    fn frame_global_resource_pass(&self) -> Option<FrameGlobalResourcePass> {
        Some(FrameGlobalResourcePass::LightCookieAtlas)
    }
}
