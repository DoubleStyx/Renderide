use std::borrow::Cow;

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
            .pass_frame
            .shared
            .frame_resources
            .has_light_cookie_requests())
    }

    fn record(
        &self,
        ctx: &mut crate::render_graph::context::EncoderPassCtx<'_, '_, '_>,
    ) -> Result<(), crate::render_graph::error::RenderPassError> {
        ctx.pass_frame
            .shared
            .frame_resources
            .encode_light_cookie_atlas(
                ctx.device,
                ctx.encoder,
                ctx.pass_frame.shared.asset_resources,
                ctx.profiler,
            );
        Ok(())
    }

    fn phase(&self) -> crate::render_graph::pass::PassPhase {
        crate::render_graph::pass::PassPhase::FrameGlobal
    }
}
