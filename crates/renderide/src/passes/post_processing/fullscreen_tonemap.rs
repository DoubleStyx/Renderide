//! Shared `RasterPass` scaffold for fullscreen blits that sample one HDR D2-array texture and
//! write a fullscreen triangle into a color attachment (ACES tonemap, AgX tonemap).
//!
//! Each tonemap pass owns its own pipeline cache (specialized via labels and shader sources) but
//! shares the same `setup` declarations and the same `record` recipe -- only the pass name, the
//! profiling scope, and the cache type differ. The free helpers here capture that common shape.

use std::sync::Arc;

use crate::passes::helpers::{
    color_attachment, missing_pass_resource, read_fragment_sampled_texture,
    transient_output_format_or,
};
use crate::render_graph::context::RasterPassCtx;
use crate::render_graph::error::{RenderPassError, SetupError};
use crate::render_graph::pass::PassBuilder;
use crate::render_graph::resources::TextureHandle;

/// Default fallback color-attachment format used when the transient texture has not been
/// resolved yet (setup-time inspection).
const DEFAULT_OUTPUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Declares the sampled input + cleared color attachment that every fullscreen-tonemap pass needs.
pub(super) fn setup_fullscreen_d2_array_pass(
    b: &mut PassBuilder<'_>,
    input: TextureHandle,
    output: TextureHandle,
) -> Result<(), SetupError> {
    read_fragment_sampled_texture(b, input);
    color_attachment(b, output, wgpu::LoadOp::Clear(wgpu::Color::BLACK));
    Ok(())
}

/// Records the shared fullscreen-triangle draw for a D2-array-sampled tonemap pass.
///
/// `pipeline_fn` and `bind_group_fn` close over the caller's pipeline cache and supply the cached
/// per-format pipeline and (texture, multiview) bind group respectively. The caller is responsible
/// for opening any `profiling::scope!` it wants attributed to its own module path.
pub(super) fn record_fullscreen_d2_array_blit(
    pass_name: &str,
    ctx: &RasterPassCtx<'_, '_>,
    rpass: &mut wgpu::RenderPass<'_>,
    input: TextureHandle,
    output: TextureHandle,
    pipeline_fn: impl FnOnce(&wgpu::Device, wgpu::TextureFormat, bool) -> Arc<wgpu::RenderPipeline>,
    bind_group_fn: impl FnOnce(&wgpu::Device, &wgpu::Texture, bool) -> wgpu::BindGroup,
) -> Result<(), RenderPassError> {
    let frame = &ctx.frame;
    let graph_resources = ctx.graph_resources;
    let Some(tex) = graph_resources.transient_texture(input) else {
        return Err(missing_pass_resource(
            pass_name,
            format_args!("missing transient input {input:?}"),
        ));
    };
    let target_format = transient_output_format_or(output, graph_resources, DEFAULT_OUTPUT_FORMAT);
    let multiview = frame.view.multiview_stereo;
    let pipeline = pipeline_fn(ctx.device, target_format, multiview);
    let bind_group = bind_group_fn(ctx.device, &tex.texture, multiview);
    rpass.set_pipeline(pipeline.as_ref());
    rpass.set_bind_group(0, &bind_group, &[]);
    rpass.draw(0..3, 0..1);
    Ok(())
}
