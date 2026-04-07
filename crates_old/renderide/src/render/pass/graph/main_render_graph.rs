//! Default main-window render graph construction.

use crate::render::pass::clustered_light::ClusteredLightPass;
use crate::render::pass::composite::CompositePass;
use crate::render::pass::fullscreen_filter::FullscreenFilterPlaceholderPass;
use crate::render::pass::mesh_pass::MeshRenderPass;
use crate::render::pass::overlay_pass::OverlayRenderPass;
use crate::render::pass::rt_shadow_compute::RtShadowComputePass;
use crate::render::pass::rtao_blur::RtaoBlurPass;
use crate::render::pass::rtao_compute::RtaoComputePass;

use super::builder::GraphBuilder;
use super::resources::GraphBuildError;
use super::runtime::RenderGraph;

/// Builds the main window render graph for the RTAO MRT variant or the direct-to-surface variant.
///
/// When `rtao_mrt_graph` is true, inserts [`RtaoComputePass`], [`RtaoBlurPass`], and
/// [`CompositePass`] between mesh and overlay, and uses [`MeshRenderPass::with_rtao_mrt_graph`]
/// with `true`. When false, the mesh pass uses `false` and writes color and depth to the
/// surface and edges mesh directly to overlay; RTAO passes are omitted.
///
/// When `fullscreen_filter_hook` is true and `rtao_mrt_graph` is false, inserts
/// [`FullscreenFilterPlaceholderPass`] between mesh and overlay.
///
/// When `rt_shadow_compute` is true (requires `rtao_mrt_graph`), inserts [`RtShadowComputePass`]
/// after the mesh pass and before RTAO compute so the atlas is filled from the current frame’s
/// G-buffer; PBR samples that atlas on the **following** frame when atlas mode is active.
///
/// [`crate::render::pass::graph::RenderGraphContext::enable_rtao_mrt`] must be set to match this variant at execute time so
/// MRT textures are allocated only when the graph expects them.
pub fn build_main_render_graph(
    rtao_mrt_graph: bool,
    rt_shadow_compute: bool,
    fullscreen_filter_hook: bool,
) -> Result<RenderGraph, GraphBuildError> {
    let mut builder = GraphBuilder::new();
    let clustered = builder.add_pass(Box::new(ClusteredLightPass::new()));
    let mesh = builder.add_pass(Box::new(MeshRenderPass::with_rtao_mrt_graph(
        rtao_mrt_graph,
    )));
    let overlay = builder.add_pass(Box::new(OverlayRenderPass::new()));
    builder.add_edge(clustered, mesh);

    if rtao_mrt_graph {
        let after_mesh = if rt_shadow_compute {
            let rt_shadow = builder.add_pass(Box::new(RtShadowComputePass::new()));
            builder.add_edge(mesh, rt_shadow);
            rt_shadow
        } else {
            mesh
        };
        let rtao = builder.add_pass(Box::new(RtaoComputePass::new()));
        let rtao_blur = builder.add_pass(Box::new(RtaoBlurPass::new()));
        let composite = builder.add_pass(Box::new(CompositePass::new()));
        builder.add_edge(after_mesh, rtao);
        builder.add_edge(rtao, rtao_blur);
        builder.add_edge(rtao_blur, composite);
        builder.add_edge(composite, overlay);
        builder.build_with_special_passes(Some(composite), Some(overlay))
    } else if fullscreen_filter_hook {
        let filter = builder.add_pass(Box::new(FullscreenFilterPlaceholderPass::new()));
        builder.add_edge(mesh, filter);
        builder.add_edge(filter, overlay);
        builder.build_with_special_passes(None, Some(overlay))
    } else {
        builder.add_edge(mesh, overlay);
        builder.build_with_special_passes(None, Some(overlay))
    }
}
