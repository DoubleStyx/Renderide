//! Canonical main render graph: imports, transient declarations, pass topology, and the default
//! post-processing chain.
//!
//! This module is the *application* of the render-graph framework: it wires the renderer's
//! built-in passes together to produce the frame the host expects (mesh deform -> clustered lights
//! -> forward -> Hi-Z -> post-processing -> HDR scene-color compose). The framework primitives it
//! consumes (builder, compiled graph, resources, post-processing chain) live in their respective
//! sibling modules.

mod default_chain;
mod edges;
mod gtao;
mod handles;
mod passes;

#[cfg(test)]
mod tests;

use crate::render_graph::GraphCacheKey;
use crate::render_graph::builder::GraphBuilder;
use crate::render_graph::compiled::CompiledRenderGraph;
use crate::render_graph::error::GraphBuildError;
use crate::render_graph::validation::RenderGraphValidationMode;

pub(crate) use handles::MainGraphPostProcessingResources;

use default_chain::build_default_post_processing_chain;
use edges::add_main_graph_edges;
use handles::{MainGraphHandles, import_main_graph_resources};
use passes::register_main_graph_passes;

/// Declares blackboard slots that graph preparation seeds before per-view pass recording.
fn seed_main_graph_blackboard(builder: &mut GraphBuilder) {
    builder.seed_blackboard::<crate::graph_inputs::PerViewFramePlanSlot>("per-view frame setup");
    builder.seed_blackboard::<crate::graph_inputs::MsaaViewsSlot>(
        "per-view graph resource resolution",
    );
    builder.seed_blackboard::<crate::passes::WorldMeshForwardPlanSlot>("world-mesh frame planning");
    builder.seed_blackboard::<crate::passes::post_processing::settings_slots::GtaoSettingsSlot>(
        "live post-processing settings",
    );
    builder.seed_blackboard::<crate::passes::post_processing::settings_slots::BloomSettingsSlot>(
        "live post-processing settings",
    );
    builder
        .seed_blackboard::<crate::passes::post_processing::settings_slots::MotionBlurSettingsSlot>(
            "live post-processing settings",
        );
    builder
        .seed_blackboard::<crate::passes::post_processing::settings_slots::AutoExposureSettingsSlot>(
            "live post-processing settings",
        );
}

/// Orchestrates registration of every pass, the default post-processing chain, the compose pass,
/// and dependency edge wiring before compiling the graph.
fn add_main_graph_passes_and_edges(
    mut builder: GraphBuilder,
    h: MainGraphHandles,
    post_processing_settings: &crate::config::PostProcessingSettings,
    post_processing_resources: &MainGraphPostProcessingResources,
    msaa_sample_count: u8,
    multiview_stereo: bool,
) -> Result<CompiledRenderGraph, GraphBuildError> {
    seed_main_graph_blackboard(&mut builder);
    let passes = register_main_graph_passes(
        &mut builder,
        &h,
        post_processing_settings,
        msaa_sample_count,
    );
    let chain = build_default_post_processing_chain(
        &h,
        post_processing_settings,
        multiview_stereo,
        post_processing_resources,
        passes.gtao_normals.as_ref().map(|node| node.view_normals),
    );
    let chain_output =
        chain.build_into_graph(&mut builder, h.scene_color_hdr, post_processing_settings);
    let compose_input = chain_output.final_handle();
    let compose = builder.add_raster_pass(Box::new(crate::passes::SceneColorComposePass::new(
        crate::passes::SceneColorComposeGraphResources {
            scene_color_hdr: h.scene_color_hdr,
            post_processed_scene_color_hdr: compose_input,
            frame_color: h.color,
        },
    )));
    add_main_graph_edges(&mut builder, &passes, chain_output, compose);
    builder.build()
}

/// Builds the main frame graph: mesh deform compute, clustered lights, world forward, Hi-Z readback,
/// then HDR scene-color compose into the display target.
///
/// Forward MSAA transients use [`crate::render_graph::resources::TransientExtent::Backbuffer`] and
/// [`crate::render_graph::resources::TransientSampleCount::Frame`] so sizes match the current view
/// at execute time. HDR scene color uses
/// [`crate::render_graph::resources::TransientTextureFormat::SceneColorHdr`]; the resolved format
/// follows [`crate::config::RenderingSettings::scene_color_format`] at execute time (see
/// [`GraphCacheKey::scene_color_format`] for graph-cache identity). `key` still drives graph-cache
/// identity ([`GraphCacheKey::surface_format`], [`GraphCacheKey::multiview_stereo`],
/// [`GraphCacheKey::msaa_sample_count`]). Imported sources resolve at execute time via
/// [`crate::backend::FrameResourceManager`].
#[cfg(test)]
pub fn build_main_graph(
    key: GraphCacheKey,
    post_processing_settings: &crate::config::PostProcessingSettings,
) -> Result<CompiledRenderGraph, GraphBuildError> {
    build_main_graph_with_resources(
        key,
        post_processing_settings,
        &MainGraphPostProcessingResources::default(),
        RenderGraphValidationMode::default(),
    )
}

/// Builds the main frame graph using caller-owned post-processing resources.
pub(crate) fn build_main_graph_with_resources(
    key: GraphCacheKey,
    post_processing_settings: &crate::config::PostProcessingSettings,
    post_processing_resources: &MainGraphPostProcessingResources,
    validation_mode: RenderGraphValidationMode,
) -> Result<CompiledRenderGraph, GraphBuildError> {
    logger::info!(
        "main render graph: scene color HDR format = {:?}, post-processing = {} effect(s)",
        key.scene_color_format,
        key.post_processing.active_count()
    );
    let mut builder = GraphBuilder::with_validation_mode(validation_mode);
    let handles = import_main_graph_resources(&mut builder);
    let msaa_handles = [handles.forward_msaa_depth, handles.forward_msaa_depth_r32];
    let mut graph = add_main_graph_passes_and_edges(
        builder,
        handles,
        post_processing_settings,
        post_processing_resources,
        key.msaa_sample_count,
        key.multiview_stereo,
    )?;
    graph.set_main_graph_msaa_transient_handles(msaa_handles);
    Ok(graph)
}
