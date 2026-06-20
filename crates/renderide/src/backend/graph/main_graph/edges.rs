//! Edge wiring for the main render graph.

use crate::render_graph::builder::GraphBuilder;
use crate::render_graph::ids::PassId;
use crate::render_graph::post_process_chain;

use super::passes::MainGraphPassIds;

fn connect_post_processing_edges(
    builder: &mut GraphBuilder,
    forward_tail: PassId,
    chain_output: post_process_chain::ChainOutput,
    compose: PassId,
) {
    if let Some((first_post, last_post)) = chain_output.pass_range() {
        builder.add_edge(forward_tail, first_post);
        builder.add_edge(last_post, compose);
    } else {
        builder.add_edge(forward_tail, compose);
    }
}

/// Adds every dependency edge between main render graph passes, optional GTAO normals, and the
/// post-processing chain leading into compose.
pub(super) fn add_main_graph_edges(
    builder: &mut GraphBuilder,
    passes: &MainGraphPassIds,
    chain_output: post_process_chain::ChainOutput,
    compose: PassId,
    desktop_overlay: PassId,
) {
    builder.add_edge(passes.light_cookies, passes.shadows);
    builder.add_edge(passes.deform, passes.shadows);
    builder.add_edge(passes.shadows, passes.clustered);
    builder.add_edge(passes.clustered, passes.depth_prepass);
    builder.add_edge(passes.depth_prepass, passes.forward_opaque);
    if let Some(gtao) = passes.gtao.as_ref() {
        let gtao_input = if let Some(pre_depth_resolve) = gtao.pre_depth_resolve {
            builder.add_edge(passes.forward_opaque, pre_depth_resolve);
            pre_depth_resolve
        } else {
            passes.forward_opaque
        };
        builder.add_edge(gtao_input, gtao.normal_pass);
        builder.add_edge(gtao.normal_pass, gtao.range.first);
        builder.add_edge(gtao.range.last, passes.depth_snapshot);
    } else {
        builder.add_edge(passes.forward_opaque, passes.depth_snapshot);
    }
    builder.add_edge(passes.depth_snapshot, passes.forward_intersect);
    let hiz_input = if let Some(occlusion_depth_resolve) = passes.occlusion_depth_resolve {
        builder.add_edge(passes.forward_intersect, occlusion_depth_resolve);
        occlusion_depth_resolve
    } else {
        passes.forward_intersect
    };
    builder.add_edge(hiz_input, passes.hiz);
    builder.add_edge(passes.hiz, passes.forward_transparent_sequence);
    let forward_tail = if let Some(depth_resolve) = passes.depth_resolve {
        builder.add_edge(passes.forward_transparent_sequence, depth_resolve);
        depth_resolve
    } else {
        passes.forward_transparent_sequence
    };
    connect_post_processing_edges(builder, forward_tail, chain_output, compose);
    builder.add_edge(compose, desktop_overlay);
}
