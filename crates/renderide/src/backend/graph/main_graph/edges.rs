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
) {
    builder.add_edge(passes.deform, passes.clustered);
    builder.add_edge(passes.clustered, passes.depth_prepass);
    builder.add_edge(passes.depth_prepass, passes.forward_opaque);
    if let Some(gtao_normals) = passes.gtao_normals.as_ref() {
        builder.add_edge(passes.forward_opaque, gtao_normals.pass);
        builder.add_edge(gtao_normals.pass, passes.depth_snapshot);
    } else {
        builder.add_edge(passes.forward_opaque, passes.depth_snapshot);
    }
    builder.add_edge(passes.depth_snapshot, passes.forward_intersect);
    builder.add_edge(
        passes.forward_intersect,
        passes.forward_transparent_sequence,
    );
    builder.add_edge(passes.forward_transparent_sequence, passes.depth_resolve);
    builder.add_edge(passes.depth_resolve, passes.hiz);
    connect_post_processing_edges(builder, passes.hiz, chain_output, compose);
}
