//! Render graph DAG: [`ResourceSlot`], [`GraphBuilder`], [`RenderGraph`], and [`RenderPass`].

mod builder;
mod context;
mod execute;
mod ids;
mod main_render_graph;
mod pass_trait;
mod resources;
mod runtime;
mod views;

#[cfg(test)]
mod tests;

pub use builder::GraphBuilder;
pub use context::{RenderGraphContext, RenderPassContext};
pub use ids::{GraphNodeId, PassId, SubgraphId, SubgraphLabel};
pub use main_render_graph::build_main_render_graph;
pub use pass_trait::RenderPass;
pub use resources::{GraphBuildError, PassResources, ResourceSlot};
pub use runtime::{ExecutionUnit, LabeledSubgraph, RenderGraph};
pub use views::{MrtViews, RenderTargetViews};
