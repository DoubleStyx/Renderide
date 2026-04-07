//! Pass and subgraph identifiers for graph edges and scheduling.

use std::borrow::Cow;

/// Opaque identifier for a pass in the graph. Used for declaring edges.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PassId(pub(crate) usize);

/// Human-readable label for a subgraph node (e.g. `"main_view"`, `"reflection_probe"`).
///
/// Used for debugging and for namespaced pass names in tests (`label/pass_name`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SubgraphLabel(Cow<'static, str>);

impl SubgraphLabel {
    /// Returns the label as a string slice.
    pub fn as_str(&self) -> &str {
        self.0.as_ref()
    }
}

impl From<&'static str> for SubgraphLabel {
    fn from(s: &'static str) -> Self {
        Self(Cow::Borrowed(s))
    }
}

impl From<String> for SubgraphLabel {
    fn from(s: String) -> Self {
        Self(Cow::Owned(s))
    }
}

/// Opaque identifier returned by [`crate::render::pass::graph::GraphBuilder::add_subgraph`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SubgraphId(pub usize);

/// Endpoint for [`crate::render::pass::graph::GraphBuilder::add_edge`]: a root-level pass or a subgraph instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GraphNodeId {
    /// A pass added with [`crate::render::pass::graph::GraphBuilder::add_pass`].
    Pass(PassId),
    /// A subgraph added with [`crate::render::pass::graph::GraphBuilder::add_subgraph`].
    Subgraph(SubgraphId),
}

impl From<PassId> for GraphNodeId {
    fn from(value: PassId) -> Self {
        GraphNodeId::Pass(value)
    }
}

impl From<SubgraphId> for GraphNodeId {
    fn from(value: SubgraphId) -> Self {
        GraphNodeId::Subgraph(value)
    }
}
