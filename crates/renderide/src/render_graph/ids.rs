//! Stable identifiers for passes registered on a [`GraphBuilder`](super::GraphBuilder).
//!
//! Phase 2 may introduce subgraph or phase ids; v1 uses only [`PassId`].

/// Opaque id returned by [`super::GraphBuilder::add_pass`] for dependency edges.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PassId(pub usize);
