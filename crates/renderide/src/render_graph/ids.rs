//! Stable identifiers for passes registered on a [`GraphBuilder`](super::GraphBuilder).
//!
//! Render graph v2 also assigns pass groups for frame-global and per-view scopes.

/// Opaque id returned by [`super::GraphBuilder::add_pass`] for dependency edges.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PassId(pub usize);

/// Opaque id returned by [`super::GraphBuilder::group`] for grouped scheduling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GroupId(pub usize);
