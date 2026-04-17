//! Declared reads and writes for each [`super::pass::RenderPass`].

use super::handles::ResourceId;

/// Declared reads and writes for a render pass.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PassResources {
    /// Logical resource handles this pass reads from.
    pub reads: Vec<ResourceId>,
    /// Logical resource handles this pass writes to.
    pub writes: Vec<ResourceId>,
}
