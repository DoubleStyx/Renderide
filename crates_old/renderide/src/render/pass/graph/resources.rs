//! Resource slot declarations and graph build errors.

use super::ids::PassId;

/// Resource slot identifier for pass resource declarations.
///
/// Passes declare which slots they read and write; the graph can use this for
/// validation and scheduling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResourceSlot {
    /// Clustered shading cluster buffers (counts, indices).
    ClusterBuffers,
    /// Light buffer for clustered shading.
    LightBuffer,
    /// MRT color texture (mesh pass output, composite input).
    Color,
    /// MRT position G-buffer.
    Position,
    /// MRT normal G-buffer.
    Normal,
    /// Raw AO texture (RTAO compute output, blur input).
    AoRaw,
    /// Blurred AO texture (blur output, composite input).
    Ao,
    /// Final surface output.
    Surface,
    /// Depth buffer.
    Depth,
}

/// Declared reads and writes for a render pass.
#[derive(Clone, Debug, Default)]
pub struct PassResources {
    /// Resource slots this pass reads from.
    pub reads: Vec<ResourceSlot>,
    /// Resource slots this pass writes to.
    pub writes: Vec<ResourceSlot>,
}

/// Errors that can occur when building a render graph.
#[derive(Debug, thiserror::Error)]
pub enum GraphBuildError {
    /// The graph contains a cycle; topological sort is impossible.
    #[error("cycle detected in render graph")]
    CycleDetected,

    /// A pass reads a resource slot that no earlier pass produces.
    #[error("pass {pass:?} reads {slot:?} but no earlier pass writes it")]
    MissingDependency {
        /// Pass that requires the missing dependency.
        pass: PassId,
        /// Resource slot that has no producer.
        slot: ResourceSlot,
    },
}
