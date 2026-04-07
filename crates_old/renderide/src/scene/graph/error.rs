//! Error types for scene graph operations.

use thiserror::Error;

/// Error returned by scene graph operations.
#[derive(Debug, Error)]
pub enum SceneError {
    /// Shared memory access failed.
    #[error("Shared memory access: {0}")]
    SharedMemoryAccess(String),
    /// Cycle detected in transform hierarchy.
    #[error("Cycle detected in scene {scene_id} at transform {transform_id}")]
    CycleDetected { scene_id: i32, transform_id: i32 },
    /// Scene not found when one was expected.
    #[error("Scene {scene_id} not found")]
    SceneNotFound { scene_id: i32 },
}
