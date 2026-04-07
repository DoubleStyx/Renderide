//! Errors from scene / hierarchy operations.

use thiserror::Error;

/// Failure applying host scene or transform data.
#[derive(Debug, Error)]
pub enum SceneError {
    /// Shared memory read for a transform batch failed.
    #[error("shared memory: {0}")]
    SharedMemoryAccess(String),
    /// Per-transform hierarchy cycle while computing world matrices.
    #[error("cycle in scene {scene_id} at transform {transform_id}")]
    CycleDetected {
        /// Host render space id.
        scene_id: i32,
        /// Dense transform index.
        transform_id: i32,
    },
    /// A scene was missing from the registry when required.
    #[error("scene {scene_id} not found")]
    SceneNotFound {
        /// Host render space id.
        scene_id: i32,
    },
}
