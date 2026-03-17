//! Error types for scene graph operations.

/// Error returned by scene graph operations.
#[derive(Debug)]
pub enum SceneError {
    /// Shared memory access failed.
    SharedMemoryAccess(String),
    /// Cycle detected in transform hierarchy.
    CycleDetected { scene_id: i32, transform_id: i32 },
}

impl std::fmt::Display for SceneError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SceneError::SharedMemoryAccess(msg) => write!(f, "Shared memory access: {}", msg),
            SceneError::CycleDetected {
                scene_id,
                transform_id,
            } => {
                write!(
                    f,
                    "Cycle detected in scene {} at transform {}",
                    scene_id, transform_id
                )
            }
        }
    }
}

impl std::error::Error for SceneError {}
