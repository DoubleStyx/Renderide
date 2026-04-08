//! [`RasterPipelineKind`]: which WGSL program backs mesh rasterization for a host shader route.

use std::sync::Arc;

/// Raster pipeline identity for mesh draws: one embedded WGSL target per shader, or the debug fallback.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RasterPipelineKind {
    /// Composed WGSL stem under `shaders/target/` (for example `ui_textunlit_default`), built at compile time.
    EmbeddedStem(Arc<str>),
    /// World-space normals when the host shader has no embedded target.
    DebugWorldNormals,
}
