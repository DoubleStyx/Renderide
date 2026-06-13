//! [`RasterPipelineKind`]: which WGSL program backs mesh rasterization for a host shader route.

use std::sync::Arc;

/// Raster pipeline identity for mesh draws: one embedded WGSL target per shader, or the null fallback.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RasterPipelineKind {
    /// Composed WGSL package stem (for example `ui_textunlit_default`).
    EmbeddedStem(Arc<str>),
    /// Object-space black/grey checkerboard fallback when the host shader has no embedded target.
    Null,
}
