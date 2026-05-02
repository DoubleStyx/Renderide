//! Material batch-key identity for world-mesh draw ordering and binding.

use crate::materials::{
    MaterialBlendMode, MaterialRenderState, RasterFrontFace, RasterPipelineKind,
};

/// Groups draws that can share the same raster pipeline, material bind data, and Unity render-queue
/// ordering bucket (Unity material +
/// [`MaterialPropertyBlock`](https://docs.unity3d.com/ScriptReference/MaterialPropertyBlock.html)-style slot0).
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct MaterialDrawBatchKey {
    /// Resolved from host `set_shader` -> [`crate::materials::resolve_raster_pipeline`].
    pub pipeline: RasterPipelineKind,
    /// Host shader asset id from material `set_shader` (or `-1` when unknown).
    pub shader_asset_id: i32,
    /// Material asset id for this renderer material slot (or `-1` when missing).
    pub material_asset_id: i32,
    /// Per-slot property block id when present; `None` is distinct from `Some` for batching.
    pub property_block_slot0: Option<i32>,
    /// Skinned deform path uses different vertex buffers.
    pub skinned: bool,
    /// Front-face winding selected from the draw's model transform.
    pub front_face: RasterFrontFace,
    /// Whether the draw uses point-list rasterization.
    pub point_topology: bool,
    /// Whether the embedded stem needs a UV0 vertex stream for the active shader permutation.
    pub embedded_needs_uv0: bool,
    /// Whether the embedded stem needs a color vertex stream at `@location(3)`.
    pub embedded_needs_color: bool,
    /// Whether the embedded stem needs a UV1 vertex stream at `@location(5)` without the full extended stream set.
    pub embedded_needs_uv1: bool,
    /// Whether the embedded stem needs extra UI streams at `@location(4..=7)`.
    pub embedded_needs_extended_vertex_streams: bool,
    /// Whether the material requires the intersection subpass with a depth snapshot.
    pub embedded_requires_intersection_pass: bool,
    /// Whether the shader samples the scene-depth snapshot through frame globals.
    pub embedded_uses_scene_depth_snapshot: bool,
    /// Whether the shader samples the scene-color snapshot through frame globals.
    pub embedded_uses_scene_color_snapshot: bool,
    /// Effective Unity render queue after material override / fallback resolution.
    pub render_queue: i32,
    /// Runtime color, stencil, and depth state for this material/property-block pair.
    pub render_state: MaterialRenderState,
    /// Resolved material blend mode for pipeline selection and diagnostics.
    pub blend_mode: MaterialBlendMode,
    /// Transparent alpha-blended UI/text stems should preserve stable canvas order.
    pub alpha_blended: bool,
}

/// Computes a 64-bit content hash for `key` used by the draw-sort comparator's primary tiebreaker.
///
/// Uses [`ahash::AHasher`] so the hash is deterministic for a given build, fast in the hot
/// draw-prep loop, and avoids leaking `RandomState` salt through Rust's default `BuildHasher`.
#[inline]
pub fn compute_batch_key_hash(key: &MaterialDrawBatchKey) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = ahash::AHasher::default();
    key.hash(&mut h);
    h.finish()
}
