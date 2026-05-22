//! Material batch-key identity for world-mesh draw ordering and binding.

use crate::materials::{
    EmbeddedTangentFallbackMode, MaterialBlendMode, MaterialRenderState, RasterFrontFace,
    RasterPipelineKind, RasterPrimitiveTopology, SceneColorSnapshotMode,
    UNITY_RENDER_QUEUE_TRANSPARENT, UNITY_TRANSPARENT_RENDER_QUEUE_MIN,
};

use super::transparent::TransparentMaterialClass;

/// Groups draws that can share the same raster pipeline, material bind data, and Unity render-queue
/// ordering bucket (Unity material +
/// [`MaterialPropertyBlock`](https://docs.unity3d.com/ScriptReference/MaterialPropertyBlock.html)-style slot0).
#[derive(Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
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
    /// Primitive topology selected from the mesh's per-submesh
    /// [`crate::shared::SubmeshTopology`]. `wgpu` bakes
    /// [`wgpu::PrimitiveState::topology`] into the render pipeline, so two draws of the same
    /// shader/material that differ in topology must build separate pipelines.
    pub primitive_topology: RasterPrimitiveTopology,
    /// Whether the embedded stem needs a UV0 vertex stream for the active shader permutation.
    pub embedded_needs_uv0: bool,
    /// Whether the embedded stem needs a color vertex stream at `@location(3)`.
    pub embedded_needs_color: bool,
    /// Whether the embedded stem needs a UV1 vertex stream at `@location(5)`.
    pub embedded_needs_uv1: bool,
    /// Whether the embedded stem needs a tangent vertex stream at `@location(4)`.
    pub embedded_needs_tangent: bool,
    /// Tangent fallback policy for lazy tangent upload.
    pub embedded_tangent_fallback_mode: EmbeddedTangentFallbackMode,
    /// Whether the tangent stream carries raw shader payload instead of a geometric tangent.
    pub embedded_raw_tangent_payload: bool,
    /// Whether the normal stream carries raw shader payload instead of a lighting normal.
    pub embedded_raw_normal_payload: bool,
    /// Whether the embedded stem needs a UV2 vertex stream at `@location(6)`.
    pub embedded_needs_uv2: bool,
    /// Whether the embedded stem needs a UV3 vertex stream at `@location(7)`.
    pub embedded_needs_uv3: bool,
    /// Whether the embedded stem needs the packed UV0-UV7 stream.
    pub embedded_needs_wide_uvs: bool,
    /// Whether the embedded stem needs any stream outside UV0/color/UV1.
    pub embedded_needs_extended_vertex_streams: bool,
    /// Whether the material requires the intersection subpass with a depth snapshot.
    pub embedded_requires_intersection_pass: bool,
    /// Whether the shader samples the scene-depth snapshot through frame globals.
    pub embedded_uses_scene_depth_snapshot: bool,
    /// Whether the shader samples the scene-color snapshot through frame globals.
    pub embedded_uses_scene_color_snapshot: bool,
    /// How the shader expects scene-color snapshots to be refreshed.
    pub scene_color_snapshot_mode: SceneColorSnapshotMode,
    /// Effective Unity render queue after material override / fallback resolution.
    pub render_queue: i32,
    /// Runtime color, stencil, and depth state for this material/property-block pair.
    pub render_state: MaterialRenderState,
    /// Resolved material blend mode for pipeline selection and diagnostics.
    pub blend_mode: MaterialBlendMode,
    /// Transparent alpha-blended UI/text stems should preserve stable canvas order.
    pub alpha_blended: bool,
    /// Renderer-local transparent behavior class inferred from existing material and shader state.
    pub transparent_class: TransparentMaterialClass,
}

impl MaterialDrawBatchKey {
    #[inline]
    pub fn is_transparent(&self) -> bool {
        render_queue_is_transparent(self.render_queue, self.blend_mode.is_transparent())
    }
}

#[inline]
pub fn render_queue_is_transparent(render_queue: i32, transparent_blend_mode: bool) -> bool {
    render_queue >= UNITY_RENDER_QUEUE_TRANSPARENT
        || (transparent_blend_mode && render_queue >= UNITY_TRANSPARENT_RENDER_QUEUE_MIN)
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

#[cfg(test)]
mod tests {
    use super::{MaterialBlendMode, MaterialDrawBatchKey};
    use crate::materials::{UNITY_RENDER_QUEUE_TRANSPARENT, UNITY_TRANSPARENT_RENDER_QUEUE_MIN};

    #[test]
    fn transparent_sorting_starts_at_transparent_queue_for_opaque_blend() {
        let mut key = MaterialDrawBatchKey {
            render_queue: UNITY_RENDER_QUEUE_TRANSPARENT - 1,
            blend_mode: MaterialBlendMode::Opaque,
            ..Default::default()
        };
        assert!(!key.is_transparent());
        key.render_queue = UNITY_RENDER_QUEUE_TRANSPARENT;
        assert!(key.is_transparent());
    }

    #[test]
    fn transparent_sorting_starts_at_lower_transparent_queue_for_non_opaque_blend() {
        let mut key = MaterialDrawBatchKey {
            render_queue: UNITY_TRANSPARENT_RENDER_QUEUE_MIN - 1,
            blend_mode: MaterialBlendMode::UnityBlend { src: 5, dst: 10 },
            ..Default::default()
        };
        assert!(!key.is_transparent());
        key.render_queue = UNITY_TRANSPARENT_RENDER_QUEUE_MIN;
        assert!(key.is_transparent());
    }
}
