//! Material batch-key identity for world-mesh draw ordering and binding.

use crate::materials::{
    EmbeddedTangentFallbackMode, MaterialBlendMode, MaterialRenderState, RasterFrontFace,
    RasterPipelineKind, RasterPrimitiveTopology, SceneColorSnapshotMode,
    UNITY_TRANSPARENT_RENDER_QUEUE_MIN,
};

use super::transparent::TransparentMaterialClass;

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
    /// Whether the material declares intersection-depth behavior and needs the depth snapshot.
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
    /// Returns whether this draw should use transparent distance sorting within its render queue.
    #[inline]
    pub fn uses_transparent_sorting(&self) -> bool {
        render_queue_uses_transparent_sorting(self.render_queue)
    }

    /// Returns whether this draw belongs after the skybox/background split.
    #[inline]
    pub fn records_after_skybox(&self) -> bool {
        self.uses_transparent_sorting() || self.embedded_uses_scene_color_snapshot
    }

    /// Returns whether this draw needs strict order-sensitive submission within its phase.
    #[inline]
    pub fn requires_strict_order(&self) -> bool {
        self.uses_transparent_sorting() || self.embedded_uses_scene_color_snapshot
    }
}

/// Returns whether a render queue should use Unity transparent distance sorting.
#[inline]
pub(super) fn render_queue_uses_transparent_sorting(render_queue: i32) -> bool {
    render_queue >= UNITY_TRANSPARENT_RENDER_QUEUE_MIN
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
    use super::MaterialBlendMode;
    use crate::materials::UNITY_TRANSPARENT_RENDER_QUEUE_MIN;
    use crate::world_mesh::test_fixtures::{DummyDrawItemSpec, dummy_world_mesh_draw_item};

    fn key(alpha_blended: bool) -> crate::world_mesh::MaterialDrawBatchKey {
        dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 1,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 1,
            node_id: 0,
            slot_index: 0,
            collect_order: 0,
            alpha_blended,
        })
        .batch_key
    }

    #[test]
    fn transparent_sorting_starts_after_geometry_last_for_opaque_blend() {
        let mut key = key(false);
        key.render_queue = UNITY_TRANSPARENT_RENDER_QUEUE_MIN - 1;
        key.blend_mode = MaterialBlendMode::Opaque;

        assert!(!key.uses_transparent_sorting());
        assert!(!key.records_after_skybox());
        assert!(!key.requires_strict_order());

        key.render_queue = UNITY_TRANSPARENT_RENDER_QUEUE_MIN;
        assert!(key.uses_transparent_sorting());
        assert!(key.records_after_skybox());
        assert!(key.requires_strict_order());
    }

    #[test]
    fn transparent_sorting_starts_at_lower_transparent_queue_for_non_opaque_blend() {
        let mut key = key(false);
        key.render_queue = UNITY_TRANSPARENT_RENDER_QUEUE_MIN - 1;
        key.blend_mode = MaterialBlendMode::UnityBlend { src: 5, dst: 10 };

        assert!(!key.uses_transparent_sorting());
        assert!(!key.records_after_skybox());
        assert!(!key.requires_strict_order());

        key.render_queue = UNITY_TRANSPARENT_RENDER_QUEUE_MIN;
        assert!(key.uses_transparent_sorting());
        assert!(key.records_after_skybox());
        assert!(key.requires_strict_order());
    }

    #[test]
    fn low_queue_alpha_blend_stays_before_skybox() {
        let mut key = key(true);
        key.render_queue = UNITY_TRANSPARENT_RENDER_QUEUE_MIN - 1;
        key.blend_mode = MaterialBlendMode::StemDefault;

        assert!(!key.uses_transparent_sorting());
        assert!(!key.records_after_skybox());
        assert!(!key.requires_strict_order());
    }
}
