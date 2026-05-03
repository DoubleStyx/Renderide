//! Cache key types and per-instance need flags for the GPU skin cache.

use crate::scene::{MeshRendererInstanceId, RenderSpaceId};

/// Source renderer list for a deformable mesh instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SkinCacheRendererKind {
    /// Static mesh renderer table.
    Static,
    /// Skinned mesh renderer table.
    Skinned,
}

/// Stable key for a deformable mesh instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SkinCacheKey {
    /// Render space that owns the renderer.
    pub space_id: RenderSpaceId,
    /// Renderer table selected by this key.
    pub renderer_kind: SkinCacheRendererKind,
    /// Renderer-local identity that survives dense table reindexing.
    pub instance_id: MeshRendererInstanceId,
}

impl SkinCacheKey {
    /// Builds a skin-cache key from draw/deform identity fields.
    pub fn new(
        space_id: RenderSpaceId,
        renderer_kind: SkinCacheRendererKind,
        instance_id: MeshRendererInstanceId,
    ) -> Self {
        Self {
            space_id,
            renderer_kind,
            instance_id,
        }
    }

    /// Builds a skin-cache key from a draw's `skinned` flag.
    pub fn from_draw_parts(
        space_id: RenderSpaceId,
        skinned: bool,
        instance_id: MeshRendererInstanceId,
    ) -> Self {
        let renderer_kind = if skinned {
            SkinCacheRendererKind::Skinned
        } else {
            SkinCacheRendererKind::Static
        };
        Self::new(space_id, renderer_kind, instance_id)
    }
}

/// Whether blendshape and/or skinning compute runs for this instance (drives arena layout).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EntryNeed {
    /// Sparse blendshape scatter runs.
    pub needs_blend: bool,
    /// Linear blend skinning runs.
    pub needs_skin: bool,
    /// Blendshape scatter needs a normal output stream before final drawing or skinning.
    pub needs_blend_normals: bool,
    /// Final tangent output is needed for this deform entry.
    pub needs_tangents: bool,
    /// Blendshape scatter needs a tangent output stream before final drawing or skinning.
    pub needs_blend_tangents: bool,
}

impl EntryNeed {
    /// Returns whether a final normal range is required.
    #[inline]
    pub fn needs_normals(self) -> bool {
        self.needs_skin || self.needs_blend_normals
    }

    /// Returns whether blendshape output must feed skinning through a temp position range.
    #[inline]
    pub fn needs_temp_positions(self) -> bool {
        self.needs_blend && self.needs_skin
    }

    /// Returns whether blendshape normal deltas must feed skinning through a temp normal range.
    #[inline]
    pub fn needs_temp_normals(self) -> bool {
        self.needs_skin && self.needs_blend_normals
    }

    /// Returns whether blendshape tangent deltas must feed skinning through a temp tangent range.
    #[inline]
    pub fn needs_temp_tangents(self) -> bool {
        self.needs_skin && self.needs_tangents && self.needs_blend_tangents
    }
}

#[cfg(test)]
mod tests {
    //! CPU-only skin-cache key identity tests.

    use super::*;

    #[test]
    fn key_distinguishes_static_and_skinned_renderer_tables() {
        let instance_id = MeshRendererInstanceId(12);
        let static_key =
            SkinCacheKey::new(RenderSpaceId(7), SkinCacheRendererKind::Static, instance_id);
        let skinned_key = SkinCacheKey::new(
            RenderSpaceId(7),
            SkinCacheRendererKind::Skinned,
            instance_id,
        );

        assert_ne!(static_key, skinned_key);
    }

    #[test]
    fn key_distinguishes_two_renderers_on_the_same_transform_by_instance_id() {
        let first = SkinCacheKey::new(
            RenderSpaceId(7),
            SkinCacheRendererKind::Skinned,
            MeshRendererInstanceId(1),
        );
        let second = SkinCacheKey::new(
            RenderSpaceId(7),
            SkinCacheRendererKind::Skinned,
            MeshRendererInstanceId(2),
        );

        assert_ne!(first, second);
    }
}
