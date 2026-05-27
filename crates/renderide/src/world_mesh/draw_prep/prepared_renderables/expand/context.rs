//! Shared render-space expansion context.

use crate::gpu_pools::MeshPool;
use crate::scene::{RenderSpaceId, SceneCoordinator};
use crate::shared::RenderingContext;

use super::super::FramePreparedDraw;

/// Frame-time inputs that stay constant across all renderers in one render space.
pub(super) struct ExpandCtx<'a> {
    /// Destination prepared draw buffer.
    pub(super) out: &'a mut Vec<FramePreparedDraw>,
    /// Scene snapshot being expanded.
    pub(super) scene: &'a SceneCoordinator,
    /// Resident GPU mesh lookup table.
    pub(super) mesh_pool: &'a MeshPool,
    /// Render context used for transform and material override resolution.
    pub(super) render_context: RenderingContext,
    /// Render space being expanded.
    pub(super) space_id: RenderSpaceId,
    /// Whether `space_id` is an overlay space.
    pub(super) space_is_overlay: bool,
}

impl<'a> ExpandCtx<'a> {
    /// Reborrows the destination buffer while retaining the immutable frame inputs.
    #[cfg(test)]
    pub(super) fn reborrow(&mut self) -> ExpandCtx<'_> {
        ExpandCtx {
            out: self.out,
            scene: self.scene,
            mesh_pool: self.mesh_pool,
            render_context: self.render_context,
            space_id: self.space_id,
            space_is_overlay: self.space_is_overlay,
        }
    }
}
