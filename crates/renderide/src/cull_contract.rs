//! Shared CPU culling contracts used by world-mesh preparation and Hi-Z occlusion.

use std::sync::Arc;

use glam::Mat4;
use hashbrown::HashMap;

use crate::scene::RenderSpaceId;

/// Projection matrices shared by all render spaces for a frame before multiplying per-space view.
#[derive(Clone, Copy, Debug)]
pub struct WorldMeshCullProjParams {
    /// Reverse-Z perspective for the main desktop / non-stereo path.
    pub world_proj: Mat4,
    /// Orthographic overlay projection used by overlay draws.
    pub overlay_proj: Mat4,
    /// OpenXR per-eye view-projection when VR is active.
    pub vr_stereo: Option<(Mat4, Mat4)>,
}

impl WorldMeshCullProjParams {
    /// Applies a projection-space transform to every projection matrix in this culling bundle.
    pub(crate) fn map_projection_matrices(
        &self,
        mut map_projection: impl FnMut(Mat4) -> Mat4,
    ) -> Self {
        Self {
            world_proj: map_projection(self.world_proj),
            overlay_proj: map_projection(self.overlay_proj),
            vr_stereo: self
                .vr_stereo
                .map(|(left, right)| (map_projection(left), map_projection(right))),
        }
    }
}

/// View and projection snapshot from the frame that produced the Hi-Z depth buffer.
///
/// The per-space view table is stored as [`Arc<HashMap<...>>`] so per-view clones are refcount
/// bumps rather than full hash table copies when secondary cameras fan out across workers.
#[derive(Clone, Debug)]
pub struct HiZTemporalState {
    /// Cull projection bundle from the depth-authoring frame.
    pub prev_cull: WorldMeshCullProjParams,
    /// World-to-camera view matrix per render space at that frame.
    pub prev_view_by_space: Arc<HashMap<RenderSpaceId, Mat4>>,
    /// Hi-Z mip0 size in texels after downscaling from the full depth attachment.
    pub depth_viewport_px: (u32, u32),
}
