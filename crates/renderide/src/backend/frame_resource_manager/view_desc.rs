//! View-specific inputs for resolving the light pack used by one render view.

use glam::Mat4;

use crate::camera::{HostCameraFrame, ViewId};
use crate::shared::RenderingContext;
use crate::world_mesh::WorldMeshCullProjParams;

/// Per-view culling inputs used to reject punctual light influence volumes before clustering.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FrameLightCullDesc {
    /// Camera frame that supplies the view transform and stereo state for culling.
    pub host_camera: HostCameraFrame,
    /// Projection bundle matching the target convention used by this view.
    pub proj: WorldMeshCullProjParams,
}

/// View-specific inputs for resolving the light pack used by one render view.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FrameLightViewDesc {
    /// Stable identity of the render view receiving this light pack.
    pub view_id: ViewId,
    /// Render context used by draw collection for this view.
    pub render_context: RenderingContext,
    /// Optional render-space scope for offscreen cameras/tasks.
    pub render_space_filter: Option<crate::scene::RenderSpaceId>,
    /// Head-output transform used when resolving overlay-space world matrices.
    pub head_output_transform: Mat4,
    /// Whether this view should pack shadow metadata for contributing lights.
    pub render_shadows: bool,
    /// Optional frustum inputs for light influence-volume culling.
    pub cull: Option<FrameLightCullDesc>,
}
