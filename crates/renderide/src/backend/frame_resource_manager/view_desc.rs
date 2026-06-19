//! View-specific inputs for resolving the light pack used by one render view.

use glam::Mat4;

use crate::camera::{HostCameraFrame, ViewId};
use crate::shared::RenderingContext;
use crate::world_mesh::{ViewLayerPolicy, ViewRenderSpaceScope, WorldMeshCullProjParams};

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
    /// Render-space scope for this view's light collection.
    pub render_space_scope: ViewRenderSpaceScope,
    /// Layer/private-space visibility policy matching this view's draw collection.
    pub layer_policy: ViewLayerPolicy,
    /// Whether this view has an explicit selective root list.
    pub has_selective_roots: bool,
    /// Head-output transform used when resolving overlay-space world matrices.
    pub head_output_transform: Mat4,
    /// Whether this view should pack shadow metadata for contributing lights.
    pub render_shadows: bool,
    /// Optional frustum inputs for light influence-volume culling.
    pub cull: Option<FrameLightCullDesc>,
}
