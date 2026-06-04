//! View-specific inputs for resolving the light pack used by one render view.

use glam::Mat4;

use crate::camera::ViewId;
use crate::shared::RenderingContext;

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
}
