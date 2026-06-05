//! Stable logical identities for renderer views.
//!
//! These types name views so view-scoped resources (temporal state, hierarchical-Z, history
//! textures, occlusion buffers) can be keyed independently of the GPU target they currently
//! render into. They are populated by per-tick view planning and consumed by the render graph
//! and pass orchestration.

use crate::scene::RenderSpaceId;

/// Stable logical identity for one secondary camera view.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SecondaryCameraId {
    /// Render space containing the camera.
    pub render_space_id: RenderSpaceId,
    /// Dense host camera renderable index within the render space.
    pub renderable_index: i32,
}

impl SecondaryCameraId {
    /// Builds a secondary-camera id from the host render-space and dense camera row.
    #[inline]
    pub const fn new(render_space_id: RenderSpaceId, renderable_index: i32) -> Self {
        Self {
            render_space_id,
            renderable_index,
        }
    }
}

/// Stable logical identity for one camera portal view.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CameraPortalId {
    /// Render space containing the camera portal.
    pub render_space_id: RenderSpaceId,
    /// Dense host camera-portal renderable index within the render space.
    pub renderable_index: i32,
}

impl CameraPortalId {
    /// Builds a camera-portal id from the host render-space and dense portal row.
    #[inline]
    pub const fn new(render_space_id: RenderSpaceId, renderable_index: i32) -> Self {
        Self {
            render_space_id,
            renderable_index,
        }
    }
}

/// Stable logical identity for one host camera readback task view.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CameraRenderTaskViewId {
    /// Render space requested by the host task.
    pub render_space_id: RenderSpaceId,
    /// Dense index within the drained host task batch.
    pub task_index: i32,
}

impl CameraRenderTaskViewId {
    /// Builds a camera readback view id from the host render-space and task batch index.
    #[inline]
    pub const fn new(render_space_id: RenderSpaceId, task_index: i32) -> Self {
        Self {
            render_space_id,
            task_index,
        }
    }
}

/// Stable logical identity for one Camera360 cubemap face.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Camera360RenderTaskFaceViewId {
    /// Render space requested by the host task.
    pub render_space_id: RenderSpaceId,
    /// Dense index within the drained host task batch.
    pub task_index: i32,
    /// Cubemap face index in host `BitmapCube` order.
    pub face_index: u8,
}

impl Camera360RenderTaskFaceViewId {
    /// Builds a Camera360 cubemap face view id.
    #[inline]
    pub const fn new(render_space_id: RenderSpaceId, task_index: i32, face_index: u8) -> Self {
        Self {
            render_space_id,
            task_index,
            face_index,
        }
    }
}

/// Stable logical identity for one reflection-probe cubemap bake face.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ReflectionProbeRenderTaskViewId {
    /// Render space requested by the host task.
    pub render_space_id: RenderSpaceId,
    /// Host reflection-probe bake task id.
    pub render_task_id: i32,
    /// Cubemap face index in host `BitmapCube` order.
    pub face_index: u8,
}

impl ReflectionProbeRenderTaskViewId {
    /// Builds a reflection-probe bake face view id.
    #[inline]
    pub const fn new(render_space_id: RenderSpaceId, render_task_id: i32, face_index: u8) -> Self {
        Self {
            render_space_id,
            render_task_id,
            face_index,
        }
    }
}

/// Identifies one logical render view for view-scoped resources and temporal state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ViewId {
    /// Main window or OpenXR multiview (shared primary-view state).
    Main,
    /// Desktop dashboard overlay camera pass for the main window.
    MainOverlay,
    /// Secondary camera, tracked independently from the render target asset it writes.
    SecondaryCamera(SecondaryCameraId),
    /// Camera portal, tracked independently from the render target asset it writes.
    CameraPortal(CameraPortalId),
    /// One-shot host camera readback task view.
    CameraRenderTask(CameraRenderTaskViewId),
    /// One-shot Camera360 cubemap bake face view.
    Camera360RenderTaskFace(Camera360RenderTaskFaceViewId),
    /// One-shot reflection-probe cubemap bake face view.
    ReflectionProbeRenderTask(ReflectionProbeRenderTaskViewId),
}

impl ViewId {
    /// Builds the stable logical identity for one secondary camera view.
    #[inline]
    pub const fn secondary_camera(render_space_id: RenderSpaceId, renderable_index: i32) -> Self {
        Self::SecondaryCamera(SecondaryCameraId::new(render_space_id, renderable_index))
    }

    /// Builds the stable logical identity for one camera portal view.
    #[inline]
    pub const fn camera_portal(render_space_id: RenderSpaceId, renderable_index: i32) -> Self {
        Self::CameraPortal(CameraPortalId::new(render_space_id, renderable_index))
    }

    /// Builds the stable logical identity for one camera readback task view.
    #[inline]
    pub const fn camera_render_task(render_space_id: RenderSpaceId, task_index: i32) -> Self {
        Self::CameraRenderTask(CameraRenderTaskViewId::new(render_space_id, task_index))
    }

    /// Builds the stable logical identity for one Camera360 cubemap bake face.
    #[inline]
    pub const fn camera360_render_task_face(
        render_space_id: RenderSpaceId,
        task_index: i32,
        face_index: u8,
    ) -> Self {
        Self::Camera360RenderTaskFace(Camera360RenderTaskFaceViewId::new(
            render_space_id,
            task_index,
            face_index,
        ))
    }

    /// Builds the stable logical identity for one reflection-probe bake face view.
    #[inline]
    pub const fn reflection_probe_render_task(
        render_space_id: RenderSpaceId,
        render_task_id: i32,
        face_index: u8,
    ) -> Self {
        Self::ReflectionProbeRenderTask(ReflectionProbeRenderTaskViewId::new(
            render_space_id,
            render_task_id,
            face_index,
        ))
    }

    /// Render space that owns this view, when the view is scoped to one host render space.
    pub const fn render_space_id(self) -> Option<RenderSpaceId> {
        match self {
            Self::Main | Self::MainOverlay => None,
            Self::SecondaryCamera(id) => Some(id.render_space_id),
            Self::CameraPortal(id) => Some(id.render_space_id),
            Self::CameraRenderTask(id) => Some(id.render_space_id),
            Self::Camera360RenderTaskFace(id) => Some(id.render_space_id),
            Self::ReflectionProbeRenderTask(id) => Some(id.render_space_id),
        }
    }
}
