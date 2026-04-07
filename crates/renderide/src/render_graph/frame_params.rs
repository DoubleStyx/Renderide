//! Per-frame parameters shared across render graph passes (scene, backend, surface state).

use glam::Mat4;

use crate::backend::RenderBackend;
use crate::scene::SceneCoordinator;
use crate::shared::HeadOutputDevice;

/// Latest camera-related fields from host [`crate::shared::FrameSubmitData`], updated each `frame_submit`.
#[derive(Clone, Copy, Debug)]
pub struct HostCameraFrame {
    /// Host lock-step frame index (`-1` before the first submit in standalone).
    pub frame_index: i32,
    /// Near clip distance from the host frame submission.
    pub near_clip: f32,
    /// Far clip distance from the host frame submission.
    pub far_clip: f32,
    /// Vertical field of view in **degrees** (matches host `desktopFOV`).
    pub desktop_fov_degrees: f32,
    /// Whether the host reported VR output as active for this frame.
    pub vr_active: bool,
    /// Init-time head output device selected by the host.
    pub output_device: HeadOutputDevice,
    /// `(orthographic_half_height, near, far)` from the first [`crate::shared::CameraRenderTask`] whose
    /// parameters use orthographic projection (overlay main-camera ortho override).
    pub primary_ortho_task: Option<(f32, f32, f32)>,
    /// When [`Self::vr_active`] and OpenXR supplies views, per-eye view–projection (reverse-Z), mapping
    /// **stage** space to clip. World mesh passes combine this with object transforms; the host
    /// `view_transform` is **not** multiplied again for stereo world draws (see `world_mesh_forward`).
    pub stereo_view_proj: Option<(Mat4, Mat4)>,
    /// Legacy Unity `HeadOutput.transform` in renderer world space.
    ///
    /// Overlay render spaces are positioned relative to this transform each frame
    /// (`RenderingManager.HandleFrameUpdate -> RenderSpace.UpdateOverlayPositioning`).
    pub head_output_transform: Mat4,
}

impl Default for HostCameraFrame {
    fn default() -> Self {
        Self {
            frame_index: -1,
            near_clip: 0.01,
            far_clip: 10_000.0,
            desktop_fov_degrees: 60.0,
            vr_active: false,
            output_device: HeadOutputDevice::screen,
            primary_ortho_task: None,
            stereo_view_proj: None,
            head_output_transform: Mat4::IDENTITY,
        }
    }
}

/// Data passes need beyond raw GPU handles: host scene, backend pools, and main-surface formats.
pub struct FrameRenderParams<'a> {
    /// World caches and mesh renderables after [`SceneCoordinator::flush_world_caches`].
    pub scene: &'a SceneCoordinator,
    /// GPU pools, materials, and deform scratch buffers.
    pub backend: &'a mut RenderBackend,
    /// Depth attachment for the main forward pass.
    pub depth_view: &'a wgpu::TextureView,
    /// Swapchain / main color format.
    pub surface_format: wgpu::TextureFormat,
    /// Main surface extent in pixels (`width`, `height`) for projection.
    pub viewport_px: (u32, u32),
    /// Clip planes, FOV, and ortho task hint from the last host frame submission.
    pub host_camera: HostCameraFrame,
    /// When `true`, the forward pass targets 2-layer array attachments and may use multiview.
    pub multiview_stereo: bool,
}
