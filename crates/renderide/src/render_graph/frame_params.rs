//! Per-frame parameters shared across render graph passes (scene, backend, surface state).

use glam::{Mat4, Vec3};

use crate::backend::RenderBackend;
use crate::materials::MaterialPipelineDesc;
use crate::pipelines::ShaderPermutation;
use crate::scene::SceneCoordinator;
use crate::shared::HeadOutputDevice;

use super::world_mesh_draw_prep::{
    CameraTransformDrawFilter, WorldMeshDrawCollection, WorldMeshDrawItem,
};
use super::OutputDepthMode;

/// Identifies which Hi-Z / occlusion slot a view uses (main vs host render texture).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OcclusionViewId {
    /// Main window or OpenXR multiview (shared Hi-Z GPU state in [`crate::backend::OcclusionSystem`]).
    Main,
    /// Secondary camera writing to a host render texture (per-RT Hi-Z state).
    OffscreenRenderTexture(i32),
}

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
    /// Per-eye **view** matrices (world-to-view, with handedness fix applied) when stereo is active.
    ///
    /// Populated alongside [`Self::stereo_view_proj`] so the clustered lighting compute pass can
    /// decompose view and projection per eye without re-deriving from HMD poses.
    pub stereo_views: Option<(Mat4, Mat4)>,
    /// Legacy Unity `HeadOutput.transform` in renderer world space.
    ///
    /// Overlay render spaces are positioned relative to this transform each frame
    /// (`RenderingManager.HandleFrameUpdate -> RenderSpace.UpdateOverlayPositioning`).
    pub head_output_transform: Mat4,
    /// When set, non-VR mesh forward uses this world-to-view instead of the main render-space view.
    ///
    /// Used for secondary (render-texture) cameras so [`super::passes::world_mesh_forward::vp::compute_per_draw_vp_triple`]
    /// matches the offscreen projection, and so CPU frustum and Hi-Z temporal culling
    /// ([`super::world_mesh_cull`]) use the same world-to-view as the depth pyramid author pass.
    pub secondary_camera_world_to_view: Option<Mat4>,
    /// Optional override for cluster + forward projection (reverse-Z perspective or ortho).
    ///
    /// When both [`Self::cluster_view_override`] and [`Self::cluster_proj_override`] are set,
    /// [`super::cluster_frame::cluster_frame_params`] uses them instead of the desktop main-space camera.
    pub cluster_view_override: Option<Mat4>,
    /// Optional override projection for clustered light assignment (reverse-Z).
    pub cluster_proj_override: Option<Mat4>,
    /// World position for `@group(0)` camera uniforms when the secondary camera is active.
    pub secondary_camera_world_position: Option<Vec3>,
    /// Skips Hi-Z temporal state and uses uncull or frustum-only paths for secondary RT passes.
    pub suppress_occlusion_temporal: bool,
}

impl Default for HostCameraFrame {
    fn default() -> Self {
        Self {
            frame_index: -1,
            near_clip: 0.01,
            far_clip: 10_000.0,
            desktop_fov_degrees: 60.0,
            vr_active: false,
            output_device: HeadOutputDevice::Screen,
            primary_ortho_task: None,
            stereo_view_proj: None,
            stereo_views: None,
            head_output_transform: Mat4::IDENTITY,
            secondary_camera_world_to_view: None,
            cluster_view_override: None,
            cluster_proj_override: None,
            secondary_camera_world_position: None,
            suppress_occlusion_temporal: false,
        }
    }
}

/// Pipeline state resolved during world-mesh forward preparation.
pub struct WorldMeshForwardPipelineState {
    /// Whether this view records multiview raster passes.
    pub use_multiview: bool,
    /// Material pipeline descriptor for this view's color/depth/sample state.
    pub pass_desc: MaterialPipelineDesc,
    /// Shader permutation used by material pipeline lookup.
    pub shader_perm: ShaderPermutation,
}

/// Per-view forward-pass preparation shared by future split graph nodes.
pub struct PreparedWorldMeshForwardFrame {
    /// Sorted world mesh draw items for this view.
    pub draws: Vec<WorldMeshDrawItem>,
    /// Draw indices that can be recorded in the opaque forward pass.
    pub regular_indices: Vec<usize>,
    /// Draw indices that need the post-depth-snapshot intersection pass.
    pub intersect_indices: Vec<usize>,
    /// Pipeline format/sample/multiview state.
    pub pipeline: WorldMeshForwardPipelineState,
    /// Whether indexed draws may use base instance.
    pub supports_base_instance: bool,
    /// Whether the opaque/clear forward subpass was already recorded by a split graph node.
    pub opaque_recorded: bool,
    /// Whether the scene-depth snapshot for intersection draws was already recorded by a split graph node.
    pub depth_snapshot_recorded: bool,
    /// Whether the intersection/color-resolve tail raster was already recorded by a split graph node.
    pub tail_raster_recorded: bool,
}

/// Data passes need beyond raw GPU handles: host scene, backend pools, and main-surface formats.
pub struct FrameRenderParams<'a> {
    /// World caches and mesh renderables after [`SceneCoordinator::flush_world_caches`].
    pub scene: &'a SceneCoordinator,
    /// GPU pools, materials, and deform scratch buffers.
    pub backend: &'a mut RenderBackend,
    /// Backing depth texture for the main forward pass (copy source for scene-depth snapshots).
    pub depth_texture: &'a wgpu::Texture,
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
    /// Optional transform filter for secondary cameras (selective / exclude lists).
    pub transform_draw_filter: Option<CameraTransformDrawFilter>,
    /// When rendering a secondary camera to a host [`crate::resources::GpuRenderTexture`], the asset id
    /// of the **color target** being written. Materials must not sample that same render texture in the
    /// same pass (wgpu forbids `TEXTURE_BINDING` + `RENDER_ATTACHMENT` on one subresource); embedded
    /// bind resolves fall back to a white placeholder for this id.
    pub offscreen_write_render_texture_asset_id: Option<i32>,
    /// When set (e.g. secondary RT cameras), [`crate::render_graph::passes::WorldMeshForwardPass`] skips
    /// draw collection and uses this list instead.
    pub prefetched_world_mesh_draws: Option<WorldMeshDrawCollection>,
    /// Prepared forward state for multi-node world-mesh forward execution.
    pub prepared_world_mesh_forward: Option<PreparedWorldMeshForwardFrame>,
    /// Which Hi-Z pyramid / temporal slot this view reads and writes.
    pub occlusion_view: OcclusionViewId,
    /// Effective raster sample count for mesh forward (1 = off). Clamped to the GPU max for this view.
    pub sample_count: u32,
}

impl<'a> FrameRenderParams<'a> {
    /// Output depth layout for Hi-Z and occlusion ([`OutputDepthMode::from_multiview_stereo`]).
    pub fn output_depth_mode(&self) -> OutputDepthMode {
        OutputDepthMode::from_multiview_stereo(self.multiview_stereo)
    }
}
